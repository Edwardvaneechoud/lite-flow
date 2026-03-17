"""Full integration test: 100k-row CSV through the entire Floe pipeline.

Generates synthetic data, writes to CSV, then exercises:
  - lazy read_csv (schema without materialization)
  - filter, select, with_column, sort, group_by, join, window, head
  - display() without materializing
  - collect() and materialization lifecycle
  - streaming write back to CSV
  - optimizer (filter pushdown, column pruning)
  - performance timing for every operation
"""
import csv
import os
import random
import time

import pytest

from pyfloe import (
    col,
    lit,
    read_csv,
    row_number,
    when,
)

N_ROWS = 1_000_00
REGIONS = ["EU", "US", "APAC", "LATAM", "MEA"]
PRODUCTS = [f"Product_{chr(65 + i)}" for i in range(20)]  # Product_A .. Product_T
SEGMENTS = ["Enterprise", "SMB", "Startup", "Government"]

# ── helpers ──────────────────────────────────────────────────

def _timer(label):
    """Context manager that prints elapsed time."""
    class Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self.t0
            print(f"  {label}: {self.elapsed:.4f}s")
    return Timer()


def generate_csv(path, n=N_ROWS):
    """Write a synthetic CSV with n rows."""
    rng = random.Random(42)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "customer_id", "product", "amount", "region", "segment"])
        for i in range(1, n + 1):
            w.writerow([
                i,
                rng.randint(1, 5000),
                rng.choice(PRODUCTS),
                round(rng.uniform(1.0, 1000.0), 2),
                rng.choice(REGIONS),
                rng.choice(SEGMENTS),
            ])


def generate_customers_csv(path, n_customers=5000):
    """Write a small customers dimension table."""
    rng = random.Random(99)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "name", "tier"])
        for cid in range(1, n_customers + 1):
            w.writerow([cid, f"Customer_{cid}", rng.choice(["Gold", "Silver", "Bronze"])])


# ── tests ────────────────────────────────────────────────────

def test_full_100k_integration(tmp_path):
    """End-to-end: generate → read → transform → aggregate → write → verify."""

    orders_csv = str(tmp_path / "orders_100k.csv")
    customers_csv = str(tmp_path / "customers.csv")
    output_csv = str(tmp_path / "result.csv")
    # ── 1. Generate test data ────────────────────────────────
    print("\n=== 100k Integration Test ===\n")

    with _timer("Generate 100k orders CSV"):
        generate_csv(orders_csv)
    with _timer("Generate 5k customers CSV"):
        generate_customers_csv(customers_csv)

    assert os.path.exists(orders_csv)
    assert os.path.exists(customers_csv)

    # ── 2. Lazy read — schema without materialization ────────
    print()
    with _timer("read_csv (orders, schema only)"):
        orders = read_csv(orders_csv)
        schema = orders.schema
        group = orders.group_by("product").agg(col("amount").sum())
    non_optimized = group.filter(col("product") == lit("Product_A"))
    print(non_optimized.explain(optimized=False))
    with _timer("aggregate (group by product)"):
        non_optimized.display(optimize=False)

    with _timer("aggregate (group by product)"):
        non_optimized.display(optimize=True)
    print(non_optimized.explain(optimized=True))
    assert not orders.is_materialized, "read_csv should not materialize"
    assert schema.column_names == ["order_id", "customer_id", "product", "amount", "region", "segment"]
    assert schema.dtypes["order_id"] is int
    assert schema.dtypes["amount"] is float
    assert schema.dtypes["product"] is str
    print(f"  Schema: {schema._repr_short()}")

    with _timer("read_csv (customers, schema only)"):
        customers = read_csv(customers_csv)
        _ = customers.schema
    assert not customers.is_materialized

    # ── 3. display() without materializing ───────────────────
    print()
    with _timer("display(n=5) on lazy orders"):
        orders.display(n=5)
    assert not orders.is_materialized, "display() must not materialize"

    # ── 4. Filter ────────────────────────────────────────────
    print()
    with _timer("Filter amount > 500 + collect"):
        big_orders = orders.filter(col("amount") > 500).collect()
    assert big_orders.is_materialized
    n_big = len(big_orders)
    print(f"  {n_big} orders with amount > 500 (out of {N_ROWS})")
    assert 0 < n_big < N_ROWS

    # ── 5. Select + with_column ──────────────────────────────
    print()
    with _timer("Select + with_column (tax, size) + collect"):
        enriched = (
            orders
            .select("order_id", "amount", "region")
            .with_column("tax", col("amount") * 0.2)
            .with_column("size",
                when(col("amount") > 500, "large")
                .when(col("amount") > 100, "medium")
                .otherwise("small")
            ).collect()

        )
    assert enriched.columns == ["order_id", "amount", "region", "tax", "size"]
    sample = enriched[0]
    assert abs(sample["tax"] - sample["amount"] * 0.2) < 0.001
    print(f"  Enriched: {len(enriched)} rows, cols={enriched.columns}")

    # ── 6. Group-by aggregation ──────────────────────────────
    print()
    with _timer("Group by region → sum, count, mean, min, max"):
        region_stats = (
            orders
            .group_by("region")
            .agg(
                col("amount").sum().alias("total"),
                col("order_id").count().alias("n_orders"),
                col("amount").mean().alias("avg_amount"),
                col("amount").min().alias("min_amount"),
                col("amount").max().alias("max_amount"),
            )
            .sort("region")
            .collect()
        )
    assert len(region_stats) == len(REGIONS)
    region_stats.display()
    total_orders = sum(r["n_orders"] for r in region_stats)
    assert total_orders == N_ROWS, f"Expected {N_ROWS} total, got {total_orders}"

    # ── 7. Multi-key group-by ────────────────────────────────
    print()
    with _timer("Group by (region, segment) → sum"):
        multi = (
            orders
            .group_by("region", "segment")
            .agg(col("amount").sum().alias("total"))
            .sort("region", "segment")
            .collect()
        )
    assert len(multi) == len(REGIONS) * len(SEGMENTS)
    print(f"  {len(multi)} groups")

    # ── 8. Join (hash) ───────────────────────────────────────
    print()
    with _timer("Hash join orders ⋈ customers on customer_id"):
        joined = orders.join(customers, on="customer_id", how="inner").collect()
    assert "name" in joined.columns
    assert "tier" in joined.columns
    assert len(joined) == N_ROWS  # all customers exist
    print(f"  Joined: {len(joined)} rows, cols={joined.columns}")

    # ── 9. Sort ──────────────────────────────────────────────
    print()
    with _timer("Sort by amount descending"):
        sorted_ff = orders.sort("amount", ascending=False).collect()
    amounts = [r["amount"] for r in sorted_ff]
    assert amounts == sorted(amounts, reverse=True)
    print(f"  Top 3 amounts: {amounts[:3]}")

    # ── 10. Window function: row_number per region ───────────
    print()
    with _timer("Window: row_number() over(partition_by=region, order_by=amount)"):
        windowed = (
            orders
            .select("order_id", "amount", "region")
            .with_column("rn",
                row_number().over(partition_by="region", order_by="amount")
            )
            .collect()
        )
    assert "rn" in windowed.columns
    # Check that row numbers within each region start at 1
    by_region = {}
    for r in windowed:
        by_region.setdefault(r["region"], []).append(r["rn"])
    for reg, rns in by_region.items():
        assert min(rns) == 1, f"Region {reg}: min rn should be 1"
    print(f"  Windowed: {len(windowed)} rows")

    # ── 11. Head (early termination) ─────────────────────────
    print()
    with _timer("head(10) on lazy pipeline"):
        top10 = orders.filter(col("amount") > 100).head(10)
    assert len(top10) == 10
    assert not orders.is_materialized, "head() should not materialize original"

    # ── 12. Chained pipeline (lazy until terminal) ───────────
    print()
    pipeline = (
        orders
        .filter(col("region") == "EU")
        .select("order_id", "amount", "product", "region")
        .with_column("discounted", col("amount") * 0.9)
        .sort("amount", ascending=False)
    )
    assert not pipeline.is_materialized
    _ = pipeline.schema
    assert not pipeline.is_materialized, "schema should not materialize"

    with _timer("Chained pipeline (filter→select→with_column→sort) collect"):
        pipeline.collect()
    assert pipeline.is_materialized
    print(f"  EU orders: {len(pipeline)} rows")
    assert all(r["region"] == "EU" for r in pipeline)

    # ── 13. Optimizer ────────────────────────────────────────
    print()
    heavy = (
        orders
        .join(customers, on="customer_id")
        .filter(col("region") == "EU")
        .select("order_id", "name", "amount", "region")
    )
    unopt_plan = heavy.explain()
    opt_plan = heavy.explain(optimized=True)
    print("  Unoptimized plan (first 3 lines):")
    for line in unopt_plan.split("\n")[:3]:
        print(f"    {line}")
    print("  Optimized plan (first 3 lines):")
    for line in opt_plan.split("\n")[:3]:
        print(f"    {line}")

    with _timer("Optimized pipeline collect"):
        opt_result = heavy.optimize().collect()
    with _timer("Unoptimized pipeline collect"):
        unopt_result = heavy.collect()
    assert len(opt_result) == len(unopt_result)
    print(f"  Both produce {len(opt_result)} rows")

    # ── 14. Write result to CSV (streaming) ──────────────────
    print()
    with _timer("Write region_stats to CSV"):
        region_stats.to_csv(output_csv)

    assert os.path.exists(output_csv)
    with open(output_csv) as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    assert len(rows) == len(REGIONS)
    print(f"  Wrote {len(rows)} rows to {os.path.basename(output_csv)}")

    # ── 15. Repr lifecycle ───────────────────────────────────
    print()
    lazy = orders.filter(col("amount") > 999)
    r = repr(lazy)
    assert "lazy" in r, f"Lazy repr should contain 'lazy': {r}"
    lazy.collect()
    r2 = repr(lazy)
    assert "materialized" in r2, f"Collected repr should contain 'materialized': {r2}"
    print(f"  Lazy repr:        {r.splitlines()[0]}")
    print(f"  Materialized repr: {r2.splitlines()[0]}")

    # ── 16. display() on collected data ──────────────────────
    print()
    print("  display() on region_stats:")
    region_stats.display()

    print("\n=== All 100k integration checks passed ===\n")


if __name__ == "__main__":
    pytest.main([__file__])