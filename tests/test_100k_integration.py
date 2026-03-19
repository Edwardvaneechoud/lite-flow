"""Integration tests: 100k-row CSV through the full pyfloe pipeline.

Generates synthetic data once per session, then exercises every major feature
in isolated test functions so failures are independent and easy to diagnose.
"""

import csv
import os
import random

import pytest

from pyfloe import (
    LazyFrame,
    Stream,
    col,
    from_iter,
    lit,
    read_csv,
    row_number,
    when,
)

N_ROWS = 100_000
REGIONS = ["EU", "US", "APAC", "LATAM", "MEA"]
PRODUCTS = [f"Product_{chr(65 + i)}" for i in range(20)]  # Product_A .. Product_T
SEGMENTS = ["Enterprise", "SMB", "Startup", "Government"]


# ── fixtures ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("integration")


@pytest.fixture(scope="module")
def orders_csv(data_dir):
    path = str(data_dir / "orders.csv")
    rng = random.Random(42)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "customer_id", "product", "amount", "region", "segment"])
        for i in range(1, N_ROWS + 1):
            w.writerow(
                [
                    i,
                    rng.randint(1, 5000),
                    rng.choice(PRODUCTS),
                    round(rng.uniform(1.0, 1000.0), 2),
                    rng.choice(REGIONS),
                    rng.choice(SEGMENTS),
                ]
            )
    return path


@pytest.fixture(scope="module")
def customers_csv(data_dir):
    path = str(data_dir / "customers.csv")
    rng = random.Random(99)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "name", "tier"])
        for cid in range(1, 5001):
            w.writerow([cid, f"Customer_{cid}", rng.choice(["Gold", "Silver", "Bronze"])])
    return path


@pytest.fixture()
def orders(orders_csv):
    """Fresh lazy LazyFrame from orders CSV (not shared across tests)."""
    return read_csv(orders_csv)


@pytest.fixture()
def customers(customers_csv):
    """Fresh lazy LazyFrame from customers CSV."""
    return read_csv(customers_csv)


# ── schema & laziness ───────────────────────────────────────


class TestSchemaAndLaziness:
    def test_read_csv_is_lazy(self, orders):
        _ = orders.schema
        assert not orders.is_materialized

    def test_schema_columns(self, orders):
        schema = orders.schema
        assert schema.column_names == [
            "order_id",
            "customer_id",
            "product",
            "amount",
            "region",
            "segment",
        ]

    def test_schema_dtypes(self, orders):
        dtypes = orders.schema.dtypes
        assert dtypes["order_id"] is int
        assert dtypes["customer_id"] is int
        assert dtypes["amount"] is float
        assert dtypes["product"] is str
        assert dtypes["region"] is str
        assert dtypes["segment"] is str

    def test_display_does_not_materialize(self, orders):
        orders.display(n=5)
        assert not orders.is_materialized


# ── filter ──────────────────────────────────────────────────


class TestFilter:
    def test_filter_reduces_rows(self, orders):
        big = orders.filter(col("amount") > 500).collect()
        assert 0 < len(big) < N_ROWS

    def test_filter_values_are_correct(self, orders):
        big = orders.filter(col("amount") > 500).collect()
        assert all(r["amount"] > 500 for r in big)

    def test_filter_with_equality(self, orders):
        eu = orders.filter(col("region") == lit("EU")).collect()
        assert len(eu) > 0
        assert all(r["region"] == "EU" for r in eu)

    def test_chained_filters(self, orders):
        result = orders.filter(col("region") == lit("US")).filter(col("amount") > 200).collect()
        assert all(r["region"] == "US" and r["amount"] > 200 for r in result)

    def test_filter_matches_nothing(self, orders):
        empty = orders.filter(col("amount") > 99999).collect()
        assert len(empty) == 0


# ── select & with_column ────────────────────────────────────


class TestSelectAndWithColumn:
    def test_select_keeps_only_chosen_columns(self, orders):
        selected = orders.select("order_id", "amount").collect()
        assert selected.columns == ["order_id", "amount"]
        assert len(selected) == N_ROWS

    def test_with_column_arithmetic(self, orders):
        enriched = (
            orders.select("order_id", "amount").with_column("tax", col("amount") * 0.2).collect()
        )
        assert "tax" in enriched.columns
        for r in enriched.head(100):
            assert abs(r["tax"] - r["amount"] * 0.2) < 0.01

    def test_with_column_when_expression(self, orders):
        result = (
            orders.select("order_id", "amount")
            .with_column(
                "size",
                when(col("amount") > 500, "large")
                .when(col("amount") > 100, "medium")
                .otherwise("small"),
            )
            .collect()
        )
        for r in result:
            if r["amount"] > 500:
                assert r["size"] == "large"
            elif r["amount"] > 100:
                assert r["size"] == "medium"
            else:
                assert r["size"] == "small"


# ── group_by / aggregation ──────────────────────────────────


class TestGroupBy:
    def test_group_by_region_count(self, orders):
        stats = orders.group_by("region").agg(col("order_id").count().alias("n")).collect()
        assert len(stats) == len(REGIONS)
        total = sum(r["n"] for r in stats)
        assert total == N_ROWS

    def test_group_by_region_aggregations(self, orders):
        stats = (
            orders.group_by("region")
            .agg(
                col("amount").sum().alias("total"),
                col("amount").mean().alias("avg"),
                col("amount").min().alias("lo"),
                col("amount").max().alias("hi"),
            )
            .sort("region")
            .collect()
        )
        for r in stats:
            assert r["total"] > 0
            assert r["lo"] <= r["avg"] <= r["hi"]
            assert r["lo"] >= 1.0
            assert r["hi"] <= 1000.0

    def test_multi_key_group_by(self, orders):
        multi = (
            orders.group_by("region", "segment").agg(col("amount").sum().alias("total")).collect()
        )
        assert len(multi) == len(REGIONS) * len(SEGMENTS)

    def test_group_by_then_filter(self, orders):
        """Filter after aggregation (pushdown candidate)."""
        result = (
            orders.group_by("product")
            .agg(col("amount").sum().alias("total"))
            .filter(col("product") == lit("Product_A"))
            .collect()
        )
        assert len(result) == 1
        assert result[0]["product"] == "Product_A"
        assert result[0]["total"] > 0


# ── join ────────────────────────────────────────────────────


class TestJoin:
    def test_inner_join_all_matched(self, orders, customers):
        joined = orders.join(customers, on="customer_id", how="inner").collect()
        assert len(joined) == N_ROWS
        assert "name" in joined.columns
        assert "tier" in joined.columns

    def test_join_values_correct(self, orders, customers):
        sample = orders.head(100)
        joined = sample.join(customers.collect(), on="customer_id", how="inner").collect()
        for r in joined:
            assert r["name"] == f"Customer_{r['customer_id']}"

    def test_left_join_preserves_all_left_rows(self, orders, customers):
        joined = orders.join(customers, on="customer_id", how="left").collect()
        assert len(joined) == N_ROWS


# ── sort ────────────────────────────────────────────────────


class TestSort:
    def test_sort_descending(self, orders):
        result = orders.sort("amount", ascending=False).collect()
        amounts = [r["amount"] for r in result]
        assert amounts == sorted(amounts, reverse=True)

    def test_sort_ascending(self, orders):
        result = orders.select("order_id", "amount").sort("amount", ascending=True).collect()
        amounts = [r["amount"] for r in result]
        assert amounts == sorted(amounts)


# ── window function ─────────────────────────────────────────


class TestWindow:
    def test_row_number_starts_at_1(self, orders):
        windowed = (
            orders.select("order_id", "amount", "region")
            .with_column(
                "rn",
                row_number().over(partition_by="region", order_by="amount"),
            )
            .collect()
        )
        by_region: dict[str, list[int]] = {}
        for r in windowed:
            by_region.setdefault(r["region"], []).append(r["rn"])
        for region, rns in by_region.items():
            assert min(rns) == 1, f"Region {region}: min rn should be 1"
            assert max(rns) == len(rns), f"Region {region}: max rn should equal count"

    def test_row_number_is_contiguous(self, orders):
        windowed = (
            orders.select("order_id", "amount", "region")
            .with_column(
                "rn",
                row_number().over(partition_by="region", order_by="amount"),
            )
            .collect()
        )
        by_region: dict[str, list[int]] = {}
        for r in windowed:
            by_region.setdefault(r["region"], []).append(r["rn"])
        for region, rns in by_region.items():
            assert sorted(rns) == list(range(1, len(rns) + 1))


# ── head (early termination) ───────────────────────────────


class TestHead:
    def test_head_returns_n_rows(self, orders):
        top = orders.filter(col("amount") > 100).head(10)
        assert len(top) == 10

    def test_head_does_not_materialize_source(self, orders):
        _ = orders.head(5)
        assert not orders.is_materialized


# ── chained pipeline ───────────────────────────────────────


class TestChainedPipeline:
    def test_lazy_until_collect(self, orders):
        pipeline = (
            orders.filter(col("region") == lit("EU"))
            .select("order_id", "amount", "product", "region")
            .with_column("discounted", col("amount") * 0.9)
            .sort("amount", ascending=False)
        )
        assert not pipeline.is_materialized
        _ = pipeline.schema
        assert not pipeline.is_materialized

    def test_chained_pipeline_correctness(self, orders):
        result = (
            orders.filter(col("region") == lit("EU"))
            .select("order_id", "amount", "product", "region")
            .with_column("discounted", col("amount") * 0.9)
            .sort("amount", ascending=False)
            .collect()
        )
        assert result.columns == ["order_id", "amount", "product", "region", "discounted"]
        assert len(result) > 0
        assert all(r["region"] == "EU" for r in result)
        for r in result.head(50):
            assert abs(r["discounted"] - r["amount"] * 0.9) < 0.01
        amounts = [r["amount"] for r in result]
        assert amounts == sorted(amounts, reverse=True)


# ── optimizer ───────────────────────────────────────────────


class TestOptimizer:
    def test_optimized_and_unoptimized_same_length(self, orders, customers):
        pipeline = (
            orders.join(customers, on="customer_id")
            .filter(col("region") == lit("EU"))
            .select("order_id", "name", "amount", "region")
        )
        opt_result = pipeline.optimize().collect()
        unopt_result = pipeline.collect()
        assert len(opt_result) == len(unopt_result)
        assert opt_result.columns == unopt_result.columns

    def test_optimized_and_unoptimized_same_values(self, orders_csv, customers_csv):
        """Use fresh LazyFrames to ensure independent execution paths."""

        def build_pipeline():
            return (
                read_csv(orders_csv)
                .join(read_csv(customers_csv), on="customer_id")
                .filter(col("region") == lit("EU"))
                .select("order_id", "name", "amount", "region")
                .sort("order_id")
            )

        opt_result = build_pipeline().optimize().collect()
        unopt_result = build_pipeline().collect()
        assert len(opt_result) == len(unopt_result)
        for opt_row, unopt_row in zip(opt_result, unopt_result):
            assert opt_row == unopt_row

    def test_explain_plans_exist(self, orders, customers):
        pipeline = (
            orders.join(customers, on="customer_id")
            .filter(col("region") == lit("EU"))
            .select("order_id", "name", "amount", "region")
        )
        unopt = pipeline.explain(optimized=False)
        opt = pipeline.explain(optimized=True)
        assert isinstance(unopt, str) and len(unopt) > 0
        assert isinstance(opt, str) and len(opt) > 0

    def test_filter_pushdown_correctness(self, orders_csv):
        """Filter pushdown through aggregation should produce identical results."""

        def build():
            return (
                read_csv(orders_csv)
                .group_by("product")
                .agg(col("amount").sum().alias("total"))
                .filter(col("product") == lit("Product_A"))
            )

        result_opt = build().optimize().collect()
        result_unopt = build().collect()
        assert len(result_opt) == 1
        assert len(result_unopt) == 1
        assert result_opt[0]["total"] == result_unopt[0]["total"]


# ── repr lifecycle ──────────────────────────────────────────


class TestReprLifecycle:
    def test_lazy_repr(self, orders):
        lazy = orders.filter(col("amount") > 999)
        r = repr(lazy)
        assert "lazy" in r.lower() or "LazyFrame" in r

    def test_materialized_repr(self, orders):
        flow = orders.filter(col("amount") > 999)
        flow.collect()
        r = repr(flow)
        assert "materialized" in r.lower() or "rows" in r.lower()


# ── CSV round-trip ──────────────────────────────────────────


class TestCsvRoundTrip:
    def test_write_and_read_back(self, orders, data_dir):
        output = str(data_dir / "roundtrip.csv")
        stats = (
            orders.group_by("region")
            .agg(
                col("amount").sum().alias("total"),
                col("order_id").count().alias("n_orders"),
            )
            .sort("region")
            .collect()
        )
        stats.to_csv(output)
        assert os.path.exists(output)

        reloaded = read_csv(output).collect()
        assert len(reloaded) == len(REGIONS)
        assert set(r["region"] for r in reloaded) == set(REGIONS)
        total = sum(r["n_orders"] for r in reloaded)
        assert total == N_ROWS


# ── streaming (from_iter / Stream) ──────────────────────────


class TestStreaming:
    def test_from_iter_basic(self):
        data = [{"x": i, "y": i * 2} for i in range(1000)]
        result = from_iter(iter(data), columns=["x", "y"]).collect()
        assert len(result) == 1000
        assert result[0]["y"] == 0

    def test_stream_filter_and_collect(self, orders_csv):
        result = (
            Stream.from_csv(orders_csv)
            .filter(col("amount") > 900)
            .select("order_id", "amount")
            .collect()
        )
        assert len(result) > 0
        assert all(r["amount"] > 900 for r in result)

    def test_stream_to_csv(self, orders_csv, data_dir):
        output = str(data_dir / "stream_out.csv")
        (
            Stream.from_csv(orders_csv)
            .filter(col("region") == lit("EU"))
            .select("order_id", "amount", "region")
            .to_csv(output)
        )
        assert os.path.exists(output)
        reloaded = read_csv(output).collect()
        assert len(reloaded) > 0
        assert all(r["region"] == "EU" for r in reloaded)


# ── edge cases ──────────────────────────────────────────────


class TestEdgeCases:
    def test_group_by_single_group(self):
        data = [{"group": "A", "val": i} for i in range(100)]
        result = LazyFrame(data).group_by("group").agg(col("val").sum().alias("total")).collect()
        assert len(result) == 1
        assert result[0]["total"] == sum(range(100))

    def test_multiple_collects_are_idempotent(self, orders):
        flow = orders.filter(col("amount") > 500)
        flow.collect()
        n1 = len(flow)
        flow.collect()
        n2 = len(flow)
        assert n1 == n2
