"""Comprehensive tests for the floe package."""

from pyfloe import (
    ColumnSchema,
    LazyFrame,
    LazySchema,
    TypedLazyFrame,
    col,
    dense_rank,
    rank,
    row_number,
    when,
)

ORDERS = [
    {"order_id": 1, "customer_id": 101, "product": "Widget A", "amount": 250.0, "region": "EU"},
    {"order_id": 2, "customer_id": 102, "product": "Widget B", "amount": 75.5, "region": "US"},
    {"order_id": 3, "customer_id": 101, "product": "Widget C", "amount": 180.0, "region": "EU"},
    {"order_id": 4, "customer_id": 103, "product": "Widget A", "amount": 320.0, "region": "US"},
    {"order_id": 5, "customer_id": 104, "product": "Widget B", "amount": 45.0, "region": "EU"},
    {"order_id": 6, "customer_id": 102, "product": "Widget C", "amount": 510.0, "region": "US"},
]
CUSTOMERS = [
    {"customer_id": 101, "name": "Acme Corp", "segment": "Enterprise"},
    {"customer_id": 102, "name": "Bob's Shop", "segment": "SMB"},
    {"customer_id": 103, "name": "MegaCo", "segment": "Enterprise"},
    {"customer_id": 104, "name": "Tiny Ltd", "segment": "SMB"},
]


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

customers = (c for c in CUSTOMERS)


def test_schema_without_materialization():

    lf = LazyFrame(CUSTOMERS)
    s = lf.schema
    assert (
        not lf.is_materialized
    )  # schema from ScanNode doesn't need materialization — data is already in ScanNode
    assert s.column_names == ["customer_id", "name", "segment"]
    assert s.dtypes["customer_id"] is int
    assert s.dtypes["name"] is str
    assert s.dtypes["segment"] is str


def test_schema_propagates_through_select():
    lf = LazyFrame(ORDERS).select("order_id", "amount")
    s = lf.schema
    assert s.column_names == ["order_id", "amount"]
    assert s.dtypes["order_id"] is int
    assert not lf.is_materialized


def test_schema_propagates_through_filter():
    lf = LazyFrame(ORDERS).filter(col("amount") > 100)
    s = lf.schema
    assert s.column_names == ["order_id", "customer_id", "product", "amount", "region"]
    assert not lf.is_materialized


def test_schema_propagates_through_with_column():
    lf = LazyFrame(ORDERS).with_column("double_amt", col("amount") * 2)
    s = lf.schema
    assert "double_amt" in s.column_names
    assert s.dtypes["double_amt"] is float
    assert not lf.is_materialized


def test_schema_propagates_through_join():
    o = LazyFrame(ORDERS)
    c = LazyFrame(CUSTOMERS)
    joined = o.join(c, on="customer_id")
    s = joined.schema
    assert "name" in s.column_names
    assert "segment" in s.column_names
    assert "right_customer_id" in s.column_names
    assert not joined.is_materialized


def test_schema_propagates_through_group_by_agg():
    lf = (
        LazyFrame(ORDERS)
        .group_by("region")
        .agg(
            col("amount").sum().alias("total"),
            col("order_id").count().alias("n"),
        )
    )
    s = lf.schema
    assert s.column_names == ["region", "total", "n"]
    assert s.dtypes["total"] is float  # sum of float
    assert s.dtypes["n"] is int  # count
    assert not lf.is_materialized


def test_schema_propagates_through_rename():
    lf = LazyFrame(ORDERS).rename({"amount": "price", "region": "area"})
    s = lf.schema
    assert "price" in s.column_names
    assert "area" in s.column_names
    assert "amount" not in s.column_names
    assert not lf.is_materialized


def test_schema_propagates_through_sort():
    lf = LazyFrame(ORDERS).sort("amount")
    assert lf.schema.column_names == LazyFrame(ORDERS).columns
    assert not lf.is_materialized


def test_schema_through_chained_ops():
    pipeline = (
        LazyFrame(ORDERS)
        .select("order_id", "amount", "region")
        .filter(col("amount") > 100)
        .with_column("tax", col("amount") * 0.2)
        .rename({"amount": "subtotal"})
    )
    s = pipeline.schema
    assert s.column_names == ["order_id", "subtotal", "region", "tax"]
    assert s.dtypes["tax"] is float
    assert not pipeline.is_materialized


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════


def test_col_lit_filter():
    result = LazyFrame(ORDERS).filter(col("amount") > 100).to_pylist()
    assert len(result) == 4
    assert all(r["amount"] > 100 for r in result)


def test_compound_filter_with_and():
    result = LazyFrame(ORDERS).filter((col("region") == "US") & (col("amount") > 100)).to_pylist()
    assert len(result) == 2
    assert all(r["region"] == "US" and r["amount"] > 100 for r in result)


def test_arithmetic_expressions():
    result = LazyFrame(ORDERS).with_column("double", col("amount") * 2).to_pylist()
    for r in result:
        assert r["double"] == r["amount"] * 2


def test_cast_expression():
    result = LazyFrame(ORDERS).with_column("amt_str", col("amount").cast(str)).to_pylist()
    assert result[0]["amt_str"] == "250.0"


def test_is_null_is_not_null():
    data = [{"x": 1, "y": None}, {"x": 2, "y": "hello"}]
    result = LazyFrame(data).filter(col("y").is_not_null()).to_pylist()
    assert len(result) == 1
    assert result[0]["x"] == 2


def test_is_in():
    result = LazyFrame(ORDERS).filter(col("region").is_in(["EU"])).to_pylist()
    assert len(result) == 3


def test_when_otherwise():
    lf = LazyFrame(ORDERS).with_column(
        "size",
        when(col("amount") > 200, "large").when(col("amount") > 100, "medium").otherwise("small"),
    )
    result = lf.to_pylist()
    for r in result:
        if r["amount"] > 200:
            assert r["size"] == "large"
        elif r["amount"] > 100:
            assert r["size"] == "medium"
        else:
            assert r["size"] == "small"


def test_negation():
    result = LazyFrame(ORDERS).with_column("neg", -col("amount")).to_pylist()
    assert result[0]["neg"] == -250.0


def test_reverse_arithmetic_lit_col():
    result = LazyFrame(ORDERS).with_column("plus100", 100 + col("amount")).to_pylist()
    assert result[0]["plus100"] == 350.0


def test_expression_repr():
    e = (col("amount") > 100) & (col("region") == "US")
    r = repr(e)
    assert "amount" in r
    assert "region" in r


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════


def test_str_upper_str_lower():
    result = LazyFrame(ORDERS).with_column("upper", col("product").str.upper()).to_pylist()
    assert result[0]["upper"] == "WIDGET A"


def test_str_contains():
    result = LazyFrame(ORDERS).filter(col("product").str.contains("Widget A")).to_pylist()
    assert len(result) == 2


def test_str_startswith_str_endswith():
    r1 = LazyFrame(ORDERS).filter(col("product").str.startswith("Widget")).to_pylist()
    assert len(r1) == 6
    r2 = LazyFrame(ORDERS).filter(col("product").str.endswith("A")).to_pylist()
    assert len(r2) == 2


def test_str_replace():
    result = (
        LazyFrame(ORDERS)
        .with_column("renamed", col("product").str.replace("Widget", "Gadget"))
        .to_pylist()
    )
    assert result[0]["renamed"] == "Gadget A"


def test_str_len():
    result = LazyFrame(ORDERS).with_column("name_len", col("product").str.len()).to_pylist()
    assert result[0]["name_len"] == 8  # "Widget A"


def test_str_slice():
    result = LazyFrame(ORDERS).with_column("first3", col("product").str.slice(0, 3)).to_pylist()
    assert result[0]["first3"] == "Wid"


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════


def test_group_by_agg_with_sum_count_mean():
    result = (
        LazyFrame(ORDERS)
        .group_by("region")
        .agg(
            col("amount").sum().alias("total"),
            col("order_id").count().alias("n"),
            col("amount").mean().alias("avg"),
        )
        .sort("region")
        .to_pylist()
    )
    eu = [r for r in result if r["region"] == "EU"][0]
    us = [r for r in result if r["region"] == "US"][0]
    assert eu["total"] == 475.0
    assert eu["n"] == 3
    assert us["total"] == 905.5


def test_group_by_agg_with_min_max_n_unique():
    result = (
        LazyFrame(ORDERS)
        .group_by("region")
        .agg(
            col("amount").min().alias("min_amt"),
            col("amount").max().alias("max_amt"),
            col("product").n_unique().alias("uniq_products"),
        )
        .sort("region")
        .to_pylist()
    )
    eu = [r for r in result if r["region"] == "EU"][0]
    assert eu["min_amt"] == 45.0
    assert eu["max_amt"] == 250.0


def test_legacy_group_by_api():
    result = LazyFrame(ORDERS).group_by("region", agg_func=sum, on_cols="amount").to_pylist()
    assert len(result) == 2
    totals = {r["region"]: r["amount"] for r in result}
    assert totals["EU"] == 475.0
    assert totals["US"] == 905.5


def test_group_by_missing_column_raises():
    import pytest

    lf = LazyFrame(ORDERS).select("order_id", "amount")
    with pytest.raises(ValueError, match=r"group_by column\(s\)"):
        lf.group_by("region")


def test_group_by_agg_missing_column_raises():
    import pytest

    lf = LazyFrame(ORDERS).select("order_id")
    with pytest.raises(ValueError, match=r"references column\(s\)"):
        lf.group_by("order_id").agg(col("amount").sum())


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════


def test_row_number_over_partition():
    lf = LazyFrame(ORDERS).with_column(
        "rn", row_number().over(partition_by="region", order_by="amount")
    )
    result = lf.to_pylist()
    eu_rows = sorted([r for r in result if r["region"] == "EU"], key=lambda r: r["rn"])
    assert [r["rn"] for r in eu_rows] == [1, 2, 3]
    assert eu_rows[0]["amount"] == 45.0  # smallest first


def test_rank_with_ties():
    data = [
        {"name": "a", "score": 10, "group": "x"},
        {"name": "b", "score": 20, "group": "x"},
        {"name": "c", "score": 20, "group": "x"},
        {"name": "d", "score": 30, "group": "x"},
    ]
    result = (
        LazyFrame(data)
        .with_column("r", rank().over(partition_by="group", order_by="score"))
        .to_pylist()
    )
    ranks = {r["name"]: r["r"] for r in result}
    assert ranks["a"] == 1
    assert ranks["b"] == 2
    assert ranks["c"] == 2  # tie
    assert ranks["d"] == 4  # skip 3


def test_dense_rank():
    data = [
        {"name": "a", "score": 10, "g": "x"},
        {"name": "b", "score": 20, "g": "x"},
        {"name": "c", "score": 20, "g": "x"},
        {"name": "d", "score": 30, "g": "x"},
    ]
    result = (
        LazyFrame(data)
        .with_column("dr", dense_rank().over(partition_by="g", order_by="score"))
        .to_pylist()
    )
    ranks = {r["name"]: r["dr"] for r in result}
    assert ranks["d"] == 3  # no skip


def test_window_agg_sum_over_partition():
    result = (
        LazyFrame(ORDERS)
        .with_column("region_total", col("amount").sum().over(partition_by="region"))
        .to_pylist()
    )
    for r in result:
        if r["region"] == "EU":
            assert r["region_total"] == 475.0
        else:
            assert r["region_total"] == 905.5


def test_cumsum():
    data = [{"x": 1, "v": 10}, {"x": 2, "v": 20}, {"x": 3, "v": 30}]
    result = (
        LazyFrame(data).with_column("running", col("v").cumsum().over(order_by="x")).to_pylist()
    )
    vals = sorted(result, key=lambda r: r["x"])
    assert [r["running"] for r in vals] == [10, 30, 60]


def test_lag_lead():
    data = [{"x": 1, "v": 10}, {"x": 2, "v": 20}, {"x": 3, "v": 30}]
    result = (
        LazyFrame(data)
        .with_column("prev", col("v").lag(1, default=0).over(order_by="x"))
        .to_pylist()
    )
    vals = sorted(result, key=lambda r: r["x"])
    assert [r["prev"] for r in vals] == [0, 10, 20]

    result2 = (
        LazyFrame(data)
        .with_column("next_v", col("v").lead(1, default=-1).over(order_by="x"))
        .to_pylist()
    )
    vals2 = sorted(result2, key=lambda r: r["x"])
    assert [r["next_v"] for r in vals2] == [20, 30, -1]


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════


def test_explain_shows_plan_tree():
    plan = (
        LazyFrame(ORDERS).filter(col("amount") > 100).select("order_id", "amount").sort("amount")
    )
    text = plan.explain()
    assert "Filter" in text
    assert "Project" in text
    assert "Sort" in text
    assert "Scan" in text


def test_explain_with_join():
    o = LazyFrame(ORDERS)
    c = LazyFrame(CUSTOMERS)
    joined = o.join(c, on="customer_id").filter(col("segment") == "Enterprise")
    text = joined.explain()
    assert "Join" in text
    assert "Filter" in text


def test_optimized_explain_pushes_filter_down():
    o = LazyFrame(ORDERS)
    c = LazyFrame(CUSTOMERS)
    pipeline = (
        o.join(c, on="customer_id").filter(col("region") == "EU")  # only touches left side
    )
    unopt = pipeline.explain()
    opt = pipeline.explain(optimized=True)
    # In optimized plan, filter should appear INSIDE the join (below it)
    unopt_lines = unopt.strip().split("\n")
    opt_lines = opt.strip().split("\n")
    # Unoptimized: Filter is on top
    assert "Filter" in unopt_lines[0]
    # Optimized: Filter should be pushed below Join
    assert "Join" in opt_lines[0]


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════


def test_filter_pushdown_past_project():
    pipeline = LazyFrame(ORDERS).select("order_id", "amount", "region").filter(col("amount") > 100)
    opt = pipeline.optimize()
    plan_text = opt.explain()
    lines = plan_text.strip().split("\n")
    # Filter should be pushed below Project
    assert "Project" in lines[0]
    assert "Filter" in lines[1]


def test_filter_pushdown_into_join_left_side():
    o = LazyFrame(ORDERS)
    c = LazyFrame(CUSTOMERS)
    pipeline = o.join(c, on="customer_id").filter(col("region") == "EU")
    opt = pipeline.optimize()
    text = opt.explain()
    # Filter on "region" (left side) should be pushed into left branch
    lines = text.strip().split("\n")
    assert "Join" in lines[0]


def test_filter_pushdown_into_join_right_side():
    o = LazyFrame(ORDERS)
    c = LazyFrame(CUSTOMERS)
    pipeline = o.join(c, on="customer_id").filter(col("segment") == "Enterprise")
    opt = pipeline.optimize()
    text = opt.explain()
    lines = text.strip().split("\n")
    assert "Join" in lines[0]  # filter pushed inside


def test_optimized_execution_produces_correct_results():
    o = LazyFrame(ORDERS)
    c = LazyFrame(CUSTOMERS)
    pipeline = (
        o.join(c, on="customer_id")
        .filter(col("region") == "EU")
        .select("order_id", "name", "amount")
    )
    unopt_result = pipeline.to_pylist()
    opt_result = pipeline.optimize().to_pylist()
    # Same results (possibly different order)
    assert len(unopt_result) == len(opt_result)
    assert set(r["order_id"] for r in unopt_result) == set(r["order_id"] for r in opt_result)


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════


def test_typed_returns_typedfloe():
    from typing import TypedDict

    class Order(TypedDict):
        order_id: int
        amount: float
        region: str

    orders = LazyFrame(ORDERS).typed(Order)
    assert isinstance(orders, TypedLazyFrame)
    assert orders.row_type is Order


def test_typedfloe_preserves_type_through_filter():
    from typing import TypedDict

    class Order(TypedDict):
        order_id: int
        amount: float
        region: str

    orders = LazyFrame(ORDERS).typed(Order)
    filtered = orders.filter(col("amount") > 100)
    assert isinstance(filtered, TypedLazyFrame)
    assert filtered.row_type is Order


def test_validate_catches_schema_mismatch():
    from typing import TypedDict

    class WrongSchema(TypedDict):
        nonexistent: int
        amount: str  # wrong type

    try:
        LazyFrame(ORDERS).validate(WrongSchema)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "missing column" in str(e)


def test_validate_passes_for_correct_schema():
    from typing import TypedDict

    class Order(TypedDict):
        order_id: int
        amount: float

    LazyFrame(ORDERS).validate(Order)  # should not raise


def test_typedfloe_repr_shows_type_name():
    from typing import TypedDict

    class Order(TypedDict):
        order_id: int

    orders = LazyFrame(ORDERS).typed(Order)
    assert "TypedLazyFrame[Order]" in repr(orders)


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════


def test_select_columns():
    result = LazyFrame(ORDERS).select("order_id", "amount").to_pylist()
    assert set(result[0].keys()) == {"order_id", "amount"}
    assert len(result) == 6


def test_drop_columns():
    result = LazyFrame(ORDERS).drop("customer_id", "product").columns
    assert "customer_id" not in result
    assert "product" not in result


def test_sort_ascending_and_descending():
    result = LazyFrame(ORDERS).sort("amount", ascending=False).to_pylist()
    amounts = [r["amount"] for r in result]
    assert amounts == sorted(amounts, reverse=True)


def test_join_inner():
    o = LazyFrame(ORDERS)
    c = LazyFrame(CUSTOMERS)
    result = o.join(c, on="customer_id", how="inner").to_pylist()
    assert len(result) == 6
    assert all("name" in r for r in result)


def test_join_left_with_nulls():
    left = LazyFrame([{"id": 1, "v": "a"}, {"id": 2, "v": "b"}])
    right = LazyFrame([{"id": 1, "info": "found"}])
    result = left.join(right, on="id", how="left").to_pylist()
    assert len(result) == 2
    matched = [r for r in result if r.get("info") is not None]
    assert len(matched) == 1


def test_union():
    a = LazyFrame(ORDERS[:3])
    b = LazyFrame(ORDERS[3:])
    result = a.union(b).to_pylist()
    assert len(result) == 6


def test_explode():
    data = [{"id": 1, "tags": ["a", "b"]}, {"id": 2, "tags": ["c"]}, {"id": 3, "tags": None}]
    result = LazyFrame(data).explode("tags").to_pylist()
    assert len(result) == 4


def test_head():
    result = LazyFrame(ORDERS).head(3).to_pylist()
    assert len(result) == 3


def test_getitem_string_select():
    lf = LazyFrame(ORDERS)["amount"]
    assert lf.columns == ["amount"]


def test_getitem_int_row_dict():
    row = LazyFrame(ORDERS)[0]
    assert row["order_id"] == 1


def test_getitem_slice_floe():
    lf = LazyFrame(ORDERS)[1:3]
    assert len(lf) == 2


def test_to_pydict():
    d = LazyFrame(ORDERS).select("order_id").to_pydict()
    assert d["order_id"] == [1, 2, 3, 4, 5, 6]


def test_to_batches_lazy_frame():
    data = [{"a": 1}, {"a": 2}, {"a": 3}]
    lf = LazyFrame(data)
    batches = list(lf.to_batches())
    assert batches == [data]


def test_to_batches_materialized_frame():
    data = [{"a": 1}, {"a": 2}]
    lf = LazyFrame(data)
    lf.collect()
    batches = list(lf.to_batches())
    assert batches == [data]


def test_apply_to_specific_columns():
    result = LazyFrame(ORDERS).apply(str, columns=["amount"]).to_pylist()
    assert result[0]["amount"] == "250.0"
    assert isinstance(result[0]["order_id"], int)  # untouched


def test_apply_to_all_columns():
    result = LazyFrame(ORDERS).apply(str).to_pylist()
    assert result[0]["order_id"] == "1"


def test_rename_columns():
    result = LazyFrame(ORDERS).rename({"amount": "price"}).columns
    assert "price" in result
    assert "amount" not in result


def test_with_columns_multiple():
    result = (
        LazyFrame(ORDERS)
        .with_columns(
            double=col("amount") * 2,
            upper_region=col("region").str.upper(),
        )
        .to_pylist()
    )
    assert result[0]["double"] == 500.0
    assert result[0]["upper_region"] == "EU"


def test_legacy_filter_with_lambda():
    result = LazyFrame(ORDERS).filter("amount", _filter=lambda a: a > 100).to_pylist()
    assert len(result) == 4


def test_legacy_filter_with_value():
    result = LazyFrame(ORDERS).filter("region", _filter="EU").to_pylist()
    assert len(result) == 3


def test_legacy_filter_multi_column_lambda():
    result = (
        LazyFrame(ORDERS)
        .filter(["region", "product"], _filter=lambda r, p: r == "US" and p == "Widget A")
        .to_pylist()
    )
    assert len(result) == 1


def test_construction_from_objects_with_dict():
    class Row:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    lf = LazyFrame([Row(1, "a"), Row(2, "b")])
    assert lf.columns == ["x", "y"]
    assert len(lf) == 2


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════


def test_chained_ops_stay_lazy():
    pipeline = (
        LazyFrame(ORDERS)
        .select("order_id", "amount", "region")
        .filter(col("amount") > 100)
        .with_column("tax", col("amount") * 0.2)
        .sort("amount")
    )
    assert not pipeline.is_materialized


def test_repr_does_not_materialize():
    pipeline = LazyFrame(ORDERS).filter(col("amount") > 100)
    r = repr(pipeline)
    assert "lazy" in r
    assert not pipeline.is_materialized


def test_schema_does_not_materialize_operations():
    pipeline = LazyFrame(ORDERS).filter(col("amount") > 100).select("order_id", "amount")
    _ = pipeline.schema  # should not materialize the filter
    assert not pipeline.is_materialized


def test_collect_materializes():
    pipeline = LazyFrame(ORDERS).filter(col("amount") > 100)
    pipeline.collect()
    assert pipeline.is_materialized
    assert "materialized" in repr(pipeline)


def test_display_uses_materialized_data():
    stored_schema = LazyFrame(ORDERS).select("order_id", "amount").schema

    class ExplodingPlan:
        def execute(self):
            raise AssertionError("execute should not be called")

        def execute_batched(self):
            raise AssertionError("execute_batched should not be called")

        def schema(self):
            return stored_schema

        def fast_count(self):
            return None

    lf = LazyFrame(ORDERS).select("order_id", "amount")
    lf.collect()
    lf._plan = ExplodingPlan()
    lf._optimized = None
    lf.display(n=1)


def test_repeated_collect_is_idempotent():
    pipeline = LazyFrame(ORDERS).filter(col("amount") > 100)
    pipeline.collect()
    n1 = len(pipeline.raw_data)
    pipeline.collect()
    assert len(pipeline.raw_data) == n1


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════


def test_lazyschema_select_drop_rename_merge():
    s = LazySchema(
        {
            "a": ColumnSchema("a", int),
            "b": ColumnSchema("b", str),
            "c": ColumnSchema("c", float),
        }
    )
    assert s.select(["a", "b"]).column_names == ["a", "b"]
    assert s.drop(["c"]).column_names == ["a", "b"]
    assert s.rename({"a": "x"}).column_names == ["x", "b", "c"]

    s2 = LazySchema({"b": ColumnSchema("b", int), "d": ColumnSchema("d", str)})
    merged = s.merge(s2)
    assert "right_b" in merged.column_names
    assert "d" in merged.column_names


def test_lazyschema_from_data():
    rows = [(1, "hello", 3.14), (2, None, 2.71)]
    s = LazySchema.from_data(["x", "y", "z"], rows)
    assert s.dtypes["x"] is int
    assert s.dtypes["y"] is str
    assert s["y"].nullable is True
    assert s["x"].nullable is False


# ═══════════════════════════════════════════════════════════
# Polars-style with_column / with_columns
# ═══════════════════════════════════════════════════════════


def test_with_column_aliased_expr():
    result = LazyFrame(ORDERS).with_column((col("amount") * 0.2).alias("tax")).to_pylist()
    for r in result:
        assert r["tax"] == r["amount"] * 0.2


def test_with_column_auto_derived_name():
    result = LazyFrame(ORDERS).with_column(col("amount") * 2).to_pylist()
    assert result[0]["amount"] == 500.0
    assert result[1]["amount"] == 151.0


def test_with_column_plain_col_expr():
    data = [{"a": 1, "b": 2}]
    result = LazyFrame(data).with_column(col("a").alias("a_copy")).to_pylist()
    assert result[0]["a_copy"] == 1


def test_with_column_backward_compat():
    result = LazyFrame(ORDERS).with_column("tax", col("amount") * 0.2).to_pylist()
    for r in result:
        assert r["tax"] == r["amount"] * 0.2


def test_with_column_no_name_raises():
    import pytest

    from pyfloe.expr import Lit

    with pytest.raises(ValueError, match="Cannot infer output name"):
        LazyFrame(ORDERS).with_column(Lit(42)).to_pylist()


def test_with_columns_positional_aliased():
    result = (
        LazyFrame(ORDERS)
        .with_columns(
            (col("amount") * 2).alias("double"),
            col("region").str.upper().alias("upper_region"),
        )
        .to_pylist()
    )
    assert result[0]["double"] == 500.0
    assert result[0]["upper_region"] == "EU"


def test_with_columns_mixed_positional_and_kwargs():
    result = (
        LazyFrame(ORDERS)
        .with_columns(
            (col("amount") * 2).alias("double"),
            upper_region=col("region").str.upper(),
        )
        .to_pylist()
    )
    assert result[0]["double"] == 500.0
    assert result[0]["upper_region"] == "EU"


def test_with_columns_kwargs_backward_compat():
    result = (
        LazyFrame(ORDERS)
        .with_columns(
            double=col("amount") * 2,
            upper_region=col("region").str.upper(),
        )
        .to_pylist()
    )
    assert result[0]["double"] == 500.0
    assert result[0]["upper_region"] == "EU"


# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
