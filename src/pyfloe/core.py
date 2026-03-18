from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import islice
from typing import (
    Any,
    Generic,
    TypeVar,
    get_type_hints,
)

from .expr import AggExpr, AggFunc, Col, Expr, JoinHow, WindowExpr
from .plan import (
    AggNode,
    ApplyNode,
    ExplodeNode,
    FilterNode,
    JoinNode,
    Optimizer,
    PivotNode,
    PlanNode,
    ProjectNode,
    RenameNode,
    ScanNode,
    SortedAggNode,
    SortedMergeJoinNode,
    SortNode,
    UnionNode,
    UnpivotNode,
    WindowNode,
    WithColumnNode,
)
from .schema import LazySchema

T = TypeVar("T")


def _dicts_to_tuples(data: list[dict]) -> tuple[list[str], list[tuple]]:
    if not data:
        return [], []
    all_keys: list = []
    seen: set = set()
    for d in data:
        for k in d:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    rows = [tuple(d.get(k) for k in all_keys) for d in data]
    return all_keys, rows


class LazyGroupBy:
    """Builder for grouped aggregation operations.

    Created by calling `LazyFrame.group_by()`. Use `.agg()` to specify
    aggregation expressions and produce a result LazyFrame.
    """

    def __init__(self, lf: LazyFrame, group_cols: list[str], sorted: bool = False):
        self._lf = lf
        self._group_cols = group_cols
        self._sorted = sorted

    def agg(self, *agg_exprs: AggExpr) -> LazyFrame:
        """Apply aggregation expressions to each group.

        Args:
            *agg_exprs: One or more aggregation expressions, e.g.
                ``col("amount").sum().alias("total")``.

        Returns:
            A new LazyFrame with one row per group.

        Raises:
            TypeError: If any argument is not an AggExpr.

        Examples:
            >>> from pyfloe import LazyFrame, col
            >>> orders = LazyFrame([
            ...     {"region": "EU", "amount": 250},
            ...     {"region": "EU", "amount": 180},
            ...     {"region": "US", "amount": 320},
            ... ])
            >>> orders.group_by("region").agg(
            ...     col("amount").sum().alias("total"),
            ...     col("amount").count().alias("n"),
            ... ).sort("region").to_pylist()
            [{'region': 'EU', 'total': 430, 'n': 2}, {'region': 'US', 'total': 320, 'n': 1}]
        """
        for i, expr in enumerate(agg_exprs):
            if not isinstance(expr, AggExpr):
                raise TypeError(
                    f"agg() argument {i + 1} is a {type(expr).__name__}, not an AggExpr. "
                    f'Use an aggregation method like col("x").sum(), .count(), '
                    f".mean(), .min(), .max(), .first(), .last(), .n_unique()"
                )
        if self._sorted:
            node = SortedAggNode(self._lf._plan, self._group_cols, list(agg_exprs))
        else:
            node = AggNode(self._lf._plan, self._group_cols, list(agg_exprs))
        return LazyFrame._from_plan(node)


class LazyFrame:
    """A lazy, composable dataframe.

    Operations on a LazyFrame build a query plan without executing it.
    Data flows only when you call a materialization method like
    `.collect()`, `.to_pylist()`, or `.to_csv()`.

    Examples:
        >>> lf = LazyFrame([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
        >>> lf.filter(col("age") > 28).to_pylist()
        [{'name': 'Alice', 'age': 30}]
    """

    __slots__ = ("_plan", "_materialized", "_name", "_optimized")

    def __init__(self, raw_data: list[dict] | list = None, *, name: str = None):
        """Create a LazyFrame from in-memory data.

        Args:
            raw_data: Input data as a list of dicts, list of tuples,
                list of objects with ``__dict__``, or a dict of columns.
            name: Optional name for the LazyFrame.

        Examples:
            From a list of dicts:

            >>> lf = LazyFrame([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
            >>> lf.columns
            ['name', 'age']

            From a dict of columns:

            >>> lf = LazyFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
            >>> lf.to_pylist()
            [{'x': 1, 'y': 'a'}, {'x': 2, 'y': 'b'}, {'x': 3, 'y': 'c'}]

            From a list of tuples (auto-named columns):

            >>> lf = LazyFrame([(1, "a"), (2, "b")])
            >>> lf.columns
            ['col_0', 'col_1']
        """
        self._name = name
        self._materialized: list[tuple] | None = None
        self._optimized: PlanNode | None = None
        if raw_data is None:
            self._plan: PlanNode | None = None
            return
        if isinstance(raw_data, dict):
            cols = list(raw_data.keys())
            if not cols:
                self._plan = ScanNode([], [])
                return
            values = [list(v) if not isinstance(v, list) else v for v in raw_data.values()]
            n_rows = len(values[0])
            rows = [tuple(values[c][r] for c in range(len(cols))) for r in range(n_rows)]
            self._plan = ScanNode(rows, cols)
        elif isinstance(raw_data, list):
            if not raw_data:
                self._plan = ScanNode([], [])
                return
            first = raw_data[0]
            if isinstance(first, dict):
                cols, rows = _dicts_to_tuples(raw_data)
            elif hasattr(first, "__dict__") and not isinstance(first, tuple):
                cols, rows = _dicts_to_tuples([r.__dict__ for r in raw_data])
            elif isinstance(first, (tuple, list)):
                width = len(first)
                cols = [f"col_{i}" for i in range(width)]
                rows = [tuple(r) for r in raw_data]
            else:
                cols = ["value"]
                rows = [(v,) for v in raw_data]
            self._plan = ScanNode(rows, cols)
        else:
            raise ValueError(
                "Provide a list of dicts, list of tuples, list of scalars, or a dict of columns"
            )

    @classmethod
    def _from_plan(cls, plan: PlanNode, name: str = None) -> LazyFrame:
        lf = cls.__new__(cls)
        lf._plan = plan
        lf._materialized = None
        lf._name = name
        lf._optimized = None
        return lf

    @property
    def _exec_plan(self) -> PlanNode:
        if self._optimized is None:
            self._optimized = Optimizer().optimize(self._plan)
        return self._optimized

    @property
    def schema(self) -> LazySchema:
        """Output schema of this LazyFrame, computed without touching data.

        Examples:
            >>> lf = LazyFrame([{"name": "Alice", "age": 30}])
            >>> lf.schema.column_names
            ['name', 'age']
            >>> lf.schema.dtypes
            {'name': <class 'str'>, 'age': <class 'int'>}
        """
        return self._plan.schema()

    @property
    def columns(self) -> list[str]:
        """List of column names.

        Examples:
            >>> LazyFrame([{"x": 1, "y": 2}]).columns
            ['x', 'y']
        """
        return self.schema.column_names

    @property
    def dtypes(self) -> dict[str, type]:
        """Mapping of column names to their Python types.

        Examples:
            >>> LazyFrame([{"name": "Alice", "age": 30}]).dtypes
            {'name': <class 'str'>, 'age': <class 'int'>}
        """
        return self.schema.dtypes

    def explain(self, optimized: bool = False) -> str:
        """Return a string representation of the query plan tree.

        Args:
            optimized: If True, show the plan after optimization
                (filter pushdown, column pruning).

        Examples:
            >>> lf = LazyFrame([{"a": 1}]).filter(col("a") > 0).select("a")
            >>> print(lf.explain())  # doctest: +SKIP
            Project [a]
              Filter [(col("a") > 0)]
                Scan [a] (1 rows)
        """
        plan = self._plan
        if optimized:
            plan = Optimizer().optimize(plan)
        return plan.explain()

    def print_explain(self, optimized: bool = False):
        """Print the query plan tree to stdout.

        Shortcut for ``print(lf.explain())``.

        Args:
            optimized: If True, show the plan after optimization.
        """
        print(self.explain(optimized))

    def select(self, *args: str | Expr) -> LazyFrame:
        """Select columns by name or expression.

        Args:
            *args: Column names as strings, or Expr objects.

        Returns:
            A new LazyFrame with only the selected columns.

        Examples:
            Select by column name:

            >>> lf = LazyFrame([{"a": 1, "b": 2, "c": 3}])
            >>> lf.select("a", "c").to_pylist()
            [{'a': 1, 'c': 3}]

            Select with expressions:

            >>> lf.select(col("a"), (col("b") + col("c")).alias("sum")).to_pylist()
            [{'a': 1, 'sum': 5}]
        """
        if all(isinstance(a, str) for a in args):
            return LazyFrame._from_plan(ProjectNode(self._plan, list(args)))
        exprs = [Col(a) if isinstance(a, str) else a for a in args]
        return LazyFrame._from_plan(ProjectNode(self._plan, exprs=[e for e in exprs]))

    def filter(self, predicate_or_col=None, _filter=None, **kwargs) -> LazyFrame:
        """Filter rows matching a predicate expression.

        Args:
            predicate_or_col: An Expr that evaluates to a boolean per row,
                e.g. ``col("amount") > 100``.

        Returns:
            A new LazyFrame with only matching rows.

        Examples:
            >>> orders = LazyFrame([
            ...     {"product": "A", "amount": 250, "region": "EU"},
            ...     {"product": "B", "amount": 75,  "region": "US"},
            ...     {"product": "C", "amount": 180, "region": "EU"},
            ... ])
            >>> orders.filter(col("amount") > 100).to_pylist()
            [{'product': 'A', 'amount': 250, 'region': 'EU'}, {'product': 'C', 'amount': 180, 'region': 'EU'}]

            Compound filters with ``&`` and ``|``:

            >>> orders.filter((col("region") == "EU") & (col("amount") > 200)).to_pylist()
            [{'product': 'A', 'amount': 250, 'region': 'EU'}]
        """
        if isinstance(predicate_or_col, Expr) and _filter is None:
            return LazyFrame._from_plan(FilterNode(self._plan, predicate_or_col))

        if _filter is not None and predicate_or_col is not None:
            import operator as _op

            from .expr import BinaryExpr
            from .expr import Col as _Col
            from .expr import Lit as _Lit

            if callable(_filter):
                cols = predicate_or_col
                if isinstance(cols, str):
                    cols = [cols]
                elif isinstance(cols, tuple):
                    cols = list(cols)
                [_Col(c) for c in cols]

                class _LegacyPredicate(Expr):
                    def __init__(self, col_names, func):
                        self._col_names = col_names
                        self._func = func

                    def eval(self, row, col_map):
                        vals = [row[col_map[c]] for c in self._col_names]
                        if len(vals) == 1:
                            return self._func(vals[0])
                        return self._func(*vals)

                    def required_columns(self):
                        return set(self._col_names)

                    def output_dtype(self, schema):
                        return bool

                    def __repr__(self):
                        return f"λ({', '.join(self._col_names)})"

                pred = _LegacyPredicate(cols, _filter)
                return LazyFrame._from_plan(FilterNode(self._plan, pred))
            else:
                val = _filter
                col_name = (
                    predicate_or_col if isinstance(predicate_or_col, str) else predicate_or_col[0]
                )
                pred = BinaryExpr(_Col(col_name), _Lit(val), _op.eq, "==")
                return LazyFrame._from_plan(FilterNode(self._plan, pred))

        raise ValueError("Provide an Expr predicate or use legacy filter(col, _filter=...)")

    def with_column(self, name: str, expr: Expr) -> LazyFrame:
        """Add a computed column to the LazyFrame.

        Args:
            name: Name for the new column.
            expr: Expression to compute the column values.

        Returns:
            A new LazyFrame with the additional column.

        Examples:
            >>> lf = LazyFrame([{"price": 100}, {"price": 200}])
            >>> lf.with_column("tax", col("price") * 0.2).to_pylist()
            [{'price': 100, 'tax': 20.0}, {'price': 200, 'tax': 40.0}]
        """
        if isinstance(expr, WindowExpr):
            return LazyFrame._from_plan(WindowNode(self._plan, expr, name))
        return LazyFrame._from_plan(WithColumnNode(self._plan, name, expr))

    def with_columns(self, **kwargs: Expr) -> LazyFrame:
        """Add multiple computed columns at once.

        Args:
            **kwargs: Column name to expression mappings.

        Returns:
            A new LazyFrame with the additional columns.

        Examples:
            >>> lf = LazyFrame([{"amount": 250, "region": "eu"}])
            >>> lf.with_columns(
            ...     double=col("amount") * 2,
            ...     upper_region=col("region").str.upper(),
            ... ).to_pylist()
            [{'amount': 250, 'region': 'eu', 'double': 500, 'upper_region': 'EU'}]
        """
        lf = self
        for name, expr in kwargs.items():
            lf = lf.with_column(name, expr)
        return lf

    def drop(self, *columns: str) -> LazyFrame:
        """Remove columns from the LazyFrame.

        Args:
            *columns: Column names to drop.

        Returns:
            A new LazyFrame without the specified columns.

        Examples:
            >>> lf = LazyFrame([{"a": 1, "b": 2, "c": 3}])
            >>> lf.drop("b", "c").columns
            ['a']
        """
        keep = [c for c in self.columns if c not in set(columns)]
        return LazyFrame._from_plan(ProjectNode(self._plan, keep))

    def rename(self, mapping: dict[str, str]) -> LazyFrame:
        """Rename columns.

        Args:
            mapping: Old name to new name mapping.

        Returns:
            A new LazyFrame with renamed columns.

        Examples:
            >>> lf = LazyFrame([{"amount": 100, "region": "EU"}])
            >>> lf.rename({"amount": "price", "region": "area"}).columns
            ['price', 'area']
        """
        return LazyFrame._from_plan(RenameNode(self._plan, mapping))

    def sort(self, *by: str, ascending: bool | list[bool] = True) -> LazyFrame:
        """Sort rows by one or more columns.

        Args:
            *by: Column names to sort by.
            ascending: Sort direction. A single bool applies to all
                columns; a list specifies per-column direction.

        Returns:
            A new LazyFrame with sorted rows.

        Examples:
            >>> lf = LazyFrame([{"name": "C"}, {"name": "A"}, {"name": "B"}])
            >>> lf.sort("name").to_pylist()
            [{'name': 'A'}, {'name': 'B'}, {'name': 'C'}]

            Descending sort:

            >>> lf.sort("name", ascending=False).to_pylist()
            [{'name': 'C'}, {'name': 'B'}, {'name': 'A'}]
        """
        by_list = list(by)
        if isinstance(ascending, bool):
            asc_list = [ascending] * len(by_list)
        else:
            asc_list = list(ascending)
        return LazyFrame._from_plan(SortNode(self._plan, by_list, asc_list))

    def join(
        self,
        other: LazyFrame,
        on: str | list[str] = None,
        left_on: str | list[str] = None,
        right_on: str | list[str] = None,
        how: JoinHow = "inner",
        sorted: bool = False,
        left_cols=None,
        right_cols=None,
    ) -> LazyFrame:
        """Join with another LazyFrame.

        Args:
            other: Right-side LazyFrame to join with.
            on: Column name(s) present in both sides.
            left_on: Column name(s) on the left side.
            right_on: Column name(s) on the right side.
            how: Join type — ``'inner'``, ``'left'``, or ``'full'``.
            sorted: If True, use sort-merge join (O(1) memory for
                pre-sorted inputs) instead of hash join.

        Returns:
            A new LazyFrame with columns from both sides.

        Examples:
            >>> orders = LazyFrame([{"id": 1, "cust": 101}, {"id": 2, "cust": 102}])
            >>> customers = LazyFrame([{"cust": 101, "name": "Alice"}])
            >>> orders.join(customers, on="cust", how="left").to_pylist()
            [{'id': 1, 'cust': 101, 'cust': 101, 'name': 'Alice'}, {'id': 2, 'cust': 102, 'cust': None, 'name': None}]

            Different key names on each side:

            >>> left = LazyFrame([{"order_id": 1, "customer_id": 10}])
            >>> right = LazyFrame([{"cid": 10, "name": "Alice"}])
            >>> left.join(right, left_on="customer_id", right_on="cid").to_pylist()
            [{'order_id': 1, 'customer_id': 10, 'cid': 10, 'name': 'Alice'}]
        """
        if left_cols is not None:
            left_on = left_cols
        if right_cols is not None:
            right_on = right_cols

        if on is not None:
            left_on = on
            right_on = on
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]
        if right_on is None:
            right_on = left_on

        if sorted:
            return LazyFrame._from_plan(
                SortedMergeJoinNode(self._plan, other._plan, left_on, right_on, how)
            )
        return LazyFrame._from_plan(JoinNode(self._plan, other._plan, left_on, right_on, how))

    def group_by(
        self, *columns: str, sorted: bool = False, **legacy_kwargs
    ) -> LazyGroupBy | LazyFrame:
        """Group by one or more columns.

        Args:
            *columns: Column names to group by.
            sorted: If True, use streaming sorted aggregation
                (requires input sorted by group columns).

        Returns:
            A LazyGroupBy — call ``.agg()`` to specify aggregations.

        Examples:
            >>> orders = LazyFrame([
            ...     {"region": "EU", "amount": 250},
            ...     {"region": "EU", "amount": 180},
            ...     {"region": "US", "amount": 320},
            ... ])
            >>> orders.group_by("region").agg(
            ...     col("amount").sum().alias("total"),
            ... ).sort("region").to_pylist()
            [{'region': 'EU', 'total': 430}, {'region': 'US', 'total': 320}]
        """
        cols = list(columns)

        agg_func = legacy_kwargs.get("agg_func")
        on_cols = legacy_kwargs.get("on_cols")
        if agg_func is not None:
            if isinstance(on_cols, str):
                on_cols = [on_cols]
            elif on_cols is None:
                on_cols = [c for c in self.columns if c not in cols]
            agg_exprs = []
            for oc in on_cols:

                class _LegacyAgg(AggExpr):
                    def __init__(self, col_name, func):
                        super().__init__(Col(col_name), "agg", func)
                        self._alias = col_name

                agg_exprs.append(_LegacyAgg(oc, agg_func))
            return LazyFrame._from_plan(AggNode(self._plan, cols, agg_exprs))

        return LazyGroupBy(self, cols, sorted=sorted)

    def explode(self, column: str) -> LazyFrame:
        """Unnest a list column into separate rows.

        Each element in the list becomes its own row, with all other
        column values duplicated.

        Args:
            column: Name of the column containing lists.

        Returns:
            A new LazyFrame with one row per list element.

        Examples:
            >>> lf = LazyFrame([
            ...     {"id": 1, "tags": ["a", "b"]},
            ...     {"id": 2, "tags": ["c"]},
            ... ])
            >>> lf.explode("tags").to_pylist()
            [{'id': 1, 'tags': 'a'}, {'id': 1, 'tags': 'b'}, {'id': 2, 'tags': 'c'}]
        """
        return LazyFrame._from_plan(ExplodeNode(self._plan, column))

    def pivot(
        self,
        index: str | list[str],
        on: str,
        values: str,
        agg: AggFunc = "first",
        columns: list[str] | None = None,
    ) -> LazyFrame:
        """Pivot (reshape long to wide).

        Args:
            index: Column(s) to keep as row identifiers.
            on: Column whose unique values become new column headers.
            values: Column whose values fill the pivoted cells.
            agg: Aggregation function name (``'first'``, ``'sum'``, etc.).
            columns: Explicit list of pivot column values (auto-detected if None).

        Returns:
            A new LazyFrame in wide format.

        Examples:
            >>> lf = LazyFrame([
            ...     {"name": "Alice", "subject": "math", "score": 90},
            ...     {"name": "Alice", "subject": "english", "score": 85},
            ...     {"name": "Bob", "subject": "math", "score": 78},
            ...     {"name": "Bob", "subject": "english", "score": 92},
            ... ])
            >>> lf.pivot(index="name", on="subject", values="score",
            ...          columns=["math", "english"]).sort("name").to_pylist()
            [{'name': 'Alice', 'math': 90, 'english': 85}, {'name': 'Bob', 'math': 78, 'english': 92}]
        """
        if isinstance(index, str):
            index = [index]
        return LazyFrame._from_plan(PivotNode(self._plan, index, on, values, agg, columns))

    def unpivot(
        self,
        id_columns: str | list[str],
        value_columns: str | list[str] | None = None,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> LazyFrame:
        """Unpivot (reshape wide to long). Also available as ``.melt()``.

        Args:
            id_columns: Column(s) to keep as identifiers.
            value_columns: Column(s) to unpivot. If None, all non-id columns.
            variable_name: Name for the new column holding original column names.
            value_name: Name for the new column holding the values.

        Returns:
            A new LazyFrame in long format.

        Examples:
            >>> lf = LazyFrame([
            ...     {"name": "Alice", "math": 90, "english": 85},
            ...     {"name": "Bob", "math": 78, "english": 92},
            ... ])
            >>> lf.unpivot("name", ["math", "english"]).sort("name", "variable").to_pylist()
            [{'name': 'Alice', 'variable': 'english', 'value': 85}, {'name': 'Alice', 'variable': 'math', 'value': 90}, {'name': 'Bob', 'variable': 'english', 'value': 92}, {'name': 'Bob', 'variable': 'math', 'value': 78}]
        """
        if isinstance(id_columns, str):
            id_columns = [id_columns]
        if value_columns is None:
            value_columns = [c for c in self.columns if c not in set(id_columns)]
        elif isinstance(value_columns, str):
            value_columns = [value_columns]
        return LazyFrame._from_plan(
            UnpivotNode(self._plan, id_columns, value_columns, variable_name, value_name)
        )

    melt = unpivot

    def union(self, other: LazyFrame) -> LazyFrame:
        """Stack rows from another LazyFrame below this one.

        Both LazyFrames must have the same columns.

        Args:
            other: LazyFrame to append.

        Returns:
            A new LazyFrame with rows from both inputs.

        Examples:
            >>> a = LazyFrame([{"x": 1}, {"x": 2}])
            >>> b = LazyFrame([{"x": 3}])
            >>> a.union(b).to_pylist()
            [{'x': 1}, {'x': 2}, {'x': 3}]
        """
        return LazyFrame._from_plan(UnionNode([self._plan, other._plan]))

    def apply(
        self, func: Callable, columns: list[str] = None, output_dtype: type = None
    ) -> LazyFrame:
        """Apply a function to column values.

        Args:
            func: Function to apply to each cell value.
            columns: Columns to apply to. If None, applies to all columns.
            output_dtype: Expected output type (for schema inference).

        Returns:
            A new LazyFrame with the function applied.

        Examples:
            Apply to specific columns:

            >>> lf = LazyFrame([{"name": "Alice", "age": 30}])
            >>> lf.apply(str, columns=["age"]).to_pylist()
            [{'name': 'Alice', 'age': '30'}]

            Apply to all columns:

            >>> lf.apply(str).to_pylist()
            [{'name': 'Alice', 'age': '30'}]
        """
        return LazyFrame._from_plan(ApplyNode(self._plan, func, columns, output_dtype))

    def read(self, columns: str | list[str]) -> LazyFrame:
        """Alias for :meth:`select`. Select columns by name.

        Args:
            columns: Column name or list of column names to select.

        Returns:
            A new LazyFrame with only the specified columns.
        """
        if isinstance(columns, str):
            columns = [columns]
        return self.select(*columns)

    def head(self, n: int = 5, optimize: bool = True) -> LazyFrame:
        """Return the first n rows as a new materialized LazyFrame.

        Args:
            n: Number of rows.
            optimize: If True, run the query optimizer first.

        Returns:
            A new materialized LazyFrame containing the first *n* rows.

        Examples:
            >>> lf = LazyFrame([{"x": i} for i in range(100)])
            >>> lf.head(3).to_pylist()
            [{'x': 0}, {'x': 1}, {'x': 2}]
        """
        plan = self._exec_plan if optimize else self._plan
        rows = list(islice(plan.execute(), n))
        return LazyFrame._from_plan(ScanNode(rows, self.columns, self.schema))

    def optimize(self) -> LazyFrame:
        """Return a new LazyFrame with an optimized query plan.

        Applies filter pushdown and column pruning.

        Returns:
            A new LazyFrame wrapping the optimized plan.

        Examples:
            >>> lf = LazyFrame([{"a": 1, "b": 2}]).select("a").filter(col("a") > 0)
            >>> opt = lf.optimize()
            >>> opt.to_pylist()
            [{'a': 1}]
        """
        optimized_plan = Optimizer().optimize(self._plan)
        return LazyFrame._from_plan(optimized_plan, self._name)

    @property
    def is_materialized(self) -> bool:
        """Whether the query plan has been executed and data is cached.

        Returns True after calling :meth:`collect` or :meth:`to_pylist`.
        """
        return self._materialized is not None

    @property
    def raw_data(self) -> list[tuple]:
        if self._materialized is None:
            data: list = []
            for chunk in self._plan.execute_batched():
                data.extend(chunk)
            self._materialized = data
        return self._materialized

    def collect(self, optimize: bool = True) -> LazyFrame:
        """Materialize the query plan and cache the results.

        After calling collect, subsequent operations use the cached data.
        Calling collect multiple times is safe and idempotent.

        Args:
            optimize: If True, run the query optimizer first.

        Returns:
            Self, with data materialized.

        Examples:
            >>> lf = LazyFrame([{"x": 1}, {"x": 2}]).filter(col("x") > 0)
            >>> lf.is_materialized
            False
            >>> lf.collect()  # doctest: +SKIP
            LazyFrame [2 rows × 1 cols] (materialized)
            >>> lf.is_materialized
            True
        """
        if self._materialized is None:
            plan = self._exec_plan if optimize else self._plan
            data: list = []
            for chunk in plan.execute_batched():
                data.extend(chunk)
            self._materialized = data
        return self

    def count(self, optimize: bool = True) -> int:
        """Return the total number of rows.

        Uses fast-path counting when possible (e.g. for in-memory data)
        without materializing all rows.

        Args:
            optimize: If True, run the query optimizer first.

        Returns:
            The row count as an integer.

        Examples:
            >>> LazyFrame([{"x": 1}, {"x": 2}, {"x": 3}]).count()
            3
        """
        n = self._known_length
        if n is not None:
            return n
        plan = self._exec_plan if optimize else self._plan
        n = plan.fast_count()
        if n is not None:
            return n
        total = 0
        for chunk in plan.execute_batched():
            total += len(chunk)
        return total

    def to_pylist(self) -> list[dict]:
        """Materialize and return data as a list of dicts.

        Examples:
            >>> LazyFrame([{"a": 1, "b": 2}]).to_pylist()
            [{'a': 1, 'b': 2}]
        """
        cols = self.columns
        return [{cols[i]: v for i, v in enumerate(row)} for row in self.raw_data]

    def to_pydict(self) -> dict[str, list]:
        """Materialize and return data as a dict of column lists.

        Examples:
            >>> LazyFrame([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]).to_pydict()
            {'x': [1, 2], 'y': ['a', 'b']}
        """
        cols = self.columns
        data = self.raw_data
        return {c: [row[i] for row in data] for i, c in enumerate(cols)}

    def to_tuples(self) -> list[tuple]:
        """Materialize and return data as a list of tuples.

        Examples:
            >>> LazyFrame([{"x": 1, "y": "a"}]).to_tuples()
            [(1, 'a')]
        """
        return list(self.raw_data)

    def to_csv(
        self, path: str, *, delimiter: str = ",", header: bool = True, encoding: str = "utf-8"
    ):
        """Stream the query plan to a CSV file with constant memory.

        Data is written row-by-row without buffering the entire dataset,
        so this works for arbitrarily large pipelines.

        Args:
            path: Output file path.
            delimiter: Field delimiter character.
            header: Whether to write a header row.
            encoding: File encoding.

        Examples:
            >>> lf = LazyFrame([{"name": "Alice", "age": 30}])
            >>> lf.to_csv("/tmp/output.csv")  # doctest: +SKIP
        """
        from .io import _to_csv_impl

        _to_csv_impl(self, path, delimiter, header, encoding)

    def to_tsv(self, path: str, **kwargs):
        """Stream the query plan to a TSV (tab-separated) file.

        Equivalent to ``lf.to_csv(path, delimiter='\\t')``.

        Args:
            path: Output file path.
            **kwargs: Additional arguments passed to :meth:`to_csv`.
        """
        kwargs.setdefault("delimiter", "\t")
        self.to_csv(path, **kwargs)

    def to_jsonl(self, path: str, *, encoding: str = "utf-8"):
        """Stream the query plan to a JSON Lines file.

        Args:
            path: Output file path.
            encoding: File encoding.
        """
        from .io import _to_jsonl_impl

        _to_jsonl_impl(self, path, encoding)

    def to_json(self, path: str, *, encoding: str = "utf-8", indent: int = None):
        """Write data as a JSON array.

        Args:
            path: Output file path.
            encoding: File encoding.
            indent: JSON indentation level.
        """
        from .io import _to_json_impl

        _to_json_impl(self, path, encoding, indent)

    def to_parquet(self, path: str, **kwargs):
        """Write data to a Parquet file (requires pyarrow).

        Args:
            path: Output file path.
            **kwargs: Additional arguments passed to pyarrow.
        """
        from .io import _to_parquet_impl

        _to_parquet_impl(self, path, **kwargs)

    def __iter__(self) -> Iterator[dict]:
        cols = self.columns
        source = (
            self._materialized if self._materialized is not None else self._exec_plan.execute()
        )
        for row in source:
            yield {cols[i]: v for i, v in enumerate(row)}

    def __len__(self) -> int:
        n = self._known_length
        if n is not None:
            return n
        raise TypeError(
            "len() requires materialized data. "
            "Call .collect() first, or use .head(n) to peek at rows."
        )

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.select(key)
        if isinstance(key, int):
            row = self.raw_data[key]
            return {c: v for c, v in zip(self.columns, row)}
        if isinstance(key, slice):
            rows = self.raw_data[key]
            return LazyFrame._from_plan(ScanNode(rows, self.columns, self.schema))
        raise TypeError(f"Invalid key type: {type(key)}")

    @property
    def _known_length(self) -> int | None:
        if self._materialized is not None:
            return len(self._materialized)
        if isinstance(self._plan, ScanNode):
            return len(self._plan._data)
        return None

    def __repr__(self):
        cols = self.columns
        ncols = len(cols)
        col_str = ", ".join(cols[:8])
        if ncols > 8:
            col_str += f", ... +{ncols - 8} more"
        n = self._known_length
        if self._materialized is not None:
            return f"LazyFrame [{n} rows × {ncols} cols] (materialized)\nColumns: [{col_str}]"
        if n is not None:
            return f"LazyFrame [{n} rows × {ncols} cols]\nColumns: [{col_str}]"
        return f"LazyFrame [? rows × {ncols} cols] (lazy)\nColumns: [{col_str}]"

    def _repr_short(self):
        n = self._known_length
        if n is not None:
            return f"LazyFrame[{n}×{len(self.columns)}]"
        return f"LazyFrame[?×{len(self.columns)}]"

    def display(self, n: int = 20, max_col_width: int = 30, optimize: bool = True) -> None:
        """Print a formatted table of the first n rows.

        Args:
            n: Maximum number of rows to display.
            max_col_width: Truncate cell values longer than this.
            optimize: If True, run the query optimizer first.

        Examples:
            >>> lf = LazyFrame([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
            >>> lf.display()  # doctest: +SKIP
            name  | age
            ------+----
            Alice | 30
            Bob   | 25
        """
        cols = self.columns
        plan = self._exec_plan if optimize else self._plan
        sample = list(islice(plan.execute(), n))
        total = self._known_length

        def _fmt(v):
            s = "" if v is None else str(v)
            return s if len(s) <= max_col_width else s[: max_col_width - 1] + "…"

        str_rows = [[_fmt(v) for v in row] for row in sample]

        widths = [len(c) for c in cols]
        for row in str_rows:
            for i, cell in enumerate(row):
                if len(cell) > widths[i]:
                    widths[i] = len(cell)

        hdr = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
        sep = "-+-".join("-" * w for w in widths)
        lines = [hdr, sep]
        for row in str_rows:
            lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(cols))))

        if total is not None and total > n:
            lines.append(f"... {total - n} more rows")
        elif total is None and len(sample) == n:
            lines.append(f"... (showing first {n} rows)")

        print("\n".join(lines))

    def typed(self, row_type: type[T]) -> TypedLazyFrame[T]:
        """Wrap this LazyFrame as a TypedLazyFrame for IDE-friendly typed results.

        Operations that preserve the schema (filter, sort, head) return
        a TypedLazyFrame, so ``.to_pylist()`` returns ``list[T]`` in type checkers.

        Args:
            row_type: A TypedDict class describing the row schema.

        Returns:
            A TypedLazyFrame wrapping the same query plan.

        Examples:
            >>> from typing import TypedDict
            >>> class Order(TypedDict):
            ...     order_id: int
            ...     amount: float
            >>> orders = LazyFrame([{"order_id": 1, "amount": 99.9}]).typed(Order)
            >>> isinstance(orders, TypedLazyFrame)
            True
        """
        return TypedLazyFrame._from_typed(self._plan, row_type, self._name)

    def validate(self, row_type: type) -> LazyFrame:
        """Validate the schema against a TypedDict type.

        Args:
            row_type: A TypedDict class. Each key is checked against
                the LazyFrame's schema for presence and type compatibility.

        Returns:
            Self, if validation passes.

        Raises:
            TypeError: If the schema doesn't match the TypedDict.

        Examples:
            >>> from typing import TypedDict
            >>> class Order(TypedDict):
            ...     order_id: int
            ...     amount: float
            >>> LazyFrame([{"order_id": 1, "amount": 9.9}]).validate(Order)  # doctest: +SKIP
            LazyFrame [1 rows × 2 cols]
        """
        hints = get_type_hints(row_type)
        schema = self.schema
        errors = []
        for col_name, expected_type in hints.items():
            if col_name not in schema:
                errors.append(f"  missing column: {col_name}")
            else:
                actual = schema[col_name].dtype
                if actual != expected_type and expected_type not in (Any,):
                    errors.append(
                        f"  {col_name}: expected {expected_type.__name__}, got {actual.__name__}"
                    )
        if errors:
            raise TypeError("Schema validation failed:\n" + "\n".join(errors))
        return self


class TypedLazyFrame(LazyFrame, Generic[T]):
    """A LazyFrame with a known row type for static type checking.

    Created via ``LazyFrame.typed(MyTypedDict)``. Operations that preserve
    the schema (filter, sort, head) return a TypedLazyFrame, so
    ``.to_pylist()`` returns ``list[T]`` in type checkers.
    """

    __slots__ = ("_row_type",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._row_type = None

    @classmethod
    def _from_typed(cls, plan: PlanNode, row_type: type[T], name: str = None) -> TypedLazyFrame[T]:
        lf = cls.__new__(cls)
        lf._plan = plan
        lf._materialized = None
        lf._name = name
        lf._row_type = row_type
        return lf

    @property
    def row_type(self) -> type[T]:
        return self._row_type

    def to_pylist(self) -> list[T]:
        return super().to_pylist()

    def collect(self, optimize: bool = True) -> TypedLazyFrame[T]:
        super().collect(optimize)
        return self

    def filter(self, *args, **kwargs) -> TypedLazyFrame[T]:
        result = super().filter(*args, **kwargs)
        return TypedLazyFrame._from_typed(result._plan, self._row_type, result._name)

    def sort(self, *args, **kwargs) -> TypedLazyFrame[T]:
        result = super().sort(*args, **kwargs)
        return TypedLazyFrame._from_typed(result._plan, self._row_type, result._name)

    def head(self, n: int = 5, optimize: bool = True) -> TypedLazyFrame[T]:
        result = super().head(n, optimize)
        return TypedLazyFrame._from_typed(result._plan, self._row_type, result._name)

    def __repr__(self):
        base = super().__repr__()
        type_name = self._row_type.__name__ if self._row_type else "?"
        return base.replace("LazyFrame", f"TypedLazyFrame[{type_name}]", 1)
