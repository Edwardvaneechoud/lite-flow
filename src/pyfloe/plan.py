from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from itertools import accumulate, chain, compress, groupby, islice
from operator import itemgetter
from typing import Any

from .expr import (
    AggExpr,
    AggFunc,
    CumExpr,
    Expr,
    JoinHow,
    OffsetExpr,
    RankExpr,
    WindowExpr,
)
from .schema import ColumnSchema, LazySchema

_BATCH_SIZE = 1024


def _batched(iterable: Any, n: int = _BATCH_SIZE) -> Iterator[list]:
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def _make_key_fn(indices: list[int]) -> Callable[[tuple], tuple]:
    if len(indices) == 1:
        idx = indices[0]
        return lambda row: (row[idx],)
    return itemgetter(*indices)


class PlanNode:
    """Abstract base class for all query plan nodes.

    Every node in the query plan tree inherits from ``PlanNode`` and
    implements the volcano / iterator execution model.  Data flows
    upward through the tree: a parent node pulls rows from its children
    by calling ``execute()`` or ``execute_batched()``.
    """

    __slots__ = ()

    def schema(self) -> LazySchema:
        """Return the output schema of this node.

        Returns:
            The schema describing columns and types produced by this node.
        """
        raise NotImplementedError

    def execute(self) -> Iterator[tuple]:
        """Yield rows one at a time by flattening batched output.

        Returns:
            An iterator of tuples, each representing one row.
        """
        return chain.from_iterable(self.execute_batched())

    def execute_batched(self) -> Iterator[list[tuple]]:
        """Yield rows in batches of up to ``_BATCH_SIZE``.

        Returns:
            An iterator of lists, where each list contains up to 1024 row tuples.
        """
        raise NotImplementedError

    def fast_count(self) -> int | None:
        """Return the exact row count without executing, if known.

        Returns:
            The row count, or ``None`` if it cannot be determined cheaply.
        """
        return None

    def children(self) -> list[PlanNode]:
        """Return the direct child nodes of this plan node.

        Returns:
            A list of child ``PlanNode`` instances (empty for leaf nodes).
        """
        return []

    def explain(self, indent: int = 0) -> str:
        """Return a human-readable representation of the plan tree.

        Args:
            indent: Current indentation level (used for recursive calls).

        Returns:
            A multi-line string showing this node and all descendants.
        """
        prefix = "  " * indent
        lines = [f"{prefix}{self._explain_self()}"]
        for child in self.children():
            lines.append(child.explain(indent + 1))
        return "\n".join(lines)

    def _explain_self(self) -> str:
        return self.__class__.__name__


class ScanNode(PlanNode):
    """Leaf node that reads from an in-memory list of row tuples.

    This is the most common data source node, created when a ``LazyFrame``
    is constructed from Python data.  It supports ``fast_count`` because
    the full dataset is already materialised.

    Args:
        data: List of row tuples.
        columns: Column names corresponding to tuple positions.
        lazy_schema: Optional pre-computed schema; inferred from *data* if omitted.
    """

    __slots__ = ("_data", "_columns", "_schema")

    def __init__(self, data: list, columns: list[str], lazy_schema: LazySchema | None = None):
        self._data = data
        self._columns = list(columns)
        self._schema = lazy_schema

    def schema(self) -> LazySchema:
        if self._schema is None:
            self._schema = LazySchema.from_data(self._columns, self._data)
        return self._schema

    def execute_batched(self) -> Iterator[list[tuple]]:
        data = self._data
        for i in range(0, len(data), _BATCH_SIZE):
            yield data[i : i + _BATCH_SIZE]

    def fast_count(self) -> int | None:
        return len(self._data)

    def _explain_self(self) -> str:
        n = len(self._data)
        cols = self._columns
        if len(cols) <= 3:
            col_str = ", ".join(cols)
        else:
            col_str = f"{', '.join(cols[:3])}, ... +{len(cols) - 3} more"
        return f"Scan [{col_str}] ({n} rows)"


class IteratorSourceNode(PlanNode):
    """Leaf node that reads from a lazily-evaluated iterator factory.

    Each call to ``execute_batched`` invokes the factory to produce a
    fresh iterator, enabling repeatable reads from streaming sources
    such as file readers.

    Args:
        columns: Column names for the produced rows.
        lazy_schema: Schema describing the output columns.
        iterator_factory: A zero-argument callable that returns an iterator of row tuples.
        source_label: Descriptive label shown in ``explain`` output.
    """

    __slots__ = ("_columns", "_schema", "_factory", "_source_label")

    def __init__(
        self,
        columns: list[str],
        lazy_schema: LazySchema,
        iterator_factory: Callable[[], Iterator[tuple]],
        source_label: str = "Iterator",
    ):
        self._columns = columns
        self._schema = lazy_schema
        self._factory = iterator_factory
        self._source_label = source_label

    def schema(self) -> LazySchema:
        return self._schema

    def execute_batched(self) -> Iterator[list[tuple]]:
        return _batched(self._factory())

    def _explain_self(self) -> str:
        cols = self._columns
        if len(cols) <= 3:
            col_str = ", ".join(cols)
        else:
            col_str = f"{', '.join(cols[:3])}, ... +{len(cols) - 3} more"
        return f"{self._source_label} [{col_str}]"


class ProjectNode(PlanNode):
    """Selects or reorders columns, or evaluates computed expressions.

    When *columns* is provided, only those columns are kept (a SQL ``SELECT``).
    When *exprs* is provided, each expression is evaluated to produce a new
    set of output columns.

    Args:
        child: Input plan node.
        columns: Column names to select.  Mutually exclusive with *exprs*.
        exprs: Expressions to evaluate.  Mutually exclusive with *columns*.
    """

    __slots__ = ("child", "_columns", "_exprs")

    def __init__(
        self, child: PlanNode, columns: list[str] | None = None, exprs: list[Expr] | None = None
    ):
        self.child = child
        self._columns = columns
        self._exprs = exprs

    def schema(self) -> LazySchema:
        parent = self.child.schema()
        if self._columns:
            return parent.select(self._columns)
        if self._exprs:
            cols = {}
            for expr in self._exprs:
                name = expr.output_name() or repr(expr)
                dtype = expr.output_dtype(parent)
                cols[name] = ColumnSchema(name, dtype)
            return LazySchema(cols)
        return parent

    def execute_batched(self) -> Iterator[list[tuple]]:
        parent_cols = self.child.schema().column_names
        col_map = {n: i for i, n in enumerate(parent_cols)}

        if self._columns:
            indices = [col_map[c] for c in self._columns if c in col_map]
            n = len(indices)
            if n == 0:
                return
            elif n == 1:
                idx = indices[0]
                for chunk in self.child.execute_batched():
                    yield [(row[idx],) for row in chunk]
            else:
                getter = itemgetter(*indices)
                for chunk in self.child.execute_batched():
                    yield list(map(getter, chunk))
        elif self._exprs:
            compiled = [e.compile(col_map) for e in self._exprs]
            saw_rows = False
            for chunk in self.child.execute_batched():
                if chunk:
                    saw_rows = True
                yield [tuple(fn(row) for fn in compiled) for row in chunk]
            if not saw_rows and all(not e.required_columns() for e in self._exprs):
                dummy = (None,) * len(parent_cols)
                yield [tuple(fn(dummy) for fn in compiled)]

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        if self._columns:
            return f"Project [{', '.join(self._columns)}]"
        if self._exprs is not None:
            return f"Project [{', '.join(repr(e) for e in self._exprs)}]"
        return "Project []"


class FilterNode(PlanNode):
    """Filters rows by evaluating a boolean predicate expression.

    Only rows for which *predicate* evaluates to a truthy value are
    passed through.  The output schema is unchanged.

    Args:
        child: Input plan node.
        predicate: Boolean expression used to filter rows.
    """

    __slots__ = ("child", "predicate")

    def __init__(self, child: PlanNode, predicate: Expr):
        self.child = child
        self.predicate = predicate

    def schema(self) -> LazySchema:
        return self.child.schema()

    def execute_batched(self) -> Iterator[list[tuple]]:
        col_map = {n: i for i, n in enumerate(self.child.schema().column_names)}
        pred_fn = self.predicate.compile(col_map)
        for chunk in self.child.execute_batched():
            filtered = list(compress(chunk, map(pred_fn, chunk)))
            if filtered:
                yield filtered

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        return f"Filter [{self.predicate}]"


class LimitNode(PlanNode):
    """Limits the number of rows passed through from its child.

    This is the lazy equivalent of ``head(n)`` — it truncates the
    output after *n* rows without materialising the upstream plan.

    Args:
        child: Input plan node.
        n: Maximum number of rows to emit.
    """

    __slots__ = ("child", "n")

    def __init__(self, child: PlanNode, n: int):
        self.child = child
        self.n = n

    def schema(self) -> LazySchema:
        return self.child.schema()

    def execute_batched(self) -> Iterator[list[tuple]]:
        remaining = self.n
        for chunk in self.child.execute_batched():
            if remaining <= 0:
                return
            if len(chunk) <= remaining:
                remaining -= len(chunk)
                yield chunk
            else:
                yield chunk[:remaining]
                return

    def fast_count(self) -> int | None:
        child_count = self.child.fast_count()
        if child_count is not None:
            return min(self.n, child_count)
        return None

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        return f"Limit [{self.n}]"


class JoinNode(PlanNode):
    """Hash-based join of two input plan nodes.

    Materialises the right side into a hash table keyed on the join
    columns, then streams the left side and probes the table.
    Supports ``inner``, ``left``, and ``full`` join modes.

    Args:
        left: Left input plan node.
        right: Right input plan node.
        left_on: Join-key column names from the left side.
        right_on: Join-key column names from the right side.
        how: Join type — ``'inner'``, ``'left'``, or ``'full'``.
    """

    __slots__ = ("left", "right", "left_on", "right_on", "how")

    def __init__(
        self,
        left: PlanNode,
        right: PlanNode,
        left_on: list[str],
        right_on: list[str],
        how: JoinHow = "inner",
    ):
        self.left = left
        self.right = right
        self.left_on = left_on
        self.right_on = right_on
        self.how = how

    def schema(self) -> LazySchema:
        return self.left.schema().merge(self.right.schema())

    def execute_batched(self) -> Iterator[list[tuple]]:
        rs = self.right.schema()
        r_map = {n: i for i, n in enumerate(rs.column_names)}
        r_idx = [r_map[c] for c in self.right_on]
        r_key = _make_key_fn(r_idx)
        right_ht: dict[tuple, list] = defaultdict(list)
        for chunk in self.right.execute_batched():
            for row in chunk:
                right_ht[r_key(row)].append(row)

        ls = self.left.schema()
        l_map = {n: i for i, n in enumerate(ls.column_names)}
        l_idx = [l_map[c] for c in self.left_on]
        l_key = _make_key_fn(l_idx)
        null_right = (None,) * len(rs.column_names)
        null_left = (None,) * len(ls.column_names)

        matched_keys: set = set()
        buf: list = []
        for chunk in self.left.execute_batched():
            for left_row in chunk:
                key = l_key(left_row)
                matches = right_ht.get(key)
                if matches:
                    matched_keys.add(key)
                    for right_row in matches:
                        buf.append(left_row + right_row)
                elif self.how in ("left", "full"):
                    buf.append(left_row + null_right)
                if len(buf) >= _BATCH_SIZE:
                    yield buf
                    buf = []

        if self.how == "full":
            for key, rows in right_ht.items():
                if key not in matched_keys:
                    for right_row in rows:
                        buf.append(null_left + right_row)
                    if len(buf) >= _BATCH_SIZE:
                        yield buf
                        buf = []

        if buf:
            yield buf

    def children(self) -> list[PlanNode]:
        return [self.left, self.right]

    def _explain_self(self) -> str:
        return f"Join [{self.how}] {self.left_on} = {self.right_on}"


class AggNode(PlanNode):
    """Hash-based group-by aggregation node.

    Groups rows by *group_by* columns using a hash map and maintains
    running accumulators for each aggregation expression.  Memory usage
    is O(k) where k is the number of distinct groups.

    Args:
        child: Input plan node.
        group_by: Column names to group by.
        agg_exprs: Aggregation expressions to evaluate per group.
    """

    __slots__ = ("child", "group_by", "agg_exprs")

    def __init__(self, child: PlanNode, group_by: list[str], agg_exprs: list[AggExpr]):
        self.child = child
        self.group_by = group_by
        self.agg_exprs = agg_exprs

    def schema(self) -> LazySchema:
        parent = self.child.schema()
        cols = {}
        for name in self.group_by:
            cols[name] = parent[name]
        for agg in self.agg_exprs:
            name = agg.output_name()
            cols[name] = ColumnSchema(name, agg.output_dtype(parent))
        return LazySchema(cols)

    def execute_batched(self) -> Iterator[list[tuple]]:
        parent_cols = self.child.schema().column_names
        col_map = {n: i for i, n in enumerate(parent_cols)}
        g_idx = [col_map[c] for c in self.group_by]
        key_fn = _make_key_fn(g_idx)
        n_aggs = len(self.agg_exprs)

        compiled_evals = [agg.expr.compile(col_map) for agg in self.agg_exprs]
        updaters = [_make_updater(agg) for agg in self.agg_exprs]

        accumulators: dict[tuple, list] = {}
        _init = self.agg_exprs

        for chunk in self.child.execute_batched():
            for row in chunk:
                key = key_fn(row)
                try:
                    accs = accumulators[key]
                except KeyError:
                    accs = [_init_acc(agg) for agg in _init]
                    accumulators[key] = accs
                for i in range(n_aggs):
                    updaters[i](accs[i], compiled_evals[i](row))

        buf: list = []
        for key, accs in accumulators.items():
            agg_vals = tuple(_finalize_acc(accs[i], self.agg_exprs[i]) for i in range(n_aggs))
            buf.append(key + agg_vals)
            if len(buf) >= _BATCH_SIZE:
                yield buf
                buf = []
        if buf:
            yield buf

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        aggs = ", ".join(repr(a) for a in self.agg_exprs)
        return f"Aggregate by=[{', '.join(self.group_by)}] aggs=[{aggs}]"


def _init_acc(agg: AggExpr) -> dict:
    kind = agg.agg_name
    if kind == "sum":
        return {"s": 0}
    elif kind == "count":
        return {"n": 0}
    elif kind == "mean":
        return {"s": 0.0, "n": 0}
    elif kind == "min":
        return {"v": None}
    elif kind == "max":
        return {"v": None}
    elif kind == "first":
        return {"v": None, "set": False}
    elif kind == "last":
        return {"v": None}
    elif kind == "n_unique":
        return {"s": set()}
    else:
        return {"vals": []}


def _update_acc(acc: dict, agg: AggExpr, val: Any) -> None:
    kind = agg.agg_name
    if kind == "sum":
        if val is not None:
            acc["s"] += val
    elif kind == "count":
        if val is not None:
            acc["n"] += 1
    elif kind == "mean":
        if val is not None:
            acc["s"] += val
            acc["n"] += 1
    elif kind == "min":
        if val is not None:
            if acc["v"] is None or val < acc["v"]:
                acc["v"] = val
    elif kind == "max":
        if val is not None:
            if acc["v"] is None or val > acc["v"]:
                acc["v"] = val
    elif kind == "first":
        if not acc["set"]:
            acc["v"] = val
            acc["set"] = True
    elif kind == "last":
        acc["v"] = val
    elif kind == "n_unique":
        if val is not None:
            acc["s"].add(val)
    else:
        acc["vals"].append(val)


def _finalize_acc(acc: dict, agg: AggExpr) -> Any:
    kind = agg.agg_name
    if kind == "sum":
        return acc["s"]
    elif kind == "count":
        return acc["n"]
    elif kind == "mean":
        return acc["s"] / acc["n"] if acc["n"] else 0.0
    elif kind in ("min", "max", "first", "last"):
        return acc["v"]
    elif kind == "n_unique":
        return len(acc["s"])
    else:
        return agg.eval_agg(acc["vals"])


def _init_pivot_acc(agg_name: AggFunc) -> dict:
    if agg_name == "sum":
        return {"s": 0}
    elif agg_name == "count":
        return {"n": 0}
    elif agg_name == "mean":
        return {"s": 0.0, "n": 0}
    elif agg_name == "min":
        return {"v": None}
    elif agg_name == "max":
        return {"v": None}
    elif agg_name == "first":
        return {"v": None, "set": False}
    elif agg_name == "last":
        return {"v": None}
    else:
        return {"v": None, "set": False}


def _update_pivot_acc(acc: dict, agg_name: AggFunc, val: Any) -> None:
    if agg_name == "sum":
        if val is not None:
            acc["s"] += val
    elif agg_name == "count":
        if val is not None:
            acc["n"] += 1
    elif agg_name == "mean":
        if val is not None:
            acc["s"] += val
            acc["n"] += 1
    elif agg_name == "min":
        if val is not None:
            if acc["v"] is None or val < acc["v"]:
                acc["v"] = val
    elif agg_name == "max":
        if val is not None:
            if acc["v"] is None or val > acc["v"]:
                acc["v"] = val
    elif agg_name == "first":
        if not acc["set"]:
            acc["v"] = val
            acc["set"] = True
    elif agg_name == "last":
        acc["v"] = val


def _finalize_pivot_acc(acc: dict, agg_name: AggFunc) -> Any:
    if agg_name == "sum":
        return acc["s"]
    elif agg_name == "count":
        return acc["n"]
    elif agg_name == "mean":
        return acc["s"] / acc["n"] if acc["n"] else 0.0
    elif agg_name in ("min", "max", "first", "last"):
        return acc["v"]
    return acc.get("v")


def _make_updater(agg: AggExpr) -> Callable[[dict, Any], None]:
    kind = agg.agg_name
    if kind == "sum":

        def _update(acc, val):
            if val is not None:
                acc["s"] += val
    elif kind == "count":

        def _update(acc, val):
            if val is not None:
                acc["n"] += 1
    elif kind == "mean":

        def _update(acc, val):
            if val is not None:
                acc["s"] += val
                acc["n"] += 1
    elif kind == "min":

        def _update(acc, val):
            if val is not None:
                if acc["v"] is None or val < acc["v"]:
                    acc["v"] = val
    elif kind == "max":

        def _update(acc, val):
            if val is not None:
                if acc["v"] is None or val > acc["v"]:
                    acc["v"] = val
    elif kind == "first":

        def _update(acc, val):
            if not acc["set"]:
                acc["v"] = val
                acc["set"] = True
    elif kind == "last":

        def _update(acc, val):
            acc["v"] = val
    elif kind == "n_unique":

        def _update(acc, val):
            if val is not None:
                acc["s"].add(val)
    else:
        _agg = agg

        def _update(acc, val):
            acc["vals"].append(val)

    return _update


class SortedAggNode(PlanNode):
    """Streaming group-by aggregation for pre-sorted input.

    Assumes the child yields rows already sorted by the *group_by*
    columns.  Groups are detected by key changes, so only one
    accumulator set is active at a time — O(1) memory per group.

    Args:
        child: Input plan node (must be pre-sorted by *group_by*).
        group_by: Column names to group by.
        agg_exprs: Aggregation expressions to evaluate per group.
    """

    __slots__ = ("child", "group_by", "agg_exprs")

    def __init__(self, child: PlanNode, group_by: list[str], agg_exprs: list[AggExpr]):
        self.child = child
        self.group_by = group_by
        self.agg_exprs = agg_exprs

    def schema(self) -> LazySchema:
        parent = self.child.schema()
        cols = {}
        for name in self.group_by:
            cols[name] = parent[name]
        for agg in self.agg_exprs:
            name = agg.output_name()
            cols[name] = ColumnSchema(name, agg.output_dtype(parent))
        return LazySchema(cols)

    def execute_batched(self) -> Iterator[list[tuple]]:
        parent_cols = self.child.schema().column_names
        col_map = {n: i for i, n in enumerate(parent_cols)}
        g_idx = [col_map[c] for c in self.group_by]
        n_aggs = len(self.agg_exprs)
        key_func = _make_key_fn(g_idx)

        compiled_evals = [agg.expr.compile(col_map) for agg in self.agg_exprs]
        updaters = [_make_updater(agg) for agg in self.agg_exprs]

        buf: list = []
        for key, group_rows in groupby(self.child.execute(), key=key_func):
            accs = [_init_acc(agg) for agg in self.agg_exprs]
            for row in group_rows:
                for i in range(n_aggs):
                    updaters[i](accs[i], compiled_evals[i](row))
            buf.append(
                key + tuple(_finalize_acc(accs[i], self.agg_exprs[i]) for i in range(n_aggs))
            )
            if len(buf) >= _BATCH_SIZE:
                yield buf
                buf = []
        if buf:
            yield buf

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        aggs = ", ".join(repr(a) for a in self.agg_exprs)
        return f"SortedAggregate by=[{', '.join(self.group_by)}] aggs=[{aggs}]"


class SortedMergeJoinNode(PlanNode):
    """Sort-merge join for pre-sorted inputs.

    Both inputs must be sorted on their respective join keys.  Two
    cursors advance in lockstep, emitting matches when keys are equal.
    Memory usage is O(1) for one-to-one joins and O(g) for groups with
    many-to-many matches.

    Args:
        left: Left input plan node (sorted by *left_on*).
        right: Right input plan node (sorted by *right_on*).
        left_on: Join-key column names from the left side.
        right_on: Join-key column names from the right side.
        how: Join type — ``'inner'``, ``'left'``, or ``'full'``.
    """

    __slots__ = ("left", "right", "left_on", "right_on", "how")

    def __init__(
        self,
        left: PlanNode,
        right: PlanNode,
        left_on: list[str],
        right_on: list[str],
        how: JoinHow = "inner",
    ):
        self.left = left
        self.right = right
        self.left_on = left_on
        self.right_on = right_on
        self.how = how

    def schema(self) -> LazySchema:
        return self.left.schema().merge(self.right.schema())

    def execute_batched(self) -> Iterator[list[tuple]]:
        return _batched(self._merge_rows())

    def _merge_rows(self) -> Iterator[tuple]:
        ls = self.left.schema()
        rs = self.right.schema()
        l_map = {n: i for i, n in enumerate(ls.column_names)}
        r_map = {n: i for i, n in enumerate(rs.column_names)}
        l_idx = [l_map[c] for c in self.left_on]
        r_idx = [r_map[c] for c in self.right_on]
        null_right = (None,) * len(rs.column_names)
        null_left = (None,) * len(ls.column_names)

        l_key = _make_key_fn(l_idx)
        r_key = _make_key_fn(r_idx)

        left_iter = self.left.execute()
        right_iter = self.right.execute()

        try:
            l_row = next(left_iter)
        except StopIteration:
            if self.how == "full":
                for r_row in right_iter:
                    yield null_left + r_row
            return

        try:
            r_row = next(right_iter)
        except StopIteration:
            if self.how in ("left", "full"):
                yield l_row + null_right
                for l_row in left_iter:
                    yield l_row + null_right
            return

        lk = l_key(l_row)
        rk = r_key(r_row)
        l_exhausted = False
        r_exhausted = False

        while not l_exhausted and not r_exhausted:
            if lk < rk:
                if self.how in ("left", "full"):
                    yield l_row + null_right
                try:
                    l_row = next(left_iter)
                    lk = l_key(l_row)
                except StopIteration:
                    l_exhausted = True
            elif lk > rk:
                if self.how == "full":
                    yield null_left + r_row
                try:
                    r_row = next(right_iter)
                    rk = r_key(r_row)
                except StopIteration:
                    r_exhausted = True
            else:
                match_key = lk
                left_group = [l_row]
                right_group = [r_row]

                while True:
                    try:
                        l_row = next(left_iter)
                        lk = l_key(l_row)
                        if lk == match_key:
                            left_group.append(l_row)
                        else:
                            break
                    except StopIteration:
                        l_exhausted = True
                        break

                while True:
                    try:
                        r_row = next(right_iter)
                        rk = r_key(r_row)
                        if rk == match_key:
                            right_group.append(r_row)
                        else:
                            break
                    except StopIteration:
                        r_exhausted = True
                        break

                for lr in left_group:
                    for rr in right_group:
                        yield lr + rr

        if not l_exhausted and self.how in ("left", "full"):
            yield l_row + null_right
            for l_row in left_iter:
                yield l_row + null_right

        if not r_exhausted and self.how == "full":
            yield null_left + r_row
            for r_row in right_iter:
                yield null_left + r_row

    def children(self) -> list[PlanNode]:
        return [self.left, self.right]

    def _explain_self(self) -> str:
        return f"SortedMergeJoin [{self.how}] {self.left_on} = {self.right_on}"


class SortNode(PlanNode):
    """Sorts all rows by one or more columns using Timsort.

    Materialises the full input before sorting.  Multi-column sorts
    are composed via sequential stable sorts in reverse priority order.

    Args:
        child: Input plan node.
        by: Column names to sort by.
        ascending: Sort direction per column.  Defaults to ascending for all.
    """

    __slots__ = ("child", "by", "ascending")

    def __init__(self, child: PlanNode, by: list[str], ascending: list[bool] | None = None):
        self.child = child
        self.by = by
        self.ascending = ascending or [True] * len(by)

    def schema(self) -> LazySchema:
        return self.child.schema()

    def execute_batched(self) -> Iterator[list[tuple]]:
        col_map = {n: i for i, n in enumerate(self.child.schema().column_names)}
        indices = [col_map[c] for c in self.by]
        data: list = []
        for chunk in self.child.execute_batched():
            data.extend(chunk)
        for idx, asc in reversed(list(zip(indices, self.ascending))):
            data.sort(key=lambda r: (r[idx] is None, r[idx]), reverse=not asc)
        for i in range(0, len(data), _BATCH_SIZE):
            yield data[i : i + _BATCH_SIZE]

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        parts = [f"{c} {'↑' if a else '↓'}" for c, a in zip(self.by, self.ascending)]
        return f"Sort [{', '.join(parts)}]"


class ExplodeNode(PlanNode):
    """Unnests a list-valued column into separate rows.

    For each row, the list in *column* is expanded so that every element
    produces its own row.  ``None`` values pass through unchanged.

    Args:
        child: Input plan node.
        column: Name of the list-valued column to explode.
    """

    __slots__ = ("child", "column")

    def __init__(self, child: PlanNode, column: str):
        self.child = child
        self.column = column

    def schema(self) -> LazySchema:
        return self.child.schema()

    def execute_batched(self) -> Iterator[list[tuple]]:
        col_map = {n: i for i, n in enumerate(self.child.schema().column_names)}
        idx = col_map[self.column]
        buf: list = []
        for chunk in self.child.execute_batched():
            for row in chunk:
                vals = row[idx]
                if vals is None:
                    buf.append(row)
                else:
                    for val in vals:
                        buf.append(row[:idx] + (val,) + row[idx + 1 :])
                if len(buf) >= _BATCH_SIZE:
                    yield buf
                    buf = []
        if buf:
            yield buf

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        return f"Explode [{self.column}]"


class UnpivotNode(PlanNode):
    """Converts columns into rows (melt / unpivot).

    Keeps *id_columns* fixed and transforms each *value_columns* entry
    into a ``(variable, value)`` row pair, similar to pandas ``melt``
    or SQL ``UNPIVOT``.

    Args:
        child: Input plan node.
        id_columns: Columns to keep as identifiers.
        value_columns: Columns to unpivot into rows.
        variable_name: Name for the new column holding the original column names.
        value_name: Name for the new column holding the values.
    """

    __slots__ = ("child", "id_columns", "value_columns", "variable_name", "value_name")

    def __init__(
        self,
        child: PlanNode,
        id_columns: list[str],
        value_columns: list[str],
        variable_name: str = "variable",
        value_name: str = "value",
    ):
        self.child = child
        self.id_columns = id_columns
        self.value_columns = value_columns
        self.variable_name = variable_name
        self.value_name = value_name

    def schema(self) -> LazySchema:
        parent = self.child.schema()
        cols: dict[str, ColumnSchema] = {}
        for name in self.id_columns:
            cols[name] = parent[name]
        cols[self.variable_name] = ColumnSchema(self.variable_name, str)
        dtypes = {parent[c].dtype for c in self.value_columns}
        if len(dtypes) == 1:
            val_dtype = dtypes.pop()
        elif dtypes <= {int, float}:
            val_dtype = float
        else:
            val_dtype = str
        cols[self.value_name] = ColumnSchema(self.value_name, val_dtype)
        return LazySchema(cols)

    def execute_batched(self) -> Iterator[list[tuple]]:
        parent_cols = self.child.schema().column_names
        col_map = {n: i for i, n in enumerate(parent_cols)}
        id_idx = [col_map[c] for c in self.id_columns]
        val_idx = [(col_map[c], c) for c in self.value_columns]
        id_getter = itemgetter(*id_idx) if len(id_idx) > 1 else None
        single_id = id_idx[0] if len(id_idx) == 1 else None
        buf: list = []
        for chunk in self.child.execute_batched():
            for row in chunk:
                if id_getter:
                    id_vals = id_getter(row)
                elif single_id is not None:
                    id_vals = (row[single_id],)
                else:
                    id_vals = ()
                for vi, vname in val_idx:
                    buf.append(id_vals + (vname, row[vi]))
                if len(buf) >= _BATCH_SIZE:
                    yield buf
                    buf = []
        if buf:
            yield buf

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        return (
            f"Unpivot id=[{', '.join(self.id_columns)}] "
            f"values=[{', '.join(self.value_columns)}] "
            f"→ {self.variable_name}, {self.value_name}"
        )


class PivotNode(PlanNode):
    """Converts rows into columns (pivot / spread).

    Groups rows by *index* columns and spreads the distinct values in
    *on* into new columns, aggregating *values* with *agg_name*.
    Pivot columns are auto-discovered from the data unless *columns*
    is explicitly provided.

    Args:
        child: Input plan node.
        index: Columns to keep as the row index.
        on: Column whose distinct values become new column headers.
        values: Column containing the values to fill the pivoted cells.
        agg_name: Aggregation function applied when multiple values map to
            the same cell.
        columns: Explicit list of pivot columns.  Auto-discovered if ``None``.
    """

    __slots__ = (
        "child",
        "index",
        "on",
        "values",
        "agg_name",
        "columns",
        "_cached_data",
        "_cached_columns",
    )

    def __init__(
        self,
        child: PlanNode,
        index: list[str],
        on: str,
        values: str,
        agg_name: AggFunc = "first",
        columns: list[str] | None = None,
    ):
        self.child = child
        self.index = index
        self.on = on
        self.values = values
        self.agg_name = agg_name
        self.columns = columns
        self._cached_data: list[tuple] | None = None
        self._cached_columns: list[str] | None = None

    def _discover_columns(self) -> list[str]:
        if self.columns is not None:
            return self.columns
        if self._cached_columns is not None:
            return self._cached_columns
        parent_cols = self.child.schema().column_names
        col_map = {n: i for i, n in enumerate(parent_cols)}
        on_idx = col_map[self.on]
        data: list[tuple] = []
        seen_vals: list = []
        seen_set: set = set()
        for chunk in self.child.execute_batched():
            data.extend(chunk)
            for row in chunk:
                v = row[on_idx]
                if v not in seen_set:
                    seen_set.add(v)
                    seen_vals.append(str(v))
        self._cached_data = data
        self._cached_columns = seen_vals
        return seen_vals

    def schema(self) -> LazySchema:
        parent = self.child.schema()
        cols: dict[str, ColumnSchema] = {}
        for name in self.index:
            cols[name] = parent[name]
        val_dtype = parent[self.values].dtype
        for pv in self._discover_columns():
            cols[pv] = ColumnSchema(pv, val_dtype, nullable=True)
        return LazySchema(cols)

    def execute_batched(self) -> Iterator[list[tuple]]:
        parent_cols = self.child.schema().column_names
        col_map = {n: i for i, n in enumerate(parent_cols)}
        idx_indices = [col_map[c] for c in self.index]
        on_idx = col_map[self.on]
        val_idx = col_map[self.values]
        key_fn = _make_key_fn(idx_indices)
        pivot_cols = self._discover_columns()
        pv_map = {v: i for i, v in enumerate(pivot_cols)}
        n_pv = len(pivot_cols)
        agg_name = self.agg_name

        groups: dict[tuple, list[dict]] = {}

        source = self._cached_data if self._cached_data is not None else None
        if source is not None:
            for row in source:
                key = key_fn(row)
                if key not in groups:
                    groups[key] = [_init_pivot_acc(agg_name) for _ in range(n_pv)]
                pv_key = str(row[on_idx])
                if pv_key in pv_map:
                    _update_pivot_acc(groups[key][pv_map[pv_key]], agg_name, row[val_idx])
        else:
            for chunk in self.child.execute_batched():
                for row in chunk:
                    key = key_fn(row)
                    if key not in groups:
                        groups[key] = [_init_pivot_acc(agg_name) for _ in range(n_pv)]
                    pv_key = str(row[on_idx])
                    if pv_key in pv_map:
                        _update_pivot_acc(groups[key][pv_map[pv_key]], agg_name, row[val_idx])

        buf: list = []
        for key, accs in groups.items():
            vals = tuple(_finalize_pivot_acc(a, agg_name) for a in accs)
            buf.append(key + vals)
            if len(buf) >= _BATCH_SIZE:
                yield buf
                buf = []
        if buf:
            yield buf

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        cols_str = str(self.columns) if self.columns else "auto"
        return (
            f"Pivot index=[{', '.join(self.index)}] on={self.on} "
            f"values={self.values} agg={self.agg_name} columns={cols_str}"
        )


class ApplyNode(PlanNode):
    """Applies a scalar function to one or more columns.

    When *columns* is provided, only those columns are transformed;
    otherwise *func* is applied to every value in every column.

    Args:
        child: Input plan node.
        func: Callable applied element-wise to each value.
        columns: Columns to transform.  If ``None``, all columns are transformed.
        output_dtype: Optional output type to record in the schema.
    """

    __slots__ = ("child", "func", "_columns", "_output_dtype")

    def __init__(
        self,
        child: PlanNode,
        func: Callable[..., Any],
        columns: list[str] | None = None,
        output_dtype: type | None = None,
    ):
        self.child = child
        self.func = func
        self._columns = columns
        self._output_dtype = output_dtype

    def schema(self) -> LazySchema:
        parent = self.child.schema()
        if self._output_dtype and self._columns:
            s = parent
            for c in self._columns:
                s = s.with_dtype(c, self._output_dtype)
            return s
        return parent

    def execute_batched(self) -> Iterator[list[tuple]]:
        col_map = {n: i for i, n in enumerate(self.child.schema().column_names)}
        if self._columns:
            target = frozenset(col_map[c] for c in self._columns if c in col_map)
            fn = self.func
            for chunk in self.child.execute_batched():
                yield [
                    tuple(fn(v) if i in target else v for i, v in enumerate(row)) for row in chunk
                ]
        else:
            fn = self.func
            for chunk in self.child.execute_batched():
                yield [tuple(map(fn, row)) for row in chunk]

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        if self._columns:
            return f"Apply [{', '.join(self._columns)}]"
        return "Apply [all]"


class WithColumnNode(PlanNode):
    """Appends a new computed column to the output.

    Evaluates *expr* for every row and adds the result as a new column
    named *name*.  The output schema is the parent schema plus the new
    column.

    Args:
        child: Input plan node.
        name: Name for the new column.
        expr: Expression to evaluate for each row.
    """

    __slots__ = ("child", "_name", "_expr")

    def __init__(self, child: PlanNode, name: str, expr: Expr):
        self.child = child
        self._name = name
        self._expr = expr

    def schema(self) -> LazySchema:
        parent = self.child.schema()
        dtype = self._expr.output_dtype(parent)
        return parent.with_column(self._name, dtype)

    def execute_batched(self) -> Iterator[list[tuple]]:
        parent_cols = self.child.schema().column_names
        col_map = {n: i for i, n in enumerate(parent_cols)}
        fn = self._expr.compile(col_map)
        if self._name in col_map:
            idx = col_map[self._name]
            for chunk in self.child.execute_batched():
                yield [row[:idx] + (fn(row),) + row[idx + 1 :] for row in chunk]
        else:
            for chunk in self.child.execute_batched():
                yield [row + (fn(row),) for row in chunk]

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        return f"WithColumn [{self._name} = {self._expr}]"


class RenameNode(PlanNode):
    """Renames columns without modifying data.

    A zero-cost metadata-only operation — the underlying rows are passed
    through unchanged, only the schema is updated.

    Args:
        child: Input plan node.
        mapping: Dictionary mapping old column names to new names.
    """

    __slots__ = ("child", "mapping")

    def __init__(self, child: PlanNode, mapping: dict[str, str]):
        self.child = child
        self.mapping = mapping

    def schema(self) -> LazySchema:
        return self.child.schema().rename(self.mapping)

    def execute_batched(self) -> Iterator[list[tuple]]:
        return self.child.execute_batched()

    def fast_count(self) -> int | None:
        return self.child.fast_count()

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        pairs = [f"{k}→{v}" for k, v in self.mapping.items()]
        return f"Rename [{', '.join(pairs)}]"


class UnionNode(PlanNode):
    """Concatenates rows from multiple input nodes (SQL ``UNION ALL``).

    The output schema is taken from the first child.  All children
    must produce rows with compatible schemas.

    Args:
        children_nodes: List of input plan nodes to concatenate.
    """

    __slots__ = ("_children",)

    def __init__(self, children_nodes: list[PlanNode]):
        self._children = children_nodes

    def schema(self) -> LazySchema:
        return self._children[0].schema()

    def execute_batched(self) -> Iterator[list[tuple]]:
        return chain.from_iterable(c.execute_batched() for c in self._children)

    def children(self) -> list[PlanNode]:
        return self._children

    def _explain_self(self) -> str:
        return f"Union [{len(self._children)} inputs]"


class WindowNode(PlanNode):
    """Evaluates a window function and appends the result as a new column.

    Implements the sort-partition-scan pattern: materialises all rows,
    partitions by key columns, sorts within each partition, then computes
    the window expression (rank, cumulative, offset, or aggregate).

    Args:
        child: Input plan node.
        window_expr: The window expression to evaluate.
        output_name: Name for the new column containing window results.
    """

    __slots__ = ("child", "window_expr", "_output_name")

    def __init__(self, child: PlanNode, window_expr: WindowExpr, output_name: str):
        self.child = child
        self.window_expr = window_expr
        self._output_name = output_name

    def schema(self) -> LazySchema:
        parent = self.child.schema()
        dtype = self.window_expr.output_dtype(parent)
        return parent.with_column(self._output_name, dtype)

    def execute_batched(self) -> Iterator[list[tuple]]:
        parent_schema = self.child.schema()
        col_map = {n: i for i, n in enumerate(parent_schema.column_names)}
        data: list = []
        for chunk in self.child.execute_batched():
            data.extend(chunk)
        if not data:
            return

        wexpr = self.window_expr
        p_idx = [col_map[p] for p in wexpr.partition_by]
        o_idx = [col_map[o] for o in wexpr.order_by]

        p_key = _make_key_fn(p_idx) if p_idx else None
        (
            itemgetter(*o_idx)
            if len(o_idx) > 1
            else (lambda x, _i=o_idx[0]: x[1][_i])
            if o_idx
            else None
        )
        if len(o_idx) > 1:
            _o_getter = itemgetter(*o_idx)

            def o_sort_key(x):
                return _o_getter(x[1])
        elif o_idx:
            _oi = o_idx[0]

            def o_sort_key(x):
                return x[1][_oi]
        else:
            o_sort_key = None  # type: ignore[assignment]

        partitions: dict[tuple, list] = defaultdict(list)
        for i, row in enumerate(data):
            key = p_key(row) if p_key else ()
            partitions[key].append((i, row))

        window_values: list[Any] = [None] * len(data)
        inner = wexpr.expr

        for _key, part in partitions.items():
            if o_sort_key is not None:
                part.sort(key=o_sort_key)

            if isinstance(inner, AggExpr):
                val_fn = inner.expr.compile(col_map)
                values = [val_fn(r) for _, r in part]
                agg_val = inner.eval_agg(values)
                for orig_i, _ in part:
                    window_values[orig_i] = agg_val

            elif isinstance(inner, RankExpr):
                prev_val = None
                rank_val = 0
                dense_val = 0
                o_key_fn = _make_key_fn(o_idx) if o_idx else None
                for pos, (orig_i, row) in enumerate(part):
                    cur = o_key_fn(row) if o_key_fn else pos
                    if inner.kind == "row_number":
                        window_values[orig_i] = pos + 1
                    elif inner.kind == "rank":
                        if cur != prev_val:
                            rank_val = pos + 1
                            prev_val = cur
                        window_values[orig_i] = rank_val
                    elif inner.kind == "dense_rank":
                        if cur != prev_val:
                            dense_val += 1
                            prev_val = cur
                        window_values[orig_i] = dense_val

            elif isinstance(inner, CumExpr):
                val_fn = inner.expr.compile(col_map)
                values = [val_fn(r) for _, r in part]
                if inner.kind == "cumsum":

                    def op(a, b):
                        return a + b
                elif inner.kind == "cummax":

                    def op(a, b):
                        return max(a, b)
                elif inner.kind == "cummin":

                    def op(a, b):
                        return min(a, b)
                else:

                    def op(a, b):
                        return b

                if not any(v is None for v in values):
                    running = list(accumulate(values, op))
                else:
                    running = []
                    acc = None
                    for v in values:
                        if v is None:
                            running.append(acc)
                        elif acc is None:
                            acc = v
                            running.append(acc)
                        else:
                            acc = op(acc, v)
                            running.append(acc)

                for (orig_i, _), val in zip(part, running):
                    window_values[orig_i] = val

            elif isinstance(inner, OffsetExpr):
                val_fn = inner.expr.compile(col_map)
                vals = [val_fn(r) for _, r in part]
                n = len(vals)
                for pos, (orig_i, _) in enumerate(part):
                    src = pos + inner.offset
                    if 0 <= src < n:
                        window_values[orig_i] = vals[src]
                    else:
                        window_values[orig_i] = inner.default

        n_data = len(data)
        for i in range(0, n_data, _BATCH_SIZE):
            end = min(i + _BATCH_SIZE, n_data)
            yield [data[j] + (window_values[j],) for j in range(i, end)]

    def children(self) -> list[PlanNode]:
        return [self.child]

    def _explain_self(self) -> str:
        return f"Window [{self.window_expr}] → {self._output_name}"


class Optimizer:
    """Rule-based query optimizer.

    Applies two optimization passes to a query plan:

    1. **Filter pushdown** — moves filter nodes closer to data sources,
       reducing the number of rows processed by downstream operations.
    2. **Column pruning** — removes unused columns early in the plan,
       reducing memory usage.
    """

    def optimize(self, plan: PlanNode) -> PlanNode:
        """Apply all optimization passes to a plan tree.

        Args:
            plan: Root node of the query plan to optimize.

        Returns:
            A new, semantically equivalent plan tree with optimizations applied.
        """
        plan = self._push_filters(plan)
        needed = set(plan.schema().column_names)
        plan = self._prune_columns(plan, needed)
        return plan

    def _push_filters(self, node: PlanNode) -> PlanNode:
        if isinstance(node, FilterNode):
            child = self._push_filters(node.child)
            return self._try_push_filter(node.predicate, child)
        if isinstance(node, ProjectNode):
            return ProjectNode(self._push_filters(node.child), node._columns, node._exprs)
        if isinstance(node, JoinNode):
            return JoinNode(
                self._push_filters(node.left),
                self._push_filters(node.right),
                node.left_on,
                node.right_on,
                node.how,
            )
        if isinstance(node, LimitNode):
            return LimitNode(self._push_filters(node.child), node.n)
        if isinstance(node, SortNode):
            return SortNode(self._push_filters(node.child), node.by, node.ascending)
        if isinstance(node, WithColumnNode):
            return WithColumnNode(self._push_filters(node.child), node._name, node._expr)
        if isinstance(node, AggNode):
            return AggNode(self._push_filters(node.child), node.group_by, node.agg_exprs)
        if isinstance(node, SortedAggNode):
            return SortedAggNode(self._push_filters(node.child), node.group_by, node.agg_exprs)
        if isinstance(node, PivotNode):
            return PivotNode(
                self._push_filters(node.child),
                node.index,
                node.on,
                node.values,
                node.agg_name,
                node.columns,
            )
        if isinstance(node, UnpivotNode):
            return UnpivotNode(
                self._push_filters(node.child),
                node.id_columns,
                node.value_columns,
                node.variable_name,
                node.value_name,
            )
        return node

    def _try_push_filter(self, predicate: Expr, child: PlanNode) -> PlanNode:
        needed = predicate.required_columns()

        if isinstance(child, ProjectNode) and child._columns:
            child_has = set(child.child.schema().column_names)
            if needed <= child_has:
                pushed = FilterNode(child.child, predicate)
                return ProjectNode(pushed, child._columns, child._exprs)

        if isinstance(child, (AggNode, SortedAggNode)):
            group_keys = set(child.group_by)
            if needed <= group_keys:
                pushed = FilterNode(child.child, predicate)
                return type(child)(pushed, child.group_by, child.agg_exprs)

        if isinstance(child, JoinNode):
            left_cols = set(child.left.schema().column_names)
            right_cols = set(child.right.schema().column_names)
            if needed <= left_cols:
                return JoinNode(
                    FilterNode(child.left, predicate),
                    child.right,
                    child.left_on,
                    child.right_on,
                    child.how,
                )
            if needed <= right_cols:
                return JoinNode(
                    child.left,
                    FilterNode(child.right, predicate),
                    child.left_on,
                    child.right_on,
                    child.how,
                )

        if isinstance(child, PivotNode):
            if needed <= set(child.index):
                pushed = FilterNode(child.child, predicate)
                return PivotNode(
                    pushed, child.index, child.on, child.values, child.agg_name, child.columns
                )

        if isinstance(child, UnpivotNode):
            if needed <= set(child.id_columns):
                pushed = FilterNode(child.child, predicate)
                return UnpivotNode(
                    pushed,
                    child.id_columns,
                    child.value_columns,
                    child.variable_name,
                    child.value_name,
                )

        return FilterNode(child, predicate)

    def _prune_columns(self, node: PlanNode, needed: set[str]) -> PlanNode:
        if isinstance(node, ScanNode):
            available = set(node._columns)
            prune_to = [c for c in node._columns if c in needed]
            if len(prune_to) < len(available) and prune_to:
                return ProjectNode(node, prune_to)
            return node

        if isinstance(node, ProjectNode) and node._columns:
            pruned_cols = [c for c in node._columns if c in needed]
            if not pruned_cols:
                pruned_cols = node._columns
            child_needed = needed | set(pruned_cols)
            return ProjectNode(self._prune_columns(node.child, child_needed), pruned_cols)

        if isinstance(node, LimitNode):
            return LimitNode(self._prune_columns(node.child, needed), node.n)

        if isinstance(node, FilterNode):
            filter_needs = node.predicate.required_columns()
            return FilterNode(
                self._prune_columns(node.child, needed | filter_needs),
                node.predicate,
            )

        if isinstance(node, JoinNode):
            left_cols = set(node.left.schema().column_names)
            right_cols = set(node.right.schema().column_names)
            left_needed = (needed & left_cols) | set(node.left_on)
            right_needed = (needed & right_cols) | set(node.right_on)
            for c in needed:
                if c.startswith("right_"):
                    base = c[6:]
                    if base in right_cols:
                        right_needed.add(base)
            return JoinNode(
                self._prune_columns(node.left, left_needed),
                self._prune_columns(node.right, right_needed),
                node.left_on,
                node.right_on,
                node.how,
            )

        if isinstance(node, WithColumnNode):
            expr_needs = node._expr.required_columns()
            child_needed = (needed - {node._name}) | expr_needs
            return WithColumnNode(
                self._prune_columns(node.child, child_needed),
                node._name,
                node._expr,
            )

        if isinstance(node, AggNode):
            child_needed = set(node.group_by)
            for agg in node.agg_exprs:
                child_needed |= agg.required_columns()
            return AggNode(
                self._prune_columns(node.child, child_needed),
                node.group_by,
                node.agg_exprs,
            )

        if isinstance(node, WindowNode):
            child_needed = needed | node.window_expr.required_columns()
            return WindowNode(
                self._prune_columns(node.child, child_needed),
                node.window_expr,
                node._output_name,
            )

        if isinstance(node, PivotNode):
            child_needed = set(node.index) | {node.on, node.values}
            return PivotNode(
                self._prune_columns(node.child, child_needed),
                node.index,
                node.on,
                node.values,
                node.agg_name,
                node.columns,
            )

        if isinstance(node, UnpivotNode):
            child_needed = set(node.id_columns) | set(node.value_columns)
            return UnpivotNode(
                self._prune_columns(node.child, child_needed),
                node.id_columns,
                node.value_columns,
                node.variable_name,
                node.value_name,
            )

        return node
