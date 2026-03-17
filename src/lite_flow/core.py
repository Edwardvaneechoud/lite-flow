from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import islice
from typing import (
    Any,
    Generic,
    TypeVar,
    get_type_hints,
)

from .expr import AggExpr, Col, Expr, WindowExpr
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

T = TypeVar('T')


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


class GroupByBuilder:
    def __init__(self, ff: Floe, group_cols: list[str], sorted: bool = False):
        self._ff = ff
        self._group_cols = group_cols
        self._sorted = sorted

    def agg(self, *agg_exprs: AggExpr) -> Floe:
        for i, expr in enumerate(agg_exprs):
            if not isinstance(expr, AggExpr):
                raise TypeError(
                    f'agg() argument {i + 1} is a {type(expr).__name__}, not an AggExpr. '
                    f'Use an aggregation method like col("x").sum(), .count(), '
                    f'.mean(), .min(), .max(), .first(), .last(), .n_unique()'
                )
        if self._sorted:
            node = SortedAggNode(self._ff._plan, self._group_cols, list(agg_exprs))
        else:
            node = AggNode(self._ff._plan, self._group_cols, list(agg_exprs))
        return Floe._from_plan(node)


class Floe:
    __slots__ = ('_plan', '_materialized', '_name', '_optimized')

    def __init__(self, raw_data: list[dict] | list = None, *,
                 name: str = None):
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
            values = [list(v) if not isinstance(v, list) else v
                      for v in raw_data.values()]
            n_rows = len(values[0])
            rows = [tuple(values[c][r] for c in range(len(cols)))
                    for r in range(n_rows)]
            self._plan = ScanNode(rows, cols)
        elif isinstance(raw_data, list):
            if not raw_data:
                self._plan = ScanNode([], [])
                return
            first = raw_data[0]
            if isinstance(first, dict):
                cols, rows = _dicts_to_tuples(raw_data)
            elif hasattr(first, '__dict__') and not isinstance(first, tuple):
                cols, rows = _dicts_to_tuples([r.__dict__ for r in raw_data])
            elif isinstance(first, (tuple, list)):
                width = len(first)
                cols = [f'col_{i}' for i in range(width)]
                rows = [tuple(r) for r in raw_data]
            else:
                cols = ['value']
                rows = [(v,) for v in raw_data]
            self._plan = ScanNode(rows, cols)
        else:
            raise ValueError(
                'Provide a list of dicts, list of tuples, list of scalars, '
                'or a dict of columns'
            )

    @classmethod
    def _from_plan(cls, plan: PlanNode, name: str = None) -> Floe:
        ff = cls.__new__(cls)
        ff._plan = plan
        ff._materialized = None
        ff._name = name
        ff._optimized = None
        return ff

    @property
    def _exec_plan(self) -> PlanNode:
        if self._optimized is None:
            self._optimized = Optimizer().optimize(self._plan)
        return self._optimized

    @property
    def schema(self) -> LazySchema:
        return self._plan.schema()

    @property
    def columns(self) -> list[str]:
        return self.schema.column_names

    @property
    def dtypes(self) -> dict[str, type]:
        return self.schema.dtypes

    def explain(self, optimized: bool = False) -> str:
        plan = self._plan
        if optimized:
            plan = Optimizer().optimize(plan)
        return plan.explain()

    def print_explain(self, optimized: bool = False):
        print(self.explain(optimized))

    def select(self, *args: str | Expr) -> Floe:
        if all(isinstance(a, str) for a in args):
            return Floe._from_plan(ProjectNode(self._plan, list(args)))
        exprs = [Col(a) if isinstance(a, str) else a for a in args]
        return Floe._from_plan(ProjectNode(self._plan, exprs=[e for e in exprs]))

    def filter(self, predicate_or_col=None, _filter=None, **kwargs) -> Floe:
        if isinstance(predicate_or_col, Expr) and _filter is None:
            return Floe._from_plan(FilterNode(self._plan, predicate_or_col))

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
                        return f'λ({", ".join(self._col_names)})'

                pred = _LegacyPredicate(cols, _filter)
                return Floe._from_plan(FilterNode(self._plan, pred))
            else:
                val = _filter
                col_name = predicate_or_col if isinstance(predicate_or_col, str) else predicate_or_col[0]
                pred = BinaryExpr(_Col(col_name), _Lit(val), _op.eq, '==')
                return Floe._from_plan(FilterNode(self._plan, pred))

        raise ValueError('Provide an Expr predicate or use legacy filter(col, _filter=...)')

    def with_column(self, name: str, expr: Expr) -> Floe:
        if isinstance(expr, WindowExpr):
            return Floe._from_plan(WindowNode(self._plan, expr, name))
        return Floe._from_plan(WithColumnNode(self._plan, name, expr))

    def with_columns(self, **kwargs: Expr) -> Floe:
        ff = self
        for name, expr in kwargs.items():
            ff = ff.with_column(name, expr)
        return ff

    def drop(self, *columns: str) -> Floe:
        keep = [c for c in self.columns if c not in set(columns)]
        return Floe._from_plan(ProjectNode(self._plan, keep))

    def rename(self, mapping: dict[str, str]) -> Floe:
        return Floe._from_plan(RenameNode(self._plan, mapping))

    def sort(self, *by: str, ascending: bool | list[bool] = True) -> Floe:
        by_list = list(by)
        if isinstance(ascending, bool):
            asc_list = [ascending] * len(by_list)
        else:
            asc_list = list(ascending)
        return Floe._from_plan(SortNode(self._plan, by_list, asc_list))

    def join(self, other: Floe, on: str | list[str] = None,
             left_on: str | list[str] = None,
             right_on: str | list[str] = None,
             how: str = 'inner',
             sorted: bool = False,
             left_cols=None, right_cols=None) -> Floe:
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
            return Floe._from_plan(
                SortedMergeJoinNode(self._plan, other._plan, left_on, right_on, how)
            )
        return Floe._from_plan(
            JoinNode(self._plan, other._plan, left_on, right_on, how)
        )

    def group_by(self, *columns: str, sorted: bool = False, **legacy_kwargs) -> GroupByBuilder | Floe:
        cols = list(columns)

        agg_func = legacy_kwargs.get('agg_func')
        on_cols = legacy_kwargs.get('on_cols')
        if agg_func is not None:
            if isinstance(on_cols, str):
                on_cols = [on_cols]
            elif on_cols is None:
                on_cols = [c for c in self.columns if c not in cols]
            agg_exprs = []
            for oc in on_cols:
                class _LegacyAgg(AggExpr):
                    def __init__(self, col_name, func):
                        super().__init__(Col(col_name), 'agg', func)
                        self._alias = col_name
                agg_exprs.append(_LegacyAgg(oc, agg_func))
            return Floe._from_plan(AggNode(self._plan, cols, agg_exprs))

        return GroupByBuilder(self, cols, sorted=sorted)

    def explode(self, column: str) -> Floe:
        return Floe._from_plan(ExplodeNode(self._plan, column))

    def pivot(self, index: str | list[str], on: str, values: str,
              agg: str = 'first', columns: list[str] | None = None) -> Floe:
        if isinstance(index, str):
            index = [index]
        return Floe._from_plan(
            PivotNode(self._plan, index, on, values, agg, columns)
        )

    def unpivot(self, id_columns: str | list[str],
                value_columns: str | list[str] | None = None,
                variable_name: str = 'variable',
                value_name: str = 'value') -> Floe:
        if isinstance(id_columns, str):
            id_columns = [id_columns]
        if value_columns is None:
            value_columns = [c for c in self.columns if c not in set(id_columns)]
        elif isinstance(value_columns, str):
            value_columns = [value_columns]
        return Floe._from_plan(
            UnpivotNode(self._plan, id_columns, value_columns,
                        variable_name, value_name)
        )

    melt = unpivot

    def union(self, other: Floe) -> Floe:
        return Floe._from_plan(UnionNode([self._plan, other._plan]))

    def apply(self, func: Callable, columns: list[str] = None,
              output_dtype: type = None) -> Floe:
        return Floe._from_plan(ApplyNode(self._plan, func, columns, output_dtype))

    def read(self, columns: str | list[str]) -> Floe:
        if isinstance(columns, str):
            columns = [columns]
        return self.select(*columns)

    def head(self, n: int = 5, optimize: bool = True) -> Floe:
        plan = self._exec_plan if optimize else self._plan
        rows = list(islice(plan.execute(), n))
        return Floe._from_plan(ScanNode(rows, self.columns, self.schema))

    def optimize(self) -> Floe:
        optimized_plan = Optimizer().optimize(self._plan)
        return Floe._from_plan(optimized_plan, self._name)

    @property
    def is_materialized(self) -> bool:
        return self._materialized is not None

    @property
    def raw_data(self) -> list[tuple]:
        if self._materialized is None:
            data: list = []
            for chunk in self._plan.execute_batched():
                data.extend(chunk)
            self._materialized = data
        return self._materialized

    def collect(self, optimize: bool = True) -> Floe:
        if self._materialized is None:
            plan = self._exec_plan if optimize else self._plan
            data: list = []
            for chunk in plan.execute_batched():
                data.extend(chunk)
            self._materialized = data
        return self

    def count(self, optimize: bool = True) -> int:
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
        cols = self.columns
        return [{cols[i]: v for i, v in enumerate(row)} for row in self.raw_data]

    def to_pydict(self) -> dict[str, list]:
        cols = self.columns
        data = self.raw_data
        return {c: [row[i] for row in data] for i, c in enumerate(cols)}

    def to_tuples(self) -> list[tuple]:
        return list(self.raw_data)

    def to_csv(self, path: str, *, delimiter: str = ',',
               header: bool = True, encoding: str = 'utf-8'):
        from .io import _to_csv_impl
        _to_csv_impl(self, path, delimiter, header, encoding)

    def to_tsv(self, path: str, **kwargs):
        kwargs.setdefault('delimiter', '\t')
        self.to_csv(path, **kwargs)

    def to_jsonl(self, path: str, *, encoding: str = 'utf-8'):
        from .io import _to_jsonl_impl
        _to_jsonl_impl(self, path, encoding)

    def to_json(self, path: str, *, encoding: str = 'utf-8', indent: int = None):
        from .io import _to_json_impl
        _to_json_impl(self, path, encoding, indent)

    def to_parquet(self, path: str, **kwargs):
        from .io import _to_parquet_impl
        _to_parquet_impl(self, path, **kwargs)

    def __iter__(self) -> Iterator[dict]:
        cols = self.columns
        source = self._materialized if self._materialized is not None else self._exec_plan.execute()
        for row in source:
            yield {cols[i]: v for i, v in enumerate(row)}

    def __len__(self) -> int:
        n = self._known_length
        if n is not None:
            return n
        raise TypeError(
            'len() requires materialized data. '
            'Call .collect() first, or use .head(n) to peek at rows.'
        )

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.select(key)
        if isinstance(key, int):
            row = self.raw_data[key]
            return {c: v for c, v in zip(self.columns, row)}
        if isinstance(key, slice):
            rows = self.raw_data[key]
            return Floe._from_plan(ScanNode(rows, self.columns, self.schema))
        raise TypeError(f'Invalid key type: {type(key)}')

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
        col_str = ', '.join(cols[:8])
        if ncols > 8:
            col_str += f', ... +{ncols - 8} more'
        n = self._known_length
        if self._materialized is not None:
            return f'Floe [{n} rows × {ncols} cols] (materialized)\nColumns: [{col_str}]'
        if n is not None:
            return f'Floe [{n} rows × {ncols} cols]\nColumns: [{col_str}]'
        return f'Floe [? rows × {ncols} cols] (lazy)\nColumns: [{col_str}]'

    def _repr_short(self):
        n = self._known_length
        if n is not None:
            return f'Floe[{n}×{len(self.columns)}]'
        return f'Floe[?×{len(self.columns)}]'

    def display(self, n: int = 20, max_col_width: int = 30,
                optimize: bool = True) -> None:
        cols = self.columns
        plan = self._exec_plan if optimize else self._plan
        sample = list(islice(plan.execute(), n))
        total = self._known_length

        def _fmt(v):
            s = '' if v is None else str(v)
            return s if len(s) <= max_col_width else s[:max_col_width - 1] + '…'

        str_rows = [[_fmt(v) for v in row] for row in sample]

        widths = [len(c) for c in cols]
        for row in str_rows:
            for i, cell in enumerate(row):
                if len(cell) > widths[i]:
                    widths[i] = len(cell)

        hdr = ' | '.join(c.ljust(widths[i]) for i, c in enumerate(cols))
        sep = '-+-'.join('-' * w for w in widths)
        lines = [hdr, sep]
        for row in str_rows:
            lines.append(' | '.join(row[i].ljust(widths[i]) for i in range(len(cols))))

        if total is not None and total > n:
            lines.append(f'... {total - n} more rows')
        elif total is None and len(sample) == n:
            lines.append(f'... (showing first {n} rows)')

        print('\n'.join(lines))

    def typed(self, row_type: type[T]) -> TypedFloe[T]:
        return TypedFloe._from_typed(self._plan, row_type, self._name)

    def validate(self, row_type: type) -> Floe:
        hints = get_type_hints(row_type)
        schema = self.schema
        errors = []
        for col_name, expected_type in hints.items():
            if col_name not in schema:
                errors.append(f'  missing column: {col_name}')
            else:
                actual = schema[col_name].dtype
                if actual != expected_type and expected_type not in (Any,):
                    errors.append(f'  {col_name}: expected {expected_type.__name__}, got {actual.__name__}')
        if errors:
            raise TypeError('Schema validation failed:\n' + '\n'.join(errors))
        return self


class TypedFloe(Floe, Generic[T]):
    __slots__ = ('_row_type',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._row_type = None

    @classmethod
    def _from_typed(cls, plan: PlanNode, row_type: type[T],
                    name: str = None) -> TypedFloe[T]:
        ff = cls.__new__(cls)
        ff._plan = plan
        ff._materialized = None
        ff._name = name
        ff._row_type = row_type
        return ff

    @property
    def row_type(self) -> type[T]:
        return self._row_type

    def to_pylist(self) -> list[T]:
        return super().to_pylist()

    def collect(self, optimize: bool = True) -> TypedFloe[T]:
        super().collect(optimize)
        return self

    def filter(self, *args, **kwargs) -> TypedFloe[T]:
        result = super().filter(*args, **kwargs)
        return TypedFloe._from_typed(result._plan, self._row_type, result._name)

    def sort(self, *args, **kwargs) -> TypedFloe[T]:
        result = super().sort(*args, **kwargs)
        return TypedFloe._from_typed(result._plan, self._row_type, result._name)

    def head(self, n: int = 5, optimize: bool = True) -> TypedFloe[T]:
        result = super().head(n, optimize)
        return TypedFloe._from_typed(result._plan, self._row_type, result._name)

    def __repr__(self):
        base = super().__repr__()
        type_name = self._row_type.__name__ if self._row_type else '?'
        return base.replace('Floe', f'TypedFloe[{type_name}]', 1)
