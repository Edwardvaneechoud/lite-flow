from __future__ import annotations

import operator as _op
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .schema import LazySchema


def _ensure_expr(val) -> Expr:
    if isinstance(val, Expr):
        return val
    return Lit(val)


def _standardize_str_list(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    return list(val)


class Expr:
    def eval(self, row: tuple, col_map: dict) -> Any:
        raise NotImplementedError

    def compile(self, col_map: dict):
        return lambda row: self.eval(row, col_map)

    def required_columns(self) -> set[str]:
        raise NotImplementedError

    def output_dtype(self, schema: LazySchema) -> type:
        raise NotImplementedError

    def output_name(self) -> str | None:
        return None

    def __bool__(self):
        raise TypeError(
            'Cannot use an Expr as a Python bool. '
            'Use & / | for logical ops, and .collect() to evaluate.'
        )

    def __gt__(self, other):  return BinaryExpr(self, _ensure_expr(other), _op.gt, '>')
    def __ge__(self, other):  return BinaryExpr(self, _ensure_expr(other), _op.ge, '>=')
    def __lt__(self, other):  return BinaryExpr(self, _ensure_expr(other), _op.lt, '<')
    def __le__(self, other):  return BinaryExpr(self, _ensure_expr(other), _op.le, '<=')
    def __eq__(self, other):  return BinaryExpr(self, _ensure_expr(other), _op.eq, '==')
    def __ne__(self, other):  return BinaryExpr(self, _ensure_expr(other), _op.ne, '!=')

    def __and__(self, other): return BinaryExpr(self, _ensure_expr(other), lambda a, b: a and b, '&')
    def __or__(self, other):  return BinaryExpr(self, _ensure_expr(other), lambda a, b: a or b, '|')
    def __invert__(self):     return UnaryExpr(self, _op.not_, '~')

    def __add__(self, other):      return BinaryExpr(self, _ensure_expr(other), _op.add, '+')
    def __radd__(self, other):     return BinaryExpr(_ensure_expr(other), self, _op.add, '+')
    def __sub__(self, other):      return BinaryExpr(self, _ensure_expr(other), _op.sub, '-')
    def __rsub__(self, other):     return BinaryExpr(_ensure_expr(other), self, _op.sub, '-')
    def __mul__(self, other):      return BinaryExpr(self, _ensure_expr(other), _op.mul, '*')
    def __rmul__(self, other):     return BinaryExpr(_ensure_expr(other), self, _op.mul, '*')
    def __truediv__(self, other):  return BinaryExpr(self, _ensure_expr(other), _op.truediv, '/')
    def __rtruediv__(self, other): return BinaryExpr(_ensure_expr(other), self, _op.truediv, '/')
    def __floordiv__(self, other): return BinaryExpr(self, _ensure_expr(other), _op.floordiv, '//')
    def __mod__(self, other):      return BinaryExpr(self, _ensure_expr(other), _op.mod, '%')
    def __neg__(self):             return UnaryExpr(self, _op.neg, '-')

    def alias(self, name: str) -> AliasExpr:
        return AliasExpr(self, name)

    def cast(self, dtype: type) -> CastExpr:
        return CastExpr(self, dtype)

    def is_null(self) -> UnaryExpr:
        return UnaryExpr(self, lambda x: x is None, 'is_null')

    def is_not_null(self) -> UnaryExpr:
        return UnaryExpr(self, lambda x: x is not None, 'is_not_null')

    def is_in(self, values) -> UnaryExpr:
        s = frozenset(values)
        return UnaryExpr(self, lambda x: x in s, f'is_in({list(values)!r})')

    @property
    def str(self) -> StringAccessor:
        return StringAccessor(self)

    @property
    def dt(self) -> DateTimeAccessor:
        return DateTimeAccessor(self)

    def sum(self)     -> AggExpr: return AggExpr(self, 'sum', sum)
    def mean(self)    -> AggExpr: return AggExpr(self, 'mean', lambda v: sum(v)/len(v) if v else 0)
    def min(self)     -> AggExpr: return AggExpr(self, 'min', min)
    def max(self)     -> AggExpr: return AggExpr(self, 'max', max)
    def count(self)   -> AggExpr: return AggExpr(self, 'count', len)
    def first(self)   -> AggExpr: return AggExpr(self, 'first', lambda v: v[0] if v else None)
    def last(self)    -> AggExpr: return AggExpr(self, 'last', lambda v: v[-1] if v else None)
    def n_unique(self)-> AggExpr: return AggExpr(self, 'n_unique', lambda v: len(set(v)))

    def cumsum(self)  -> CumExpr: return CumExpr(self, 'cumsum')
    def cummax(self)  -> CumExpr: return CumExpr(self, 'cummax')
    def cummin(self)  -> CumExpr: return CumExpr(self, 'cummin')
    def lag(self, n: int = 1, default=None) -> OffsetExpr: return OffsetExpr(self, -n, default)
    def lead(self, n: int = 1, default=None) -> OffsetExpr: return OffsetExpr(self, n, default)

    def __hash__(self):
        return id(self)


class Col(Expr):
    __slots__ = ('name',)

    def __init__(self, name: str):
        self.name = name

    def eval(self, row, col_map):
        return row[col_map[self.name]]

    def compile(self, col_map):
        idx = col_map[self.name]
        return lambda row: row[idx]

    def required_columns(self):
        return {self.name}

    def output_dtype(self, schema):
        if self.name in schema:
            return schema[self.name].dtype
        return str

    def output_name(self):
        return self.name

    def __repr__(self):
        return f'col("{self.name}")'


class Lit(Expr):
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def eval(self, row, col_map):
        return self.value

    def compile(self, col_map):
        v = self.value
        return lambda row: v

    def required_columns(self):
        return set()

    def output_dtype(self, schema):
        if self.value is None:
            return type(None)
        return type(self.value)

    def __repr__(self):
        return repr(self.value)


class BinaryExpr(Expr):
    __slots__ = ('left', 'right', 'op', 'op_str')

    def __init__(self, left: Expr, right: Expr, op, op_str: str):
        self.left = left
        self.right = right
        self.op = op
        self.op_str = op_str

    def eval(self, row, col_map):
        lv = self.left.eval(row, col_map)
        rv = self.right.eval(row, col_map)
        if lv is None or rv is None:
            return None
        return self.op(lv, rv)

    def compile(self, col_map):
        left_fn = self.left.compile(col_map)
        right_fn = self.right.compile(col_map)
        op = self.op
        if self.op_str in ('&', '|'):
            return lambda row: op(left_fn(row), right_fn(row))
        def _eval(row):
            lv = left_fn(row)
            rv = right_fn(row)
            if lv is None or rv is None:
                return None
            return op(lv, rv)
        return _eval

    def required_columns(self):
        return self.left.required_columns() | self.right.required_columns()

    def output_dtype(self, schema):
        if self.op_str in ('>', '>=', '<', '<=', '==', '!=', '&', '|'):
            return bool
        return self.left.output_dtype(schema)

    def __repr__(self):
        return f'({self.left} {self.op_str} {self.right})'


class UnaryExpr(Expr):
    __slots__ = ('operand', 'op', 'op_str')

    def __init__(self, operand: Expr, op, op_str: str):
        self.operand = operand
        self.op = op
        self.op_str = op_str

    def eval(self, row, col_map):
        return self.op(self.operand.eval(row, col_map))

    def compile(self, col_map):
        inner = self.operand.compile(col_map)
        op = self.op
        return lambda row: op(inner(row))

    def required_columns(self):
        return self.operand.required_columns()

    def output_dtype(self, schema):
        if self.op_str in ('~', 'is_null', 'is_not_null') or self.op_str.startswith('str.'):
            if self.op_str.startswith('str.contains') or self.op_str.startswith('str.starts') or self.op_str.startswith('str.ends'):
                return bool
            if self.op_str == 'str.len':
                return int
        return self.operand.output_dtype(schema)

    def output_name(self):
        return self.operand.output_name()

    def __repr__(self):
        return f'{self.op_str}({self.operand})'


class AliasExpr(Expr):
    __slots__ = ('expr', '_name')

    def __init__(self, expr: Expr, name: str):
        self.expr = expr
        self._name = name

    def eval(self, row, col_map):
        return self.expr.eval(row, col_map)

    def compile(self, col_map):
        return self.expr.compile(col_map)

    def required_columns(self):
        return self.expr.required_columns()

    def output_dtype(self, schema):
        return self.expr.output_dtype(schema)

    def output_name(self):
        return self._name

    def __repr__(self):
        return f'{self.expr}.alias("{self._name}")'


class CastExpr(Expr):
    __slots__ = ('expr', '_dtype')

    def __init__(self, expr: Expr, dtype: type):
        self.expr = expr
        self._dtype = dtype

    def eval(self, row, col_map):
        val = self.expr.eval(row, col_map)
        if val is None:
            return None
        return self._dtype(val)

    def compile(self, col_map):
        inner = self.expr.compile(col_map)
        dtype = self._dtype
        def _eval(row):
            val = inner(row)
            return None if val is None else dtype(val)
        return _eval

    def required_columns(self):
        return self.expr.required_columns()

    def output_dtype(self, schema):
        return self._dtype

    def output_name(self):
        return self.expr.output_name()

    def __repr__(self):
        return f'{self.expr}.cast({self._dtype.__name__})'


class AggExpr(Expr):
    __slots__ = ('expr', 'agg_name', 'agg_func', '_alias')

    def __init__(self, expr: Expr, agg_name: str, agg_func):
        self.expr = expr
        self.agg_name = agg_name
        self.agg_func = agg_func
        self._alias: str | None = None

    def eval_agg(self, values: list) -> Any:
        filtered = [v for v in values if v is not None]
        if not filtered:
            return None
        return self.agg_func(filtered)

    def eval(self, row, col_map):
        return self.expr.eval(row, col_map)

    def required_columns(self):
        return self.expr.required_columns()

    def output_dtype(self, schema):
        if self.agg_name in ('count', 'n_unique'):
            return int
        if self.agg_name == 'mean':
            return float
        return self.expr.output_dtype(schema)

    def output_name(self):
        base = self.expr.output_name() or 'value'
        return self._alias or f'{base}_{self.agg_name}'

    def alias(self, name: str) -> AggExpr:
        self._alias = name
        return self

    def over(self, partition_by=None, order_by=None) -> WindowExpr:
        return WindowExpr(self, partition_by, order_by)

    def __repr__(self):
        a = f'.alias("{self._alias}")' if self._alias else ''
        return f'{self.expr}.{self.agg_name}(){a}'


class WindowExpr(Expr):
    __slots__ = ('expr', 'partition_by', 'order_by', '_alias')

    def __init__(self, expr: Expr, partition_by=None, order_by=None):
        self.expr = expr
        self.partition_by: list[str] = _standardize_str_list(partition_by)
        self.order_by: list[str] = _standardize_str_list(order_by)
        self._alias: str | None = None

    def required_columns(self) -> set[str]:
        cols = self.expr.required_columns()
        cols.update(self.partition_by)
        cols.update(self.order_by)
        return cols

    def output_dtype(self, schema):
        return self.expr.output_dtype(schema)

    def output_name(self):
        return self._alias or self.expr.output_name()

    def alias(self, name: str) -> WindowExpr:
        self._alias = name
        return self

    def eval(self, row, col_map):
        raise RuntimeError('WindowExpr must be evaluated via WindowNode')

    def __repr__(self):
        parts = []
        if self.partition_by:
            parts.append(f'partition_by={self.partition_by}')
        if self.order_by:
            parts.append(f'order_by={self.order_by}')
        a = f'.alias("{self._alias}")' if self._alias else ''
        return f'{self.expr}.over({", ".join(parts)}){a}'


class RankExpr(Expr):
    __slots__ = ('kind', '_alias')

    def __init__(self, kind: str):
        self.kind = kind
        self._alias: str | None = None

    def over(self, partition_by=None, order_by=None) -> WindowExpr:
        return WindowExpr(self, partition_by, order_by)

    def required_columns(self):
        return set()

    def output_dtype(self, schema):
        return int

    def output_name(self):
        return self._alias or self.kind

    def alias(self, name: str) -> RankExpr:
        self._alias = name
        return self

    def eval(self, row, col_map):
        raise RuntimeError('RankExpr must be evaluated via WindowNode')

    def __repr__(self):
        return f'{self.kind}()'


class CumExpr(Expr):
    __slots__ = ('expr', 'kind', '_alias')

    def __init__(self, expr: Expr, kind: str):
        self.expr = expr
        self.kind = kind
        self._alias: str | None = None

    def over(self, partition_by=None, order_by=None) -> WindowExpr:
        return WindowExpr(self, partition_by, order_by)

    def required_columns(self):
        return self.expr.required_columns()

    def output_dtype(self, schema):
        return self.expr.output_dtype(schema)

    def output_name(self):
        base = self.expr.output_name() or 'value'
        return self._alias or f'{base}_{self.kind}'

    def alias(self, name: str) -> CumExpr:
        self._alias = name
        return self

    def eval(self, row, col_map):
        raise RuntimeError('CumExpr must be evaluated via WindowNode')

    def __repr__(self):
        return f'{self.expr}.{self.kind}()'


class OffsetExpr(Expr):
    __slots__ = ('expr', 'offset', 'default', '_alias')

    def __init__(self, expr: Expr, offset: int, default=None):
        self.expr = expr
        self.offset = offset
        self.default = default
        self._alias: str | None = None

    def over(self, partition_by=None, order_by=None) -> WindowExpr:
        return WindowExpr(self, partition_by, order_by)

    def required_columns(self):
        return self.expr.required_columns()

    def output_dtype(self, schema):
        return self.expr.output_dtype(schema)

    def output_name(self):
        base = self.expr.output_name() or 'value'
        kind = 'lag' if self.offset < 0 else 'lead'
        return self._alias or f'{base}_{kind}_{abs(self.offset)}'

    def alias(self, name: str) -> OffsetExpr:
        self._alias = name
        return self

    def eval(self, row, col_map):
        raise RuntimeError('OffsetExpr must be evaluated via WindowNode')

    def __repr__(self):
        kind = 'lag' if self.offset < 0 else 'lead'
        return f'{self.expr}.{kind}({abs(self.offset)})'


class StringAccessor:
    __slots__ = ('_expr',)

    def __init__(self, expr: Expr):
        self._expr = expr

    def _unary(self, fn, name):
        return UnaryExpr(self._expr, fn, name)

    def upper(self):    return self._unary(lambda x: x.upper() if isinstance(x, str) else x, 'str.upper')
    def lower(self):    return self._unary(lambda x: x.lower() if isinstance(x, str) else x, 'str.lower')
    def strip(self):    return self._unary(lambda x: x.strip() if isinstance(x, str) else x, 'str.strip')
    def len(self):      return self._unary(lambda x: len(x) if isinstance(x, str) else 0, 'str.len')
    def title(self):    return self._unary(lambda x: x.title() if isinstance(x, str) else x, 'str.title')

    def contains(self, pat):
        return self._unary(lambda x: pat in x if isinstance(x, str) else False, f'str.contains("{pat}")')

    def startswith(self, prefix):
        return self._unary(lambda x: x.startswith(prefix) if isinstance(x, str) else False, f'str.startswith("{prefix}")')

    def endswith(self, suffix):
        return self._unary(lambda x: x.endswith(suffix) if isinstance(x, str) else False, f'str.endswith("{suffix}")')

    def replace(self, old, new):
        return self._unary(lambda x: x.replace(old, new) if isinstance(x, str) else x, f'str.replace("{old}","{new}")')

    def slice(self, start=None, end=None):
        return self._unary(lambda x: x[start:end] if isinstance(x, str) else x, f'str[{start}:{end}]')


class DateTimeAccessor:
    __slots__ = ('_expr',)

    def __init__(self, expr: Expr):
        self._expr = expr

    def _unary(self, fn, name, dtype=int):
        return _DtUnaryExpr(self._expr, fn, name, dtype)

    def year(self):        return self._unary(lambda x: x.year, 'dt.year')
    def month(self):       return self._unary(lambda x: x.month, 'dt.month')
    def day(self):         return self._unary(lambda x: x.day, 'dt.day')
    def hour(self):        return self._unary(lambda x: x.hour, 'dt.hour')
    def minute(self):      return self._unary(lambda x: x.minute, 'dt.minute')
    def second(self):      return self._unary(lambda x: x.second, 'dt.second')
    def microsecond(self): return self._unary(lambda x: x.microsecond, 'dt.microsecond')
    def weekday(self):     return self._unary(lambda x: x.weekday(), 'dt.weekday')
    def isoweekday(self):  return self._unary(lambda x: x.isoweekday(), 'dt.isoweekday')

    def day_name(self):
        _DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return self._unary(lambda x: _DAYS[x.weekday()], 'dt.day_name', str)

    def month_name(self):
        _MONTHS = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
        return self._unary(lambda x: _MONTHS[x.month], 'dt.month_name', str)

    def quarter(self):
        return self._unary(lambda x: (x.month - 1) // 3 + 1, 'dt.quarter')

    def week(self):
        return self._unary(lambda x: x.isocalendar()[1], 'dt.week')

    def day_of_year(self):
        return self._unary(lambda x: x.timetuple().tm_yday, 'dt.day_of_year')

    def date(self):
        return self._unary(lambda x: x.date() if hasattr(x, 'date') else x, 'dt.date', _date)

    def time(self):
        return self._unary(lambda x: x.time() if hasattr(x, 'time') else x, 'dt.time', _time)

    def truncate(self, unit: str):
        from datetime import datetime as _dt
        if unit == 'year':
            def fn(x):
                return _dt(x.year, 1, 1)
        elif unit == 'month':
            def fn(x):
                return _dt(x.year, x.month, 1)
        elif unit == 'day':
            def fn(x):
                return _dt(x.year, x.month, x.day)
        elif unit == 'hour':
            def fn(x):
                return _dt(x.year, x.month, x.day, x.hour)
        elif unit == 'minute':
            def fn(x):
                return _dt(x.year, x.month, x.day, x.hour, x.minute)
        else:
            raise ValueError(f'Unknown truncation unit: {unit!r}. Use year/month/day/hour/minute.')
        return self._unary(fn, f'dt.truncate("{unit}")', _datetime)

    def strftime(self, fmt: str):
        return self._unary(lambda x: x.strftime(fmt), f'dt.strftime("{fmt}")', str)

    def epoch_seconds(self):
        from datetime import datetime as _dt
        from datetime import timezone as _tz
        _EPOCH = _dt(1970, 1, 1, tzinfo=_tz.utc)
        def _to_epoch(x):
            if x.tzinfo is None:
                from datetime import timezone
                x = x.replace(tzinfo=timezone.utc)
            return (x - _EPOCH).total_seconds()
        return self._unary(_to_epoch, 'dt.epoch_seconds', float)

    def add_days(self, n):
        from datetime import timedelta
        d = timedelta(days=n)
        return self._unary(lambda x: x + d, f'dt.add_days({n})', _datetime)

    def add_hours(self, n):
        from datetime import timedelta
        d = timedelta(hours=n)
        return self._unary(lambda x: x + d, f'dt.add_hours({n})', _datetime)

    def add_minutes(self, n):
        from datetime import timedelta
        d = timedelta(minutes=n)
        return self._unary(lambda x: x + d, f'dt.add_minutes({n})', _datetime)

    def add_seconds(self, n):
        from datetime import timedelta
        d = timedelta(seconds=n)
        return self._unary(lambda x: x + d, f'dt.add_seconds({n})', _datetime)


class _DtUnaryExpr(UnaryExpr):
    __slots__ = ('_dtype',)

    def __init__(self, operand, op, op_str, dtype=int):
        super().__init__(operand, op, op_str)
        self._dtype = dtype

    def output_dtype(self, schema):
        return self._dtype

    def eval(self, row, col_map):
        val = self.operand.eval(row, col_map)
        if val is None:
            return None
        return self.op(val)

    def compile(self, col_map):
        inner = self.operand.compile(col_map)
        op = self.op
        def _eval(row):
            val = inner(row)
            return None if val is None else op(val)
        return _eval


from datetime import date as _date
from datetime import datetime as _datetime
from datetime import time as _time

_DATETIME_FORMATS = [
    '%Y-%m-%dT%H:%M:%S.%f',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%d %H:%M:%S.%f',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M',
    '%Y-%m-%d',
    '%m/%d/%Y %H:%M:%S',
    '%m/%d/%Y',
    '%d/%m/%Y %H:%M:%S',
    '%d/%m/%Y',
    '%Y%m%d',
    '%d-%b-%Y',
    '%b %d, %Y',
    '%B %d, %Y',
]

def _try_parse_datetime(value: str):
    v = value.strip()
    if not v or len(v) < 6:
        return None
    for fmt in _DATETIME_FORMATS:
        try:
            return _datetime.strptime(v, fmt), fmt
        except ValueError:
            continue
    return None


def _detect_datetime_format(values: list):
    non_empty = [v for v in values if v and v.strip()]
    if not non_empty:
        return None

    candidate_fmt = None
    for v in non_empty[:10]:
        result = _try_parse_datetime(v)
        if result:
            candidate_fmt = result[1]
            break

    if candidate_fmt is None:
        return None

    hits = 0
    for v in non_empty:
        try:
            _datetime.strptime(v.strip(), candidate_fmt)
            hits += 1
        except ValueError:
            pass

    if hits / len(non_empty) >= 0.8:
        return candidate_fmt
    return None


def col(name: str) -> Col:
    return Col(name)

def lit(value) -> Lit:
    return Lit(value)

def rank() -> RankExpr:
    return RankExpr('rank')

def dense_rank() -> RankExpr:
    return RankExpr('dense_rank')

def row_number() -> RankExpr:
    return RankExpr('row_number')

def when(condition: Expr, then_val) -> WhenExpr:
    return WhenExpr(condition, _ensure_expr(then_val))


class WhenExpr(Expr):
    __slots__ = ('_branches', '_otherwise')

    def __init__(self, condition: Expr, then_val: Expr):
        self._branches = [(condition, then_val)]
        self._otherwise: Expr = Lit(None)

    def when(self, condition: Expr, then_val) -> WhenExpr:
        self._branches.append((condition, _ensure_expr(then_val)))
        return self

    def otherwise(self, val) -> WhenExpr:
        self._otherwise = _ensure_expr(val)
        return self

    def eval(self, row, col_map):
        for cond, val in self._branches:
            if cond.eval(row, col_map):
                return val.eval(row, col_map)
        return self._otherwise.eval(row, col_map)

    def compile(self, col_map):
        compiled_branches = [(c.compile(col_map), v.compile(col_map))
                            for c, v in self._branches]
        otherwise_fn = self._otherwise.compile(col_map)
        def _eval(row):
            for cond_fn, val_fn in compiled_branches:
                if cond_fn(row):
                    return val_fn(row)
            return otherwise_fn(row)
        return _eval

    def required_columns(self):
        cols = set()
        for cond, val in self._branches:
            cols |= cond.required_columns()
            cols |= val.required_columns()
        cols |= self._otherwise.required_columns()
        return cols

    def output_dtype(self, schema):
        return self._branches[0][1].output_dtype(schema)

    def __repr__(self):
        parts = [f'when({c}, {v})' for c, v in self._branches]
        return '.'.join(parts) + f'.otherwise({self._otherwise})'
