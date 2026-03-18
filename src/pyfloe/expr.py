from __future__ import annotations

import operator as _op
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .schema import LazySchema

AggFunc = Literal['sum', 'mean', 'min', 'max', 'count', 'first', 'last', 'n_unique']
JoinHow = Literal['inner', 'left', 'full']
RankKind = Literal['rank', 'dense_rank', 'row_number']
CumKind = Literal['cumsum', 'cummax', 'cummin']
DateTruncUnit = Literal['year', 'month', 'day', 'hour', 'minute']


class Agg:
    """Aggregation function names for use with ``pivot(agg=...)`` and group-by.

    Examples:
        >>> lf.pivot(index="name", on="subject", values="score", agg=Agg.sum)
    """
    sum: AggFunc = 'sum'
    mean: AggFunc = 'mean'
    min: AggFunc = 'min'
    max: AggFunc = 'max'
    count: AggFunc = 'count'
    first: AggFunc = 'first'
    last: AggFunc = 'last'
    n_unique: AggFunc = 'n_unique'


class Join:
    """Join type names for use with ``join(how=...)``.

    Examples:
        >>> orders.join(customers, on="cust", how=Join.left)
    """
    inner: JoinHow = 'inner'
    left: JoinHow = 'left'
    full: JoinHow = 'full'


class DateTrunc:
    """Truncation units for use with ``col("ts").dt.truncate(...)``.

    Examples:
        >>> col("ts").dt.truncate(DateTrunc.month)
    """
    year: DateTruncUnit = 'year'
    month: DateTruncUnit = 'month'
    day: DateTruncUnit = 'day'
    hour: DateTruncUnit = 'hour'
    minute: DateTruncUnit = 'minute'


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
    """Base class for all expressions in the query plan.

    Expressions are composable AST nodes that represent computations
    on column values. They support arithmetic operators, comparisons,
    logical operators, and method chaining for aggregations, window
    functions, string operations, and datetime operations.

    Examples:
        Arithmetic:

        >>> col("price") * 0.9  # doctest: +SKIP
        (col("price") * 0.9)

        Comparisons:

        >>> col("age") > 18  # doctest: +SKIP
        (col("age") > 18)

        Logical operators:

        >>> (col("age") > 18) & (col("active") == True)  # doctest: +SKIP
        ((col("age") > 18) & (col("active") == True))
    """

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

    def __gt__(self, other) -> BinaryExpr:  return BinaryExpr(self, _ensure_expr(other), _op.gt, '>')
    def __ge__(self, other) -> BinaryExpr:  return BinaryExpr(self, _ensure_expr(other), _op.ge, '>=')
    def __lt__(self, other) -> BinaryExpr:  return BinaryExpr(self, _ensure_expr(other), _op.lt, '<')
    def __le__(self, other) -> BinaryExpr:  return BinaryExpr(self, _ensure_expr(other), _op.le, '<=')
    def __eq__(self, other) -> BinaryExpr:  return BinaryExpr(self, _ensure_expr(other), _op.eq, '==')
    def __ne__(self, other) -> BinaryExpr:  return BinaryExpr(self, _ensure_expr(other), _op.ne, '!=')

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
        """Rename the output column of this expression.

        Args:
            name: New column name.

        Examples:
            >>> (col("price") * 0.2).alias("tax")  # doctest: +SKIP
            (col("price") * 0.2).alias("tax")
        """
        return AliasExpr(self, name)

    def cast(self, dtype: type) -> CastExpr:
        """Cast the expression value to a different type.

        Args:
            dtype: Target Python type (e.g. ``int``, ``str``, ``float``).

        Examples:
            >>> col("amount").cast(str)  # doctest: +SKIP
            col("amount").cast(str)
        """
        return CastExpr(self, dtype)

    def is_null(self) -> UnaryExpr:
        """Test whether the value is None.

        Returns:
            A boolean expression that is True for None values.

        Examples:
            >>> lf = LazyFrame([{"x": 1}, {"x": None}])  # doctest: +SKIP
            >>> lf.filter(col("x").is_null()).to_pylist()  # doctest: +SKIP
            [{'x': None}]
        """
        return UnaryExpr(self, lambda x: x is None, 'is_null')

    def is_not_null(self) -> UnaryExpr:
        """Test whether the value is not None.

        Returns:
            A boolean expression that is True for non-None values.

        Examples:
            >>> lf = LazyFrame([{"x": 1}, {"x": None}])  # doctest: +SKIP
            >>> lf.filter(col("x").is_not_null()).to_pylist()  # doctest: +SKIP
            [{'x': 1}]
        """
        return UnaryExpr(self, lambda x: x is not None, 'is_not_null')

    def is_in(self, values) -> UnaryExpr:
        """Test whether the value is in a set of values.

        Args:
            values: Collection of values to test membership against.

        Returns:
            A boolean expression.

        Examples:
            >>> lf = LazyFrame([{"r": "EU"}, {"r": "US"}, {"r": "AP"}])  # doctest: +SKIP
            >>> lf.filter(col("r").is_in(["EU", "US"])).to_pylist()  # doctest: +SKIP
            [{'r': 'EU'}, {'r': 'US'}]
        """
        s = frozenset(values)
        return UnaryExpr(self, lambda x: x in s, f'is_in({list(values)!r})')

    @property
    def str(self) -> StringAccessor:
        """Access string methods on this expression.

        Returns:
            A :class:`StringAccessor` providing ``.upper()``, ``.lower()``,
            ``.contains()``, ``.replace()``, etc.

        Examples:
            >>> col("name").str.upper()  # doctest: +SKIP
            str.upper(col("name"))
        """
        return StringAccessor(self)

    @property
    def dt(self) -> DateTimeAccessor:
        """Access datetime methods on this expression.

        Returns:
            A :class:`DateTimeAccessor` providing ``.year()``, ``.month()``,
            ``.truncate()``, ``.add_days()``, etc.

        Examples:
            >>> col("ts").dt.year()  # doctest: +SKIP
            dt.year(col("ts"))
        """
        return DateTimeAccessor(self)

    def sum(self) -> AggExpr:
        """Sum of non-null values. Use inside ``group_by().agg()`` or with ``.over()``.

        Examples:
            >>> col("amount").sum().alias("total")  # doctest: +SKIP
        """
        return AggExpr(self, 'sum', sum)

    def mean(self) -> AggExpr:
        """Mean of non-null values. Use inside ``group_by().agg()`` or with ``.over()``.

        Examples:
            >>> col("score").mean().alias("avg_score")  # doctest: +SKIP
        """
        return AggExpr(self, 'mean', lambda v: sum(v)/len(v) if v else 0)

    def min(self) -> AggExpr:
        """Minimum of non-null values. Use inside ``group_by().agg()`` or with ``.over()``.

        Examples:
            >>> col("price").min().alias("lowest")  # doctest: +SKIP
        """
        return AggExpr(self, 'min', min)

    def max(self) -> AggExpr:
        """Maximum of non-null values. Use inside ``group_by().agg()`` or with ``.over()``.

        Examples:
            >>> col("price").max().alias("highest")  # doctest: +SKIP
        """
        return AggExpr(self, 'max', max)

    def count(self) -> AggExpr:
        """Count of non-null values. Use inside ``group_by().agg()`` or with ``.over()``.

        Examples:
            >>> col("order_id").count().alias("n_orders")  # doctest: +SKIP
        """
        return AggExpr(self, 'count', len)

    def first(self) -> AggExpr:
        """First non-null value in the group. Use inside ``group_by().agg()``.

        Examples:
            >>> col("name").first().alias("first_name")  # doctest: +SKIP
        """
        return AggExpr(self, 'first', lambda v: v[0] if v else None)

    def last(self) -> AggExpr:
        """Last non-null value in the group. Use inside ``group_by().agg()``.

        Examples:
            >>> col("name").last().alias("last_name")  # doctest: +SKIP
        """
        return AggExpr(self, 'last', lambda v: v[-1] if v else None)

    def n_unique(self) -> AggExpr:
        """Count of distinct non-null values. Use inside ``group_by().agg()``.

        Examples:
            >>> col("product").n_unique().alias("unique_products")  # doctest: +SKIP
        """
        return AggExpr(self, 'n_unique', lambda v: len(set(v)))

    def cumsum(self) -> CumExpr:
        """Cumulative sum. Use with ``.over()`` to create a window expression.

        Examples:
            >>> col("amount").cumsum().over(order_by="date")  # doctest: +SKIP
        """
        return CumExpr(self, 'cumsum')

    def cummax(self) -> CumExpr:
        """Cumulative maximum. Use with ``.over()`` to create a window expression.

        Examples:
            >>> col("score").cummax().over(order_by="round")  # doctest: +SKIP
        """
        return CumExpr(self, 'cummax')

    def cummin(self) -> CumExpr:
        """Cumulative minimum. Use with ``.over()`` to create a window expression.

        Examples:
            >>> col("score").cummin().over(order_by="round")  # doctest: +SKIP
        """
        return CumExpr(self, 'cummin')

    def lag(self, n: int = 1, default=None) -> OffsetExpr:
        """Access the value from *n* rows before in the window.

        Must be used with ``.over()`` to specify ordering.

        Args:
            n: Number of rows to look back.
            default: Value to use when there is no previous row.

        Examples:
            >>> col("value").lag(1, default=0).over(order_by="id")  # doctest: +SKIP
        """
        return OffsetExpr(self, -n, default)

    def lead(self, n: int = 1, default=None) -> OffsetExpr:
        """Access the value from *n* rows ahead in the window.

        Must be used with ``.over()`` to specify ordering.

        Args:
            n: Number of rows to look ahead.
            default: Value to use when there is no subsequent row.

        Examples:
            >>> col("value").lead(1, default=0).over(order_by="id")  # doctest: +SKIP
        """
        return OffsetExpr(self, n, default)

    def __hash__(self):
        return id(self)


class Col(Expr):
    """A column reference expression.

    Evaluates to the value of the named column in each row.

    Attributes:
        name: The column name this expression refers to.
    """

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
    """A literal value expression.

    Always evaluates to the same constant value, regardless of the row.

    Attributes:
        value: The constant value.
    """

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
    """An expression combining two operands with a binary operator.

    Created implicitly via operators like ``col("a") + col("b")``
    or ``col("x") > 100``.
    """

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
    """An expression applying a unary operator to a single operand.

    Created implicitly via operators like ``~col("active")``
    or methods like ``col("x").is_null()``.
    """

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
    """An expression with a renamed output column.

    Created via ``expr.alias("new_name")``.
    """

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
    """An expression that casts its value to a target type.

    Created via ``expr.cast(int)`` or ``expr.cast(str)``.
    """

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
    """An aggregation expression (sum, mean, count, etc.).

    Created via methods like ``col("amount").sum()`` and used
    inside ``group_by().agg()`` or as a window function via ``.over()``.
    """

    __slots__ = ('expr', 'agg_name', 'agg_func', '_alias')

    def __init__(self, expr: Expr, agg_name: AggFunc, agg_func):
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
        """Rename the output column of this aggregation.

        Args:
            name: New column name.

        Examples:
            >>> col("amount").sum().alias("total_amount")  # doctest: +SKIP
        """
        self._alias = name
        return self

    def over(self, partition_by=None, order_by=None) -> WindowExpr:
        """Turn this aggregation into a window function.

        Args:
            partition_by: Column name(s) to partition by.
            order_by: Column name(s) to order by within each partition.

        Returns:
            A WindowExpr for use with :meth:`~pyfloe.LazyFrame.with_column`.

        Examples:
            >>> col("amount").sum().over(partition_by="region")  # doctest: +SKIP
        """
        return WindowExpr(self, partition_by, order_by)

    def __repr__(self):
        a = f'.alias("{self._alias}")' if self._alias else ''
        return f'{self.expr}.{self.agg_name}(){a}'


class WindowExpr(Expr):
    """A window function expression.

    Created by calling ``.over()`` on an AggExpr, RankExpr,
    CumExpr, or OffsetExpr. Evaluated by the WindowNode in
    the query plan.
    """

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
    """A ranking expression (row_number, rank, or dense_rank).

    Must be used with ``.over()`` to create a WindowExpr.
    """

    __slots__ = ('kind', '_alias')

    def __init__(self, kind: RankKind):
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
    """A cumulative expression (cumsum, cummax, cummin).

    Must be used with ``.over()`` to create a WindowExpr.
    """

    __slots__ = ('expr', 'kind', '_alias')

    def __init__(self, expr: Expr, kind: CumKind):
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
    """A lag/lead expression for offset-based row access.

    Created via ``col("x").lag(1)`` or ``col("x").lead(1)``.
    Must be used with ``.over()`` to create a WindowExpr.
    """

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
    """String methods accessor for expressions.

    Access via ``col("name").str``. Provides methods like
    ``.upper()``, ``.lower()``, ``.contains()``, ``.replace()``, etc.

    Examples:
        >>> from pyfloe import LazyFrame, col
        >>> lf = LazyFrame([{"name": "Alice"}, {"name": "Bob"}])
        >>> lf.with_column("upper", col("name").str.upper()).to_pylist()
        [{'name': 'Alice', 'upper': 'ALICE'}, {'name': 'Bob', 'upper': 'BOB'}]
    """

    __slots__ = ('_expr',)

    def __init__(self, expr: Expr):
        self._expr = expr

    def _unary(self, fn, name):
        return UnaryExpr(self._expr, fn, name)

    def upper(self):
        """Convert to uppercase.

        Examples:
            >>> col("name").str.upper()  # doctest: +SKIP
        """
        return self._unary(lambda x: x.upper() if isinstance(x, str) else x, 'str.upper')

    def lower(self):
        """Convert to lowercase.

        Examples:
            >>> col("name").str.lower()  # doctest: +SKIP
        """
        return self._unary(lambda x: x.lower() if isinstance(x, str) else x, 'str.lower')

    def strip(self):
        """Strip leading and trailing whitespace.

        Examples:
            >>> col("text").str.strip()  # doctest: +SKIP
        """
        return self._unary(lambda x: x.strip() if isinstance(x, str) else x, 'str.strip')

    def len(self):
        """Return the length of the string.

        Examples:
            >>> col("name").str.len()  # doctest: +SKIP
        """
        return self._unary(lambda x: len(x) if isinstance(x, str) else 0, 'str.len')

    def title(self):
        """Convert to title case.

        Examples:
            >>> col("name").str.title()  # doctest: +SKIP
        """
        return self._unary(lambda x: x.title() if isinstance(x, str) else x, 'str.title')

    def contains(self, pat):
        """Test if string contains a pattern.

        Args:
            pat: Substring to search for.

        Returns:
            A boolean expression.

        Examples:
            >>> lf = LazyFrame([{"product": "Widget A"}, {"product": "Gadget B"}])  # doctest: +SKIP
            >>> lf.filter(col("product").str.contains("Widget")).to_pylist()  # doctest: +SKIP
            [{'product': 'Widget A'}]
        """
        return self._unary(lambda x: pat in x if isinstance(x, str) else False, f'str.contains("{pat}")')

    def startswith(self, prefix):
        """Test if string starts with a prefix.

        Args:
            prefix: Prefix to check for.

        Returns:
            A boolean expression.

        Examples:
            >>> col("name").str.startswith("A")  # doctest: +SKIP
        """
        return self._unary(lambda x: x.startswith(prefix) if isinstance(x, str) else False, f'str.startswith("{prefix}")')

    def endswith(self, suffix):
        """Test if string ends with a suffix.

        Args:
            suffix: Suffix to check for.

        Returns:
            A boolean expression.

        Examples:
            >>> col("file").str.endswith(".csv")  # doctest: +SKIP
        """
        return self._unary(lambda x: x.endswith(suffix) if isinstance(x, str) else False, f'str.endswith("{suffix}")')

    def replace(self, old, new):
        """Replace occurrences of a substring.

        Args:
            old: Substring to find.
            new: Replacement string.

        Examples:
            >>> lf = LazyFrame([{"name": "Widget A"}])  # doctest: +SKIP
            >>> lf.with_column("renamed", col("name").str.replace("Widget", "Gadget")).to_pylist()  # doctest: +SKIP
            [{'name': 'Widget A', 'renamed': 'Gadget A'}]
        """
        return self._unary(lambda x: x.replace(old, new) if isinstance(x, str) else x, f'str.replace("{old}","{new}")')

    def slice(self, start=None, end=None):
        """Extract a substring by position.

        Args:
            start: Start index (inclusive).
            end: End index (exclusive).

        Examples:
            >>> lf = LazyFrame([{"name": "Alice"}])  # doctest: +SKIP
            >>> lf.with_column("first3", col("name").str.slice(0, 3)).to_pylist()  # doctest: +SKIP
            [{'name': 'Alice', 'first3': 'Ali'}]
        """
        return self._unary(lambda x: x[start:end] if isinstance(x, str) else x, f'str[{start}:{end}]')


class DateTimeAccessor:
    """Datetime methods accessor for expressions.

    Access via ``col("ts").dt``. Provides component extraction
    (``.year()``, ``.month()``), truncation (``.truncate()``),
    formatting (``.strftime()``), and arithmetic (``.add_days()``).

    All methods handle None values gracefully, returning None.

    Examples:
        >>> from pyfloe import LazyFrame, col
        >>> from datetime import datetime
        >>> lf = LazyFrame([{"ts": datetime(2024, 1, 15, 8, 30)}])
        >>> lf.with_column("year", col("ts").dt.year()).to_pylist()
        [{'ts': datetime.datetime(2024, 1, 15, 8, 30), 'year': 2024}]
    """

    __slots__ = ('_expr',)

    def __init__(self, expr: Expr):
        self._expr = expr

    def _unary(self, fn, name, dtype=int):
        return _DtUnaryExpr(self._expr, fn, name, dtype)

    def year(self):
        """Extract the year component."""
        return self._unary(lambda x: x.year, 'dt.year')

    def month(self):
        """Extract the month (1–12)."""
        return self._unary(lambda x: x.month, 'dt.month')

    def day(self):
        """Extract the day of the month (1–31)."""
        return self._unary(lambda x: x.day, 'dt.day')

    def hour(self):
        """Extract the hour (0–23)."""
        return self._unary(lambda x: x.hour, 'dt.hour')

    def minute(self):
        """Extract the minute (0–59)."""
        return self._unary(lambda x: x.minute, 'dt.minute')

    def second(self):
        """Extract the second (0–59)."""
        return self._unary(lambda x: x.second, 'dt.second')

    def microsecond(self):
        """Extract the microsecond (0–999999)."""
        return self._unary(lambda x: x.microsecond, 'dt.microsecond')

    def weekday(self):
        """Day of the week (Monday=0, Sunday=6)."""
        return self._unary(lambda x: x.weekday(), 'dt.weekday')

    def isoweekday(self):
        """ISO day of the week (Monday=1, Sunday=7)."""
        return self._unary(lambda x: x.isoweekday(), 'dt.isoweekday')

    def day_name(self):
        """Full name of the day (e.g. ``'Monday'``).

        Examples:
            >>> col("ts").dt.day_name()  # doctest: +SKIP
        """
        _DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return self._unary(lambda x: _DAYS[x.weekday()], 'dt.day_name', str)

    def month_name(self):
        """Full name of the month (e.g. ``'January'``).

        Examples:
            >>> col("ts").dt.month_name()  # doctest: +SKIP
        """
        _MONTHS = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
        return self._unary(lambda x: _MONTHS[x.month], 'dt.month_name', str)

    def quarter(self):
        """Quarter of the year (1–4).

        Examples:
            >>> col("ts").dt.quarter()  # doctest: +SKIP
        """
        return self._unary(lambda x: (x.month - 1) // 3 + 1, 'dt.quarter')

    def week(self):
        """ISO week number (1–53).

        Examples:
            >>> col("ts").dt.week()  # doctest: +SKIP
        """
        return self._unary(lambda x: x.isocalendar()[1], 'dt.week')

    def day_of_year(self):
        """Day of the year (1–366).

        Examples:
            >>> col("ts").dt.day_of_year()  # doctest: +SKIP
        """
        return self._unary(lambda x: x.timetuple().tm_yday, 'dt.day_of_year')

    def date(self):
        """Extract the date part, discarding the time.

        Returns a ``datetime.date`` object.

        Examples:
            >>> col("ts").dt.date()  # doctest: +SKIP
        """
        return self._unary(lambda x: x.date() if hasattr(x, 'date') else x, 'dt.date', _date)

    def time(self):
        """Extract the time part, discarding the date.

        Returns a ``datetime.time`` object.

        Examples:
            >>> col("ts").dt.time()  # doctest: +SKIP
        """
        return self._unary(lambda x: x.time() if hasattr(x, 'time') else x, 'dt.time', _time)

    def truncate(self, unit: DateTruncUnit):
        """Truncate the datetime to a given unit.

        Args:
            unit: One of ``'year'``, ``'month'``, ``'day'``, ``'hour'``,
                ``'minute'``.

        Examples:
            Truncate to month:

            >>> from datetime import datetime
            >>> lf = LazyFrame([{"ts": datetime(2024, 3, 15, 14, 30)}])  # doctest: +SKIP
            >>> lf.with_column("mo", col("ts").dt.truncate("month")).to_pylist()  # doctest: +SKIP
            [{'ts': ..., 'mo': datetime.datetime(2024, 3, 1, 0, 0)}]

        Raises:
            ValueError: If *unit* is not recognized.
        """
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
        """Format datetime as a string.

        Args:
            fmt: Format string (e.g. ``'%Y/%m/%d'``).

        Examples:
            >>> col("ts").dt.strftime("%Y-%m-%d")  # doctest: +SKIP
        """
        return self._unary(lambda x: x.strftime(fmt), f'dt.strftime("{fmt}")', str)

    def epoch_seconds(self):
        """Convert to Unix epoch seconds (float).

        Naive datetimes are treated as UTC.

        Examples:
            >>> col("ts").dt.epoch_seconds()  # doctest: +SKIP
        """
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
        """Add *n* days to the datetime.

        Args:
            n: Number of days (can be negative).

        Examples:
            >>> col("ts").dt.add_days(7)  # doctest: +SKIP
        """
        from datetime import timedelta
        d = timedelta(days=n)
        return self._unary(lambda x: x + d, f'dt.add_days({n})', _datetime)

    def add_hours(self, n):
        """Add *n* hours to the datetime.

        Args:
            n: Number of hours (can be negative).

        Examples:
            >>> col("ts").dt.add_hours(3)  # doctest: +SKIP
        """
        from datetime import timedelta
        d = timedelta(hours=n)
        return self._unary(lambda x: x + d, f'dt.add_hours({n})', _datetime)

    def add_minutes(self, n):
        """Add *n* minutes to the datetime.

        Args:
            n: Number of minutes (can be negative).

        Examples:
            >>> col("ts").dt.add_minutes(30)  # doctest: +SKIP
        """
        from datetime import timedelta
        d = timedelta(minutes=n)
        return self._unary(lambda x: x + d, f'dt.add_minutes({n})', _datetime)

    def add_seconds(self, n):
        """Add *n* seconds to the datetime.

        Args:
            n: Number of seconds (can be negative).

        Examples:
            >>> col("ts").dt.add_seconds(90)  # doctest: +SKIP
        """
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
    """Create a column reference expression.

    This is the primary way to reference columns in filter, with_column,
    and aggregation expressions.

    Args:
        name: The column name to reference.

    Examples:
        >>> from pyfloe import LazyFrame, col
        >>> lf = LazyFrame([{"x": 10, "y": 20}])
        >>> lf.with_column("total", col("x") + col("y")).to_pylist()
        [{'x': 10, 'y': 20, 'total': 30}]
    """
    return Col(name)

def lit(value) -> Lit:
    """Create a literal value expression.

    Wraps a Python value as an expression for use in computations.

    Args:
        value: The constant value.

    Examples:
        >>> from pyfloe import LazyFrame, col, lit
        >>> lf = LazyFrame([{"x": 10}])
        >>> lf.filter(col("x") > lit(5)).to_pylist()
        [{'x': 10}]
    """
    return Lit(value)

def rank() -> RankExpr:
    """Create a rank window function expression.

    Equal values receive the same rank with gaps (e.g. 1, 2, 2, 4).
    Use with ``.over()`` to specify partitioning and ordering.

    Examples:
        >>> from pyfloe import LazyFrame, rank
        >>> data = [{"name": "a", "score": 10}, {"name": "b", "score": 20},
        ...         {"name": "c", "score": 20}, {"name": "d", "score": 30}]
        >>> LazyFrame(data).with_column("r", rank().over(order_by="score")).to_pylist()  # doctest: +SKIP
        [{'name': 'a', 'score': 10, 'r': 1}, {'name': 'b', 'score': 20, 'r': 2}, {'name': 'c', 'score': 20, 'r': 2}, {'name': 'd', 'score': 30, 'r': 4}]
    """
    return RankExpr('rank')

def dense_rank() -> RankExpr:
    """Create a dense_rank window function expression.

    Like :func:`rank` but without gaps in the ranking sequence
    (e.g. 1, 2, 2, 3).

    Examples:
        >>> from pyfloe import LazyFrame, dense_rank
        >>> data = [{"name": "a", "score": 10}, {"name": "b", "score": 20},
        ...         {"name": "c", "score": 20}, {"name": "d", "score": 30}]
        >>> LazyFrame(data).with_column("dr", dense_rank().over(order_by="score")).to_pylist()  # doctest: +SKIP
        [{'name': 'a', ..., 'dr': 1}, {'name': 'b', ..., 'dr': 2}, {'name': 'c', ..., 'dr': 2}, {'name': 'd', ..., 'dr': 3}]
    """
    return RankExpr('dense_rank')

def row_number() -> RankExpr:
    """Create a row_number window function expression.

    Assigns a sequential integer to each row within its partition,
    starting at 1.

    Examples:
        >>> from pyfloe import LazyFrame, row_number
        >>> data = [{"region": "EU", "amount": 100}, {"region": "EU", "amount": 200}]
        >>> LazyFrame(data).with_column("rn",
        ...     row_number().over(partition_by="region", order_by="amount")
        ... ).to_pylist()  # doctest: +SKIP
        [{'region': 'EU', 'amount': 100, 'rn': 1}, {'region': 'EU', 'amount': 200, 'rn': 2}]
    """
    return RankExpr('row_number')

def when(condition: Expr, then_val) -> WhenExpr:
    """Begin a conditional expression (SQL CASE WHEN).

    Chain with ``.when()`` for additional branches and ``.otherwise()``
    for the default value.

    Args:
        condition: Boolean expression for the first branch.
        then_val: Value to return when condition is true.

    Returns:
        A WhenExpr that can be chained with ``.when()`` and ``.otherwise()``.

    Examples:
        >>> from pyfloe import LazyFrame, col, when
        >>> lf = LazyFrame([{"amount": 250}, {"amount": 75}, {"amount": 150}])
        >>> lf.with_column("size",
        ...     when(col("amount") > 200, "large")
        ...     .when(col("amount") > 100, "medium")
        ...     .otherwise("small")
        ... ).to_pylist()
        [{'amount': 250, 'size': 'large'}, {'amount': 75, 'size': 'small'}, {'amount': 150, 'size': 'medium'}]
    """
    return WhenExpr(condition, _ensure_expr(then_val))


class WhenExpr(Expr):
    """Conditional expression (SQL CASE WHEN equivalent).

    Created via ``when(condition, value)`` and chained with
    ``.when()`` and ``.otherwise()``.
    """

    __slots__ = ('_branches', '_otherwise')

    def __init__(self, condition: Expr, then_val: Expr):
        self._branches = [(condition, then_val)]
        self._otherwise: Expr = Lit(None)

    def when(self, condition: Expr, then_val) -> WhenExpr:
        """Add another conditional branch.

        Args:
            condition: Boolean expression for this branch.
            then_val: Value to return when condition is true.
        """
        self._branches.append((condition, _ensure_expr(then_val)))
        return self

    def otherwise(self, val) -> WhenExpr:
        """Set the default value when no conditions match.

        Args:
            val: Default value or expression.
        """
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
