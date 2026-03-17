from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Iterator
from itertools import chain, islice

from .expr import Expr
from .plan import IteratorSourceNode, ScanNode
from .schema import ColumnSchema, LazySchema


def _dict_iter_to_tuple_iter(it: Iterator[dict], columns: list[str]) -> Iterator[tuple]:
    for obj in it:
        yield tuple(obj.get(c) for c in columns)


def _object_iter_to_tuple_iter(it: Iterator, columns: list[str]) -> Iterator[tuple]:
    for obj in it:
        d = obj.__dict__
        yield tuple(d.get(c) for c in columns)


def _peek_and_wrap(source, n: int = 1):
    it = iter(source)
    peeked = list(islice(it, n))

    def chained():
        return chain(peeked, it)

    return peeked, chained


def from_iter(
    source: Iterable | Callable[[], Iterable],
    *,
    columns: list[str] | None = None,
    dtypes: dict[str, type] | None = None,
    schema: LazySchema | None = None,
    source_label: str = 'Iterator',
):
    """Create a lazy Floe from any iterator or generator.

    If given a generator (single-pass), the data can only be evaluated once.
    If given a callable factory or an iterable (list, etc.), data can be
    replayed on each evaluation.

    Args:
        source: An iterable, generator, or zero-argument callable that returns
            an iterable. Items can be dicts, tuples, or objects with ``__dict__``.
        columns: Override column names.
        dtypes: Override column types as ``{name: type}`` mapping.
        schema: Explicit LazySchema to use instead of inferring.
        source_label: Label shown in ``explain()`` output.

    Returns:
        A lazy Floe.

    Examples:
        From a generator:

        >>> from pyfloe import from_iter, col
        >>> def gen():
        ...     for i in range(5):
        ...         yield {"id": i, "value": i * 1.5}
        >>> from_iter(gen()).to_pylist()
        [{'id': 0, 'value': 0.0}, {'id': 1, 'value': 1.5}, {'id': 2, 'value': 3.0}, {'id': 3, 'value': 4.5}, {'id': 4, 'value': 6.0}]

        From a replayable callable factory:

        >>> def make_data():
        ...     for i in range(3):
        ...         yield {"x": i}
        >>> ff = from_iter(make_data)  # pass the function, not the generator
        >>> ff.to_pylist() == ff.to_pylist()  # can be evaluated multiple times
        True
    """
    from .core import Floe

    is_factory = callable(source) and not isinstance(source, (list, tuple, dict))
    if not is_factory:
        if hasattr(source, '__next__'):
            peeked, chained_factory = _peek_and_wrap(source, n=10)
            if not peeked:
                cols = columns or []
                lazy_schema = schema or LazySchema()
                node = IteratorSourceNode(cols, lazy_schema, lambda: iter([]), source_label)
                return Floe._from_plan(node)

            cols, lazy_schema, item_type = _infer_from_sample(peeked, columns, dtypes, schema)

            _chained = chained_factory
            _exhausted = [False]

            def oneshot_factory():
                if _exhausted[0]:
                    return iter([])
                _exhausted[0] = True
                return _convert_iter(_chained(), cols, item_type)

            node = IteratorSourceNode(cols, lazy_schema, oneshot_factory,
                                       f'{source_label}(one-shot)')
            return Floe._from_plan(node)

        else:
            _source = source

            def factory():
                return iter(_source)
            is_factory = True
            source = factory

    if is_factory:
        sample_iter = source()
        peeked, _ = _peek_and_wrap(sample_iter, n=10)

        if not peeked:
            cols = columns or []
            lazy_schema = schema or LazySchema()
            node = IteratorSourceNode(cols, lazy_schema, lambda: iter([]), source_label)
            return Floe._from_plan(node)

        cols, lazy_schema, item_type = _infer_from_sample(peeked, columns, dtypes, schema)

        _src = source

        def replay_factory():
            return _convert_iter(_src(), cols, item_type)

        node = IteratorSourceNode(cols, lazy_schema, replay_factory,
                                   f'{source_label}(replayable)')
        return Floe._from_plan(node)


def _infer_from_sample(peeked, columns, dtypes, schema):
    first = peeked[0]

    if isinstance(first, dict):
        item_type = 'dict'
        if columns is None:
            all_keys = []
            seen = set()
            for d in peeked:
                for k in d:
                    if k not in seen:
                        seen.add(k)
                        all_keys.append(k)
            columns = all_keys
        if schema is None:
            if dtypes:
                schema = LazySchema({
                    c: ColumnSchema(c, dtypes.get(c, str)) for c in columns
                })
            else:
                schema = LazySchema.from_dicts(peeked)

    elif isinstance(first, (tuple, list)):
        item_type = 'tuple'
        if columns is None:
            columns = [f'col_{i}' for i in range(len(first))]
        if schema is None:
            if dtypes:
                schema = LazySchema({
                    c: ColumnSchema(c, dtypes.get(c, str)) for c in columns
                })
            else:
                schema = LazySchema.from_data(columns, [tuple(r) for r in peeked])

    elif hasattr(first, '__dict__'):
        item_type = 'object'
        if columns is None:
            columns = list(first.__dict__.keys())
        if schema is None:
            dicts = [o.__dict__ for o in peeked]
            if dtypes:
                schema = LazySchema({
                    c: ColumnSchema(c, dtypes.get(c, str)) for c in columns
                })
            else:
                schema = LazySchema.from_dicts(dicts)
    else:
        raise ValueError(f'Cannot infer schema from items of type {type(first)}. '
                         f'Items must be dicts, tuples, or objects with __dict__.')

    return columns, schema, item_type


def _convert_iter(it, columns, item_type):
    if item_type == 'dict':
        for obj in it:
            yield tuple(obj.get(c) for c in columns)
    elif item_type == 'tuple':
        for obj in it:
            yield tuple(obj)
    elif item_type == 'object':
        for obj in it:
            d = obj.__dict__
            yield tuple(d.get(c) for c in columns)


def from_chunks(
    chunks: Iterable[list[dict]] | Callable[[], Iterable[list[dict]]],
    *,
    columns: list[str] | None = None,
    dtypes: dict[str, type] | None = None,
    schema: LazySchema | None = None,
    source_label: str = 'Chunked',
):
    """Create a lazy Floe from batched/paginated data.

    Each chunk is a list of dicts, list of tuples, or a Floe.
    Useful for paginated APIs or batch-producing sources.

    Args:
        chunks: An iterable of chunks, or a callable that returns one.
        columns: Override column names.
        dtypes: Override column types as ``{name: type}`` mapping.
        schema: Explicit LazySchema to use instead of inferring.
        source_label: Label shown in ``explain()`` output.

    Returns:
        A lazy Floe.

    Examples:
        From a replayable chunk factory:

        >>> from pyfloe import from_chunks
        >>> def make_chunks():
        ...     yield [{"n": 1}, {"n": 2}]
        ...     yield [{"n": 3}]
        >>> ff = from_chunks(make_chunks)
        >>> ff.to_pylist()
        [{'n': 1}, {'n': 2}, {'n': 3}]

        From an iterator of chunks:

        >>> chunks = [[{"id": 1, "v": "a"}], [{"id": 2, "v": "b"}]]
        >>> from_chunks(iter(chunks)).to_pylist()
        [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'b'}]
    """
    from .core import Floe

    def flatten_dicts(chunk_iter, cols):
        def _chunk_to_rows(chunk):
            if isinstance(chunk, list) and chunk and isinstance(chunk[0], dict):
                return (tuple(row.get(c) for c in cols) for row in chunk)
            elif isinstance(chunk, Floe):
                return chunk._plan.execute()
            elif isinstance(chunk, list) and chunk and isinstance(chunk[0], (tuple, list)):
                return (tuple(row) for row in chunk)
            elif isinstance(chunk, list):
                return iter(())
            else:
                raise ValueError(f'Unexpected chunk type: {type(chunk)}')

        return chain.from_iterable(_chunk_to_rows(c) for c in chunk_iter)

    is_factory = callable(chunks) and not isinstance(chunks, (list, tuple))

    if is_factory:
        sample_chunks = chunks()
    else:
        if hasattr(chunks, '__next__'):
            peeked_chunks, chained = _peek_and_wrap(chunks, n=1)
        else:
            peeked_chunks = list(chunks)[:1]

            def chained():
                return iter(chunks)

    if is_factory:
        peeked_chunks, _ = _peek_and_wrap(sample_chunks, n=1)

    if not peeked_chunks or not peeked_chunks[0]:
        cols = columns or []
        lazy_schema = schema or LazySchema()
        node = IteratorSourceNode(cols, lazy_schema, lambda: iter([]), source_label)
        return Floe._from_plan(node)

    first_chunk = peeked_chunks[0]
    if isinstance(first_chunk, Floe):
        cols = columns or first_chunk.columns
        lazy_schema = schema or first_chunk.schema
    elif isinstance(first_chunk[0], dict):
        sample = first_chunk[:100]
        if columns is None:
            all_keys = []
            seen = set()
            for d in sample:
                for k in d:
                    if k not in seen:
                        seen.add(k)
                        all_keys.append(k)
            cols = all_keys
        else:
            cols = columns
        if schema:
            lazy_schema = schema
        elif dtypes:
            lazy_schema = LazySchema({c: ColumnSchema(c, dtypes.get(c, str)) for c in cols})
        else:
            lazy_schema = LazySchema.from_dicts(sample)
    elif isinstance(first_chunk[0], (tuple, list)):
        cols = columns or [f'col_{i}' for i in range(len(first_chunk[0]))]
        if schema:
            lazy_schema = schema
        elif dtypes:
            lazy_schema = LazySchema({c: ColumnSchema(c, dtypes.get(c, str)) for c in cols})
        else:
            lazy_schema = LazySchema.from_data(cols, [tuple(r) for r in first_chunk[:100]])
    else:
        cols = columns or []
        lazy_schema = schema or LazySchema()

    if is_factory:
        _src = chunks

        def factory():
            return flatten_dicts(_src(), cols)
    else:
        _chained = chained
        _exhausted = [False]

        def factory():
            if _exhausted[0]:
                return iter([])
            _exhausted[0] = True
            return flatten_dicts(_chained(), cols)

    node = IteratorSourceNode(cols, lazy_schema, factory, source_label)
    return Floe._from_plan(node)


class Stream:
    """A true single-pass streaming pipeline.

    Unlike Floe, Stream compiles transforms into a flat loop for
    maximum throughput. Supports filter, with_column, select, and apply.
    Results are consumed via ``.to_csv()``, ``.to_jsonl()``,
    ``.to_pylist()``, or ``.collect()``.

    Examples:
        >>> from pyfloe import Stream, col
        >>> def gen():
        ...     for i in range(100):
        ...         yield {"id": i, "value": i * 2.0}
        >>> result = (
        ...     Stream.from_iter(gen())
        ...     .filter(col("value") > 190)
        ...     .with_column("label", col("id").cast(str))
        ...     .select("id", "value", "label")
        ...     .to_pylist()
        ... )
        >>> len(result)
        4
    """

    def __init__(self, source_factory, columns: list[str],
                 schema: LazySchema, transforms: list | None = None,
                 source_columns: list[str] | None = None):
        self._source_factory = source_factory
        self._columns = columns
        self._source_columns = source_columns or columns
        self._schema = schema
        self._transforms: list = transforms or []

    @classmethod
    def from_iter(cls, source, *, columns=None, dtypes=None, schema=None):
        """Create a Stream from an iterator, iterable, or factory callable.

        Args:
            source: Data source — iterable, generator, or callable factory.
            columns: Override column names.
            dtypes: Override column types.
            schema: Explicit LazySchema.

        Returns:
            A new Stream.

        Examples:
            >>> def gen():
            ...     for i in range(10):
            ...         yield {"x": i, "y": i * 10}
            >>> Stream.from_iter(gen()).filter(col("y") > 50).to_pylist()  # doctest: +SKIP
            [{'x': 6, 'y': 60}, {'x': 7, 'y': 70}, {'x': 8, 'y': 80}, {'x': 9, 'y': 90}]
        """
        is_factory = callable(source) and not isinstance(source, (list, tuple))

        if is_factory:
            peeked, _ = _peek_and_wrap(source(), n=10)
        elif hasattr(source, '__next__'):
            peeked, chained = _peek_and_wrap(source, n=10)
            source = chained
        else:
            peeked = list(source)[:10]

        if not peeked:
            return cls(lambda: iter([]), columns or [], schema or LazySchema())

        cols, lazy_schema, item_type = _infer_from_sample(peeked, columns, dtypes, schema)

        if is_factory:
            _src = source

            def factory():
                return _convert_iter(_src(), cols, item_type)
        elif hasattr(source, '__call__'):
            _src = source

            def factory():
                return _convert_iter(_src(), cols, item_type)
        else:
            _source = source
            _exhausted = [False]

            def factory():
                if _exhausted[0]:
                    return iter([])
                _exhausted[0] = True
                return _convert_iter(iter(_source) if not hasattr(_source, '__next__') else _source(), cols, item_type)

        return cls(factory, cols, lazy_schema)

    @classmethod
    def from_csv(cls, path: str, **kwargs):
        """Create a Stream from a CSV file.

        Args:
            path: Path to the CSV file.
            **kwargs: Arguments passed to the CSV reader.

        Returns:
            A new Stream.

        Examples:
            >>> Stream.from_csv("orders.csv").filter(col("amount") > 100).to_pylist()  # doctest: +SKIP
            [{'order_id': 1, 'amount': 250.0, ...}, ...]
        """
        from .io import _read_delimited
        node = _read_delimited(path, **kwargs)
        return cls(node._row_factory, node._columns, node._schema)

    def filter(self, predicate: Expr) -> Stream:
        """Add a filter step to the pipeline.

        Args:
            predicate: Boolean expression to filter rows.

        Returns:
            A new Stream with the filter applied.

        Examples:
            >>> stream.filter(col("amount") > 100)  # doctest: +SKIP
        """
        return Stream(self._source_factory, self._columns, self._schema,
                      self._transforms + [('filter', predicate)],
                      self._source_columns)

    def with_column(self, name: str, expr: Expr) -> Stream:
        """Add a computed column step to the pipeline.

        Args:
            name: Name for the new column.
            expr: Expression to compute column values.

        Returns:
            A new Stream with the additional column.

        Examples:
            >>> stream.with_column("total", col("x") + col("y"))  # doctest: +SKIP
        """
        new_schema = self._schema.with_column(name, expr.output_dtype(self._schema))
        new_cols = self._columns + [name]
        return Stream(self._source_factory, new_cols, new_schema,
                      self._transforms + [('with_column', name, expr)],
                      self._source_columns)

    def select(self, *columns: str) -> Stream:
        """Add a column selection step to the pipeline.

        Args:
            *columns: Column names to keep.

        Returns:
            A new Stream with only the selected columns.

        Examples:
            >>> stream.select("id", "value")  # doctest: +SKIP
        """
        new_schema = self._schema.select(list(columns))
        return Stream(self._source_factory, list(columns), new_schema,
                      self._transforms + [('select', list(columns))],
                      self._source_columns)

    def apply(self, func: Callable, columns: list[str] | None = None) -> Stream:
        """Apply a function to column values in the stream.

        Args:
            func: Function to apply to each cell value.
            columns: Columns to apply to. If None, applies to all columns.

        Returns:
            A new Stream with the function applied.
        """
        return Stream(self._source_factory, self._columns, self._schema,
                      self._transforms + [('apply', func, columns)],
                      self._source_columns)

    def _build_processor(self):
        col_map = {n: i for i, n in enumerate(self._source_columns)}
        steps = []

        current_cols = list(self._source_columns)
        current_map = dict(col_map)

        for transform in self._transforms:
            kind = transform[0]

            if kind == 'filter':
                pred = transform[1]
                frozen_map = dict(current_map)
                steps.append(('filter', pred, frozen_map))

            elif kind == 'with_column':
                name, expr = transform[1], transform[2]
                frozen_map = dict(current_map)
                steps.append(('with_column', expr, frozen_map))
                current_cols.append(name)
                current_map[name] = len(current_cols) - 1

            elif kind == 'select':
                cols = transform[1]
                frozen_map = dict(current_map)
                indices = [frozen_map[c] for c in cols]
                steps.append(('select', indices))
                current_cols = list(cols)
                current_map = {c: i for i, c in enumerate(current_cols)}

            elif kind == 'apply':
                func, target_cols = transform[1], transform[2]
                if target_cols:
                    target_indices = {current_map[c] for c in target_cols}
                else:
                    target_indices = None
                steps.append(('apply', func, target_indices))

        return steps, current_cols

    def _execute(self) -> Iterator[tuple]:
        steps, _ = self._build_processor()

        for row in self._source_factory():
            skip = False
            current = row

            for step in steps:
                kind = step[0]
                if kind == 'filter':
                    _, pred, cm = step
                    if not pred.eval(current, cm):
                        skip = True
                        break
                elif kind == 'with_column':
                    _, expr, cm = step
                    current = current + (expr.eval(current, cm),)
                elif kind == 'select':
                    _, indices = step
                    current = tuple(current[i] for i in indices)
                elif kind == 'apply':
                    _, func, targets = step
                    if targets is None:
                        current = tuple(func(v) for v in current)
                    else:
                        current = tuple(func(v) if i in targets else v for i, v in enumerate(current))

            if not skip:
                yield current

    def collect(self):
        """Execute the pipeline and return a materialized Floe.

        Examples:
            >>> ff = Stream.from_iter(gen()).filter(col("x") > 5).collect()  # doctest: +SKIP
            >>> isinstance(ff, Floe)  # doctest: +SKIP
            True
        """
        from .core import Floe
        _, out_cols = self._build_processor()
        rows = list(self._execute())
        return Floe._from_plan(
            ScanNode(rows, out_cols, self._schema)
        )

    def to_pylist(self) -> list[dict]:
        """Execute the pipeline and return results as a list of dicts.

        Examples:
            >>> Stream.from_iter(gen()).filter(col("x") > 5).to_pylist()  # doctest: +SKIP
            [{'x': 6}, {'x': 7}, ...]
        """
        _, out_cols = self._build_processor()
        return [{out_cols[i]: v for i, v in enumerate(row)} for row in self._execute()]

    def to_csv(self, path: str, *, delimiter: str = ',',
               header: bool = True, encoding: str = 'utf-8'):
        """Execute the pipeline and stream results to a CSV file.

        Rows are written one-at-a-time with constant memory.

        Args:
            path: Output file path.
            delimiter: Field delimiter.
            header: Whether to write a header row.
            encoding: File encoding.

        Examples:
            >>> Stream.from_iter(gen()).filter(col("score") > 50).to_csv("/tmp/out.csv")  # doctest: +SKIP
        """
        _, out_cols = self._build_processor()
        with open(path, 'w', encoding=encoding, newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            if header:
                writer.writerow(out_cols)
            for row in self._execute():
                writer.writerow(row)

    def to_jsonl(self, path: str, *, encoding: str = 'utf-8'):
        """Execute the pipeline and stream results to a JSON Lines file.

        Args:
            path: Output file path.
            encoding: File encoding.

        Examples:
            >>> Stream.from_iter(gen()).filter(col("ts") > 10).to_jsonl("/tmp/out.jsonl")  # doctest: +SKIP
        """
        _, out_cols = self._build_processor()
        with open(path, 'w', encoding=encoding) as f:
            for row in self._execute():
                obj = {out_cols[i]: v for i, v in enumerate(row)}
                f.write(json.dumps(obj, default=str) + '\n')

    def foreach(self, func: Callable[[dict], None]):
        """Execute the pipeline and call a function for each row.

        Args:
            func: Function that receives each row as a dict.

        Examples:
            >>> collected = []
            >>> Stream.from_iter(gen()).foreach(lambda row: collected.append(row))  # doctest: +SKIP
        """
        _, out_cols = self._build_processor()
        for row in self._execute():
            obj = {out_cols[i]: v for i, v in enumerate(row)}
            func(obj)

    def count(self) -> int:
        """Execute the pipeline and return the total row count.

        Examples:
            >>> Stream.from_iter(gen()).filter(col("x") > 500).count()  # doctest: +SKIP
            499
        """
        return sum(1 for _ in self._execute())

    def take(self, n: int) -> list[dict]:
        """Execute the pipeline and return the first n rows as dicts.

        Args:
            n: Number of rows to return.

        Examples:
            >>> Stream.from_iter(gen()).take(3)  # doctest: +SKIP
            [{'x': 0}, {'x': 1}, {'x': 2}]
        """
        _, out_cols = self._build_processor()
        return [
            {out_cols[j]: v for j, v in enumerate(row)}
            for row in islice(self._execute(), n)
        ]

    @property
    def columns(self) -> list[str]:
        """List of output column names after all transforms."""
        _, out_cols = self._build_processor()
        return out_cols

    @property
    def schema(self) -> LazySchema:
        """Output schema of this Stream."""
        return self._schema

    def __repr__(self):
        _, out_cols = self._build_processor()
        n_steps = len(self._transforms)
        return f'Stream [{", ".join(out_cols)}] ({n_steps} transforms)'
