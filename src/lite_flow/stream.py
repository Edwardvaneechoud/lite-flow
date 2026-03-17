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
        from .io import _read_delimited
        node = _read_delimited(path, **kwargs)
        return cls(node._row_factory, node._columns, node._schema)

    def filter(self, predicate: Expr) -> Stream:
        return Stream(self._source_factory, self._columns, self._schema,
                      self._transforms + [('filter', predicate)],
                      self._source_columns)

    def with_column(self, name: str, expr: Expr) -> Stream:
        new_schema = self._schema.with_column(name, expr.output_dtype(self._schema))
        new_cols = self._columns + [name]
        return Stream(self._source_factory, new_cols, new_schema,
                      self._transforms + [('with_column', name, expr)],
                      self._source_columns)

    def select(self, *columns: str) -> Stream:
        new_schema = self._schema.select(list(columns))
        return Stream(self._source_factory, list(columns), new_schema,
                      self._transforms + [('select', list(columns))],
                      self._source_columns)

    def apply(self, func: Callable, columns: list[str] | None = None) -> Stream:
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
        from .core import Floe
        _, out_cols = self._build_processor()
        rows = list(self._execute())
        return Floe._from_plan(
            ScanNode(rows, out_cols, self._schema)
        )

    def to_pylist(self) -> list[dict]:
        _, out_cols = self._build_processor()
        return [{out_cols[i]: v for i, v in enumerate(row)} for row in self._execute()]

    def to_csv(self, path: str, *, delimiter: str = ',',
               header: bool = True, encoding: str = 'utf-8'):
        _, out_cols = self._build_processor()
        with open(path, 'w', encoding=encoding, newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            if header:
                writer.writerow(out_cols)
            for row in self._execute():
                writer.writerow(row)

    def to_jsonl(self, path: str, *, encoding: str = 'utf-8'):
        _, out_cols = self._build_processor()
        with open(path, 'w', encoding=encoding) as f:
            for row in self._execute():
                obj = {out_cols[i]: v for i, v in enumerate(row)}
                f.write(json.dumps(obj, default=str) + '\n')

    def foreach(self, func: Callable[[dict], None]):
        _, out_cols = self._build_processor()
        for row in self._execute():
            obj = {out_cols[i]: v for i, v in enumerate(row)}
            func(obj)

    def count(self) -> int:
        return sum(1 for _ in self._execute())

    def take(self, n: int) -> list[dict]:
        _, out_cols = self._build_processor()
        return [
            {out_cols[j]: v for j, v in enumerate(row)}
            for row in islice(self._execute(), n)
        ]

    @property
    def columns(self) -> list[str]:
        _, out_cols = self._build_processor()
        return out_cols

    @property
    def schema(self) -> LazySchema:
        return self._schema

    def __repr__(self):
        _, out_cols = self._build_processor()
        n_steps = len(self._transforms)
        return f'Stream [{", ".join(out_cols)}] ({n_steps} transforms)'
