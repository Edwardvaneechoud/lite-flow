from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

from .plan import PlanNode, ScanNode
from .schema import ColumnSchema, LazySchema


def _infer_type(value: str) -> type:
    if value == '' or value is None:
        return type(None)
    if value.lower() in ('true', 'false'):
        return bool
    try:
        int(value)
        return int
    except (ValueError, OverflowError):
        pass
    try:
        float(value)
        return float
    except ValueError:
        pass
    return str


def _cast_value(value: str, dtype: type, dt_fmt: str = None) -> Any:
    if value == '' or value is None:
        return None
    if dtype is bool:
        return value.lower() == 'true'
    if dtype is int:
        try:
            return int(value)
        except (ValueError, OverflowError):
            return value
    if dtype is float:
        try:
            return float(value)
        except ValueError:
            return value
    if dtype is datetime:
        if dt_fmt:
            try:
                return datetime.strptime(value.strip(), dt_fmt)
            except ValueError:
                return value
        from .expr import _try_parse_datetime
        result = _try_parse_datetime(value)
        if result:
            return result[0]
        return value
    return value


def _promote_types(t1: type, t2: type) -> type:
    if t1 is t2:
        return t1
    if t1 is type(None):
        return t2
    if t2 is type(None):
        return t1
    if {t1, t2} == {int, float}:
        return float
    if t1 is datetime or t2 is datetime:
        return str
    return str


def _infer_schema_from_sample(columns: list[str], sample_rows: list[list[str]],
                               ) -> tuple[LazySchema, list[type], list[str | None]]:
    from .expr import _detect_datetime_format

    n_cols = len(columns)
    col_types = [type(None)] * n_cols
    col_nullable = [False] * n_cols
    dt_formats: list[str | None] = [None] * n_cols

    for row in sample_rows:
        for i in range(min(len(row), n_cols)):
            val = row[i]
            if val == '' or val is None:
                col_nullable[i] = True
            else:
                detected = _infer_type(val)
                col_types[i] = _promote_types(col_types[i], detected)

    for i in range(n_cols):
        if col_types[i] is str:
            col_values = [row[i] for row in sample_rows if i < len(row)]
            fmt = _detect_datetime_format(col_values)
            if fmt:
                col_types[i] = datetime
                dt_formats[i] = fmt

    col_types = [t if t is not type(None) else str for t in col_types]

    schema_cols = {}
    for i, name in enumerate(columns):
        schema_cols[name] = ColumnSchema(name, col_types[i], col_nullable[i])

    return LazySchema(schema_cols), col_types, dt_formats


class _FileStreamNode(PlanNode):
    def __init__(self, columns: list[str], lazy_schema: LazySchema,
                 row_factory: Callable[[], Iterator[tuple]],
                 source_label: str = 'File',
                 row_counter: Callable[[], int] = None):
        self._columns = columns
        self._schema = lazy_schema
        self._row_factory = row_factory
        self._source_label = source_label
        self._row_counter = row_counter

    def schema(self) -> LazySchema:
        return self._schema

    def execute_batched(self) -> Iterator[list]:
        from .plan import _batched
        return _batched(self._row_factory())

    def fast_count(self):
        if self._row_counter is not None:
            return self._row_counter()
        return None

    def _explain_self(self):
        return f'{self._source_label} [{", ".join(self._columns)}]'


def _read_delimited(
    path: str,
    delimiter: str = ',',
    has_header: bool = True,
    columns: list[str] | None = None,
    encoding: str = 'utf-8',
    schema_sample_size: int = 100,
    skip_rows: int = 0,
    quotechar: str = '"',
    cast_types: bool = True,
    source_label: str = 'CSV',
) -> _FileStreamNode:
    with open(path, encoding=encoding, newline='') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)

        for _ in range(skip_rows):
            next(reader, None)

        if has_header:
            header = next(reader, None)
            if header is None:
                col_names = columns or []
                return _FileStreamNode(col_names, LazySchema(), lambda: iter([]), source_label)
            col_names = columns or [h.strip() for h in header]
            sample = []
        else:
            first = next(reader, None)
            if first is None:
                col_names = columns or []
                return _FileStreamNode(col_names, LazySchema(), lambda: iter([]), source_label)
            n = len(first)
            col_names = columns or [f'col_{i}' for i in range(n)]
            sample = [first]

        for _ in range(schema_sample_size - len(sample)):
            row = next(reader, None)
            if row is None:
                break
            sample.append(row)

    schema, col_types, dt_formats = _infer_schema_from_sample(col_names, sample)
    n_cols = len(col_names)

    def make_rows() -> Iterator[tuple]:
        with open(path, encoding=encoding, newline='') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            for _ in range(skip_rows):
                next(reader, None)
            if has_header:
                next(reader, None)
            for row in reader:
                if len(row) < n_cols:
                    row = row + [''] * (n_cols - len(row))
                if cast_types:
                    yield tuple(_cast_value(row[i], col_types[i], dt_formats[i]) for i in range(n_cols))
                else:
                    yield tuple(row[i] for i in range(n_cols))

    def count_rows() -> int:
        n = 0
        with open(path, encoding=encoding, newline='') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            for _ in range(skip_rows):
                next(reader, None)
            if has_header:
                next(reader, None)
            for _ in reader:
                n += 1
        return n

    return _FileStreamNode(col_names, schema, make_rows,
                           f'{source_label}({os.path.basename(path)})',
                           row_counter=count_rows)


def read_csv(
    path: str,
    *,
    has_header: bool = True,
    columns: list[str] | None = None,
    encoding: str = 'utf-8',
    delimiter: str = ',',
    schema_sample_size: int = 100,
    skip_rows: int = 0,
    quotechar: str = '"',
    cast_types: bool = True,
):
    """Read a CSV file as a lazy Floe.

    Types are inferred from a sample and datetime columns are auto-detected.
    Data is not loaded until you trigger evaluation.

    Args:
        path: Path to the CSV file.
        has_header: Whether the first row is a header.
        columns: Override column names.
        encoding: File encoding.
        delimiter: Field delimiter character.
        schema_sample_size: Number of rows to sample for type inference.
        skip_rows: Number of initial rows to skip.
        quotechar: Quote character for CSV fields.
        cast_types: If True, cast string values to inferred types.

    Returns:
        A lazy Floe. Data streams from disk when evaluated.

    Examples:
        >>> from pyfloe import read_csv, col
        >>> ff = read_csv("orders.csv")  # doctest: +SKIP
        >>> ff.schema.dtypes  # doctest: +SKIP
        {'order_id': <class 'int'>, 'amount': <class 'float'>, ...}
        >>> ff.filter(col("amount") > 100).to_pylist()  # doctest: +SKIP
        [{'order_id': 1, 'amount': 250.0, ...}, ...]
    """
    from .core import Floe
    node = _read_delimited(path, delimiter=delimiter, has_header=has_header,
                           columns=columns, encoding=encoding,
                           schema_sample_size=schema_sample_size,
                           skip_rows=skip_rows, quotechar=quotechar,
                           cast_types=cast_types, source_label='CSV')
    return Floe._from_plan(node)


def read_tsv(path: str, **kwargs):
    """Read a TSV (tab-separated) file as a lazy Floe.

    Equivalent to ``read_csv(path, delimiter='\\t', ...)``.

    Args:
        path: Path to the TSV file.
        **kwargs: Additional arguments passed to the CSV reader.

    Returns:
        A lazy Floe.

    Examples:
        >>> from pyfloe import read_tsv
        >>> ff = read_tsv("data.tsv")  # doctest: +SKIP
        >>> ff.to_pylist()  # doctest: +SKIP
        [{'name': 'Alice', 'score': 95}, ...]
    """
    kwargs.setdefault('delimiter', '\t')
    from .core import Floe
    node = _read_delimited(path, source_label='TSV', **kwargs)
    return Floe._from_plan(node)


def read_jsonl(
    path: str,
    *,
    encoding: str = 'utf-8',
    schema_sample_size: int = 100,
    columns: list[str] | None = None,
):
    """Read a JSON Lines file as a lazy Floe.

    Each line is parsed as a JSON object. Schema is inferred from a sample.

    Args:
        path: Path to the JSONL file.
        encoding: File encoding.
        schema_sample_size: Number of lines to sample for schema inference.
        columns: Subset of columns to read.

    Returns:
        A lazy Floe.

    Examples:
        >>> from pyfloe import read_jsonl, col
        >>> ff = read_jsonl("events.jsonl")  # doctest: +SKIP
        >>> ff.filter(col("event") == "click").to_pylist()  # doctest: +SKIP
        [{'event': 'click', 'user_id': 42, ...}, ...]

        Read only specific columns:

        >>> ff = read_jsonl("events.jsonl", columns=["event", "user_id"])  # doctest: +SKIP
    """
    from .core import Floe

    all_keys: list = []
    seen: set = set()
    sample_dicts: list[dict] = []

    with open(path, encoding=encoding) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i >= schema_sample_size:
                break
            obj = json.loads(line)
            sample_dicts.append(obj)
            for k in obj:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)

    col_names = columns or all_keys
    if not col_names:
        return Floe._from_plan(ScanNode([], []))

    schema = LazySchema.from_dicts(sample_dicts)

    if columns:
        schema = schema.select(columns)

    len(col_names)

    def make_rows() -> Iterator[tuple]:
        with open(path, encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield tuple(obj.get(k) for k in col_names)

    node = _FileStreamNode(col_names, schema, make_rows,
                           f'JSONL({os.path.basename(path)})')
    return Floe._from_plan(node)


def read_json(
    path: str,
    *,
    encoding: str = 'utf-8',
    columns: list[str] | None = None,
):
    """Read a JSON file containing a top-level array.

    Unlike :func:`read_jsonl`, this loads the entire file into memory at once.

    Args:
        path: Path to the JSON file.
        encoding: File encoding.
        columns: Subset of columns to read.

    Returns:
        A Floe containing the parsed data.

    Raises:
        ValueError: If the JSON file does not contain a top-level array.

    Examples:
        >>> from pyfloe import read_json
        >>> ff = read_json("cities.json")  # doctest: +SKIP
        >>> ff.to_pylist()  # doctest: +SKIP
        [{'city': 'NYC', 'pop': 8336817}, ...]
    """
    from .core import Floe

    with open(path, encoding=encoding) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError('JSON file must contain a top-level array')
    if not data:
        return Floe([])

    return Floe(data)


def read_fixed_width(
    path: str,
    *,
    widths: list[int],
    columns: list[str] | None = None,
    has_header: bool = False,
    encoding: str = 'utf-8',
    schema_sample_size: int = 100,
    strip: bool = True,
    cast_types: bool = True,
):
    """Read a fixed-width file as a lazy Floe.

    Args:
        path: Path to the file.
        widths: List of column widths in characters.
        columns: Override column names.
        has_header: Whether the first row is a header.
        encoding: File encoding.
        schema_sample_size: Number of rows to sample for type inference.
        strip: Whether to strip whitespace from field values.
        cast_types: If True, cast string values to inferred types.

    Returns:
        A lazy Floe.

    Examples:
        >>> from pyfloe import read_fixed_width
        >>> ff = read_fixed_width("people.txt", widths=[10, 4, 14], has_header=True)  # doctest: +SKIP
        >>> ff.to_pylist()  # doctest: +SKIP
        [{'NAME': 'Alice', 'AGE': 30, 'CITY': 'New York'}, ...]
    """
    from .core import Floe

    n_cols = len(widths)
    col_names = columns

    positions = []
    start = 0
    for w in widths:
        positions.append((start, start + w))
        start += w

    def split_line(line: str) -> list[str]:
        vals = []
        for s, e in positions:
            v = line[s:e]
            if strip:
                v = v.strip()
            vals.append(v)
        return vals

    with open(path, encoding=encoding) as f:
        if has_header:
            header_line = f.readline()
            if col_names is None:
                col_names = split_line(header_line)
        if col_names is None:
            col_names = [f'col_{i}' for i in range(n_cols)]

        sample: list[list[str]] = []
        for i, line in enumerate(f):
            if i >= schema_sample_size:
                break
            if line.strip():
                sample.append(split_line(line))

    schema, col_types, dt_formats = _infer_schema_from_sample(col_names, sample)

    def make_rows() -> Iterator[tuple]:
        with open(path, encoding=encoding) as f:
            if has_header:
                next(f, None)
            for line in f:
                if not line.strip():
                    continue
                vals = split_line(line)
                if cast_types:
                    yield tuple(_cast_value(vals[i], col_types[i], dt_formats[i]) for i in range(n_cols))
                else:
                    yield tuple(vals)

    node = _FileStreamNode(col_names, schema, make_rows,
                           f'FixedWidth({os.path.basename(path)})')
    return Floe._from_plan(node)


def read_parquet(
    path: str,
    *,
    columns: list[str] | None = None,
    batch_size: int = 10_000,
):
    """Read a Parquet file as a lazy Floe (requires pyarrow).

    Schema is read from the Parquet file metadata without loading data.

    Args:
        path: Path to the Parquet file.
        columns: Subset of columns to read.
        batch_size: Number of rows to read per batch.

    Returns:
        A lazy Floe.

    Raises:
        ImportError: If pyarrow is not installed.

    Examples:
        >>> from pyfloe import read_parquet, col
        >>> ff = read_parquet("data.parquet")  # doctest: +SKIP
        >>> ff.filter(col("score") > 85).select("name", "score").to_pylist()  # doctest: +SKIP
        [{'name': 'Alice', 'score': 95.5}, ...]

        Read only specific columns:

        >>> ff = read_parquet("data.parquet", columns=["id", "score"])  # doctest: +SKIP
    """
    from .core import Floe

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "read_parquet requires pyarrow. Install it with:\n"
            "  pip install pyarrow\n"
            "Or use read_csv/read_jsonl for zero-dependency file reading."
        )

    pf = pq.ParquetFile(path)
    arrow_schema = pf.schema_arrow

    if columns:
        selected_fields = [arrow_schema.field(c) for c in columns]
    else:
        selected_fields = [arrow_schema.field(i) for i in range(len(arrow_schema))]
        columns = [f.name for f in selected_fields]

    def _arrow_to_python(arrow_type) -> type:
        import pyarrow as pa
        if pa.types.is_integer(arrow_type):
            return int
        if pa.types.is_floating(arrow_type):
            return float
        if pa.types.is_boolean(arrow_type):
            return bool
        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return str
        if pa.types.is_timestamp(arrow_type) or pa.types.is_date(arrow_type):
            from datetime import datetime
            return datetime
        return str

    schema_cols = {}
    for f in selected_fields:
        dtype = _arrow_to_python(f.type)
        nullable = f.nullable
        schema_cols[f.name] = ColumnSchema(f.name, dtype, nullable)
    schema = LazySchema(schema_cols)

    col_names = list(schema_cols.keys())

    def make_rows() -> Iterator[tuple]:
        pf_inner = pq.ParquetFile(path)
        for batch in pf_inner.iter_batches(batch_size=batch_size, columns=col_names):
            py_cols = [batch.column(c).to_pylist() for c in col_names]
            n = batch.num_rows
            for i in range(n):
                yield tuple(py_cols[j][i] for j in range(len(col_names)))

    node = _FileStreamNode(col_names, schema, make_rows,
                           f'Parquet({os.path.basename(path)})')
    return Floe._from_plan(node)


def _to_csv_impl(ff, path: str, delimiter: str = ',',
                  header: bool = True, encoding: str = 'utf-8'):
    with open(path, 'w', encoding=encoding, newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        if header:
            writer.writerow(ff.columns)
        for chunk in ff._plan.execute_batched():
            writer.writerows(chunk)


def _to_jsonl_impl(ff, path: str, encoding: str = 'utf-8'):
    cols = ff.columns
    with open(path, 'w', encoding=encoding) as f:
        for row in ff._plan.execute():
            obj = {cols[i]: v for i, v in enumerate(row)}
            f.write(json.dumps(obj, default=str) + '\n')


def _to_json_impl(ff, path: str, encoding: str = 'utf-8', indent: int = None):
    cols = ff.columns
    rows = []
    for row in ff._plan.execute():
        rows.append({cols[i]: v for i, v in enumerate(row)})
    with open(path, 'w', encoding=encoding) as f:
        json.dump(rows, f, default=str, indent=indent)


def _to_parquet_impl(ff, path: str, **kwargs):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "to_parquet requires pyarrow. Install with: pip install pyarrow"
        )

    data = ff.to_pydict()
    table = pa.table(data)
    pq.write_table(table, path, **kwargs)
