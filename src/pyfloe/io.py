from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .plan import PlanNode, ScanNode
from .schema import ColumnSchema, LazySchema

if TYPE_CHECKING:
    from .core import LazyFrame


def _infer_type(value: str) -> type:
    """Infer the Python type of a single string value.

    Attempts to parse *value* as bool, int, float (in that order) and
    returns the first matching type.  Returns ``str`` as a fallback.
    Empty or ``None`` values yield ``type(None)``.
    """
    if value == "" or value is None:
        return type(None)
    if value.lower() in ("true", "false"):
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


def _cast_value(value: str, dtype: type, dt_fmt: str | None = None) -> Any:
    """Cast a raw string *value* to the given *dtype*.

    Returns ``None`` for empty strings.  Uses *dt_fmt* when casting to
    ``datetime``.  Falls back to the original string on parse errors.
    """
    if value == "" or value is None:
        return None
    if dtype is bool:
        return value.lower() == "true"
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
    """Return the least general type that can represent both *t1* and *t2*.

    Rules: ``None`` is transparent, ``int`` + ``float`` promotes to
    ``float``, and any other mismatch falls back to ``str``.
    """
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


def _infer_schema_from_sample(
    columns: list[str],
    sample_rows: list[list[str]],
) -> tuple[LazySchema, list[type], list[str | None]]:
    """Infer a ``LazySchema`` and per-column types from sample rows.

    Returns a tuple of ``(schema, col_types, dt_formats)`` where
    *col_types* is the inferred Python type per column and *dt_formats*
    holds the ``strptime`` format string for datetime columns (or
    ``None`` for non-datetime columns).
    """
    from .expr import _detect_datetime_format

    n_cols = len(columns)
    col_types: list[type] = [type(None)] * n_cols
    col_nullable = [False] * n_cols
    dt_formats: list[str | None] = [None] * n_cols

    for row in sample_rows:
        for i in range(min(len(row), n_cols)):
            val = row[i]
            if val == "" or val is None:
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
    """Plan node that lazily streams rows from a file.

    Wraps a *row_factory* callable that opens and reads the file on
    each execution, and an optional *row_counter* for fast ``count()``
    without full parsing.
    """

    def __init__(
        self,
        columns: list[str],
        lazy_schema: LazySchema,
        row_factory: Callable[[], Iterator[tuple]],
        source_label: str = "File",
        row_counter: Callable[[], int] | None = None,
    ) -> None:
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

    def fast_count(self) -> int | None:
        if self._row_counter is not None:
            return self._row_counter()
        return None

    def _explain_self(self) -> str:
        return f"{self._source_label} [{', '.join(self._columns)}]"


def _count_lines(
    path: str, quotechar: str, encoding: str, has_header: bool, skip_rows: int
) -> int:
    """Count data rows in a delimited file without parsing field values.

    Reads the file in binary mode in 1 MB chunks and counts newlines
    that appear outside quoted regions.  When a chunk contains no quote
    characters the count is performed entirely at C level via
    ``bytes.count``, making it significantly faster than iterating with
    ``csv.reader``.

    Args:
        path: Absolute path to the file.
        quotechar: The quote character used in the file.
        encoding: File encoding (used to encode *quotechar* to bytes).
        has_header: Whether the first row is a header (skipped if True).
        skip_rows: Number of additional leading rows to skip.

    Returns:
        The number of data rows in the file.
    """
    n = 0
    in_quote = False
    qb = quotechar.encode(encoding)
    with open(path, "rb") as f:
        if has_header:
            f.readline()
        for _ in range(skip_rows):
            f.readline()
        while chunk := f.read(1 << 20):
            if not in_quote and qb not in chunk:
                n += chunk.count(b"\n")
            else:
                parts = chunk.split(qb)
                for i, part in enumerate(parts):
                    outside = (i % 2 == 0) != in_quote
                    if outside:
                        n += part.count(b"\n")
                in_quote = (len(parts) % 2 == 0) != in_quote
    return n


def _read_delimited(
    path: str,
    delimiter: str = ",",
    has_header: bool = True,
    columns: list[str] | None = None,
    encoding: str = "utf-8",
    schema_sample_size: int = 100,
    skip_rows: int = 0,
    quotechar: str = '"',
    cast_types: bool = True,
    source_label: str = "CSV",
) -> _FileStreamNode:
    """Build a ``_FileStreamNode`` for a delimited text file.

    Opens the file once to read the header and a sample of rows for
    schema inference, then returns a lazy node whose *row_factory*
    re-opens the file on each execution.
    """
    path = os.path.expanduser(path)
    with open(path, encoding=encoding, newline="") as f:
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
            col_names = columns or [f"col_{i}" for i in range(n)]
            sample = [first]

        for _ in range(schema_sample_size - len(sample)):
            row = next(reader, None)
            if row is None:
                break
            sample.append(row)

    schema, col_types, dt_formats = _infer_schema_from_sample(col_names, sample)
    n_cols = len(col_names)

    def make_rows() -> Iterator[tuple]:
        with open(path, encoding=encoding, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            for _ in range(skip_rows):
                next(reader, None)
            if has_header:
                next(reader, None)
            for row in reader:
                if len(row) < n_cols:
                    row = row + [""] * (n_cols - len(row))
                if cast_types:
                    yield tuple(
                        _cast_value(row[i], col_types[i], dt_formats[i]) for i in range(n_cols)
                    )
                else:
                    yield tuple(row[i] for i in range(n_cols))

    def count_rows() -> int:
        return _count_lines(path, quotechar, encoding, has_header, skip_rows)

    return _FileStreamNode(
        col_names,
        schema,
        make_rows,
        f"{source_label}({os.path.basename(path)})",
        row_counter=count_rows,
    )


def read_csv(
    path: str,
    *,
    has_header: bool = True,
    columns: list[str] | None = None,
    encoding: str = "utf-8",
    delimiter: str = ",",
    schema_sample_size: int = 100,
    skip_rows: int = 0,
    quotechar: str = '"',
    cast_types: bool = True,
) -> LazyFrame:
    """Read a CSV file as a LazyFrame.

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
        A LazyFrame. Data streams from disk when evaluated.

    Examples:
        >>> import pyfloe as pf
        >>> lf = pf.read_csv("orders.csv")  # doctest: +SKIP
        >>> lf.schema.dtypes  # doctest: +SKIP
        {'order_id': <class 'int'>, 'amount': <class 'float'>, ...}
        >>> lf.filter(pf.col("amount") > 100).to_pylist()  # doctest: +SKIP
        [{'order_id': 1, 'amount': 250.0, ...}, ...]
    """
    from .core import LazyFrame

    node = _read_delimited(
        path,
        delimiter=delimiter,
        has_header=has_header,
        columns=columns,
        encoding=encoding,
        schema_sample_size=schema_sample_size,
        skip_rows=skip_rows,
        quotechar=quotechar,
        cast_types=cast_types,
        source_label="CSV",
    )
    return LazyFrame._from_plan(node)


def read_tsv(path: str, **kwargs: Any) -> LazyFrame:
    """Read a TSV (tab-separated) file as a LazyFrame.

    Equivalent to ``read_csv(path, delimiter='\\t', ...)``.

    Args:
        path: Path to the TSV file.
        **kwargs: Additional arguments passed to the CSV reader.

    Returns:
        A LazyFrame.

    Examples:
        >>> import pyfloe as pf
        >>> lf = pf.read_tsv("data.tsv")  # doctest: +SKIP
        >>> lf.to_pylist()  # doctest: +SKIP
        [{'name': 'Alice', 'score': 95}, ...]
    """
    kwargs.setdefault("delimiter", "\t")
    from .core import LazyFrame

    node = _read_delimited(path, source_label="TSV", **kwargs)
    return LazyFrame._from_plan(node)


def read_jsonl(
    path: str,
    *,
    encoding: str = "utf-8",
    schema_sample_size: int = 100,
    columns: list[str] | None = None,
) -> LazyFrame:
    """Read a JSON Lines file as a LazyFrame.

    Each line is parsed as a JSON object. Schema is inferred from a sample.

    Args:
        path: Path to the JSONL file.
        encoding: File encoding.
        schema_sample_size: Number of lines to sample for schema inference.
        columns: Subset of columns to read.

    Returns:
        A LazyFrame.

    Examples:
        >>> import pyfloe as pf
        >>> lf = pf.read_jsonl("events.jsonl")  # doctest: +SKIP
        >>> lf.filter(pf.col("event") == "click").to_pylist()  # doctest: +SKIP
        [{'event': 'click', 'user_id': 42, ...}, ...]

        Read only specific columns:

        >>> lf = pf.read_jsonl("events.jsonl", columns=["event", "user_id"])  # doctest: +SKIP
    """
    from .core import LazyFrame

    path = os.path.expanduser(path)
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
        return LazyFrame._from_plan(ScanNode([], []))

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

    node = _FileStreamNode(col_names, schema, make_rows, f"JSONL({os.path.basename(path)})")
    return LazyFrame._from_plan(node)


def read_json(
    path: str,
    *,
    encoding: str = "utf-8",
    columns: list[str] | None = None,
) -> LazyFrame:
    """Read a JSON file containing a top-level array.

    Unlike :func:`read_jsonl`, this loads the entire file into memory at once.

    Args:
        path: Path to the JSON file.
        encoding: File encoding.
        columns: Subset of columns to read.

    Returns:
        A LazyFrame containing the parsed data.

    Raises:
        ValueError: If the JSON file does not contain a top-level array.

    Examples:
        >>> import pyfloe as pf
        >>> lf = pf.read_json("cities.json")  # doctest: +SKIP
        >>> lf.to_pylist()  # doctest: +SKIP
        [{'city': 'NYC', 'pop': 8336817}, ...]
    """
    from .core import LazyFrame

    path = os.path.expanduser(path)
    with open(path, encoding=encoding) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a top-level array")
    if not data:
        return LazyFrame([])

    return LazyFrame(data)


def read_fixed_width(
    path: str,
    *,
    widths: list[int],
    columns: list[str] | None = None,
    has_header: bool = False,
    encoding: str = "utf-8",
    schema_sample_size: int = 100,
    strip: bool = True,
    cast_types: bool = True,
) -> LazyFrame:
    """Read a fixed-width file as a LazyFrame.

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
        A LazyFrame.

    Examples:
        >>> import pyfloe as pf
        >>> lf = pf.read_fixed_width("people.txt", widths=[10, 4, 14], has_header=True)  # doctest: +SKIP
        >>> lf.to_pylist()  # doctest: +SKIP
        [{'NAME': 'Alice', 'AGE': 30, 'CITY': 'New York'}, ...]
    """
    from .core import LazyFrame

    n_cols = len(widths)
    col_names = columns

    positions = []
    start = 0
    for w in widths:
        positions.append((start, start + w))
        start += w

    path = os.path.expanduser(path)

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
            col_names = [f"col_{i}" for i in range(n_cols)]

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
                    yield tuple(
                        _cast_value(vals[i], col_types[i], dt_formats[i]) for i in range(n_cols)
                    )
                else:
                    yield tuple(vals)

    node = _FileStreamNode(col_names, schema, make_rows, f"FixedWidth({os.path.basename(path)})")
    return LazyFrame._from_plan(node)


def read_parquet(
    path: str,
    *,
    columns: list[str] | None = None,
    batch_size: int = 10_000,
) -> LazyFrame:
    """Read a Parquet file as a LazyFrame (requires pyarrow).

    Schema is read from the Parquet file metadata without loading data.

    Args:
        path: Path to the Parquet file.
        columns: Subset of columns to read.
        batch_size: Number of rows to read per batch.

    Returns:
        A LazyFrame.

    Raises:
        ImportError: If pyarrow is not installed.

    Examples:
        >>> import pyfloe as pf
        >>> lf = pf.read_parquet("data.parquet")  # doctest: +SKIP
        >>> lf.filter(pf.col("score") > 85).select("name", "score").to_pylist()  # doctest: +SKIP
        [{'name': 'Alice', 'score': 95.5}, ...]

        Read only specific columns:

        >>> lf = pf.read_parquet("data.parquet", columns=["id", "score"])  # doctest: +SKIP
    """
    from .core import LazyFrame

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "read_parquet requires pyarrow. Install it with:\n"
            "  pip install pyarrow\n"
            "Or use read_csv/read_jsonl for zero-dependency file reading."
        )

    path = os.path.expanduser(path)
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

    node = _FileStreamNode(col_names, schema, make_rows, f"Parquet({os.path.basename(path)})")
    return LazyFrame._from_plan(node)


def _to_csv_impl(
    lf: LazyFrame, path: str, delimiter: str = ",", header: bool = True, encoding: str = "utf-8"
) -> None:
    """Write a LazyFrame to a delimited text file.

    Streams batches from the plan so the full dataset is never held in
    memory at once.
    """
    path = os.path.expanduser(path)
    with open(path, "w", encoding=encoding, newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        if header:
            writer.writerow(lf.columns)
        for chunk in lf._plan.execute_batched():
            writer.writerows(chunk)


def _to_jsonl_impl(lf: LazyFrame, path: str, encoding: str = "utf-8") -> None:
    path = os.path.expanduser(path)
    cols = lf.columns
    with open(path, "w", encoding=encoding) as f:
        for row in lf._plan.execute():
            obj = {cols[i]: v for i, v in enumerate(row)}
            f.write(json.dumps(obj, default=str) + "\n")


def _to_json_impl(
    lf: LazyFrame, path: str, encoding: str = "utf-8", indent: int | None = None
) -> None:
    path = os.path.expanduser(path)
    cols = lf.columns
    rows = []
    for row in lf._plan.execute():
        rows.append({cols[i]: v for i, v in enumerate(row)})
    with open(path, "w", encoding=encoding) as f:
        json.dump(rows, f, default=str, indent=indent)


def _to_parquet_impl(lf: LazyFrame, path: str, **kwargs: Any) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("to_parquet requires pyarrow. Install with: pip install pyarrow")

    path = os.path.expanduser(path)
    data = lf.to_pydict()
    table = pa.table(data)
    pq.write_table(table, path, **kwargs)
