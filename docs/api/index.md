# API Reference

Overview of the pyfloe public API, organized by topic.

## Quick reference

### Constructors

| Function | Description |
|----------|-------------|
| `LazyFrame(data)` | From list of dicts or objects |
| `read_csv(path, ...)` | Lazy CSV reader (auto-detects datetime) |
| `read_tsv(path, ...)` | Lazy TSV reader |
| `read_jsonl(path, ...)` | Lazy JSON Lines reader |
| `read_json(path, ...)` | JSON array reader |
| `read_fixed_width(path, widths, ...)` | Lazy fixed-width reader |
| `read_parquet(path, ...)` | Lazy Parquet reader (requires pyarrow) |
| `from_iter(source, ...)` | From any iterator/generator |
| `from_chunks(chunks, ...)` | From batched/paginated source |
| `Stream.from_iter(source, ...)` | True streaming pipeline |
| `Stream.from_csv(path, ...)` | Stream from CSV |

### LazyFrame methods

| Method | Lazy? | Description |
|--------|-------|-------------|
| `.select(*cols)` | ✓ | Select columns or expressions |
| `.filter(expr)` | ✓ | Filter rows |
| `.with_column(name, expr)` | ✓ | Add computed column |
| `.with_columns(**exprs)` | ✓ | Add multiple columns |
| `.drop(*cols)` | ✓ | Drop columns |
| `.rename(mapping)` | ✓ | Rename columns |
| `.sort(*cols)` | ✗ | Sort (Timsort) |
| `.join(other, on=)` | ✓ | Hash join (or sort-merge with `sorted=True`) |
| `.union(other)` | ✓ | Stack rows |
| `.explode(col)` | ✓ | Unnest lists |
| `.apply(func)` | ✓ | Apply to columns |
| `.group_by(*cols).agg(...)` | ✗ | Hash agg (or streaming with `sorted=True`) |
| `.head(n)` | partial | First n rows |
| `.optimize()` | ✓ | Optimized plan |
| `.collect()` | ✗ | Materialize |
| `.to_pylist()` | ✗ | → List[dict] |
| `.to_csv(path)` | streaming | Write CSV |
| `.to_jsonl(path)` | streaming | Write JSONL |
| `.explain()` | ✓ | Print plan |
| `.schema` | ✓ | Schema (no data) |
| `.typed(T)` | ✓ | → TypedLazyFrame[T] |
| `.validate(T)` | ✓ | Check schema |

### Accessor methods

| Accessor | Methods |
|----------|---------|
| `.str` | `upper`, `lower`, `strip`, `title`, `len`, `contains`, `startswith`, `endswith`, `replace`, `slice` |
| `.dt` | `year`, `month`, `day`, `hour`, `minute`, `second`, `microsecond`, `weekday`, `isoweekday`, `quarter`, `week`, `day_of_year`, `day_name`, `month_name`, `date`, `time`, `truncate`, `strftime`, `epoch_seconds`, `add_days`, `add_hours`, `add_minutes`, `add_seconds` |
