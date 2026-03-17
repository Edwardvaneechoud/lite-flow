# lite-flow

**Zero-dependency, lazy, type-safe dataframes for Python.**

Floe is a pure-Python dataframe library that provides lazy evaluation, an expression-based API, query optimization, window functions, datetime handling, and type safety — all without a single external dependency. It installs instantly, runs anywhere Python runs, and streams data with constant memory.

```python
from lite_flow import Floe, read_csv, col, when, row_number

result = (
    read_csv("orders.csv")                            # lazy — file not read yet
    .filter(col("amount") > 100)                      # lazy — no rows scanned
    .with_column("rank", row_number()
        .over(partition_by="region", order_by="amount"))
    .select("order_id", "region", "amount", "rank")
    .sort("region", "rank")
    .collect()                                         # NOW it runs
)

result.schema   # Schema was known before collect() — no data touched
result[0]       # {'order_id': 4, 'region': 'EU', 'amount': 180.0, 'rank': 1}
```

---

## Why Floe?

| Need | Pandas | Polars | Floe |
|------|--------|--------|----------|
| Zero dependencies | ✗ (numpy, pytz, ...) | ✗ (Rust binary) | **✓** |
| Install time | ~seconds | ~seconds | **instant** |
| Package size | ~30 MB | ~20 MB | **~25 KB** |
| Lazy evaluation | ✗ (eager) | ✓ | **✓** |
| Query optimizer | ✗ | ✓ | **✓** |
| Window functions | ✓ | ✓ | **✓** |
| Datetime auto-detection | ✓ | partial | **✓** |
| Type-safe schemas | ✗ | partial | **✓** (TypedDict + runtime) |
| Streaming I/O | partial | partial | **✓** (constant memory) |
| Pure Python | ✗ | ✗ | **✓** |
| Debugger-safe repr | ✗ (eager) | ✗ (eager) | **✓** (won't materialize) |

Floe is not trying to replace Pandas or Polars for heavy analytical workloads. It targets a real gap:

- **Library authors** who can't force users to install numpy/pyarrow
- **Serverless / Lambda** where package size and cold-start matter
- **Embedded ETL** — CLI tools, data pipelines, config processors
- **Education** — a readable query engine you can study end-to-end
- **Type safety enthusiasts** — catch column errors before runtime

---

## Installation

```bash
pip install lite-flow
```

---

## Core concepts

### Everything is lazy

Operations build a **query plan** — a tree of nodes describing *what* to do, without doing it. Data flows only when you trigger evaluation:

```python
pipeline = (
    read_csv("big_file.csv")
    .filter(col("status") == "active")
    .join(read_csv("users.csv"), on="user_id")
    .with_column("score", col("points") * 1.5)
    .select("user_id", "name", "score")
    .sort("score", ascending=False)
)

pipeline.is_materialized  # False
pipeline.schema           # Known instantly — no data touched
pipeline.explain()        # Print the plan tree
```

**Materialization triggers** (the plan runs when you ask for data):

| Method | What happens |
|--------|-------------|
| `.collect()` | Materialize and cache. Returns self. |
| `.to_pylist()` | Returns `List[dict]` |
| `.to_pydict()` | Returns `Dict[str, List]` |
| `.to_csv(path)` | Streams to file — constant memory |
| `.to_jsonl(path)` | Streams to file — constant memory |
| `len(ff)` | Counts all rows |
| `ff[0]` | Indexes into data |

**Safe in debuggers**: `repr()` on an unmaterialized Floe shows `Floe [? rows × 5 cols] (lazy)` without triggering evaluation.

### Schemas propagate without data

Every plan node knows its output schema. You get final types instantly:

```python
pipeline = (
    orders
    .filter(col("amount") > 100)
    .with_column("tax", col("amount") * 0.2)
    .join(customers, on="customer_id")
    .rename({"amount": "subtotal"})
    .select("order_id", "subtotal", "tax")
)

pipeline.schema
# Schema(
#   order_id: int
#   subtotal: float
#   tax: float
# )

pipeline.is_materialized  # False
```

---

## Expression API

Expressions are composable AST nodes. They enable type inference, optimization, and IDE autocomplete.

### Column references and literals

```python
from lite_flow import col, lit

col("amount")           # column reference
lit(42)                 # literal value
col("amount") * 1.1     # arithmetic (1.1 auto-wrapped as lit)
```

### Comparisons and logic

```python
col("amount") > 100
col("region") == "EU"
(col("amount") > 100) & (col("region") == "EU")   # AND
(col("x") < 0) | (col("x") > 100)                 # OR
~(col("active"))                                    # NOT
col("region").is_in(["EU", "APAC"])
col("value").is_null()
col("value").is_not_null()
```

### Arithmetic

```python
col("price") * col("quantity")
col("amount") + lit(100)
col("total") / col("count")
col("score") % 10
-col("delta")
100 + col("amount")                       # reverse ops work
```

### Type casting

```python
col("amount").cast(str)
col("id").cast(float)
```

### Conditional logic (CASE WHEN)

```python
from lite_flow import when

when(col("amount") > 200, "large") \
    .when(col("amount") > 100, "medium") \
    .otherwise("small")
```

### String methods

```python
col("name").str.upper()              # "ALICE"
col("name").str.lower()              # "alice"
col("name").str.strip()              # trim whitespace
col("name").str.title()              # "Alice"
col("name").str.len()                # 5
col("name").str.contains("li")       # True
col("name").str.startswith("Al")     # True
col("name").str.endswith("ce")       # True
col("name").str.replace("A", "a")    # "alice"
col("name").str.slice(0, 3)          # "Ali"
```

### Aggregations

```python
col("amount").sum()
col("amount").mean()
col("amount").min()
col("amount").max()
col("amount").count()
col("amount").n_unique()
col("amount").first()
col("amount").last()
```

Used with `group_by`:

```python
orders.group_by("region").agg(
    col("amount").sum().alias("total_revenue"),
    col("order_id").count().alias("order_count"),
    col("amount").mean().alias("avg_order"),
)
```

### Window functions

```python
from lite_flow import row_number, rank, dense_rank

# Ranking
row_number().over(partition_by="region", order_by="amount")
rank().over(partition_by="dept", order_by="salary")
dense_rank().over(order_by="score")

# Running aggregates
col("amount").cumsum().over(partition_by="region", order_by="date")
col("score").cummax().over(order_by="round")

# Lag / Lead
col("value").lag(1, default=0).over(partition_by="user", order_by="ts")
col("value").lead(1).over(order_by="ts")

# Window aggregation (partition total on every row)
col("amount").sum().over(partition_by="region")
```

---

## Datetime support

Floe auto-detects datetime columns when reading CSV files and provides a full `.dt` accessor for extraction, truncation, arithmetic, and formatting — all using Python's stdlib `datetime` module.

### Auto-detection from CSV

```python
ff = read_csv("events.csv")
ff.schema
# Schema(
#   event_id: int
#   ts: datetime       ← auto-detected from strings like "2024-01-15 08:30:00"
#   amount: float
# )
```

Detection works by sampling the first 100 rows. If ≥80% of non-empty values in a string column parse as the same datetime format, the column is typed as `datetime`. Supported formats include ISO 8601 (`2024-01-15T08:30:00`), space-separated (`2024-01-15 08:30:00`), date-only (`2024-01-15`), US (`01/15/2024`), EU (`15/01/2024`), compact (`20240115`), and named months (`Jan 15, 2024`).

### In-memory datetime data

```python
from datetime import datetime

ff = Floe([
    {"id": 1, "ts": datetime(2024, 1, 15, 10, 0), "val": 100},
    {"id": 2, "ts": datetime(2024, 6, 20, 14, 30), "val": 200},
])
ff.schema.dtypes["ts"]  # <class 'datetime'>
```

### Component extraction

```python
col("ts").dt.year()           # 2024
col("ts").dt.month()          # 1
col("ts").dt.day()            # 15
col("ts").dt.hour()           # 10
col("ts").dt.minute()         # 30
col("ts").dt.second()         # 0
col("ts").dt.microsecond()    # 0
col("ts").dt.weekday()        # 0 (Monday)
col("ts").dt.isoweekday()     # 1 (Monday)
col("ts").dt.quarter()        # 1
col("ts").dt.week()           # ISO week number
col("ts").dt.day_of_year()    # 15
col("ts").dt.day_name()       # "Monday"
col("ts").dt.month_name()     # "January"
col("ts").dt.date()           # date(2024, 1, 15)
col("ts").dt.time()           # time(10, 0)
```

### Truncation

Snap datetimes to a boundary — useful for time-series grouping:

```python
col("ts").dt.truncate("year")     # 2024-03-15 14:30:00 → 2024-01-01 00:00:00
col("ts").dt.truncate("month")    # 2024-03-15 14:30:00 → 2024-03-01 00:00:00
col("ts").dt.truncate("day")      # 2024-03-15 14:30:00 → 2024-03-15 00:00:00
col("ts").dt.truncate("hour")     # 2024-03-15 14:30:00 → 2024-03-15 14:00:00
col("ts").dt.truncate("minute")   # 2024-03-15 14:30:45 → 2024-03-15 14:30:00
```

### Formatting

```python
col("ts").dt.strftime("%Y/%m/%d")     # "2024/01/15"
col("ts").dt.strftime("%b %d, %Y")    # "Jan 15, 2024"
col("ts").dt.epoch_seconds()           # Unix timestamp as float
```

### Arithmetic

```python
col("ts").dt.add_days(7)
col("ts").dt.add_hours(3)
col("ts").dt.add_minutes(45)
col("ts").dt.add_seconds(90)
```

### Filtering by datetime

```python
from lite_flow import lit
from datetime import datetime

# Filter by component
events.filter(col("ts").dt.year() == 2024)
events.filter(col("ts").dt.quarter() == 4)

# Compare to a specific datetime
cutoff = datetime(2024, 6, 1)
events.filter(col("ts") > lit(cutoff))
```

### Grouping by time periods

```python
# Revenue by quarter
(
    read_csv("events.csv")
    .filter(col("event") == "purchase")
    .with_column("q", col("ts").dt.quarter())
    .group_by("q").agg(col("amount").sum().alias("revenue"))
    .sort("q")
)

# Event count by calendar month
(
    read_csv("events.csv")
    .with_column("month_start", col("ts").dt.truncate("month"))
    .group_by("month_start").agg(col("event_id").count().alias("n"))
    .sort("month_start")
)
```

### Null handling

All `.dt` methods return `None` for null inputs:

```python
data = [{"ts": datetime(2024, 1, 15)}, {"ts": None}]
Floe(data).with_column("month", col("ts").dt.month()).to_pylist()
# [{"ts": ..., "month": 1}, {"ts": None, "month": None}]
```

---

## Operations

### Creating a Floe

```python
ff = Floe([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
ff = Floe([user1, user2, user3])  # objects with __dict__
ff = read_csv("data.csv")
ff = from_iter(my_generator())
```

### Selecting, filtering, computing

```python
ff.select("name", "age")
ff.drop("internal_id")
ff.filter(col("amount") > 100)
ff.with_column("tax", col("amount") * 0.2)
ff.with_columns(tax=col("amount") * 0.2, q=col("ts").dt.quarter())
```

### Sorting, joining, grouping

```python
ff.sort("amount", ascending=False)

# Hash join (default) — materializes right side
orders.join(customers, on="customer_id", how="left")

# Sort-merge join — O(1) memory for pre-sorted inputs
orders.sort("id").join(customers.sort("id"), on="id", sorted=True)

# Hash aggregation (default)
ff.group_by("region").agg(col("amount").sum().alias("total"))

# Sorted streaming aggregation — O(1) memory per group
ff.sort("region").group_by("region", sorted=True).agg(
    col("amount").sum().alias("total")
)
```

### Other operations

```python
ff.rename({"old": "new"})
ff.explode("tags")
ff.union(other_ff)
ff.apply(str, columns=["amount"])
ff.head(10)
ff[0]          # first row as dict
ff[5:10]       # slice → new Floe
```

---

## File I/O

### Reading files (all lazy)

```python
from lite_flow import read_csv, read_tsv, read_jsonl, read_json, read_fixed_width

ff = read_csv("data.csv")       # lazy, types + datetime inferred
ff = read_tsv("data.tsv")
ff = read_jsonl("events.jsonl")  # streaming
ff = read_json("cities.json")    # parsed at read time
ff = read_fixed_width("report.txt", widths=[10, 20, 8, 12], has_header=True)
```

#### Parquet (optional — requires pyarrow)

```python
from lite_flow import read_parquet
ff = read_parquet("data.parquet")
ff = read_parquet("data.parquet", columns=["id", "score"])
```

### Writing files (streaming)

```python
ff.to_csv("out.csv")           # streams from plan, constant memory
ff.to_tsv("out.tsv")
ff.to_jsonl("out.jsonl")
ff.to_json("out.json", indent=2)
ff.to_parquet("out.parquet")   # requires pyarrow
```

---

## Streaming

### `from_iter` — any generator or iterator

```python
from lite_flow import from_iter, col

def fetch_events():
    for line in open("log.txt"):
        yield json.loads(line)

ff = from_iter(fetch_events())
ff.filter(col("level") == "ERROR").to_csv("errors.csv")
```

### `from_chunks` — batched / paginated sources

```python
from lite_flow import from_chunks

def fetch_pages():
    page = 1
    while True:
        rows = api.get("/users", page=page, limit=1000)
        if not rows:
            break
        yield rows
        page += 1

ff = from_chunks(fetch_pages)
```

### `Stream` — true single-pass pipeline

```python
from lite_flow import Stream, col, when

Stream.from_iter(event_source(), columns=["ts", "event", "value"]) \
    .filter(col("event") == "error") \
    .with_column("severity",
        when(col("value") > 100, "critical").otherwise("warning")) \
    .to_csv("errors.csv")
```

---

## Query plan and optimizer

```python
print(pipeline.explain())
# Project [order_id, name, amount]
#   Filter [(col("segment") == 'Enterprise')]
#     Filter [(col("region") == 'EU')]
#       Join [inner] ['customer_id'] = ['customer_id']
#         Scan [...] (6 rows)
#         Scan [...] (4 rows)

print(pipeline.explain(optimized=True))
# Project [order_id, name, amount]
#   Join [inner] ...
#     Filter [(col("region") == 'EU')]        ← pushed into left branch
#       Scan [...]
#     Filter [(col("segment") == 'Enterprise')]  ← pushed into right branch
#       Scan [...]
```

---

## Type safety

```python
from typing import TypedDict

class Order(TypedDict):
    order_id: int
    amount: float
    region: str

orders.validate(Order)           # raises TypeError with mismatches
typed = orders.typed(Order)      # → TypedFloe[Order]
typed.filter(...).to_pylist()    # IDE knows this returns List[Order]
```

---

## Algorithms and data structures

This section describes the algorithms behind each major subsystem. Everything is implemented in pure Python using only stdlib data structures.

### Execution model: volcano / iterator

Floe uses the **volcano model** (also called the iterator model or Graefe model), the same execution strategy used by most SQL databases. Each plan node implements an `execute()` method that returns a Python iterator. When a parent node asks for data, it calls `execute()` on its child, which calls its child, and so on down to the leaf (data source). Rows are pulled up one at a time through the tree.

```
to_pylist() calls → FilterNode.execute()
                        calls → JoinNode.execute()
                                    calls → ScanNode.execute()  (left)
                                    calls → ScanNode.execute()  (right)
```

Rows are never buffered between stages unless an operation requires it (sort, group-by). For a pipeline like `read_csv → filter → select → to_csv`, exactly one row is in memory at a time.

**Complexity**: O(1) memory per row for streaming operations. O(n) only when materialization is forced.

### Join: hash join vs sort-merge join

Floe provides two join algorithms, chosen via the `sorted=` parameter:

**Hash join** (default): materializes the right table into a hash map keyed on join columns, then streams the left table probing the map.

```
Build:   right_index = {}
         for row in right: right_index[key(row)].append(row)     O(m)

Probe:   for row in left:
             for match in right_index[key(row)]:
                 yield row + match                                O(n)
```

**Sort-merge join** (`sorted=True`): for inputs already sorted on the join key, two cursors advance in lockstep. Whichever has the smaller key advances; when keys match, it emits the cross product.

```
while left and right not exhausted:
    if left_key < right_key:  advance left    (emit null for left/full join)
    elif left_key > right_key: advance right  (emit null for full join)
    else: collect matching group, emit cross product
```

| | Hash join | Sort-merge join |
|---|---|---|
| Time | O(n + m) | O(n + m) if pre-sorted, O(n log n) if not |
| Memory | **O(m)** for hash table | **O(1)** base (O(g) for many-to-many groups) |
| Best when | Unsorted data, small right side | Pre-sorted data, memory-constrained |
| API | `.join(other, on="id")` | `.join(other, on="id", sorted=True)` |

For a streaming pipeline like `read_csv("sorted_log.csv").join(lookup, on="key", sorted=True).to_csv(...)`, the merge join never materializes either side — rows flow through with constant memory.

### Aggregation: hash vs sorted streaming

Like joins, Floe provides two aggregation strategies:

**Hash aggregation** (default): groups rows into a `dict` keyed by group columns, maintaining **running accumulators** per group. Unlike a naive implementation that stores all rows per group, this stores only the accumulator state (a running sum, count, min, etc.), so memory is O(k) where k = number of groups, not O(n).

```
accumulators = {}     # key → [running_sum, running_count, ...]

for row in child.execute():
    key = (row[group_col_1], ...)
    if key not in accumulators:
        accumulators[key] = init()
    update(accumulators[key], row)     # O(1) per row

for key, acc in accumulators.items():
    yield key + finalize(acc)
```

**Sorted aggregation** (`sorted=True`): when input is pre-sorted by the group key, groups appear as contiguous runs. A single cursor watches for key changes and emits each group as soon as the key changes. Memory: O(1) per group — only one accumulator lives at a time.

```
prev_key = None
for row in sorted_input:
    key = row[group_cols]
    if key != prev_key:
        if prev_key is not None: yield finalize(prev_key, acc)
        acc = init()
        prev_key = key
    update(acc, row)
yield finalize(prev_key, acc)
```

| | Hash aggregation | Sorted aggregation |
|---|---|---|
| Time | O(n) | O(n) (O(n log n) if you need to sort first) |
| Memory | **O(k)** groups | **O(1)** — one accumulator at a time |
| Best when | Unsorted data | Pre-sorted data, streaming sources |
| API | `.group_by("k").agg(...)` | `.group_by("k", sorted=True).agg(...)` |

Accumulator types: `sum` (running total), `count` (increment), `mean` (sum + count, divide at finalize), `min`/`max` (running extremum), `first` (first seen), `last` (overwrite), `n_unique` (set).

### Sort: Timsort

`SortNode` materializes all rows and delegates to Python's built-in `sorted()`, which uses **Timsort** — a hybrid merge-sort / insertion-sort that is O(n log n) worst-case but adapts to partially-sorted data (O(n) for already-sorted input).

Multi-column sorting composes key tuples. For mixed ascending/descending, a negation wrapper handles numeric columns; string columns use separate stable sorts in reverse priority order, exploiting Timsort's stability guarantee.

**Complexity**: O(n log n) time, O(n) memory.

### Window functions: sort-partition-scan

`WindowNode` implements the **sort-partition-scan** pattern used by SQL window functions:

```
1. Materialize all rows                      O(n)
2. Sort by (partition_key, order_key)         O(n log n)
3. Scan each partition (contiguous run):
     - row_number: incrementing counter       O(1) per row
     - rank: counter + gap on tie change      O(1) per row
     - dense_rank: counter, no gap            O(1) per row
     - cumsum/cummax/cummin: running fold      O(1) per row
     - lag/lead: index offset into partition   O(1) per row
     - agg over partition: one pass,           O(p) per partition
       then broadcast to all rows
4. Restore original row order                 O(n log n)
```

Partition boundaries are detected by checking when the partition key changes in the sorted sequence (no hash map needed).

**Complexity**: O(n log n) dominated by the two sorts. O(n) memory.

### Schema propagation

Each `PlanNode` computes its output schema from its input schema and the operation's semantics, without touching data:

| Node | Schema rule |
|------|------------|
| `ScanNode` | Inferred from data sample or provided |
| `FilterNode` | Pass through parent unchanged |
| `ProjectNode` | Select/reorder from parent |
| `WithColumnNode` | Append new column with inferred type |
| `JoinNode` | Merge left and right schemas |
| `AggNode` | Group-by keys + aggregate output types |
| `RenameNode` | Rename keys in parent schema |
| `SortNode`, `ExplodeNode` | Pass through unchanged |
| `WindowNode` | Append window column with inferred type |

Type inference for expressions walks the AST: `col("a") * col("b")` resolves both operands' types, then applies promotion rules (`int × float → float`). `when().otherwise()` takes the widest branch type. Aggregations have fixed output types (`count → int`, `mean → float`).

**Complexity**: O(d) where d is plan tree depth. Instantaneous for any practical pipeline.

### Query optimizer: rule-based rewriting

The optimizer makes two passes over the plan tree:

#### Pass 1: Filter pushdown

Walks top-down. When it finds a `FilterNode`, it examines the filter's required columns and tries to push it past the node below:

| Child node | Rule |
|-----------|------|
| `ProjectNode` | Push if all filter columns exist in the project's input |
| `JoinNode` | If all filter columns come from the left side, push into left branch. If all from the right, push into right. If mixed, leave in place. |
| `AggNode`, `SortNode` | Leave in place (semantics change across aggregation) |

Compound filters (`(a > 1) & (b < 5)`) are split into independent filters to maximize pushdown.

```
Before:                          After:
Filter(region='EU')              Join
  Filter(segment='Ent')            Filter(region='EU')
    Join                              Scan(orders)
      Scan(orders)                Filter(segment='Ent')
      Scan(customers)                Scan(customers)
```

#### Pass 2: Column pruning

Walks top-down, tracking which columns are needed by nodes above:

1. Root declares its output columns
2. For each child, compute minimal columns needed (filter columns + parent columns + join keys)
3. Insert `ProjectNode` at `ScanNode` to prune unused columns early

**Complexity**: Both passes are O(d) where d is tree depth.

### Type inference for CSV: two-phase detection

**Phase 1 — Basic types**: each cell is tested against a type ladder: `bool → int → float → str`. The first successful parse wins. Per-column types are promoted using a lattice: `int + float → float`, `anything + str → str`, `T + None → T(nullable)`.

**Phase 2 — Datetime detection**: columns still typed as `str` after phase 1 are tested column-wise. The first successfully-parsed value's format becomes a candidate, then all values are validated against that single format. If ≥80% parse, the column is typed as `datetime` and the detected format string is cached for fast parsing during streaming.

This avoids the cost of trying `strptime` with 14 formats on every cell — datetime detection runs once per column on the sample, not per-value.

**Complexity**: O(s × c) where s = sample size (default 100), c = column count.

### Streaming: factory-based replay

File readers store a **factory function**, not data:

```python
def make_rows():
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)             # skip header
        for row in reader:
            yield cast(row)      # one row at a time
```

Each `execute()` call invokes the factory, reopening the file and yielding a fresh generator. This means:

- `read_csv → filter → to_csv` uses O(1) memory: one row in flight at a time
- `to_pylist()` can be called repeatedly: the file is re-read each time
- `head(10)` yields 10 rows, then the generator is garbage-collected (file closes)

The `Stream` class compiles transforms into a flat loop:

```python
for row in source():
    if not predicate(row): continue
    row = row + (expr(row),)
    row = (row[2], row[0])
    writer.writerow(row)
```

No intermediate iterators or plan-tree overhead — ~30% faster than the Floe pipeline for pure streaming.

### Data representation

Rows are stored as Python **tuples** internally, not dicts:

- ~40% less memory than dicts (no key storage per row)
- Faster to create and iterate
- Hashable (needed for grouping and join keys)

Column-to-index mapping lives in the schema:

```python
columns = ["name", "age", "score"]
col_map = {"name": 0, "age": 1, "score": 2}
row = ("Alice", 30, 95.5)
row[col_map["age"]]  # → 30
```

Conversion to dicts happens only at the output boundary (`to_pylist()`, `__iter__`).

### Expression evaluation

Expressions form an AST. Each node implements `eval(row, col_map)`:

```
BinaryExpr(op=*, left=Col("price"), right=Col("qty"))
    Col("price").eval(row, col_map)  →  row[2]  →  25.0
    Col("qty").eval(row, col_map)    →  row[3]  →  4
    result: 25.0 * 4 = 100.0
```

`WhenExpr` walks branches in order (identical to SQL `CASE WHEN`):

```python
for condition, value in branches:
    if condition.eval(row, col_map):
        return value.eval(row, col_map)
return otherwise.eval(row, col_map)
```

Datetime accessor methods (`col("ts").dt.year()`) produce `_DtUnaryExpr` nodes that extract components via stdlib `datetime` attributes. Null values short-circuit to `None`.

**Complexity**: O(expression depth) per row — typically 1–5 levels.

### Summary of complexities

| Operation | Time | Memory | Algorithm |
|-----------|------|--------|-----------|
| `filter`, `select`, `with_column` | O(n) streaming | O(1) | Generator chain (volcano) |
| `join` | O(n + m) | O(m) right side | Hash join |
| `join(sorted=True)` | O(n + m) | **O(1)** | Sort-merge join |
| `group_by().agg()` | O(n) | O(k groups) | Hash agg (running accumulators) |
| `group_by(sorted=True).agg()` | O(n) | **O(1)** | Sorted streaming agg |
| `sort` | O(n log n) | O(n) | Timsort |
| `window` functions | O(n log n) | O(n) | Sort-partition-scan |
| `explain` / `schema` | O(d tree depth) | O(d) | AST walk |
| `optimize` | O(d tree depth) | O(d) | Rule-based rewrite |
| CSV type inference | O(s × c) | O(s × c) | Type ladder + datetime detection |
| File streaming | O(n) | O(1) | Factory-based generator replay |

---

## Architecture

```
lite_flow/
├── schema.py    (154 lines)  LazySchema, ColumnSchema — type propagation
├── expr.py      (740 lines)  Expression AST — col, lit, when, .str, .dt, aggregations, windows
├── plan.py      (660 lines)  Plan nodes + optimizer (volcano model)
├── core.py      (518 lines)  Floe, TypedFloe, GroupByBuilder
├── io.py        (610 lines)  File readers/writers — CSV, TSV, JSONL, JSON, Parquet
├── stream.py    (649 lines)  from_iter, from_chunks, Stream pipeline
└── __init__.py  (56 lines)   Public API
                ─────────────
                ~3,400 lines total, zero dependencies
```

---

## Complete API reference

### Constructors

| Function | Description |
|----------|-------------|
| `Floe(data)` | From list of dicts or objects |
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

### Floe methods

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
| `.typed(T)` | ✓ | → TypedFloe[T] |
| `.validate(T)` | ✓ | Check schema |

### Accessor methods

| Accessor | Methods |
|----------|---------|
| `.str` | `upper`, `lower`, `strip`, `title`, `len`, `contains`, `startswith`, `endswith`, `replace`, `slice` |
| `.dt` | `year`, `month`, `day`, `hour`, `minute`, `second`, `microsecond`, `weekday`, `isoweekday`, `quarter`, `week`, `day_of_year`, `day_name`, `month_name`, `date`, `time`, `truncate`, `strftime`, `epoch_seconds`, `add_days`, `add_hours`, `add_minutes`, `add_seconds` |

---

## Tests

190 tests across five suites:

```
python test_package.py      # 73 tests — core, expressions, windows, optimizer, types
python test_io.py           # 28 tests — CSV, TSV, JSONL, JSON, fixed-width, roundtrips
python test_streaming.py    # 32 tests — from_iter, from_chunks, Stream, memory proofs
python test_datetime.py     # 39 tests — auto-detection, accessor, truncation, arithmetic
python test_sorted.py       # 18 tests — sort-merge join, sorted agg, running accumulators, memory
```

---

## License

MIT