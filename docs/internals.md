# Internals

This section describes the algorithms behind each major subsystem. Everything is implemented in pure Python using only stdlib data structures.

## Execution model: volcano / iterator

LazyFrame uses the **volcano model** (also called the iterator model or Graefe model), the same execution strategy used by most SQL databases. Each plan node implements an `execute()` method that returns a Python iterator. When a parent node asks for data, it calls `execute()` on its child, which calls its child, and so on down to the leaf (data source). Rows are pulled up one at a time through the tree.

```
to_pylist() calls → FilterNode.execute()
                        calls → JoinNode.execute()
                                    calls → ScanNode.execute()  (left)
                                    calls → ScanNode.execute()  (right)
```

Rows are never buffered between stages unless an operation requires it (sort, group-by). For a pipeline like `read_csv → filter → select → to_csv`, exactly one row is in memory at a time.

**Complexity**: O(1) memory per row for streaming operations. O(n) only when materialization is forced.

## Join: hash join vs sort-merge join

LazyFrame provides two join algorithms, chosen via the `sorted=` parameter:

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

## Aggregation: hash vs sorted streaming

Like joins, LazyFrame provides two aggregation strategies:

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

## Sort: Timsort

`SortNode` materializes all rows and delegates to Python's built-in `sorted()`, which uses **Timsort** — a hybrid merge-sort / insertion-sort that is O(n log n) worst-case but adapts to partially-sorted data (O(n) for already-sorted input).

Multi-column sorting composes key tuples. For mixed ascending/descending, a negation wrapper handles numeric columns; string columns use separate stable sorts in reverse priority order, exploiting Timsort's stability guarantee.

**Complexity**: O(n log n) time, O(n) memory.

## Window functions: sort-partition-scan

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

## Schema propagation

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

## Query optimizer: rule-based rewriting

The optimizer makes two passes over the plan tree:

### Pass 1: Filter pushdown

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

### Pass 2: Column pruning

Walks top-down, tracking which columns are needed by nodes above:

1. Root declares its output columns
2. For each child, compute minimal columns needed (filter columns + parent columns + join keys)
3. Insert `ProjectNode` at `ScanNode` to prune unused columns early

**Complexity**: Both passes are O(d) where d is tree depth.

## Type inference for CSV: two-phase detection

**Phase 1 — Basic types**: each cell is tested against a type ladder: `bool → int → float → str`. The first successful parse wins. Per-column types are promoted using a lattice: `int + float → float`, `anything + str → str`, `T + None → T(nullable)`.

**Phase 2 — Datetime detection**: columns still typed as `str` after phase 1 are tested column-wise. The first successfully-parsed value's format becomes a candidate, then all values are validated against that single format. If ≥80% parse, the column is typed as `datetime` and the detected format string is cached for fast parsing during streaming.

This avoids the cost of trying `strptime` with 14 formats on every cell — datetime detection runs once per column on the sample, not per-value.

**Complexity**: O(s × c) where s = sample size (default 100), c = column count.

## Streaming: factory-based replay

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

No intermediate iterators or plan-tree overhead — ~30% faster than the LazyFrame pipeline for pure streaming.

## Data representation

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

## Expression evaluation

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

## Architecture

```
pyfloe/
├── schema.py    (154 lines)  LazySchema, ColumnSchema — type propagation
├── expr.py      (740 lines)  Expression AST — col, lit, when, .str, .dt, aggregations, windows
├── plan.py      (660 lines)  Plan nodes + optimizer (volcano model)
├── core.py      (518 lines)  LazyFrame, TypedLazyFrame, LazyGroupBy
├── io.py        (610 lines)  File readers/writers — CSV, TSV, JSONL, JSON, Parquet
├── stream.py    (649 lines)  from_iter, from_chunks, Stream pipeline
└── __init__.py  (56 lines)   Public API
                ─────────────
                ~3,400 lines total, zero dependencies
```

## Summary of complexities

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
