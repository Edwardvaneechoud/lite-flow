# Getting Started

**pyfloe** is a zero-dependency, lazy dataframe library for Python. It provides an expression-based API, query optimization, window functions, datetime handling, and type safety — all in pure Python.

## Installation

```bash
pip install pyfloe
```

## Quick example

```python
from pyfloe import LazyFrame, read_csv, col, when, row_number

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

**Safe in debuggers**: `repr()` on an unmaterialized LazyFrame shows `LazyFrame [? rows × 5 cols] (lazy)` without triggering evaluation.

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

## Who is it for?

- **Library authors** who can't force users to install numpy/pyarrow
- **Serverless / Lambda** where package size and cold-start matter
- **Embedded ETL** — CLI tools, data pipelines, config processors
- **Education** — a readable query engine you can study end-to-end
- **Type safety enthusiasts** — catch column errors before runtime
