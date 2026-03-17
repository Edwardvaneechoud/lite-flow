# pyfloe

**Zero-dependency, lazy dataframes for Python.**

pyfloe is a pure-Python dataframe library with lazy evaluation, query optimization, and type safety — no external dependencies required.

```python
from pyfloe import Floe, read_csv, col, row_number

result = (
    read_csv("orders.csv")
    .filter(col("amount") > 100)
    .with_column("rank", row_number()
        .over(partition_by="region", order_by="amount"))
    .select("order_id", "region", "amount", "rank")
    .sort("region", "rank")
    .collect()
)
```

## Installation

```bash
pip install pyfloe
```

## Key features

- **Lazy evaluation** — operations build a query plan; data flows only when you collect
- **Expression API** — composable column expressions with arithmetic, comparisons, string methods, and conditionals
- **Window functions** — `row_number`, `rank`, `dense_rank`, `cumsum`, `lag`, `lead`, and more
- **Datetime handling** — auto-detection from CSV, `.dt` accessor for extraction, truncation, and arithmetic
- **Streaming I/O** — read and write CSV, TSV, JSONL, JSON, and fixed-width files with constant memory
- **Query optimizer** — filter pushdown and column pruning
- **Type safety** — TypedDict validation and `TypedFloe` for IDE-friendly typed results

## Documentation

Full documentation is available at [edwardvaneechoud.github.io/pyfloe](https://edwardvaneechoud.github.io/pyfloe/).

## License

MIT
