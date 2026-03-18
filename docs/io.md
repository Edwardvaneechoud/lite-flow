# File I/O

## Reading files

All readers are lazy — data is not loaded until you trigger evaluation.

```python
import pyfloe as pf

ff = pf.read_csv("data.csv")       # lazy, types + datetime inferred
ff = pf.read_tsv("data.tsv")
ff = pf.read_jsonl("events.jsonl")  # streaming
ff = pf.read_json("cities.json")    # parsed at read time
ff = pf.read_fixed_width("report.txt", widths=[10, 20, 8, 12], has_header=True)
```

### Parquet (optional — requires pyarrow)

```python
import pyfloe as pf

ff = pf.read_parquet("data.parquet")
ff = pf.read_parquet("data.parquet", columns=["id", "score"])
```

## Writing files

Writers stream from the query plan with constant memory.

```python
ff.to_csv("out.csv")           # streams from plan, constant memory
ff.to_tsv("out.tsv")
ff.to_jsonl("out.jsonl")
ff.to_json("out.json", indent=2)
ff.to_parquet("out.parquet")   # requires pyarrow
```
