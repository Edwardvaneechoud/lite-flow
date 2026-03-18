# Streaming

## `from_iter` — any generator or iterator

```python
import pyfloe as pf

def fetch_events():
    for line in open("log.txt"):
        yield json.loads(line)

ff = pf.from_iter(fetch_events())
ff.filter(pf.col("level") == "ERROR").to_csv("errors.csv")
```

## `from_chunks` — batched / paginated sources

```python
import pyfloe as pf

def fetch_pages():
    page = 1
    while True:
        rows = api.get("/users", page=page, limit=1000)
        if not rows:
            break
        yield rows
        page += 1

ff = pf.from_chunks(fetch_pages)
```

## `Stream` — true single-pass pipeline

`Stream` compiles transforms into a flat loop for maximum throughput. No intermediate iterators or plan-tree overhead.

```python
import pyfloe as pf

pf.Stream.from_iter(event_source(), columns=["ts", "event", "value"]) \
    .filter(pf.col("event") == "error") \
    .with_column("severity",
        pf.when(pf.col("value") > 100, "critical").otherwise("warning")) \
    .to_csv("errors.csv")
```
