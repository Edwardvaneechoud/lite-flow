# Operations

## Creating a LazyFrame

```python
from pyfloe import LazyFrame, read_csv, from_iter

ff = LazyFrame([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
ff = LazyFrame([user1, user2, user3])  # objects with __dict__
ff = read_csv("data.csv")
ff = from_iter(my_generator())
```

## Selecting, filtering, computing

```python
ff.select("name", "age")
ff.drop("internal_id")
ff.filter(col("amount") > 100)
ff.with_column("tax", col("amount") * 0.2)
ff.with_columns(tax=col("amount") * 0.2, q=col("ts").dt.quarter())
```

## Sorting

```python
ff.sort("amount", ascending=False)
```

## Joining

```python
# Hash join (default) — materializes right side
orders.join(customers, on="customer_id", how="left")

# Sort-merge join — O(1) memory for pre-sorted inputs
orders.sort("id").join(customers.sort("id"), on="id", sorted=True)
```

## Grouping and aggregation

```python
# Hash aggregation (default)
ff.group_by("region").agg(col("amount").sum().alias("total"))

# Sorted streaming aggregation — O(1) memory per group
ff.sort("region").group_by("region", sorted=True).agg(
    col("amount").sum().alias("total")
)
```

## Other operations

```python
ff.rename({"old": "new"})
ff.explode("tags")
ff.union(other_ff)
ff.apply(str, columns=["amount"])
ff.head(10)
ff[0]          # first row as dict
ff[5:10]       # slice → new LazyFrame
```

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
