# Expressions

Expressions are composable AST nodes. They enable type inference, optimization, and IDE autocomplete.

## Column references and literals

```python
import pyfloe as pf

pf.col("amount")           # column reference
pf.lit(42)                 # literal value
pf.col("amount") * 1.1     # arithmetic (1.1 auto-wrapped as lit)
```

## Comparisons and logic

```python
import pyfloe as pf

pf.col("amount") > 100
pf.col("region") == "EU"
(pf.col("amount") > 100) & (pf.col("region") == "EU")   # AND
(pf.col("x") < 0) | (pf.col("x") > 100)                 # OR
~(pf.col("active"))                                      # NOT
pf.col("region").is_in(["EU", "APAC"])
pf.col("value").is_null()
pf.col("value").is_not_null()
```

## Arithmetic

```python
import pyfloe as pf

pf.col("price") * pf.col("quantity")
pf.col("amount") + pf.lit(100)
pf.col("total") / pf.col("count")
pf.col("score") % 10
-pf.col("delta")
100 + pf.col("amount")                       # reverse ops work
```

## Type casting

```python
import pyfloe as pf

pf.col("amount").cast(str)
pf.col("id").cast(float)
```

## Conditional logic (CASE WHEN)

```python
import pyfloe as pf

pf.when(pf.col("amount") > 200, "large") \
    .when(pf.col("amount") > 100, "medium") \
    .otherwise("small")
```

## String methods

```python
import pyfloe as pf

pf.col("name").str.upper()              # "ALICE"
pf.col("name").str.lower()              # "alice"
pf.col("name").str.strip()              # trim whitespace
pf.col("name").str.title()              # "Alice"
pf.col("name").str.len()                # 5
pf.col("name").str.contains("li")       # True
pf.col("name").str.startswith("Al")     # True
pf.col("name").str.endswith("ce")       # True
pf.col("name").str.replace("A", "a")    # "alice"
pf.col("name").str.slice(0, 3)          # "Ali"
```

## Aggregations

```python
import pyfloe as pf

pf.col("amount").sum()
pf.col("amount").mean()
pf.col("amount").min()
pf.col("amount").max()
pf.col("amount").count()
pf.col("amount").n_unique()
pf.col("amount").first()
pf.col("amount").last()
```

Used with `group_by`:

```python
import pyfloe as pf

orders.group_by("region").agg(
    pf.col("amount").sum().alias("total_revenue"),
    pf.col("order_id").count().alias("order_count"),
    pf.col("amount").mean().alias("avg_order"),
)
```

## Window functions

```python
import pyfloe as pf

# Ranking
pf.row_number().over(partition_by="region", order_by="amount")
pf.rank().over(partition_by="dept", order_by="salary")
pf.dense_rank().over(order_by="score")

# Running aggregates
pf.col("amount").cumsum().over(partition_by="region", order_by="date")
pf.col("score").cummax().over(order_by="round")

# Lag / Lead
pf.col("value").lag(1, default=0).over(partition_by="user", order_by="ts")
pf.col("value").lead(1).over(order_by="ts")

# Window aggregation (partition total on every row)
pf.col("amount").sum().over(partition_by="region")
```
