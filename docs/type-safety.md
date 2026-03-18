# Type Safety

pyfloe supports TypedDict-based schema validation and typed dataframes for IDE-friendly results.

## Validating schemas

```python
from typing import TypedDict

class Order(TypedDict):
    order_id: int
    amount: float
    region: str

orders.validate(Order)           # raises TypeError with mismatches
```

## TypedLazyFrame

Wrap a LazyFrame to get typed results that your IDE understands:

```python
typed = orders.typed(Order)      # → TypedLazyFrame[Order]
typed.filter(...).to_pylist()    # IDE knows this returns List[Order]
```
