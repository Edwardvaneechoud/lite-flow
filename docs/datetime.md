# Datetime Support

LazyFrame auto-detects datetime columns when reading CSV files and provides a full `.dt` accessor for extraction, truncation, arithmetic, and formatting — all using Python's stdlib `datetime` module.

## Auto-detection from CSV

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

## In-memory datetime data

```python
from datetime import datetime

ff = LazyFrame([
    {"id": 1, "ts": datetime(2024, 1, 15, 10, 0), "val": 100},
    {"id": 2, "ts": datetime(2024, 6, 20, 14, 30), "val": 200},
])
ff.schema.dtypes["ts"]  # <class 'datetime'>
```

## Component extraction

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

## Truncation

Snap datetimes to a boundary — useful for time-series grouping:

```python
col("ts").dt.truncate("year")     # 2024-03-15 14:30:00 → 2024-01-01 00:00:00
col("ts").dt.truncate("month")    # 2024-03-15 14:30:00 → 2024-03-01 00:00:00
col("ts").dt.truncate("day")      # 2024-03-15 14:30:00 → 2024-03-15 00:00:00
col("ts").dt.truncate("hour")     # 2024-03-15 14:30:00 → 2024-03-15 14:00:00
col("ts").dt.truncate("minute")   # 2024-03-15 14:30:45 → 2024-03-15 14:30:00
```

## Formatting

```python
col("ts").dt.strftime("%Y/%m/%d")     # "2024/01/15"
col("ts").dt.strftime("%b %d, %Y")    # "Jan 15, 2024"
col("ts").dt.epoch_seconds()           # Unix timestamp as float
```

## Arithmetic

```python
col("ts").dt.add_days(7)
col("ts").dt.add_hours(3)
col("ts").dt.add_minutes(45)
col("ts").dt.add_seconds(90)
```

## Filtering by datetime

```python
from pyfloe import lit
from datetime import datetime

# Filter by component
events.filter(col("ts").dt.year() == 2024)
events.filter(col("ts").dt.quarter() == 4)

# Compare to a specific datetime
cutoff = datetime(2024, 6, 1)
events.filter(col("ts") > lit(cutoff))
```

## Grouping by time periods

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

## Null handling

All `.dt` methods return `None` for null inputs:

```python
data = [{"ts": datetime(2024, 1, 15)}, {"ts": None}]
LazyFrame(data).with_column("month", col("ts").dt.month()).to_pylist()
# [{"ts": ..., "month": 1}, {"ts": None, "month": None}]
```
