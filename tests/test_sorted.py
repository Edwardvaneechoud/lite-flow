"""Tests for datetime support: parsing, expressions, accessor, IO."""
import os
import tempfile
from datetime import date, datetime, timedelta

from pyfloe import LazyFrame, col, read_csv

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_detect_datetime_column_in_csv_space_separated():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    s = lf.schema
    assert s.dtypes['ts'] is datetime, f"Expected datetime, got {s.dtypes['ts']}"
    assert s.dtypes['event_id'] is int
    assert s.dtypes['amount'] is float

def test_datetime_values_parsed_correctly():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.to_pylist()
    ts = result[0]['ts']
    assert isinstance(ts, datetime), f"Expected datetime, got {type(ts)}: {ts}"
    assert ts == datetime(2024, 1, 15, 8, 30, 0)

def test_detect_date_only_column_in_csv():
    lf = read_csv(f'{DATA_DIR}/orders_dates.csv')
    s = lf.schema
    assert s.dtypes['order_date'] is datetime
    assert s.dtypes['ship_date'] is datetime
    assert s.dtypes['total'] is float

def test_date_only_values_parsed_as_datetime():
    lf = read_csv(f'{DATA_DIR}/orders_dates.csv')
    result = lf.to_pylist()
    d = result[0]['order_date']
    assert isinstance(d, datetime)
    assert d == datetime(2024, 1, 15)

def test_schema_stays_lazy_with_datetime_columns():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    _ = lf.schema
    assert not lf.is_materialized

def test_non_datetime_string_columns_not_misdetected():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    s = lf.schema
    assert s.dtypes['event'] is str  # "login", "purchase" — not datetimes

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_dt_year_dt_month_dt_day():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_columns(
        year=col("ts").dt.year(),
        month=col("ts").dt.month(),
        day=col("ts").dt.day(),
    ).to_pylist()
    r = result[0]  # 2024-01-15 08:30:00
    assert r['year'] == 2024
    assert r['month'] == 1
    assert r['day'] == 15

def test_dt_hour_dt_minute_dt_second():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_columns(
        hour=col("ts").dt.hour(),
        minute=col("ts").dt.minute(),
        second=col("ts").dt.second(),
    ).to_pylist()
    r = result[1]  # 2024-01-15 09:15:30
    assert r['hour'] == 9
    assert r['minute'] == 15
    assert r['second'] == 30

def test_dt_weekday_dt_isoweekday():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_columns(
        wd=col("ts").dt.weekday(),
        iwd=col("ts").dt.isoweekday(),
    ).to_pylist()
    # 2024-01-15 is a Monday
    assert result[0]['wd'] == 0     # Monday = 0 in weekday()
    assert result[0]['iwd'] == 1    # Monday = 1 in isoweekday()

def test_dt_day_name_dt_month_name():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_columns(
        day_name=col("ts").dt.day_name(),
        month_name=col("ts").dt.month_name(),
    ).to_pylist()
    assert result[0]['day_name'] == 'Monday'
    assert result[0]['month_name'] == 'January'

def test_dt_quarter():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("q", col("ts").dt.quarter()).to_pylist()
    assert result[0]['q'] == 1   # January → Q1
    assert result[2]['q'] == 1   # February → Q1
    assert result[3]['q'] == 1   # March → Q1
    assert result[5]['q'] == 2   # June → Q2
    assert result[7]['q'] == 3   # September → Q3
    assert result[8]['q'] == 4   # December → Q4

def test_dt_week_iso_week_number():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("wk", col("ts").dt.week()).to_pylist()
    # 2024-01-15 is ISO week 3
    assert result[0]['wk'] == 3

def test_dt_day_of_year():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("doy", col("ts").dt.day_of_year()).to_pylist()
    assert result[0]['doy'] == 15  # Jan 15

def test_dt_strftime():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("formatted", col("ts").dt.strftime("%Y/%m/%d")).to_pylist()
    assert result[0]['formatted'] == '2024/01/15'
    assert result[8]['formatted'] == '2024/12/25'

def test_dt_date_extraction():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("d", col("ts").dt.date()).to_pylist()
    assert result[0]['d'] == date(2024, 1, 15)

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_dt_truncate_year():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("yr", col("ts").dt.truncate("year")).to_pylist()
    assert result[0]['yr'] == datetime(2024, 1, 1)

def test_dt_truncate_month():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("mo", col("ts").dt.truncate("month")).to_pylist()
    assert result[0]['mo'] == datetime(2024, 1, 1)   # Jan
    assert result[2]['mo'] == datetime(2024, 2, 1)   # Feb

def test_dt_truncate_day():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("d", col("ts").dt.truncate("day")).to_pylist()
    assert result[1]['d'] == datetime(2024, 1, 15)  # strips time

def test_dt_truncate_hour():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("h", col("ts").dt.truncate("hour")).to_pylist()
    assert result[1]['h'] == datetime(2024, 1, 15, 9, 0, 0)

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_dt_add_days():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("later", col("ts").dt.add_days(7)).to_pylist()
    orig = result[0]['ts']
    later = result[0]['later']
    assert later - orig == timedelta(days=7)

def test_dt_add_hours():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("shifted", col("ts").dt.add_hours(3)).to_pylist()
    assert result[0]['shifted'] == datetime(2024, 1, 15, 11, 30, 0)

def test_dt_add_minutes():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("shifted", col("ts").dt.add_minutes(45)).to_pylist()
    assert result[0]['shifted'] == datetime(2024, 1, 15, 9, 15, 0)

def test_dt_add_seconds():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv')
    result = lf.with_column("shifted", col("ts").dt.add_seconds(90)).to_pylist()
    assert result[0]['shifted'] == datetime(2024, 1, 15, 8, 31, 30)

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_filter_by_year():
    result = (
        read_csv(f'{DATA_DIR}/events_dt.csv')
        .filter(col("ts").dt.year() == 2024)
        .to_pylist()
    )
    assert len(result) == 10  # all rows are 2024

def test_filter_by_month():
    result = (
        read_csv(f'{DATA_DIR}/events_dt.csv')
        .filter(col("ts").dt.month() == 1)
        .to_pylist()
    )
    assert len(result) == 2  # Jan 15 events

def test_filter_by_quarter():
    result = (
        read_csv(f'{DATA_DIR}/events_dt.csv')
        .filter(col("ts").dt.quarter() == 4)
        .to_pylist()
    )
    assert len(result) == 2  # Dec events

def test_filter_events_after_a_specific_datetime():
    cutoff = datetime(2024, 6, 1)
    from pyfloe import lit
    result = (
        read_csv(f'{DATA_DIR}/events_dt.csv')
        .filter(col("ts") > lit(cutoff))
        .to_pylist()
    )
    assert all(r['ts'] > cutoff for r in result)
    assert len(result) == 5  # Jun 1 07:00, Jun 15, Sep 22, Dec 25, Dec 31

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_group_by_month():
    result = (
        read_csv(f'{DATA_DIR}/events_dt.csv')
        .with_column("month", col("ts").dt.month())
        .group_by("month").agg(
            col("event_id").count().alias("events"),
        )
        .sort("month")
        .to_pylist()
    )
    assert result[0]['month'] == 1
    assert result[0]['events'] == 2

def test_group_by_quarter_with_sum():
    result = (
        read_csv(f'{DATA_DIR}/events_dt.csv')
        .filter(col("event") == "purchase")
        .with_column("q", col("ts").dt.quarter())
        .group_by("q").agg(
            col("amount").sum().alias("revenue"),
        )
        .sort("q")
        .to_pylist()
    )
    assert result[0]['q'] == 1
    assert result[0]['revenue'] == 370.50  # 250.50 + 120.00

def test_group_by_truncated_month():
    result = (
        read_csv(f'{DATA_DIR}/events_dt.csv')
        .with_column("month_start", col("ts").dt.truncate("month"))
        .group_by("month_start").agg(
            col("event_id").count().alias("n"),
        )
        .sort("month_start")
        .to_pylist()
    )
    assert result[0]['month_start'] == datetime(2024, 1, 1)
    assert result[0]['n'] == 2

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_datetime_objects_in_floe_constructor():
    data = [
        {"id": 1, "ts": datetime(2024, 1, 15, 10, 0), "val": 100},
        {"id": 2, "ts": datetime(2024, 3, 20, 14, 30), "val": 200},
        {"id": 3, "ts": datetime(2024, 6, 1, 8, 0), "val": 300},
    ]
    lf = LazyFrame(data)
    s = lf.schema
    assert s.dtypes['ts'] is datetime
    result = lf.with_column("month", col("ts").dt.month()).to_pylist()
    assert [r['month'] for r in result] == [1, 3, 6]

def test_datetime_schema_propagation_through_pipeline():
    data = [
        {"id": 1, "ts": datetime(2024, 1, 15, 10, 0)},
        {"id": 2, "ts": datetime(2024, 3, 20, 14, 30)},
    ]
    pipeline = (
        LazyFrame(data)
        .with_column("year", col("ts").dt.year())
        .with_column("day_name", col("ts").dt.day_name())
        .select("id", "year", "day_name")
    )
    s = pipeline.schema
    assert s.dtypes['year'] is int
    assert s.dtypes['day_name'] is str
    assert not pipeline.is_materialized

def test_datetime_null_handling():
    data = [
        {"id": 1, "ts": datetime(2024, 1, 15)},
        {"id": 2, "ts": None},
        {"id": 3, "ts": datetime(2024, 6, 1)},
    ]
    lf = LazyFrame(data)
    result = lf.with_column("month", col("ts").dt.month()).to_pylist()
    assert result[0]['month'] == 1
    assert result[1]['month'] is None
    assert result[2]['month'] == 6

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_lag_on_datetime_column():
    data = [
        {"id": 1, "ts": datetime(2024, 1, 10)},
        {"id": 2, "ts": datetime(2024, 1, 15)},
        {"id": 3, "ts": datetime(2024, 1, 20)},
    ]
    result = LazyFrame(data).with_column("prev_ts",
        col("ts").lag(1).over(order_by="id")
    ).to_pylist()
    assert result[0]['prev_ts'] is None
    assert result[1]['prev_ts'] == datetime(2024, 1, 10)

def test_row_number_ordered_by_datetime():
    data = [
        {"id": 1, "ts": datetime(2024, 3, 1)},
        {"id": 2, "ts": datetime(2024, 1, 1)},
        {"id": 3, "ts": datetime(2024, 2, 1)},
    ]
    from pyfloe import row_number
    result = LazyFrame(data).with_column("rn",
        row_number().over(order_by="ts")
    ).to_pylist()
    earliest = [r for r in result if r['rn'] == 1][0]
    assert earliest['ts'] == datetime(2024, 1, 1)

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_datetime_survives_csv_write_read_roundtrip():
    lf = read_csv(f'{DATA_DIR}/events_dt.csv').select("event_id", "ts")
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name
    try:
        lf.to_csv(path)
        lf2 = read_csv(path)
        assert lf2.schema.dtypes['ts'] is datetime
        result = lf2.to_pylist()
        assert result[0]['ts'] == datetime(2024, 1, 15, 8, 30)
    finally:
        os.unlink(path)

def test_datetime_format_detection_various_formats():
    from pyfloe.expr import _detect_datetime_format
    # ISO
    assert _detect_datetime_format(['2024-01-15', '2024-02-20', '2024-03-10']) == '%Y-%m-%d'
    # ISO with time
    assert _detect_datetime_format(['2024-01-15 08:30:00', '2024-02-20 14:00:00']) == '%Y-%m-%d %H:%M:%S'
    # ISO T separator
    assert _detect_datetime_format(['2024-01-15T08:30:00', '2024-02-20T14:00:00']) == '%Y-%m-%dT%H:%M:%S'
    # Not datetime
    assert _detect_datetime_format(['hello', 'world', 'foo']) is None
    # Mixed — less than 80% parse
    assert _detect_datetime_format(['2024-01-15', 'not-a-date', 'also-not']) is None

def test_non_datetime_columns_with_numbers_not_misdetected():
    # Values like "12345" should be int, not datetime
    from pyfloe.expr import _detect_datetime_format
    assert _detect_datetime_format(['100', '200', '300']) is None
    assert _detect_datetime_format(['3.14', '2.71', '1.41']) is None


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_realistic_analytics_pipeline_with_datetime():
    result = (
        read_csv(f'{DATA_DIR}/events_dt.csv')
        .filter(col("event") == "purchase")
        .with_columns(
            quarter=col("ts").dt.quarter(),
            month_name=col("ts").dt.month_name(),
            day_name=col("ts").dt.day_name(),
            formatted=col("ts").dt.strftime("%b %d"),
        )
        .select("formatted", "month_name", "quarter", "day_name", "amount")
        .sort("amount", ascending=False)
        .to_pylist()
    )
    # Highest amount purchase: Dec 25, $500
    top = result[0]
    assert top['amount'] == 500.0
    assert top['formatted'] == 'Dec 25'
    assert top['quarter'] == 4
    assert top['day_name'] == 'Wednesday'

    # Check all purchases are sorted
    amounts = [r['amount'] for r in result]
    assert amounts == sorted(amounts, reverse=True)


# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════