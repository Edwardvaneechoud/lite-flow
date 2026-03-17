"""Tests for floe.io — file reading and writing."""
import os
import tempfile

from pyfloe import (
    Floe,
    col,
    read_csv,
    read_fixed_width,
    read_json,
    read_jsonl,
    read_tsv,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_read_csv_basic():
    ff = read_csv(f'{DATA_DIR}/orders.csv')
    assert ff.columns == ['order_id', 'customer_id', 'product', 'amount', 'region', 'active']
    result = ff.to_pylist()
    assert len(result) == 7

def test_read_csv_laziness_schema_without_materialization():
    ff = read_csv(f'{DATA_DIR}/orders.csv')
    s = ff.schema
    assert not ff.is_materialized
    assert s.column_names == ['order_id', 'customer_id', 'product', 'amount', 'region', 'active']

def test_read_csv_type_inference():
    ff = read_csv(f'{DATA_DIR}/orders.csv')
    s = ff.schema
    assert s.dtypes['order_id'] is int
    assert s.dtypes['amount'] is float
    assert s.dtypes['product'] is str
    assert s.dtypes['active'] is bool

def test_read_csv_type_casting():
    ff = read_csv(f'{DATA_DIR}/orders.csv')
    result = ff.to_pylist()
    assert result[0]['order_id'] == 1
    assert isinstance(result[0]['order_id'], int)
    assert result[0]['amount'] == 250.0
    assert isinstance(result[0]['amount'], float)
    assert result[0]['active'] is True
    assert isinstance(result[0]['active'], bool)

def test_read_csv_nullable_column():
    ff = read_csv(f'{DATA_DIR}/orders.csv')
    s = ff.schema
    assert s['product'].nullable is True  # row 7 has empty product
    result = ff.to_pylist()
    assert result[6]['product'] is None

def test_read_csv_filter_expression():
    ff = read_csv(f'{DATA_DIR}/orders.csv')
    big = ff.filter(col('amount') > 100)
    assert not big.is_materialized
    result = big.to_pylist()
    assert len(result) == 4
    assert all(r['amount'] > 100 for r in result)

def test_read_csv_chained_pipeline():
    result = (
        read_csv(f'{DATA_DIR}/orders.csv')
        .filter(col('region') == 'EU')
        .select('order_id', 'product', 'amount')
        .sort('amount', ascending=False)
        .to_pylist()
    )
    assert len(result) == 4
    amounts = [r['amount'] for r in result]
    assert amounts == sorted(amounts, reverse=True)

def test_read_csv_cast_types_false_keeps_strings():
    ff = read_csv(f'{DATA_DIR}/orders.csv', cast_types=False)
    result = ff.to_pylist()
    assert result[0]['order_id'] == '1'
    assert result[0]['amount'] == '250.00'

def test_read_csv_re_execution_generator_replays():
    ff = read_csv(f'{DATA_DIR}/orders.csv')
    r1 = ff.to_pylist()
    r2 = ff.to_pylist()  # second call — should still work
    assert r1 == r2

def test_read_csv_explain_shows_file_source():
    ff = read_csv(f'{DATA_DIR}/orders.csv')
    text = ff.explain()
    assert 'CSV' in text

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_read_tsv_basic():
    ff = read_tsv(f'{DATA_DIR}/students.tsv')
    assert ff.columns == ['name', 'score', 'grade']
    result = ff.to_pylist()
    assert len(result) == 4
    assert result[0]['name'] == 'Alice'
    assert result[0]['score'] == 95

def test_read_tsv_schema_inference():
    ff = read_tsv(f'{DATA_DIR}/students.tsv')
    s = ff.schema
    assert s.dtypes['score'] is int
    assert s.dtypes['name'] is str

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_read_jsonl_basic():
    ff = read_jsonl(f'{DATA_DIR}/events.jsonl')
    assert 'event' in ff.columns
    assert 'user_id' in ff.columns
    result = ff.to_pylist()
    assert len(result) == 5

def test_read_jsonl_laziness():
    ff = read_jsonl(f'{DATA_DIR}/events.jsonl')
    s = ff.schema
    assert not ff.is_materialized
    assert s.dtypes['user_id'] is int
    assert s.dtypes['value'] is float

def test_read_jsonl_nullable():
    ff = read_jsonl(f'{DATA_DIR}/events.jsonl')
    s = ff.schema
    assert s['value'].nullable is True

def test_read_jsonl_filter_pipeline():
    result = (
        read_jsonl(f'{DATA_DIR}/events.jsonl')
        .filter(col('event') == 'click')
        .to_pylist()
    )
    assert len(result) == 2

def test_read_jsonl_column_selection():
    ff = read_jsonl(f'{DATA_DIR}/events.jsonl', columns=['event', 'user_id'])
    assert ff.columns == ['event', 'user_id']
    result = ff.to_pylist()
    assert set(result[0].keys()) == {'event', 'user_id'}

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_read_json_array():
    ff = read_json(f'{DATA_DIR}/cities.json')
    assert 'city' in ff.columns
    result = ff.to_pylist()
    assert len(result) == 3
    assert result[0]['city'] == 'NYC'

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_read_fixed_width_basic():
    ff = read_fixed_width(
        f'{DATA_DIR}/people.txt',
        widths=[10, 4, 14],
        has_header=True,
    )
    assert ff.columns == ['NAME', 'AGE', 'CITY']
    result = ff.to_pylist()
    assert len(result) == 3
    assert result[0]['NAME'] == 'Alice'
    assert result[0]['AGE'] == 30
    assert result[0]['CITY'] == 'New York'

def test_read_fixed_width_schema_inference():
    ff = read_fixed_width(
        f'{DATA_DIR}/people.txt',
        widths=[10, 4, 14],
        has_header=True,
    )
    s = ff.schema
    assert s.dtypes['AGE'] is int
    assert s.dtypes['NAME'] is str

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_to_csv_roundtrip():
    ff = Floe([
        {"name": "Alice", "age": 30, "score": 95.5},
        {"name": "Bob", "age": 25, "score": 82.0},
    ])
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        path = f.name
    try:
        ff.to_csv(path)
        ff2 = read_csv(path)
        result = ff2.to_pylist()
        assert len(result) == 2
        assert result[0]['name'] == 'Alice'
        assert result[0]['age'] == 30
        assert result[0]['score'] == 95.5
    finally:
        os.unlink(path)

def test_to_tsv_roundtrip():
    ff = Floe([{"x": 1, "y": "hello"}, {"x": 2, "y": "world"}])
    with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False, mode='w') as f:
        path = f.name
    try:
        ff.to_tsv(path)
        ff2 = read_tsv(path)
        result = ff2.to_pylist()
        assert result == [{"x": 1, "y": "hello"}, {"x": 2, "y": "world"}]
    finally:
        os.unlink(path)

def test_to_jsonl_roundtrip():
    ff = Floe([
        {"event": "click", "value": 3.14},
        {"event": "view", "value": None},
    ])
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as f:
        path = f.name
    try:
        ff.to_jsonl(path)
        ff2 = read_jsonl(path)
        result = ff2.to_pylist()
        assert len(result) == 2
        assert result[0]['event'] == 'click'
        assert result[0]['value'] == 3.14
        assert result[1]['value'] is None
    finally:
        os.unlink(path)

def test_to_json_roundtrip():
    ff = Floe([{"a": 1}, {"a": 2}])
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        path = f.name
    try:
        ff.to_json(path, indent=2)
        ff2 = read_json(path)
        result = ff2.to_pylist()
        assert result == [{"a": 1}, {"a": 2}]
    finally:
        os.unlink(path)

def test_to_csv_streams_from_lazy_pipeline():
    """Write directly from a lazy pipeline without calling collect()."""
    ff = (
        read_csv(f'{DATA_DIR}/orders.csv')
        .filter(col('amount') > 100)
        .select('order_id', 'amount')
    )
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        path = f.name
    try:
        ff.to_csv(path)
        ff2 = read_csv(path)
        result = ff2.to_pylist()
        assert len(result) == 4
        assert all(r['amount'] > 100 for r in result)
    finally:
        os.unlink(path)

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def _write_large_csv():
    """Helper: generate a 10k-row CSV in a temp file, return path."""
    fd, path = tempfile.mkstemp(suffix='_large.csv')
    os.close(fd)
    with open(path, 'w') as f:
        f.write('id,value,category\n')
        for i in range(10_000):
            f.write(f'{i},{i * 1.5},{["A","B","C"][i % 3]}\n')
    return path

def test_large_csv_schema_from_sample_stream_all_rows():
    """Generate a 10k row CSV and verify lazy behavior."""
    path = _write_large_csv()
    try:
        ff = read_csv(path)
        # Schema inferred from first 100 rows without reading all 10k
        s = ff.schema
        assert not ff.is_materialized
        assert s.dtypes['id'] is int
        assert s.dtypes['value'] is float
        assert s.dtypes['category'] is str

        # Filter streams without materializing everything first
        big = ff.filter(col('value') > 14000)
        assert not big.is_materialized
        result = big.to_pylist()
        assert len(result) > 0
        assert all(r['value'] > 14000 for r in result)
    finally:
        os.unlink(path)

def test_large_csv_head_reads_only_first_n_rows():
    path = _write_large_csv()
    try:
        ff = read_csv(path)
        first5 = ff.head(5)
        result = first5.to_pylist()
        assert len(result) == 5
        assert result[0]['id'] == 0
    finally:
        os.unlink(path)

def test_lazy_pipeline_csv_filter_group_to_csv():
    src = _write_large_csv()
    fd, out = tempfile.mkstemp(suffix='_output.csv')
    os.close(fd)
    try:
        (
            read_csv(src)
            .filter(col('id') < 100)
            .group_by('category').agg(
                col('value').sum().alias('total'),
                col('id').count().alias('n'),
            )
            .sort('category')
        ).to_csv(out)

        result = read_csv(out).to_pylist()
        assert len(result) == 3  # A, B, C
        for r in result:
            assert r['n'] in [33, 34]  # 100 rows / 3 categories
    finally:
        os.unlink(src)
        if os.path.exists(out):
            os.unlink(out)


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

import pytest

try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def _write_test_parquet():
    """Helper: create a small parquet file, return path. Requires pyarrow."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    fd, path = tempfile.mkstemp(suffix='.parquet')
    os.close(fd)
    table = pa.table({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'score': [95.5, 82.0, 78.3, 91.2, 88.7],
    })
    pq.write_table(table, path)
    return path


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
def test_read_parquet_basic():
    from pyfloe import read_parquet
    path = _write_test_parquet()
    try:
        ff = read_parquet(path)
        assert ff.columns == ['id', 'name', 'score']
        result = ff.to_pylist()
        assert len(result) == 5
    finally:
        os.unlink(path)


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
def test_read_parquet_lazy_schema_from_metadata():
    from pyfloe import read_parquet
    path = _write_test_parquet()
    try:
        ff = read_parquet(path)
        s = ff.schema
        assert not ff.is_materialized
        assert s.dtypes['id'] is int
        assert s.dtypes['score'] is float
        assert s.dtypes['name'] is str
    finally:
        os.unlink(path)


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
def test_read_parquet_column_pruning():
    from pyfloe import read_parquet
    path = _write_test_parquet()
    try:
        ff = read_parquet(path, columns=['id', 'score'])
        assert ff.columns == ['id', 'score']
        result = ff.to_pylist()
        assert set(result[0].keys()) == {'id', 'score'}
    finally:
        os.unlink(path)


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
def test_read_parquet_with_filter_pipeline():
    from pyfloe import read_parquet
    path = _write_test_parquet()
    try:
        result = (
            read_parquet(path)
            .filter(col('score') > 85)
            .select('name', 'score')
            .sort('score', ascending=False)
            .to_pylist()
        )
        assert all(r['score'] > 85 for r in result)
    finally:
        os.unlink(path)


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
def test_to_parquet_roundtrip():
    from pyfloe import read_parquet
    ff = Floe([
        {"x": 1, "y": "hello"},
        {"x": 2, "y": "world"},
    ])
    fd, path = tempfile.mkstemp(suffix='.parquet')
    os.close(fd)
    try:
        ff.to_parquet(path)
        ff2 = read_parquet(path)
        result = ff2.to_pylist()
        assert result == [{"x": 1, "y": "hello"}, {"x": 2, "y": "world"}]
    finally:
        os.unlink(path)