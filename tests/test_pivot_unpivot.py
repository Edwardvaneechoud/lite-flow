from __future__ import annotations

import pytest

from pyfloe.core import LazyFrame
from pyfloe.expr import Col


@pytest.fixture
def long_data():
    return LazyFrame([
        {'name': 'Alice', 'subject': 'math', 'score': 90},
        {'name': 'Alice', 'subject': 'english', 'score': 85},
        {'name': 'Bob', 'subject': 'math', 'score': 78},
        {'name': 'Bob', 'subject': 'english', 'score': 92},
    ])


@pytest.fixture
def wide_data():
    return LazyFrame([
        {'name': 'Alice', 'math': 90, 'english': 85},
        {'name': 'Bob', 'math': 78, 'english': 92},
    ])


def test_basic_pivot_explicit_columns(long_data):
    result = long_data.pivot(
        index='name', on='subject', values='score',
        columns=['math', 'english'],
    ).collect()
    rows = sorted(result.to_pylist(), key=lambda r: r['name'])
    assert rows[0] == {'name': 'Alice', 'math': 90, 'english': 85}
    assert rows[1] == {'name': 'Bob', 'math': 78, 'english': 92}


def test_pivot_schema_lazy_with_columns(long_data):
    lf = long_data.pivot(
        index='name', on='subject', values='score',
        columns=['math', 'english'],
    )
    schema = lf.schema
    assert 'name' in schema
    assert 'math' in schema
    assert 'english' in schema


def test_pivot_auto_discovers_columns(long_data):
    result = long_data.pivot(
        index='name', on='subject', values='score',
    ).collect()
    cols = set(result.columns)
    assert 'name' in cols
    assert 'math' in cols
    assert 'english' in cols
    rows = sorted(result.to_pylist(), key=lambda r: r['name'])
    assert rows[0]['name'] == 'Alice'
    assert rows[0]['math'] == 90


def test_pivot_sum_aggregation():
    data = LazyFrame([
        {'city': 'NY', 'product': 'A', 'sales': 10},
        {'city': 'NY', 'product': 'A', 'sales': 20},
        {'city': 'NY', 'product': 'B', 'sales': 5},
        {'city': 'LA', 'product': 'A', 'sales': 15},
    ])
    result = data.pivot(
        index='city', on='product', values='sales',
        agg='sum', columns=['A', 'B'],
    ).collect()
    rows = sorted(result.to_pylist(), key=lambda r: r['city'])
    assert rows[0] == {'city': 'LA', 'A': 15, 'B': 0}
    assert rows[1] == {'city': 'NY', 'A': 30, 'B': 5}


def test_pivot_missing_combinations_are_none(long_data):
    extra = LazyFrame([
        {'name': 'Alice', 'subject': 'math', 'score': 90},
        {'name': 'Bob', 'subject': 'english', 'score': 92},
    ])
    result = extra.pivot(
        index='name', on='subject', values='score',
        columns=['math', 'english'],
    ).collect()
    rows = sorted(result.to_pylist(), key=lambda r: r['name'])
    assert rows[0] == {'name': 'Alice', 'math': 90, 'english': None}
    assert rows[1] == {'name': 'Bob', 'math': None, 'english': 92}


def test_basic_unpivot(wide_data):
    result = wide_data.unpivot(
        id_columns='name',
        value_columns=['math', 'english'],
    ).collect()
    rows = sorted(result.to_pylist(), key=lambda r: (r['name'], r['variable']))
    assert len(rows) == 4
    assert rows[0] == {'name': 'Alice', 'variable': 'english', 'value': 85}
    assert rows[1] == {'name': 'Alice', 'variable': 'math', 'value': 90}


def test_unpivot_schema_is_lazy(wide_data):
    lf = wide_data.unpivot(id_columns='name', value_columns=['math', 'english'])
    schema = lf.schema
    assert schema.column_names == ['name', 'variable', 'value']
    assert schema['variable'].dtype is str
    assert schema['value'].dtype is int


def test_unpivot_default_value_columns(wide_data):
    result = wide_data.unpivot(id_columns='name').collect()
    rows = sorted(result.to_pylist(), key=lambda r: (r['name'], r['variable']))
    assert len(rows) == 4
    vars_found = {r['variable'] for r in rows}
    assert vars_found == {'math', 'english'}


def test_pivot_unpivot_roundtrip(long_data):
    pivoted = long_data.pivot(
        index='name', on='subject', values='score',
        columns=['math', 'english'],
    )
    roundtrip = pivoted.unpivot(
        id_columns='name',
        value_columns=['math', 'english'],
        variable_name='subject',
        value_name='score',
    ).collect()
    rows = sorted(roundtrip.to_pylist(), key=lambda r: (r['name'], r['subject']))
    orig = sorted(long_data.to_pylist(), key=lambda r: (r['name'], r['subject']))
    assert rows == orig


def test_pivot_explain(long_data):
    plan_str = long_data.pivot(
        index='name', on='subject', values='score',
        columns=['math', 'english'],
    ).explain()
    assert 'Pivot' in plan_str
    assert 'name' in plan_str
    assert 'subject' in plan_str


def test_unpivot_explain(wide_data):
    plan_str = wide_data.unpivot(
        id_columns='name', value_columns=['math', 'english'],
    ).explain()
    assert 'Unpivot' in plan_str
    assert 'name' in plan_str


def test_filter_pushdown_through_pivot(long_data):
    lf = long_data.pivot(
        index='name', on='subject', values='score',
        columns=['math', 'english'],
    ).filter(Col('name') == 'Alice')
    optimized_plan = lf.explain(optimized=True)
    lines = optimized_plan.strip().split('\n')
    pivot_line = None
    filter_line = None
    for i, line in enumerate(lines):
        if 'Pivot' in line:
            pivot_line = i
        if 'Filter' in line:
            filter_line = i
    assert pivot_line is not None and filter_line is not None
    assert filter_line > pivot_line


def test_filter_pushdown_through_unpivot(wide_data):
    lf = wide_data.unpivot(
        id_columns='name', value_columns=['math', 'english'],
    ).filter(Col('name') == 'Bob')
    optimized_plan = lf.explain(optimized=True)
    lines = optimized_plan.strip().split('\n')
    unpivot_line = None
    filter_line = None
    for i, line in enumerate(lines):
        if 'Unpivot' in line:
            unpivot_line = i
        if 'Filter' in line:
            filter_line = i
    assert unpivot_line is not None and filter_line is not None
    assert filter_line > unpivot_line


def test_multi_index_pivot():
    data = LazyFrame([
        {'region': 'East', 'year': 2023, 'product': 'A', 'sales': 10},
        {'region': 'East', 'year': 2023, 'product': 'B', 'sales': 20},
        {'region': 'East', 'year': 2024, 'product': 'A', 'sales': 30},
        {'region': 'West', 'year': 2023, 'product': 'A', 'sales': 40},
    ])
    result = data.pivot(
        index=['region', 'year'], on='product', values='sales',
        columns=['A', 'B'],
    ).collect()
    rows = sorted(result.to_pylist(), key=lambda r: (r['region'], r['year']))
    assert rows[0] == {'region': 'East', 'year': 2023, 'A': 10, 'B': 20}
    assert rows[1] == {'region': 'East', 'year': 2024, 'A': 30, 'B': None}
    assert rows[2] == {'region': 'West', 'year': 2023, 'A': 40, 'B': None}


def test_melt_is_alias_for_unpivot(wide_data):
    r1 = wide_data.unpivot(id_columns='name', value_columns=['math', 'english']).collect()
    r2 = wide_data.melt(id_columns='name', value_columns=['math', 'english']).collect()
    assert sorted(r1.to_pylist(), key=str) == sorted(r2.to_pylist(), key=str)
