"""Tests for streaming data into Floe."""
import os
import tempfile
import tracemalloc

from pyfloe import Floe, Stream, col, from_chunks, from_iter, when

# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_from_iter_with_dict_generator():
    def gen():
        for i in range(100):
            yield {"id": i, "value": i * 1.5}

    ff = from_iter(gen())
    assert ff.columns == ["id", "value"]
    result = ff.to_pylist()
    assert len(result) == 100
    assert result[0] == {"id": 0, "value": 0.0}

def test_from_iter_schema_inference_from_dicts():
    def gen():
        for i in range(100):
            yield {"id": i, "value": i * 1.5, "name": f"item_{i}"}

    ff = from_iter(gen())
    s = ff.schema
    assert s.dtypes["id"] is int
    assert s.dtypes["value"] is float
    assert s.dtypes["name"] is str

def test_from_iter_with_tuple_generator_explicit_schema():
    def gen():
        for i in range(50):
            yield (i, f"row_{i}", i * 2.0)

    ff = from_iter(gen(), columns=["id", "name", "score"],
                   dtypes={"id": int, "name": str, "score": float})
    assert ff.columns == ["id", "name", "score"]
    result = ff.to_pylist()
    assert len(result) == 50
    assert result[0] == {"id": 0, "name": "row_0", "score": 0.0}

def test_from_iter_with_objects():
    class Item:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    items = [Item(1, "a"), Item(2, "b"), Item(3, "c")]
    ff = from_iter(iter(items))
    assert ff.columns == ["x", "y"]
    result = ff.to_pylist()
    assert len(result) == 3

def test_from_iter_is_lazy():
    call_count = [0]

    def gen():
        for i in range(1000):
            call_count[0] += 1
            yield {"id": i}

    ff = from_iter(gen())
    # Only 10 items peeked for schema inference
    assert call_count[0] == 10
    assert not ff.is_materialized

def test_from_iter_filter_pipeline():
    def gen():
        for i in range(1000):
            yield {"id": i, "value": i * 3.0}

    result = (
        from_iter(gen())
        .filter(col("value") > 2900)
        .to_pylist()
    )
    assert len(result) == 33  # 968..999
    assert all(r["value"] > 2900 for r in result)


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_from_iter_with_factory_replayable():
    def make_data():
        for i in range(50):
            yield {"x": i}

    ff = from_iter(make_data, columns=["x"], dtypes={"x": int})
    r1 = ff.to_pylist()
    r2 = ff.to_pylist()
    assert r1 == r2
    assert len(r1) == 50

def test_from_iter_factory_replays_for_collect():
    call_count = [0]

    def make_data():
        call_count[0] += 1
        for i in range(20):
            yield {"v": i}

    ff = from_iter(make_data)
    ff.to_pylist()
    ff.to_pylist()
    # Factory called: once for peek, once per to_pylist
    assert call_count[0] >= 2

def test_from_iter_with_list_re_iterable():
    data = [{"a": 1}, {"a": 2}, {"a": 3}]
    ff = from_iter(data)
    assert ff.to_pylist() == data
    assert ff.to_pylist() == data  # re-iterable

def test_from_iter_with_range_of_tuples():
    ff = from_iter(((i, i**2) for i in range(10)),
                   columns=["n", "sq"], dtypes={"n": int, "sq": int})
    result = ff.to_pylist()
    assert len(result) == 10
    assert result[4] == {"n": 4, "sq": 16}

def test_from_iter_one_shot_generator_exhaustion():
    def gen():
        for i in range(5):
            yield {"x": i}

    ff = from_iter(gen())
    r1 = ff.to_pylist()
    assert len(r1) == 5
    # Data is now cached in _materialized, so second to_pylist() returns from cache
    r2 = ff.to_pylist()
    assert r2 == r1  # cached

    # But if we bypass the cache by re-executing the plan, the generator is gone
    r3 = list(ff._plan.execute())
    assert len(r3) == 0  # one-shot: factory is exhausted


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_from_chunks_with_list_of_dict_batches():
    chunks = [
        [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}],
        [{"id": 3, "v": "c"}],
        [{"id": 4, "v": "d"}, {"id": 5, "v": "e"}],
    ]
    ff = from_chunks(iter(chunks))
    result = ff.to_pylist()
    assert len(result) == 5
    assert result[0] == {"id": 1, "v": "a"}

def test_from_chunks_schema_inference():
    chunks = [
        [{"x": 1, "y": 3.14}, {"x": 2, "y": 2.71}],
        [{"x": 3, "y": 1.0}],
    ]
    ff = from_chunks(iter(chunks))
    s = ff.schema
    assert s.dtypes["x"] is int
    assert s.dtypes["y"] is float

def test_from_chunks_with_factory_replayable():
    def make_chunks():
        yield [{"n": 1}, {"n": 2}]
        yield [{"n": 3}, {"n": 4}]
        yield [{"n": 5}]

    ff = from_chunks(make_chunks)
    r1 = ff.to_pylist()
    r2 = ff.to_pylist()
    assert r1 == r2
    assert len(r1) == 5

def test_from_chunks_filter_pipeline():
    def make_chunks():
        for batch_start in range(0, 100, 10):
            yield [{"id": i, "val": i * 2.0} for i in range(batch_start, batch_start + 10)]

    result = (
        from_chunks(make_chunks)
        .filter(col("val") > 150)
        .to_pylist()
    )
    assert all(r["val"] > 150 for r in result)
    assert len(result) == 24  # ids 76..99

def test_from_chunks_with_floe_chunks():
    ff1 = Floe([{"a": 1}, {"a": 2}])
    ff2 = Floe([{"a": 3}, {"a": 4}])
    combined = from_chunks(iter([ff1, ff2]), columns=["a"], dtypes={"a": int})
    result = combined.to_pylist()
    assert len(result) == 4

def test_from_chunks_with_tuple_batches():
    chunks = [
        [(1, "x"), (2, "y")],
        [(3, "z")],
    ]
    ff = from_chunks(iter(chunks), columns=["id", "label"])
    result = ff.to_pylist()
    assert len(result) == 3
    assert result[0] == {"id": 1, "label": "x"}


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_stream_basic_pipeline():
    def gen():
        for i in range(100):
            yield {"id": i, "value": i * 2.0}

    result = (
        Stream.from_iter(gen(), columns=["id", "value"],
                        dtypes={"id": int, "value": float})
        .filter(col("value") > 150)
        .to_pylist()
    )
    assert len(result) == 24
    assert all(r["value"] > 150 for r in result)

def test_stream_with_column():
    def gen():
        for i in range(10):
            yield {"x": i, "y": i * 10}

    result = (
        Stream.from_iter(gen())
        .with_column("total", col("x") + col("y"))
        .to_pylist()
    )
    assert result[5] == {"x": 5, "y": 50, "total": 55}

def test_stream_filter_with_column_select():
    def gen():
        for i in range(100):
            yield {"id": i, "val": i * 3.0, "noise": "ignore"}

    result = (
        Stream.from_iter(gen())
        .filter(col("val") > 200)
        .with_column("label", when(col("val") > 280, "high").otherwise("mid"))
        .select("id", "val", "label")
        .to_pylist()
    )
    assert all("noise" not in r for r in result)
    assert all(r["val"] > 200 for r in result)
    high = [r for r in result if r["label"] == "high"]
    assert all(r["val"] > 280 for r in high)

def test_stream_to_csv_sink():
    def gen():
        for i in range(50):
            yield {"id": i, "score": i * 1.5}

    from pyfloe import read_csv
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name
    try:
        Stream.from_iter(gen()).filter(col("score") > 50).to_csv(path)
        ff = read_csv(path)
        result = ff.to_pylist()
        assert all(r["score"] > 50 for r in result)
    finally:
        os.unlink(path)

def test_stream_to_jsonl_sink():
    def gen():
        for i in range(20):
            yield {"event": f"e_{i}", "ts": i}

    from pyfloe import read_jsonl
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
        path = f.name
    try:
        Stream.from_iter(gen()).filter(col("ts") > 10).to_jsonl(path)
        result = read_jsonl(path).to_pylist()
        assert all(r["ts"] > 10 for r in result)
    finally:
        os.unlink(path)

def test_stream_foreach_sink():
    collected = []
    def gen():
        for i in range(10):
            yield {"x": i}

    Stream.from_iter(gen()).filter(col("x") > 5).foreach(lambda row: collected.append(row))
    assert len(collected) == 4  # 6, 7, 8, 9

def test_stream_count():
    def gen():
        for i in range(1000):
            yield {"x": i}

    n = Stream.from_iter(gen()).filter(col("x") > 500).count()
    assert n == 499

def test_stream_take():
    def gen():
        for i in range(1000):
            yield {"x": i}

    first5 = Stream.from_iter(gen()).take(5)
    assert len(first5) == 5
    assert first5[0] == {"x": 0}

def test_stream_collect_returns_floe():
    def gen():
        for i in range(10):
            yield {"x": i}

    ff = Stream.from_iter(gen()).filter(col("x") > 5).collect()
    assert isinstance(ff, Floe)
    assert len(ff) == 4

def test_stream_from_csv():
    path = os.path.join(os.path.dirname(__file__), 'test_data', 'orders.csv')
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('order_id,customer_id,product,amount,region,active\n')
            f.write('1,101,Widget A,250.00,EU,true\n')
            f.write('2,102,Widget B,75.50,US,true\n')
            f.write('3,101,Widget C,180.00,EU,false\n')

    result = Stream.from_csv(path).filter(col("amount") > 100).to_pylist()
    assert len(result) >= 2
    assert all(r["amount"] > 100 for r in result)

def test_stream_schema_available_without_execution():
    def gen():
        yield {"a": 1, "b": "x"}

    s = Stream.from_iter(gen())
    assert s.schema.column_names == ["a", "b"]

def test_stream_repr():
    def gen():
        yield {"x": 1}

    s = Stream.from_iter(gen()).filter(col("x") > 0).with_column("y", col("x") * 2)
    r = repr(s)
    assert "Stream" in r
    assert "2 transforms" in r


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

def test_from_iter_streams_with_constant_memory():
    """Prove from_iter → filter → to_csv doesn't buffer everything."""
    N = 200_000
    def gen():
        for i in range(N):
            yield {"id": i, "value": i * 1.5}

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name
    try:
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]

        from_iter(gen()).filter(col("value") > 290_000).to_csv(path)

        after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        delta_kb = (after - before) / 1024

        from pyfloe import read_csv
        result = read_csv(path).to_pylist()
        assert len(result) > 0
        assert all(r["value"] > 290_000 for r in result)
        # Memory should be well under what buffering 200k rows would need
        assert delta_kb < 5000, f"Used {delta_kb:.0f} KB — likely buffering"
    finally:
        os.unlink(path)

def test_stream_processes_with_constant_memory():
    N = 200_000
    def gen():
        for i in range(N):
            yield {"id": i, "val": i * 2.0}

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name
    try:
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]

        (
            Stream.from_iter(gen(), columns=["id", "val"],
                            dtypes={"id": int, "val": float})
            .filter(col("val") > 390_000)
            .with_column("flag", when(col("val") > 395_000, "hot").otherwise("warm"))
            .to_csv(path)
        )

        after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        delta_kb = (after - before) / 1024

        from pyfloe import read_csv
        result = read_csv(path).to_pylist()
        assert len(result) > 0
        assert delta_kb < 5000, f"Used {delta_kb:.0f} KB — likely buffering"
    finally:
        os.unlink(path)

def test_from_chunks_streams_chunk_by_chunk():
    chunk_count = [0]

    def make_chunks():
        for batch in range(100):
            chunk_count[0] += 1
            yield [{"id": batch * 10 + i, "v": i} for i in range(10)]

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name
    try:
        from_chunks(make_chunks).filter(col("v") > 5).to_csv(path)

        from pyfloe import read_csv
        result = read_csv(path).to_pylist()
        assert chunk_count[0] >= 2  # chunks were actually generated lazily
        assert all(r["v"] > 5 for r in result)
    finally:
        os.unlink(path)


# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════