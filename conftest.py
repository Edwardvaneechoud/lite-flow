from __future__ import annotations

import os
import shutil

import pytest

_TEST_DATA = os.path.join(os.path.dirname(__file__), "tests", "test_data")


@pytest.fixture(autouse=True)
def _doctest_env(request, tmp_path, doctest_namespace):
    if request.config.option.doctestmodules or "doctest" in request.keywords:
        import pyfloe as pf

        doctest_namespace["pf"] = pf
        doctest_namespace["LazyFrame"] = pf.LazyFrame
        doctest_namespace["TypedLazyFrame"] = pf.TypedLazyFrame
        doctest_namespace["col"] = pf.col
        doctest_namespace["lit"] = pf.lit
        doctest_namespace["when"] = pf.when
        doctest_namespace["rank"] = pf.rank
        doctest_namespace["dense_rank"] = pf.dense_rank
        doctest_namespace["row_number"] = pf.row_number
        doctest_namespace["Stream"] = pf.Stream
        doctest_namespace["from_iter"] = pf.from_iter
        doctest_namespace["from_chunks"] = pf.from_chunks
        doctest_namespace["Agg"] = pf.Agg
        doctest_namespace["Join"] = pf.Join
        doctest_namespace["DateTrunc"] = pf.DateTrunc

        for fname in os.listdir(_TEST_DATA):
            src = os.path.join(_TEST_DATA, fname)
            if os.path.isfile(src):
                shutil.copy2(src, tmp_path / fname)

        shutil.copy2(
            os.path.join(_TEST_DATA, "students.tsv"),
            tmp_path / "data.tsv",
        )

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table(
                {
                    "id": [1, 2, 3, 4, 5],
                    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                    "score": [95.5, 82.0, 78.3, 91.2, 88.7],
                }
            )
            pq.write_table(table, str(tmp_path / "data.parquet"))
        except ImportError:
            pass

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        yield
        os.chdir(old_cwd)
    else:
        yield
