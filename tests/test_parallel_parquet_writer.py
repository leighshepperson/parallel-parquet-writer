from functools import partial
from pathlib import Path

import pytest
import polars as pl

from parallel_parquet_writer import (
    parallel_write_parquet,
    ExecutionStrategyType,
)


@pytest.fixture
def base_tmp_dir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture
def output_dir(tmp_path) -> Path:
    return tmp_path


def create_lazy_frame(i: int) -> pl.LazyFrame:
    return pl.DataFrame({"data": [i]}).lazy()


def failing_lazy_frame() -> None:
    raise ValueError("Intentional Failure")


@pytest.mark.parametrize("chunk_size", [1, 2, 5])
@pytest.mark.parametrize("max_workers", [1, 2])
@pytest.mark.parametrize(
    "execution_strategy_type",
    [ExecutionStrategyType.PROCESS, ExecutionStrategyType.THREAD],
)
def test_givenItemsAndChunkSize_whenParallelWriteParquet_thenChunksProcessedAndOutputWritten(
    base_tmp_dir: Path,
    output_dir: Path,
    chunk_size: int,
    max_workers: int,
    execution_strategy_type: ExecutionStrategyType,
):
    items = [partial(create_lazy_frame, i) for i in range(10)]
    output_path = output_dir / "output.parquet"

    assert not output_path.exists()

    parallel_write_parquet(
        items,
        chunk_size,
        output_path,
        base_tmp_dir,
        max_workers,
        execution_strategy_type,
    )

    assert output_path.exists()

    expected = pl.DataFrame({"data": range(10)})

    result = pl.read_parquet(output_path)

    assert expected.frame_equal(result)


def test_givenFailingItem_whenParallelWriteParquet_thenOtherItemsStillProcessed(
    base_tmp_dir, output_dir
):
    items = [partial(create_lazy_frame, i) for i in range(4)] + [
        partial(failing_lazy_frame)
    ]
    chunk_size = 2
    output_path = output_dir / "output.parquet"

    parallel_write_parquet(items, chunk_size, output_path, base_tmp_dir)

    assert output_path.exists()

    result = pl.read_parquet(output_path)
    assert (
        result.shape[0] == 4
    ), "Output should contain 4 rows, one for each successful item."


@pytest.mark.parametrize("items", [[], [partial(create_lazy_frame, 0)]])
def test_givenEdgeCases_whenParallelWriteParquet_thenHandleGracefully(
    items, base_tmp_dir, output_dir
):
    chunk_size = 2
    output_path = output_dir / "output_edge.parquet"

    parallel_write_parquet(items, chunk_size, output_path, base_tmp_dir)

    if items:
        assert output_path.exists()
        result = pl.read_parquet(output_path)
        assert result.shape[0] == len(
            items
        ), "Output should contain the same number of rows as items."
    else:
        assert (
            not output_path.exists()
        ), "Output file should not be created for empty input."
