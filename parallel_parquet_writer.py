import logging
import shutil
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Callable
import polars as pl

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ExecutionStrategyType(Enum):
    PROCESS = "Process"
    THREAD = "Thread"


def parallel_write_parquet(
    items: List[Callable[[], pl.LazyFrame]],
    chunk_size: int,
    output_path: Path,
    base_tmp_path: Path,
    max_workers: int = 4,
    execution_strategy_type: ExecutionStrategyType = ExecutionStrategyType.PROCESS,
) -> None:
    if len(items) == 0:
        return

    execution_strategy = (
        process_map
        if execution_strategy_type == ExecutionStrategyType.PROCESS
        else thread_map
    )

    with logging_redirect_tqdm():
        with TemporaryDirectory(dir=base_tmp_path) as tmp_dir:
            tmp_path = Path(tmp_dir)

            chunks = [
                items[i : i + chunk_size] for i in range(0, len(items), chunk_size)
            ]
            chunk_output_paths = [
                tmp_path / f"chunk_{i}.parquet"
                for i in range(len(items) // chunk_size + 1)
            ]

            execution_strategy(
                _chunk_processor,
                chunks,
                range(len(chunk_output_paths)),
                [path for path in chunk_output_paths],
                max_workers=max_workers,
                chunksize=1,
            )

            _atomic_write_parquet(tmp_path, output_path)


def _atomic_write_parquet(
    tmp_path: Path, output_path: Path, temp_suffix: str = ".temp"
) -> None:
    temp_file_path = output_path.with_suffix(temp_suffix)

    try:
        pl.scan_parquet(tmp_path / "*.parquet").collect(streaming=True).write_parquet(
            temp_file_path
        )
        shutil.move(str(temp_file_path), str(output_path))
    except Exception as ex:
        LOG.exception(ex)
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()


def _chunk_processor(
    chunk: List[Callable[[], pl.LazyFrame]], chunk_index: int, output_path: Path
) -> None:
    total_items = len(chunk)
    successful_count = 0
    failed_count = 0
    lfs = []

    with logging_redirect_tqdm():
        with tqdm(
            total=total_items, desc=f"Processing Chunk {chunk_index} - Get Lazy Frames", unit="get_lazy_frame"
        ) as lf_pbar:
            for get_lazy_frame in chunk:
                try:
                    lf = get_lazy_frame()
                    lfs.append(lf)
                    successful_count += 1
                except Exception as e:
                    LOG.exception(f"Error creating lazy frame in chunk {chunk_index}: {e}")
                    failed_count += 1
                finally:
                    lf_pbar.update(1)
                    lf_pbar.set_description(
                        f"Processing Chunk {chunk_index} - Fetching Lazy Frames: Successful: {successful_count}, Failed: {failed_count}, Total: {total_items}"
                    )
        with tqdm(
                total=1, desc=f"Processing Chunk {chunk_index} - Writing Parquet", unit="parquet"
        ) as parquet_pbar:
            if lfs:
                output_lfs = pl.concat(lfs)
                output_lfs.sink_parquet(output_path)
                parquet_pbar.update()
