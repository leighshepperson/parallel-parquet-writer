import os
import random
from functools import partial
from pathlib import Path
from time import sleep

import polars as pl

from parallel_parquet_writer import parallel_write_parquet


def inner(i: int):
    sleep(random.randint(0, 3))
    if i == 3 or i == 12:
        raise Exception("Bad times")
    return pl.DataFrame(
        {
            "a": range(100 * i, 100 * (i + 1)),
            "b": [f"item {j}" for j in range(100 * i, 100 * (i + 1))],
            "c": [f"item {i}" for j in range(100 * i, 100 * (i + 1))],
        }
    ).lazy()


def generate_data(i):
    return partial(inner, i)


if __name__ == "__main__":
    items = [generate_data(i) for i in range(100)]
    output_path = Path("./output")
    output_path.mkdir(parents=True, exist_ok=True)

    chunk_size = 20
    max_workers = 4

    tmp_path = Path("./tmp")
    if tmp_path.exists() and tmp_path.is_dir():
        for f in tmp_path.glob("*"):
            os.remove(f)
    else:
        tmp_path.mkdir(parents=True, exist_ok=True)

    output_path = output_path / "final_output.parquet"
    parallel_write_parquet(items, chunk_size, output_path, tmp_path, max_workers)

    df = pl.read_parquet(output_path)

    print(df)
