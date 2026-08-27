import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = Path(os.environ.get("OPENREC_TEST_DATA", PROJECT_ROOT / "data" / "test"))


def data_path(filename):
    return TEST_DATA / filename
