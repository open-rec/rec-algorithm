from pathlib import Path


current_path = Path(__file__).resolve()


def root_path():
    return current_path.parents[2]


def model_path():
    return root_path() / "model/"

