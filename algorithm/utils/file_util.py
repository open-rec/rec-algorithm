from pathlib import Path

current_path = Path(__file__).resolve()


def root_path():
    return current_path.parents[2]


def model_path():
    """
    Where checkpoints are written. The directory is gitignored, so it does not exist on a fresh
    clone — create it here rather than letting torch.save fail with FileNotFoundError.
    """
    path = root_path() / "model"
    path.mkdir(parents=True, exist_ok=True)
    return path
