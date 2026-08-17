import os
from pathlib import Path

current_path = Path(__file__).resolve()

MODEL_HOME_ENV = "OPENREC_MODEL_HOME"

RANK_DIR = "rank"
FEATURE_DIR = "feature"

# the namespace trained artifacts are filed under, keeping them clear of the pre-trained Douban
# checkpoint that sits at the root of model/rank
DEFAULT_SCENE = "default"


def root_path():
    return current_path.parents[2]


def model_path():
    """
    Legacy per-checkout model directory (`rec-algorithm/model`). Kept for callers that want a local
    scratch location; trained artifacts now go to `model_home()` instead.
    """
    path = root_path() / "model"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_home():
    """
    The shared model store — the sibling `model/` repo, holding rank checkpoints and feature spaces
    so they survive across runs instead of being regenerated.

    `OPENREC_MODEL_HOME` overrides it, which is what you want when `rec-algorithm` is installed as a
    wheel (as `rank-engine` does): `__file__` then points into site-packages and the sibling repo is
    nowhere near it. Falls back to the per-checkout directory when no shared repo is present.
    """
    override = os.environ.get(MODEL_HOME_ENV)
    if override:
        return Path(override)
    shared = root_path().parent / "model"
    if (shared / RANK_DIR).is_dir():
        return shared
    return root_path() / "model"


def rank_model_path(scene=DEFAULT_SCENE):
    """`model/rank/{scene}`, created on demand."""
    path = model_home() / RANK_DIR / scene
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_path(scene=DEFAULT_SCENE):
    """`model/feature/{scene}`, created on demand."""
    path = model_home() / FEATURE_DIR / scene
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_file_candidates(model_file):
    """
    Where the feature space belonging to `model_file` might live, best guess first.

    Checkpoints and feature spaces are stored in parallel trees (`model/rank/{scene}/lr.pth` beside
    `model/feature/{scene}/lr.features.json`), so the sidecar is not simply next to the checkpoint.
    Both layouts are offered: same-directory first, then the mirrored `rank/` -> `feature/` path.
    """
    model_file = Path(model_file)
    sidecar = model_file.with_suffix(".features.json")
    candidates = [sidecar]

    parts = list(model_file.parts)
    if RANK_DIR in parts:
        # rightmost `rank` component, so a path that happens to contain the word earlier is safe
        index = len(parts) - 1 - parts[::-1].index(RANK_DIR)
        candidates.append(Path(*parts[:index], FEATURE_DIR, *parts[index + 1:-1], sidecar.name))
    return candidates


def resolve_feature_file(model_file):
    """The first existing candidate from `feature_file_candidates`, or None."""
    for candidate in feature_file_candidates(model_file):
        if candidate.exists():
            return candidate
    return None
