"""
The canonical model feature space: which columns feed the model, in which order, over which
vocabulary.

`UserFeature`/`ItemFeature` fit an encoder on whatever frame they are handed, so the resulting
column layout is a function of *that frame*. Training fits on the CSVs while the rank engine fits on
whatever `user:*`/`item:*` keys happen to be in Redis, and the two only agree by luck: one extra city
shifts every column after it, and the trained weights land on a permuted feature space with no error
raised. `FeatureSpace` pins the layout down once, and is persisted next to the checkpoint so serving
reproduces training's encoding exactly rather than re-deriving it.

Fit offline, save alongside the model, load online:

    space = FeatureSpace()
    space.fit(users=users_df, items=items_df)
    space.save("model/lr.features.json")

    space = FeatureSpace.load("model/lr.features.json")   # rank-engine
    user_vectors, item_vectors = space.build_maps(users_df, items_df)
"""

import json
import re

import numpy as np
import pandas as pd
from algorithm.feature.event_feature import event_feature_columns
from algorithm.feature.feature_catalog import ModelFeatureSet

# column kinds
ID = "id"        # one-hot over the categories seen at fit time
NUM = "num"      # standardized scalar, missing values imputed to the fitted mean
BOOL = "bool"    # 0/1
MULTI = "multi"  # bag-of-tokens over a separator

# user `tags` used to be split on "," and item `tags`/`category` on "/", which meant the same
# concept was tokenized two different ways. Accept either so both layouts encode identically.
DEFAULT_MULTI_SEP = r"[,/]"

_TRUTHY = {"1", "true", "t", "yes", "y"}

SCHEMA_VERSION = 1


def _str_series(frame, name):
    """Missing column -> empty strings, so a frame short one field degrades instead of raising."""
    if name not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=object)
    return frame[name].map(
        lambda value: ",".join(str(item) for item in value)
        if isinstance(value, (list, tuple, set)) else value).fillna("").astype(str)


def _num_series(frame, name):
    """`errors="coerce"` turns junk (and Redis' JSON strings) into NaN rather than raising."""
    if name not in frame.columns:
        return pd.Series(np.full(len(frame), np.nan), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _bool_series(frame, name):
    if name not in frame.columns:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    raw = frame[name]
    if raw.dtype == bool:
        return raw.astype(float)
    numeric = pd.to_numeric(raw, errors="coerce")
    # strings such as "true"/"yes" coerce to NaN above; recover them before defaulting to 0
    text = raw.where(numeric.isna()).fillna("").astype(str).str.strip().str.lower()
    return np.where(numeric.notna(), numeric.fillna(0.0) != 0, text.isin(_TRUTHY)).astype(float)


class ColumnSpec(object):
    """One input column plus whatever was learned about it at fit time."""

    def __init__(self, name="", kind=ID, sep=DEFAULT_MULTI_SEP, feature_id=None):
        self.name = name
        self.kind = kind
        self.sep = sep
        self.feature_id = feature_id
        self.categories = []  # ID / MULTI vocabulary, sorted so the column order is deterministic
        self.mean = 0.0       # NUM
        self.scale = 1.0      # NUM

    @property
    def width(self):
        if self.kind in (NUM, BOOL):
            return 1
        return len(self.categories)

    def _tokenize(self, value):
        return [t for t in (part.strip() for part in re.split(self.sep, value)) if t]

    def fit(self, frame):
        if self.kind == ID:
            self.categories = sorted(set(_str_series(frame, self.name)))
        elif self.kind == MULTI:
            tokens = set()
            for value in _str_series(frame, self.name):
                tokens.update(self._tokenize(value))
            self.categories = sorted(tokens)
        elif self.kind == NUM:
            values = _num_series(frame, self.name)
            self.mean = float(values.mean()) if values.notna().any() else 0.0
            scale = float(values.std(ddof=0)) if values.notna().any() else 0.0
            # a constant column has zero variance; dividing by it would produce inf/NaN
            self.scale = scale if scale > 1e-12 else 1.0
        return self

    def transform(self, frame):
        rows = len(frame)
        if self.kind == NUM:
            values = _num_series(frame, self.name).fillna(self.mean)
            # Streaming entities can be far more active than the training population. Bound
            # standardized values to avoid uncontrolled FM quadratic extrapolation while keeping
            # ordinary observations unchanged.
            normalized = np.clip((values.values - self.mean) / self.scale, -3.0, 3.0)
            return normalized.reshape(rows, 1)
        if self.kind == BOOL:
            return np.asarray(_bool_series(frame, self.name)).reshape(rows, 1)

        out = np.zeros((rows, len(self.categories)), dtype=np.float64)
        if not self.categories:
            return out
        index = {category: i for i, category in enumerate(self.categories)}
        values = _str_series(frame, self.name)
        if self.kind == ID:
            # unknown categories stay all-zero instead of blowing up, the equivalent of
            # OneHotEncoder(handle_unknown="ignore")
            positions = values.map(index)
            known = positions.notna().values
            out[np.arange(rows)[known], positions[known].astype(int).values] = 1.0
        else:
            for row, value in enumerate(values):
                for token in self._tokenize(value):
                    position = index.get(token)
                    if position is not None:
                        out[row, position] += 1.0
        return out

    def to_dict(self):
        payload = {
            "name": self.name,
            "kind": self.kind,
            "sep": self.sep,
            "categories": self.categories,
            "mean": self.mean,
            "scale": self.scale,
        }
        if self.feature_id:
            payload["feature"] = self.feature_id
        return payload

    @classmethod
    def from_dict(cls, payload):
        spec = cls(name=payload["name"], kind=payload["kind"],
                   sep=payload.get("sep", DEFAULT_MULTI_SEP),
                   feature_id=payload.get("feature"))
        spec.categories = list(payload.get("categories", []))
        spec.mean = float(payload.get("mean", 0.0))
        spec.scale = float(payload.get("scale", 1.0)) or 1.0
        return spec


def _default_user_columns():
    # `id`/`device_id`/`name`/`phone` are deliberately absent: one-hotting them yields one column
    # per user, which is both useless to a linear model and too wide to build a tensor from.
    columns = [
        ColumnSpec("country", ID),
        ColumnSpec("city", ID),
        ColumnSpec("gender", BOOL),
        ColumnSpec("age", NUM),
        ColumnSpec("tags", MULTI),
    ]
    return columns + [ColumnSpec(name, NUM) for name in event_feature_columns("item")]


def _default_item_columns():
    # `title` and `tags` are omitted for the same reason: their vocabulary dwarfs the rest.
    columns = [
        ColumnSpec("category", MULTI),
        ColumnSpec("scene", ID),
        ColumnSpec("weight", NUM),
    ]
    return columns + [ColumnSpec(name, NUM) for name in event_feature_columns("user")]


class FeatureSpace(object):

    def __init__(self, user_columns=None, item_columns=None, catalog_version=None,
                 feature_set=None, model_type=None):
        self.user_columns = user_columns if user_columns is not None else _default_user_columns()
        self.item_columns = item_columns if item_columns is not None else _default_item_columns()
        self.catalog_version = catalog_version
        self.feature_set = feature_set
        self.model_type = model_type
        self.fitted = False

    @classmethod
    def for_model(cls, model_type):
        """Build an unfitted space from the model's catalog-backed feature-set declaration."""
        selected = ModelFeatureSet.for_model(model_type)

        def columns(items):
            return [ColumnSpec(name=definition["column"], kind=definition["kind"],
                               sep=definition.get("sep", DEFAULT_MULTI_SEP),
                               feature_id=feature_id)
                    for feature_id, definition in items]

        return cls(user_columns=columns(selected.user), item_columns=columns(selected.item),
                   catalog_version=selected.catalog_version, feature_set=selected.name,
                   model_type=selected.model_type)

    @property
    def user_width(self):
        return sum(column.width for column in self.user_columns)

    @property
    def item_width(self):
        return sum(column.width for column in self.item_columns)

    @property
    def dim(self):
        """The model's in_features — user vector and item vector concatenated."""
        return self.user_width + self.item_width

    def fit(self, users=None, items=None):
        for column in self.user_columns:
            column.fit(users)
        for column in self.item_columns:
            column.fit(items)
        self.fitted = True
        return self

    def _require_fitted(self):
        if not self.fitted:
            raise RuntimeError("FeatureSpace is not fitted yet; call fit() or load() first")

    def transform_users(self, users):
        self._require_fitted()
        return np.hstack([column.transform(users) for column in self.user_columns])

    def transform_items(self, items):
        self._require_fitted()
        return np.hstack([column.transform(items) for column in self.item_columns])

    def build_maps(self, users=None, items=None):
        """id -> encoded row, for both frames. The one place training and serving share."""
        return (
            self._build_map(users, self.transform_users(users)),
            self._build_map(items, self.transform_items(items)),
        )

    @staticmethod
    def _build_map(frame, encoded):
        if "id" not in frame.columns:
            raise KeyError("frame must carry an 'id' column to key the feature map by")
        # a duplicated id keeps the last row, matching the previous dict-comprehension behaviour
        return {raw_id: encoded[i] for i, raw_id in enumerate(frame["id"])}

    def to_dict(self):
        payload = {
            "version": SCHEMA_VERSION,
            "user": [column.to_dict() for column in self.user_columns],
            "item": [column.to_dict() for column in self.item_columns],
            "user_width": self.user_width,
            "item_width": self.item_width,
            "input_dim": self.dim,
        }
        if self.catalog_version is not None:
            payload["catalog_version"] = self.catalog_version
        if self.feature_set:
            payload["feature_set"] = self.feature_set
        if self.model_type:
            payload["model_type"] = self.model_type
        return payload

    @classmethod
    def from_dict(cls, payload):
        version = payload.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported feature space version {version}, expected {SCHEMA_VERSION}")
        space = cls(
            user_columns=[ColumnSpec.from_dict(item) for item in payload["user"]],
            item_columns=[ColumnSpec.from_dict(item) for item in payload["item"]],
            catalog_version=payload.get("catalog_version"),
            feature_set=payload.get("feature_set"),
            model_type=payload.get("model_type"),
        )
        space.fitted = True
        for key, actual in (("user_width", space.user_width), ("item_width", space.item_width),
                            ("input_dim", space.dim)):
            expected = payload.get(key)
            if expected is not None and int(expected) != actual:
                raise ValueError("feature space %s=%s, computed %s" % (key, expected, actual))
        return space

    def save(self, path):
        self._require_fitted()
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
        return str(path)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
