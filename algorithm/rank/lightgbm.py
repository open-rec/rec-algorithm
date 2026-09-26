"""LightGBM ranker for point-in-time OpenRec feature matrices."""

from pathlib import Path

import numpy as np


class LightGBMBinaryModel(object):
    """Serializable binary classifier for independent recommendation events."""

    def __init__(self, **params):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError(
                "LightGBM support requires rec-algorithm[lightgbm]"
            ) from error
        defaults = {
            "objective": "binary",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": 2,
            "verbosity": -1,
        }
        defaults.update(params)
        self.model = LGBMClassifier(**defaults)
        self.booster = None

    def fit(self, features, labels, group_ids=None, validation=None):
        kwargs = {}
        if validation is not None:
            x_val, y_val, _ = validation
            kwargs.update(eval_set=[(x_val, y_val)], eval_metric=["auc", "binary_logloss"])
        self.model.fit(features, labels, **kwargs)
        self.booster = self.model.booster_
        return self

    def predict_proba(self, features):
        if self.booster is not None:
            return np.asarray(self.booster.predict(features), dtype=np.float64)
        return np.asarray(self.model.predict_proba(features)[:, 1], dtype=np.float64)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path, n_jobs=2):
        from lightgbm import Booster
        value = cls.__new__(cls)
        value.model = None
        value.booster = Booster(model_file=str(path))
        return value


class LightGBMRankModel(object):
    """A small serializable LambdaRank wrapper with deterministic defaults."""

    def __init__(self, **params):
        try:
            from lightgbm import LGBMRanker
        except ImportError as error:
            raise ImportError(
                "LightGBM support requires rec-algorithm[lightgbm]"
            ) from error
        self.early_stopping_rounds = int(params.pop("early_stopping_rounds", 50))
        self.eval_at = tuple(params.pop("eval_at", (1, 5, 10, 20)))
        defaults = {
            "objective": "lambdarank",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": 2,
            "verbosity": -1,
        }
        defaults.update(params)
        self.model = LGBMRanker(**defaults)
        self.booster = None

    @staticmethod
    def group_sizes(group_ids):
        values = np.asarray(group_ids)
        if not len(values):
            raise ValueError("ranking groups must not be empty")
        boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
        if len(set(values.tolist())) != len(boundaries) - 1:
            raise ValueError("ranking group rows must be contiguous")
        return np.diff(boundaries).tolist()

    def fit(self, features, labels, group_ids, validation=None):
        kwargs = {}
        if validation is not None:
            from lightgbm import early_stopping, log_evaluation
            x_val, y_val, val_groups = validation
            kwargs.update(
                eval_set=[(x_val, y_val)],
                eval_group=[self.group_sizes(val_groups)],
                eval_at=list(self.eval_at),
                callbacks=[early_stopping(self.early_stopping_rounds, verbose=False),
                           log_evaluation(period=0)],
            )
        self.model.fit(
            features, labels, group=self.group_sizes(group_ids), **kwargs
        )
        self.booster = self.model.booster_
        return self

    def predict_proba(self, features):
        predictor = self.booster if self.booster is not None else self.model
        score = np.asarray(predictor.predict(features), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(score, -40, 40)))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path, n_jobs=2):
        from lightgbm import Booster
        value = cls.__new__(cls)
        value.model = None
        value.booster = Booster(model_file=str(path))
        return value
