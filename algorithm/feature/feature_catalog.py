"""Global feature catalog and model-specific feature-set selection.

The catalog describes every feature OpenRec can produce.  A feature set selects the subset a
model family consumes.  Neither file is a fitted serving artifact: vocabularies and normalization
statistics remain in the model-specific ``*.features.json`` written beside each checkpoint.
"""

import json
from pathlib import Path


DEFINITION_ROOT = Path(__file__).resolve().parent / "definitions"
CATALOG_FILE = DEFINITION_ROOT / "feature.catalog.json"
FEATURE_SET_FILES = {
    "lr": DEFINITION_ROOT / "lr.feature-set.json",
    "fm": DEFINITION_ROOT / "fm.feature-set.json",
}


class FeatureCatalog(object):

    def __init__(self, payload):
        self.payload = payload
        self.version = int(payload["version"])
        self.features = dict(payload["features"])

    @classmethod
    def load(cls, path=CATALOG_FILE):
        with open(path) as stream:
            return cls(json.load(stream))

    def require(self, feature_id, entity=None):
        feature = self.features.get(feature_id)
        if feature is None:
            raise ValueError("unknown catalog feature: %s" % feature_id)
        if entity and feature.get("entity") != entity:
            raise ValueError("feature %s belongs to %s, not %s" % (
                feature_id, feature.get("entity"), entity))
        return feature


class ModelFeatureSet(object):

    def __init__(self, payload, catalog):
        self.payload = payload
        self.name = payload["name"]
        self.model_type = payload["model_type"]
        self.catalog_version = int(payload["catalog_version"])
        if self.catalog_version != catalog.version:
            raise ValueError("feature set catalog version does not match the loaded catalog")
        self.user = self._resolve(payload.get("user", []), "user", catalog)
        self.item = self._resolve(payload.get("item", []), "item", catalog)

    @staticmethod
    def _resolve(feature_ids, entity, catalog):
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("feature set contains duplicate %s features" % entity)
        return [(feature_id, catalog.require(feature_id, entity)) for feature_id in feature_ids]

    @classmethod
    def for_model(cls, model_type, catalog=None):
        normalized = str(model_type).strip().lower()
        path = FEATURE_SET_FILES.get(normalized)
        if path is None:
            raise ValueError("no feature set is defined for model type: %s" % normalized)
        catalog = catalog or FeatureCatalog.load()
        with open(path) as stream:
            feature_set = cls(json.load(stream), catalog)
        if feature_set.model_type != normalized:
            raise ValueError("feature set model type does not match %s" % normalized)
        return feature_set
