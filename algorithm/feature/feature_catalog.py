"""Global feature catalog and model-specific feature-set selection.

The catalog describes every feature OpenRec can produce.  A feature set selects
the subset a
model family consumes.  Neither file is a fitted serving artifact: vocabularies
and normalization
statistics remain in the model-specific ``*.features.json`` written beside each
checkpoint.
"""

import json
import pkgutil
import hashlib
from pathlib import Path


DEFINITION_ROOT = Path(__file__).resolve().parent / "definitions"
CATALOG_FILE = DEFINITION_ROOT / "feature.catalog.json"
FEATURE_SET_FILES = {
    "lr": DEFINITION_ROOT / "lr.feature-set.json",
    "fm": DEFINITION_ROOT / "fm.feature-set.json",
}


def _read_definition(path):
    path = Path(path)
    if path.parent == DEFINITION_ROOT:
        raw = pkgutil.get_data("algorithm.feature.definitions", path.name)
        if raw is None:
            raise FileNotFoundError(str(path))
        return raw
    return path.read_bytes()


class FeatureCatalog(object):
    def __init__(self, payload):
        self.payload = payload
        self.version = int(payload["catalog_version"])
        self.features = {
            item["id"]: self._runtime_definition(item)
            for item in payload["features"]
        }
        self.sha256 = None

    @staticmethod
    def _runtime_definition(item):
        kind = {
            "categorical": "id",
            "multi_value": "multi",
            "hashed_text": "hash",
        }.get(item["shape"])
        if kind is None:
            kind = "bool" if item["value_type"] == "boolean" else "num"
        return dict(item, column=item["name"], kind=kind)

    @classmethod
    def load(cls, path=CATALOG_FILE):
        raw = _read_definition(path)
        catalog = cls(json.loads(raw.decode("utf-8")))
        catalog.sha256 = hashlib.sha256(raw).hexdigest()
        return catalog

    def require(self, feature_id, entity=None):
        feature = self.features.get(feature_id)
        if feature is None:
            raise ValueError("unknown catalog feature: %s" % feature_id)
        if entity and feature.get("entity") != entity:
            raise ValueError(
                "feature %s belongs to %s, not %s"
                % (feature_id, feature.get("entity"), entity)
            )
        return feature

    def fingerprint(self, feature_id):
        definition = dict(self.require(feature_id))
        definition.pop("description", None)
        policy = self.payload.get("policies", {}).get(definition.get("policy"))
        raw = json.dumps(
            {"definition": definition, "policy": policy},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(raw).hexdigest()


class ModelFeatureSet(object):
    def __init__(self, payload, catalog):
        self.payload = payload
        self.name = payload["name"]
        self.model_type = payload["model_type"]
        self.catalog_version = int(payload["catalog_version"])
        if self.catalog_version != catalog.version:
            raise ValueError(
                "feature set catalog version does not match the loaded catalog"
            )
        self.catalog_sha256 = catalog.sha256
        self.user = self._resolve(payload.get("user", []), "user", catalog)
        self.item = self._resolve(payload.get("item", []), "item", catalog)

    @staticmethod
    def _resolve(feature_ids, entity, catalog):
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError(
                "feature set contains duplicate %s features" % entity
            )
        return [
            (feature_id, catalog.require(feature_id, entity))
            for feature_id in feature_ids
        ]

    @classmethod
    def for_model(cls, model_type, catalog=None):
        normalized = str(model_type).strip().lower()
        path = FEATURE_SET_FILES.get(normalized)
        if path is None:
            raise ValueError(
                "no feature set is defined for model type: %s" % normalized
            )
        catalog = catalog or FeatureCatalog.load()
        feature_set = cls(json.loads(_read_definition(path)), catalog)
        if feature_set.model_type != normalized:
            raise ValueError(
                "feature set model type does not match %s" % normalized
            )
        return feature_set


def select_features(model_type, target_type="item", selection=None):
    """Resolve an ordered, explicitly supported source/candidate selection."""
    supported = ModelFeatureSet.for_model(model_type)
    if target_type not in ("item", "user"):
        raise ValueError("unsupported target_type")
    roles = {
        "user": supported.user,
        "candidate": supported.item
        if target_type == "item"
        else supported.user,
    }
    if selection is not None and (
        not isinstance(selection, dict) or set(selection) != set(roles)
    ):
        raise ValueError("feature_selection must contain user and candidate")
    result = {}
    for role, definitions in roles.items():
        allowed = dict(definitions)
        ids = list(allowed) if selection is None else selection[role]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(value, str) for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                "%s features must be a unique nonempty list" % role
            )
        for feature_id in ids:
            if feature_id not in allowed:
                raise ValueError(
                    "unsupported %s feature: %s" % (role, feature_id)
                )
            definition = allowed[feature_id]
            capability = definition.get("materialization", {})
            if (
                definition.get("status") != "stable"
                or not capability.get("online")
                or not capability.get("offline")
            ):
                raise ValueError(
                    "feature is not implemented online and offline: %s"
                    % feature_id
                )
        result[role] = list(ids)
    return result


def feature_catalog():
    """Expose declared capabilities, not a claim about live data readiness."""
    catalog = FeatureCatalog.load()
    models = {}
    for model_type in FEATURE_SET_FILES:
        models[model_type] = {
            target: select_features(model_type, target)
            for target in ("item", "user")
        }
    return {
        "catalog_version": catalog.version,
        "catalog_sha256": catalog.sha256,
        "features": catalog.payload["features"],
        "models": models,
        "availability": "declared_capability",
    }
