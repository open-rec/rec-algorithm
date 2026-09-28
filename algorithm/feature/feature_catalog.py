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
    "lightgbm": DEFINITION_ROOT / "lightgbm.feature-set.json",
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
        self.session = self._resolve(payload.get("session", []), "session", catalog)
        self.context = self._resolve(payload.get("context", []), "context", catalog)
        self.interaction = self._resolve(
            payload.get("interaction", []), "interaction", catalog
        )

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


def select_features(
    model_type, target_type="item", selection=None, families=None, scene=None
):
    """Resolve an ordered, explicitly supported source/candidate selection."""
    supported = ModelFeatureSet.for_model(model_type)
    if target_type not in ("item", "user"):
        raise ValueError("unsupported target_type")
    roles = {
        "user": supported.user,
        "candidate": supported.item
        if target_type == "item"
        else supported.user,
        "session": supported.session,
        "context": supported.context,
        "interaction": supported.interaction,
    }
    if selection is not None and not isinstance(selection, dict):
        raise ValueError("feature_selection must be an object")
    if selection is not None and not {"user", "candidate"}.issubset(selection):
        raise ValueError("feature_selection must contain user and candidate")
    if selection is not None and set(selection) - set(roles):
        raise ValueError("feature_selection contains an unknown role")
    result = {}
    for role, definitions in roles.items():
        allowed = dict(definitions)
        ids = list(allowed) if selection is None else selection.get(role, [])
        if (not isinstance(ids, list)
                or any(not isinstance(value, str) for value in ids)
                or len(set(ids)) != len(ids)
                or (role in ("user", "candidate") and not ids)):
            raise ValueError(
                "%s features must be a unique list; user and candidate cannot be empty" % role
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
        if families is not None:
            unknown = set(families) - set(supported.payload.get("families", []))
            if unknown:
                raise ValueError("unknown feature families: %s" % sorted(unknown))
            ids = [
                value for value in ids
                if allowed[value].get("family") in set(families)
            ]
        if scene is not None:
            ids = [
                value for value in ids
                if "all" in allowed[value].get("scenes", ["all"])
                or scene in allowed[value].get("scenes", [])
            ]
        if role in ("user", "candidate") and not ids:
            raise ValueError("%s features must be a unique nonempty list" % role)
        # Keep the established two-role gateway response byte-for-byte
        # compatible when callers submit a legacy selection. Optional roles
        # become first-class only when they actually select features.
        if ids or role in ("user", "candidate"):
            result[role] = list(ids)
    return result


# The distributed job currently exports user/candidate rows. Calendar context
# is recoverable from the label timestamp; other dynamic roles require sample
# files and/or request producers that this pipeline does not yet provide.
TRAINING_CONTEXT = {
    "context.request_hour_sin", "context.request_hour_cos",
    "context.request_weekday_sin", "context.request_weekday_cos",
}


def select_training_features(model_type, target_type="item", selection=None):
    supported = select_features(model_type, target_type)
    available = {
        role: ids if role in ("user", "candidate") else [
            feature_id for feature_id in ids if feature_id in TRAINING_CONTEXT
        ]
        for role, ids in supported.items()
    }
    if selection is None:
        # Preserve the established entity-only default; calendar context is opt-in.
        selection = {role: available[role] for role in ("user", "candidate")}
    resolved = select_features(model_type, target_type, selection)
    for role, ids in resolved.items():
        missing = set(ids) - set(available.get(role, []))
        if missing:
            raise ValueError(
                "features unavailable in cluster training: %s" % sorted(missing)
            )
    return resolved


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
        "taxonomy": catalog.payload.get("taxonomy", {}),
        "scene_presets": catalog.payload.get("scene_presets", {}),
        "models": models,
        "availability": "declared_capability",
        "training_models": {
            model: {
                target: {role: ids if role in ("user", "candidate") else [
                    feature_id for feature_id in ids if feature_id in TRAINING_CONTEXT
                ] for role, ids in roles.items()}
                for target, roles in targets.items()
            } for model, targets in models.items()
        },
    }
