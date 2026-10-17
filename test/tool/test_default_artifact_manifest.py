import json

import pytest

from tool import check_default_artifacts as checker
from tool.build_default_artifacts import copy_optional_recall


def test_cpu_bundle_and_optional_vectors_are_both_checksummed(tmp_path):
    data, model = tmp_path / "data", tmp_path / "model"
    data.mkdir()
    model.mkdir()
    for name in ("user.csv", "item.csv", "event.csv"):
        (data / name).write_text("fixture")
    for name in checker.REQUIRED:
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact")
    catalog = checker.FeatureCatalog.load()
    manifest = {"schema_version": checker.SCHEMA_VERSION,
                "build_version": checker.BUILD_VERSION,
                "inputs": {name: checker.sha256(data / name)
                           for name in ("user.csv", "item.csv", "event.csv")},
                "catalog_version": catalog.version, "catalog_sha256": catalog.sha256,
                "outputs": {name: checker.sha256(model / name) for name in checker.REQUIRED}}
    manifest_path = model / "default.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert checker.valid(data, model)
    optional = checker.OPTIONAL[0]
    (model / optional).write_text("title vectors")
    manifest["outputs"][optional] = checker.sha256(model / optional)
    manifest_path.write_text(json.dumps(manifest))
    assert checker.valid(data, model)
    (model / optional).write_text("corrupted")
    assert not checker.valid(data, model)
    (model / optional).unlink()
    assert not checker.valid(data, model)


@pytest.mark.parametrize("changed", [None, "event", "item", "vector"])
def test_preserve_only_verified_title_vectors(tmp_path, changed):
    model, stage = tmp_path / "model", tmp_path / "stage"
    name = checker.OPTIONAL[0]
    source = model / name
    source.parent.mkdir(parents=True)
    source.write_text("title vector fixture")
    manifest = {"build_version": 18, "inputs": {"item.csv": "items", "event.csv": "events"},
                "outputs": {name: checker.sha256(source)}}
    (model / "default.manifest.json").write_text(json.dumps(manifest))
    inputs = dict(manifest["inputs"])
    if changed in ("item", "event"):
        inputs[changed + ".csv"] = "changed"
    if changed == "vector":
        source.write_text("corrupt")
    copied = copy_optional_recall(model, stage, inputs)
    if changed in (None, "event"):
        assert copied == [name]
        assert (stage / name).read_bytes() == source.read_bytes()
    else:
        assert copied == []
        assert not (stage / name).exists()
