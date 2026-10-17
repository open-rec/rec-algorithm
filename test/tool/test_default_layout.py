import ast

import numpy as np
import pandas as pd
import pytest

from algorithm.recall.query_embedding import QueryEmbeddingRecall
from tool.check_default_artifacts import REQUIRED
from tool.gen_recall_data import generate


@pytest.mark.parametrize("include_text", [False, True])
def test_complete_scene_scoped_recall_and_algorithm_specific_vectors(tmp_path, monkeypatch, include_text):
    class Encoder:
        def __init__(self):
            assert include_text, "CPU bootstrap must not instantiate the language encoder"
        def encode(self, texts, **kwargs):
            return np.tile(np.array([1., 0., 0.]), (len(texts), 1))
        def metadata(self):
            return {"dimension": 3, "model_name": "test-only"}
    monkeypatch.setattr("tool.gen_text_embedding_data.TextEmbeddingEncoder", Encoder)
    users = pd.DataFrame([{"id": f"u{i}", "tags": "shared", "city": "city"} for i in range(4)])
    items = pd.DataFrame([{"id": f"i{i}", "scene": scene, "category": "books",
                           "tags": "fiction", "title": "book", "pub_time": 1}
                          for scene in ("a", "b") for i in range(3)])
    events = pd.DataFrame([{"scene": scene, "user_id": f"u{u}", "item_id": f"i{i}",
                            "type": "click", "time": 2, "id": f"{scene}-{u}-{i}", "value": 1}
                           for scene, ids in (("a", (0, 1)), ("b", (2, 3)))
                           for u in ids for i in range(3)])
    for name, frame in (("user", users), ("item", items), ("event", events)):
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    output = tmp_path / "recall"
    generate(tmp_path / "item.csv", tmp_path / "event.csv", output,
             include_text_embeddings=include_text)
    expected = {name.removeprefix("recall/") for name in REQUIRED if name.startswith("recall/") and name.endswith(".csv")}
    if include_text:
        expected.add("item/item_title_emb.csv")
    assert {str(path.relative_to(output)) for path in output.rglob("*.csv")} == expected
    for target, entity, vector_file in (("item", "item", "item_seq_emb"),
                                         ("item", "item", "item_title_emb"),
                                         ("user", "user", "user_als_emb")):
        if vector_file == "item_title_emb" and not include_text:
            continue
        query = output / target / f"{vector_file}.csv"
        frame = pd.read_csv(query)
        for scene, rows in frame.groupby("scene"):
            vectors = {row[entity]: ast.literal_eval(row["vector"]) for _, row in rows.iterrows()}
            assert all(len(vector) == (3 if vector_file in ("item_content_emb", "item_title_emb") else 10) and np.isclose(np.linalg.norm(vector), 1)
                       for vector in vectors.values())
            if target == "user":
                assert set(vectors) == ({"u0", "u1"} if scene == "a" else {"u2", "u3"})
            first = next(iter(vectors))
            assert QueryEmbeddingRecall(list(vectors), list(vectors.values())).recall(vectors[first])
    assert not (output / "user/query_embedding.csv").exists()
    assert not list(output.rglob("query_qwen_emb.csv"))
    assert not (output / "item/query_embedding.csv").exists()
    assert not (output / "item/item_content_emb.csv").exists()
    assert not (output / "item/text_embedding.manifest.json").exists()
    for target, entity in (("item", "item"), ("user", "user")):
        assert not (output / target / "sparse.csv").exists()
        frame = pd.read_csv(output / target / "text_search.csv")
        assert list(frame.columns) == ["scene", entity, "text"]
        assert frame.text.notna().all()
    for name in ("user_cf_u2u", "content_u2u", "user_als_u2u"):
        frame = pd.read_csv(output / "user" / f"{name}.csv")
        assert not frame.empty
        assert (frame.left_user != frame.right_user).all()
    with pytest.raises(ValueError, match="embedding_dim"):
        generate(tmp_path / "item.csv", tmp_path / "event.csv", output, embedding_dim=33)
