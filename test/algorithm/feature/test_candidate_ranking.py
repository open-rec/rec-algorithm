import numpy as np

from algorithm.feature.candidate_ranking import CandidateRankingFeatureBuilder


def test_features_preserve_order_and_combine_recall_evidence():
    builder = CandidateRankingFeatureBuilder(
        ["a", "b", "c"], [[1, 0], [0.8, 0.2], [0, 1]],
        primary_entities=["x", "x", "y"], tags=[["rock"], ["rock", "live"], ["jazz"]],
        popularity=[10, 5, 1], release_times=["2020-01-01", "2021-01-01", "2022-01-01"],
        texts=["alpha rock", "beta rock live", "gamma jazz"],
        channel_names=["embedding", "popular"])
    ids, features = builder.transform(
        ["c", "missing", "b"], query_vector=[1, 0], history_ids=["a"],
        channel_results={"embedding": [("b", 0.9), ("c", 0.2)],
                         "popular": [("a", 1.0), ("b", 0.5)]},
        request_position=2, request_time="2023-01-01", query_text="find beta live")
    assert ids == ["c", "b"]
    assert features.shape == (2, len(builder.feature_names))
    values = dict(zip(builder.feature_names, features[1]))
    assert values["recall.channel_count"] == 2
    assert values["recall.embedding.rank_reciprocal"] == 1
    assert values["recall.agreement_top10"] == 2
    assert values["recall.score_minmax_mnz"] >= values["recall.score_minmax_sum"]
    assert values["candidate.entity_history_rate"] == 1
    assert values["candidate.query_token_overlap_log"] > 0


def test_features_allow_missing_optional_inputs():
    builder = CandidateRankingFeatureBuilder(["a"], [[3, 4]])
    _, features = builder.transform(["a"])
    assert features.shape == (1, len(builder.BASE_FEATURE_NAMES))
    assert np.isfinite(features).all()
