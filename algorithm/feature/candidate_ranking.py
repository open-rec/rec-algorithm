"""Model-neutral features for ranking candidates from multiple recall channels."""

from collections import Counter

import numpy as np


def _tokens(value):
    if value is None:
        return set()
    if isinstance(value, str):
        return set(value.lower().split())
    return set(str(item).lower() for item in value if item is not None)


class CandidateRankingFeatureBuilder(object):
    """Materialize request-candidate features without labels or domain fields."""

    BASE_FEATURE_NAMES = (
        "request.history_length_log", "request.position_log",
        "candidate.query_similarity", "candidate.query_token_overlap_log",
        "candidate.query_token_coverage", "candidate.history_similarity_last",
        "candidate.history_similarity_max", "candidate.history_similarity_mean",
        "candidate.history_similarity_recency_weighted",
        "candidate.entity_history_count_log", "candidate.entity_history_rate",
        "candidate.tag_history_count_log", "candidate.tag_history_rate",
        "candidate.popularity_log", "candidate.popularity_rank_pct",
        "candidate.age_days_log", "recall.channel_count", "recall.score_max",
        "recall.score_mean", "recall.best_rank_reciprocal", "recall.rrf",
        "recall.rank_std", "recall.agreement_top10", "recall.agreement_top50",
        "recall.score_minmax_sum", "recall.score_minmax_mnz",
        "recall.rank_product_score",
    )

    def __init__(self, item_ids, item_vectors, primary_entities=None, tags=None,
                 popularity=None, release_times=None, texts=None,
                 channel_names=(), rrf_k=60.0):
        self.item_ids = np.asarray(item_ids).astype(str)
        if len(set(self.item_ids.tolist())) != len(self.item_ids):
            raise ValueError("item ids must be unique")
        self.index = {value: position for position, value in enumerate(self.item_ids)}
        vectors = np.asarray(item_vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(self.item_ids):
            raise ValueError("item vectors must align with item ids")
        self.vectors = vectors / np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
        )
        size = len(self.item_ids)
        self.entities = np.asarray(
            primary_entities if primary_entities is not None else [""] * size
        ).astype(str)
        self.tags = [_tokens(value) for value in (
            tags if tags is not None else [()] * size
        )]
        self.text_tokens = [_tokens(value) for value in (
            texts if texts is not None else [()] * size
        )]
        raw_popularity = np.asarray(
            popularity if popularity is not None else np.zeros(size), dtype=np.float64
        )
        raw_popularity = np.nan_to_num(raw_popularity, nan=0.0, posinf=0.0, neginf=0.0)
        self.popularity = np.log1p(np.maximum(raw_popularity, 0.0))
        order = np.argsort(np.argsort(self.popularity, kind="stable"), kind="stable")
        self.popularity_rank = order.astype(np.float32) / max(size - 1, 1)
        self.release_days = (np.full(size, np.datetime64("NaT"), dtype="datetime64[D]")
                             if release_times is None else
                             np.asarray(release_times, dtype="datetime64[D]"))
        if len(self.release_days) != size:
            raise ValueError("release times must align with item ids")
        self.channel_names = tuple(sorted(set(str(value) for value in channel_names)))
        self.rrf_k = float(rrf_k)

    @property
    def feature_names(self):
        names = list(self.BASE_FEATURE_NAMES)
        for channel in self.channel_names:
            names.extend(("recall.%s.present" % channel,
                          "recall.%s.score" % channel,
                          "recall.%s.score_minmax" % channel,
                          "recall.%s.score_z" % channel,
                          "recall.%s.rank_reciprocal" % channel,
                          "recall.%s.rank_pct" % channel,
                          "recall.%s.in_top10" % channel))
        return names

    def transform(self, candidate_ids, query_vector=None, history_ids=(),
                  channel_results=None, request_position=0, request_time=None,
                  query_text=None):
        """Return valid IDs and features in the caller's candidate order."""
        positions = [self.index[str(value)] for value in candidate_ids
                     if str(value) in self.index]
        if not positions:
            return [], np.empty((0, len(self.feature_names)), dtype=np.float32)
        candidate_vectors = self.vectors[positions]
        if query_vector is None:
            query_similarity = np.zeros(len(positions), dtype=np.float32)
        else:
            query = np.asarray(query_vector, dtype=np.float32)
            query /= max(float(np.linalg.norm(query)), 1e-12)
            query_similarity = candidate_vectors @ query
        query_tokens = _tokens(query_text)
        token_overlap = np.asarray(
            [len(query_tokens & self.text_tokens[value]) for value in positions],
            dtype=np.float32)
        token_coverage = token_overlap / max(len(query_tokens), 1)
        history_positions = [self.index[str(value)] for value in history_ids
                             if str(value) in self.index]
        if history_positions:
            similarities = candidate_vectors @ self.vectors[history_positions].T
            history_last = similarities[:, -1]
            history_max = similarities.max(axis=1)
            history_mean = similarities.mean(axis=1)
            weights = np.arange(1, len(history_positions) + 1, dtype=np.float32)
            history_weighted = similarities @ (weights / weights.sum())
        else:
            history_last = history_max = history_mean = history_weighted = np.zeros(
                len(positions), dtype=np.float32)
        entity_counts = Counter(self.entities[value] for value in history_positions
                                if self.entities[value])
        tag_counts = Counter(tag for value in history_positions for tag in self.tags[value])
        entity_count = np.asarray(
            [entity_counts.get(self.entities[value], 0) for value in positions], dtype=np.float32)
        tag_count = np.asarray(
            [sum(tag_counts.get(tag, 0) for tag in self.tags[value]) for value in positions],
            dtype=np.float32)
        history_total, tag_total = max(len(history_positions), 1), max(sum(tag_counts.values()), 1)
        if request_time is None:
            age = np.zeros(len(positions), dtype=np.float32)
        else:
            missing = np.isnat(self.release_days[positions])
            age = (np.datetime64(request_time, "D") - self.release_days[positions]).astype(
                "timedelta64[D]").astype(np.float64)
            age[missing] = 0.0
            age = np.log1p(np.maximum(age, 0.0)).astype(np.float32)
        evidence = {}
        for channel in self.channel_names:
            entries = list((channel_results or {}).get(channel, ()))
            scores = np.asarray([float(score) for _, score in entries], dtype=np.float64)
            low = float(scores.min()) if len(scores) else 0.0
            span = max(float(scores.max() - low), 1e-12) if len(scores) else 1.0
            mean = float(scores.mean()) if len(scores) else 0.0
            std = max(float(scores.std()), 1e-12) if len(scores) else 1.0
            count = max(len(entries), 1)
            evidence[channel] = {
                str(item): (float(score), rank, (float(score) - low) / span,
                            (float(score) - mean) / std, rank / count)
                for rank, (item, score) in enumerate(entries, 1)
            }
        aggregate, per_channel = [], [[] for _ in self.channel_names]
        for position in positions:
            item = self.item_ids[position]
            found = [evidence[channel][item] for channel in self.channel_names
                     if item in evidence[channel]]
            scores, ranks = [x[0] for x in found], [x[1] for x in found]
            minmax = [x[2] for x in found]
            aggregate.append((len(found), max(scores) if scores else 0.0,
                              float(np.mean(scores)) if scores else 0.0,
                              1.0 / min(ranks) if ranks else 0.0,
                              sum(1.0 / (self.rrf_k + rank) for rank in ranks),
                              float(np.std(ranks)) if ranks else 0.0,
                              sum(rank <= 10 for rank in ranks),
                              sum(rank <= 50 for rank in ranks),
                              sum(minmax), sum(minmax) * len(minmax),
                              (float(np.exp(np.mean(np.log([1.0 / rank for rank in ranks]))))
                               if ranks else 0.0)))
            for target, channel in zip(per_channel, self.channel_names):
                score, rank, normalized, zscore, rank_pct = evidence[channel].get(
                    item, (0.0, 0, 0.0, 0.0, 0.0))
                target.append((float(rank > 0), score, normalized, zscore,
                               1.0 / rank if rank else 0.0, rank_pct,
                               float(0 < rank <= 10)))
        columns = [np.full(len(positions), np.log1p(len(history_positions))),
                   np.full(len(positions), np.log1p(max(int(request_position), 0))),
                   query_similarity, np.log1p(token_overlap), token_coverage,
                   history_last, history_max, history_mean, history_weighted,
                   np.log1p(entity_count), entity_count / history_total,
                   np.log1p(tag_count), tag_count / tag_total,
                   self.popularity[positions], self.popularity_rank[positions], age,
                   np.asarray(aggregate, dtype=np.float32)]
        columns.extend(np.asarray(value, dtype=np.float32) for value in per_channel)
        return ([self.item_ids[value] for value in positions],
                np.column_stack(columns).astype(np.float32))
