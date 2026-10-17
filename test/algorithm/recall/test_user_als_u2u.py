import numpy as np
import pandas as pd
import pytest

from algorithm.recall.user_als_u2u import UserALSRecall


def test_als_shared_behaviour_neighbours_and_reproducibility():
    events = pd.DataFrame([
        ('u1', 'a'), ('u1', 'b'), ('u2', 'a'), ('u2', 'b'), ('u3', 'c'),
    ], columns=['user_id', 'item_id'])
    model = UserALSRecall(events, vector_size=3, neighbour_size=2)
    vectors = model.dump_vectors()
    repeated = UserALSRecall(events.sample(frac=1, random_state=42), vector_size=3).dump_vectors()
    assert vectors == repeated
    assert all(len(value) == 3 and np.isfinite(value).all() for value in vectors.values())
    assert model.dump_u2u()['u1'][0][0] == 'u2'
    rows = model.rows('scene')
    assert len(rows) == 6
    assert all(row['left_user'] != row['right_user'] for row in rows)
    assert set(rows[0]) == {'scene', 'left_user', 'right_user', 'score'}


def test_als_update_solves_weighted_implicit_objective():
    model = UserALSRecall(pd.DataFrame(), vector_size=2)
    factors = np.array([[1., 2.], [3., 1.], [0., 1.]])
    solved = model._solve(factors, [[0, 2]])[0]
    confidence = np.diag([2., 1., 2.])
    preference = np.array([1., 0., 1.])
    residual = (factors.T @ confidence @ factors + .2 * np.eye(2)) @ solved
    assert np.allclose(residual, factors.T @ confidence @ preference)


def test_als_empty_events_and_invalid_parameters():
    events = pd.DataFrame(columns=['user_id', 'item_id'])
    assert UserALSRecall(events).rows() == []
    with pytest.raises(ValueError):
        UserALSRecall(events, regularization=0)
