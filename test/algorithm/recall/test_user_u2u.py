import pandas as pd

from algorithm.recall.content_u2u import ContentBasedU2U
from algorithm.recall.user_cf_u2u import UserBasedU2U
from algorithm.recall.user_emb_u2u import UserEmbeddingU2U


def test_user_cf_u2u_excludes_self_and_orders_neighbours():
    events = pd.DataFrame([
        {'user_id': 'u1', 'item_id': 'a'}, {'user_id': 'u1', 'item_id': 'b'},
        {'user_id': 'u2', 'item_id': 'a'}, {'user_id': 'u2', 'item_id': 'b'},
        {'user_id': 'u3', 'item_id': 'a'},
    ])
    result = UserBasedU2U(events).dump_u2u()
    assert [row[0] for row in result['u1']] == ['u2', 'u3']
    assert all(right != left for left, rows in result.items() for right, _ in rows)


def test_content_u2u_uses_profile_fields_and_emits_serving_schema():
    users = pd.DataFrame([
        {'id': 'u1', 'city': 'hz', 'tags': 'music,film'},
        {'id': 'u2', 'city': 'hz', 'tags': 'music'},
        {'id': 'u3', 'city': 'sh', 'tags': 'sport'},
    ])
    recall = ContentBasedU2U(users)
    assert recall.dump_u2u()['u1'][0][0] == 'u2'
    assert set(recall.rows('home')[0]) == {'scene', 'left_user', 'right_user', 'score'}


def test_user_embedding_u2u_emits_vectors_and_neighbours():
    events = pd.DataFrame([
        {'user_id': 'u1', 'item_id': 'a'}, {'user_id': 'u1', 'item_id': 'b'},
        {'user_id': 'u2', 'item_id': 'a'}, {'user_id': 'u2', 'item_id': 'b'},
        {'user_id': 'u3', 'item_id': 'c'},
    ])
    recall = UserEmbeddingU2U(events, vector_size=2)
    assert recall.dump_u2u()['u1'][0][0] == 'u2'
    assert set(recall.vector_rows('home')[0]) == {'scene', 'id', 'vector'}
    assert set(recall.rows('home')[0]) == {'scene', 'left_user', 'right_user', 'score'}
