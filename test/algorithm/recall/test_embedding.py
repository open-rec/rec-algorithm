import pandas as pd

from algorithm.recall.embedding import EventEmbedding


def test_recall():
    test_embedding_size = 10
    events = pd.read_csv('../../../data/test/event.csv', header=0)
    for scene, scene_events in events.groupby('scene'):
        scene_embedding = EventEmbedding(events=scene_events, recall_size=test_embedding_size)
        recall_items = scene_embedding.recall(item_triggers=['item_4081', 'item_7441'])
        assert len(recall_items) <= test_embedding_size
        for item in recall_items:
            print(item)
