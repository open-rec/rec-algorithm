import pandas as pd

from algorithm.recall.hot import Hot


def test_recall():
    test_hot_size = 100
    events = pd.read_csv('../../../data/test/event.csv', header=0)
    for scene, scene_events in events.groupby('scene'):
        scene_hot = Hot(events=scene_events, recall_size=test_hot_size)
        recall_items = scene_hot.recall()
        assert len(recall_items) <= test_hot_size
        for item in recall_items:
            print(item)
