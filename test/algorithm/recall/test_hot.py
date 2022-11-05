import pandas as pd

from algorithm.recall.hot import Hot


def test_recall():
    test_hot_size = 100
    events = pd.read_csv('../../../data/event.csv', header=0)
    hot = Hot(events=events, recall_size=test_hot_size)
    recall_items = hot.recall()
    assert len(recall_items) == test_hot_size
    for item in recall_items:
        print(item)
