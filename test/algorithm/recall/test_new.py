import pandas as pd

from algorithm.recall.new import New


def test_recall():
    test_new_size = 100
    items = pd.read_csv('../../../data/item.csv', header=0)
    for scene, scene_items in items.groupby('scene'):
        scene_new = New(items=scene_items, recall_size=test_new_size)
        recall_items = scene_new.recall()
        assert len(recall_items) <= test_new_size
        for item in recall_items:
            print(item)
