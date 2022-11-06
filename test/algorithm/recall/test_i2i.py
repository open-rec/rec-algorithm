import pandas as pd

from algorithm.recall.i2i import ItemBasedI2I


def test_item_based_i2i():
    test_i2i_size = 10
    test_triggers = ['item_4081', 'item_7441']
    events = pd.read_csv('../../../data/event.csv', header=0)
    for scene, scene_events in events.groupby('scene'):
        scene_i2i = ItemBasedI2I(events=scene_events, recall_size=test_i2i_size)
        recall_items = scene_i2i.recall(item_triggers=test_triggers)
        assert len(recall_items) <= test_i2i_size
        for item in recall_items:
            print(item)
