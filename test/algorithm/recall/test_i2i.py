import pandas as pd

from algorithm.recall.i2i import ItemBasedI2I


def test_item_based_i2i():
    test_i2i_size = 10
    test_triggers = ['id_7491', 'id_1321']
    behaviors = pd.read_csv('../../../data/behavior.csv', header=0)
    i2i = ItemBasedI2I(behaviors=behaviors, recall_size=test_i2i_size)
    recall_items = i2i.recall(test_triggers)
    assert len(recall_items) == test_i2i_size
    for item in recall_items:
        print(item)
