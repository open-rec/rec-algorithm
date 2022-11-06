import csv

import pandas as pd

from algorithm.recall.hot import Hot
from algorithm.recall.i2i import ItemBasedI2I
from algorithm.recall.new import New


def gen_i2i_data(items, events, i2i_size, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'left_item', 'right_item', 'score'])
        scene_items_map = {scene: scene_items['id'].tolist() for scene, scene_items in items.groupby('scene')}
        for scene, scene_events in events.groupby('scene'):
            scene_i2i = ItemBasedI2I(events=scene_events, recall_size=i2i_size)
            for left_item in scene_items_map[scene]:
                recall_items = scene_i2i.recall(item_triggers=[left_item])
                for score_item in recall_items:
                    writer.writerow([scene, left_item, score_item.item, score_item.score])


def gen_embedding_data(items, events, embedding_size, filename):
    pass


def gen_hot_data(events, hot_size, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'item', 'score'])
        for scene, scene_events in events.groupby('scene'):
            scene_hot = Hot(events=scene_events, recall_size=hot_size)
            recall_items = scene_hot.recall()
            for score_item in recall_items:
                writer.writerow([scene, score_item.item, score_item.score])


def gen_new_data(items, new_size, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'item', 'score'])
        for scene, scene_items in items.groupby('scene'):
            scene_new = New(items=scene_items, recall_size=new_size)
            recall_items = scene_new.recall()
            for score_item in recall_items:
                writer.writerow([scene, score_item.item, score_item.score])


if __name__ == "__main__":
    users = pd.read_csv('../data/user.csv', header=0)
    items = pd.read_csv('../data/item.csv', header=0)
    events = pd.read_csv('../data/event.csv', header=0)

    gen_i2i_data(items, events, 50, '../data/i2i.csv')
    gen_embedding_data(items, events, 20, '../data/embedding.csv')
    gen_hot_data(events, 2000, '../data/hot.csv')
    gen_new_data(items, 2000, '../data/new.csv')
