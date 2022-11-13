import csv

import pandas as pd

from algorithm.recall.embedding import EventEmbedding
from algorithm.recall.hot import Hot
from algorithm.recall.i2i import ItemBasedI2I
from algorithm.recall.new import New


def gen_i2i_data(items, events, i2i_size, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'left_item', 'right_item', 'score'])
        #scene_items_map = {scene: scene_items['id'].tolist() for scene, scene_items in items.groupby('scene')}
        for scene, scene_events in events.groupby('scene'):
            scene_i2i = ItemBasedI2I(events=scene_events, recall_size=i2i_size)
            i2i_items = scene_i2i.dump_i2i(i2i_size)
            for left_item in i2i_items:
                for relate_item, score in i2i_items[left_item]:
                    writer.writerow([scene, left_item, relate_item, score])



def gen_embedding_data(items, events, embedding_dim, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'item', 'vector'])
        for scene, scene_events in events.groupby('scene'):
            scene_embedding = EventEmbedding(events=scene_events)
            item_vectors = scene_embedding.dump_vectors(embedding_dim)
            for item, vector in item_vectors:
                writer.writerow([scene, item, vector])


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


def gen_scene_recall(scene='test'):
    users = pd.read_csv('../data/%s/user.csv' % scene, header=0)
    items = pd.read_csv('../data/%s/item.csv' % scene, header=0)
    events = pd.read_csv('../data/%s/event.csv' % scene, header=0)

    events = events[events['type'] == 'click']

    gen_i2i_data(items, events, 20, '../data/%s/recall/i2i.csv' % scene)
    #gen_embedding_data(items, events, 10, '../data/%s/recall/embedding.csv' % scene)
    #gen_hot_data(events, 2000, '../data/%s/recall/hot.csv' % scene)
    #gen_new_data(items, 2000, '../data/%s/recall/new.csv' % scene)


if __name__ == "__main__":
    #gen_scene_recall('test')
    gen_scene_recall('douban')
