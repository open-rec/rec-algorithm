import pandas as pd

from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.user_feature import UserFeature
from algorithm.rank.lr import LRRecModel

items = pd.read_csv('../../../data/test/item.csv', header=0)
users = pd.read_csv('../../../data/test/user.csv', header=0)
events = pd.read_csv('../../../data/test/event.csv', header=0)


def test_train():
    user_feature = UserFeature(users=users, events=events)
    item_feature = ItemFeature(items=items, events=events)
    lr_model = LRRecModel(user_feature=user_feature, item_feature=item_feature, events=events)
    lr_model.train(epoch_num=3, batch_size=20)
    lr_model.save()


def test_inference():
    user_feature = UserFeature(users=users, events=events)
    item_feature = ItemFeature(items=items, events=events)
    lr_model = LRRecModel(user_feature=user_feature, item_feature=item_feature, events=events)
    lr_model.load()

    test_user = users["id"][0]
    test_item_size = 5
    test_items = []
    for i in range(test_item_size):
        test_items.append(items["id"][i])
    lr_model.score(test_user, test_items)
