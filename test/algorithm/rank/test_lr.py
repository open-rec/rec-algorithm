import pandas as pd

from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.user_feature import UserFeature
from algorithm.rank.lr import LRRecModel


def test_rank():
    items = pd.read_csv('../../../data/test/item.csv', header=0)
    users = pd.read_csv('../../../data/test/user.csv', header=0)
    events = pd.read_csv('../../../data/test/event.csv', header=0)

    user_feature = UserFeature(users=users, events=events)
    item_feature = ItemFeature(items=items, events=events)
    lr_model = LRRecModel(user_feature=user_feature, item_feature=item_feature, events=events)
    lr_model.train(epoch_num=5)
