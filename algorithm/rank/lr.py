import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.user_feature import UserFeature
from algorithm.rank.model import RecModel


class EventDataSet(Dataset):

    def __init__(self, user_feature: UserFeature = None, item_feature: ItemFeature = None, events: DataFrame = None):
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.events = events
        self.user_feature_map = None
        self.item_feature_map = None
        self.labels = None
        self.dim = 1
        self.preprocess()

    @property
    def feature_dim(self):
        return self.dim

    def preprocess(self):
        merge_events = self.events \
            .merge(self.user_feature.users, left_on="user_id", right_on="id") \
            .merge(self.item_feature.items, left_on="item_id", right_on="id")
        self.labels = merge_events[merge_events["type"].isin(["click"])]["value"].values
        user_features = torch.tensor(
            np.hstack([
                # self.user_feature.id,
                # self.user_feature.device_id,
                # self.user_feature.name,
                # self.user_feature.country,
                # self.user_feature.city,
                # self.user_feature.phone,
                # self.user_feature.id_features,
                self.user_feature.gender,
                self.user_feature.age,
                # self.user_feature.tags,
                # self.user_feature.register_time,
                # self.user_feature.login_time,
            ]),
            dtype=torch.float32
        )

        item_features = torch.tensor(
            np.hstack([
                # self.item_feature.id,
                # self.item_feature.title,
                # self.item_feature.category,
                # self.item_feature.tags,
                # self.item_feature.scene,
                # self.item_feature.pub_time,
                # self.item_feature.modify_time,
                # self.item_feature.expire_time,
                self.item_feature.status,
                self.item_feature.weight,
            ]),
            dtype=torch.float32
        )

        self.user_feature_map = {
            user_id: user_features[i]
            for i, user_id in enumerate(self.user_feature.raw_id)
        }

        self.item_feature_map = {
            item_id: item_features[i]
            for i, item_id in enumerate(self.item_feature.raw_id)
        }
        self.dim = item_features.dim() + user_features.dim()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        event = self.events.iloc[idx]
        user_feature = self.user_feature_map[event["user_id"]]
        item_feature = self.item_feature_map[event["item_id"]]
        label = torch.tensor(event["value"], dtype=torch.float32)
        return user_feature, item_feature, label

    def user_feature_by_id(self, user_id):
        return self.user_feature_map[user_id]

    def item_feature_by_id(self, item_id):
        return self.item_feature_map[item_id]


class LRModel(nn.Module):
    def __init__(self, dim=10):
        super().__init__()
        self.linear = nn.Linear(in_features=dim, out_features=1)

    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))
        return pred


class LRRecModel(RecModel):

    def __init__(self, user_feature=None, item_feature=None, events=None):
        super().__init__()
        self.dataset = EventDataSet(user_feature=user_feature, item_feature=item_feature, events=events)
        self.model = LRModel(dim=self.dataset.feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.learning_rate = 0.01

    def score(self, user_id="", item_ids=""):
        with torch.no_grad():
            user_features = self.dataset.user_feature_by_id(user_id)
            item_features = self.dataset.item_feature_by_id(item_ids)
            score = self.model(torch.tensor(user_features, dtype=torch.float32).unsqueeze(),
                               torch.tensor(item_features, dtype=torch.float32).unsqueeze())
        return score.item()

    def train(self, epoch_num=10, batch_size=5, shuffle=False):
        losser = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(epoch_num):
            for user, item, label in dataloader:
                x = torch.cat((user, item), dim=1)
                y_pred = self.model(x)
                loss = losser(y_pred.squeeze(), label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"epoch {epoch + 1}/{epoch_num}, loss:{loss.item():.4f}")
