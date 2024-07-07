import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.user_feature import UserFeature
from algorithm.rank.model import RecModel
from algorithm.utils.file_util import model_path


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
        self.events = merge_events[merge_events["type"].isin(["click", "expose"])]
        self.labels = self.events["type"].apply(lambda x: 1 if x == "click" else 0)

        user_features = np.hstack([
            # too big to build tensor
            # self.user_feature.id,
            # self.user_feature.device_id,
            # self.user_feature.name,
            self.user_feature.country,
            self.user_feature.city,
            self.user_feature.gender,
            self.user_feature.age,
            self.user_feature.tags,
        ])

        item_features = np.hstack([
            # too big to build tensor
            # self.item_feature.id,

            # cost too much memory
            # self.item_feature.title,
            # self.item_feature.tags,
            self.item_feature.category,
            self.item_feature.scene,
            self.item_feature.weight,
        ])

        self.user_feature_map = {
            user_id: user_features[i]
            for i, user_id in enumerate(self.user_feature.raw_id)
        }

        self.item_feature_map = {
            item_id: item_features[i]
            for i, item_id in enumerate(self.item_feature.raw_id)
        }
        self.dim = user_features.shape[-1] + item_features.shape[-1]

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event = self.events.iloc[idx]
        user_feature = torch.tensor(self.user_feature_map[event["user_id"]], dtype=torch.float32)
        item_feature = torch.tensor(self.item_feature_map[event["item_id"]], dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float32)
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
        self.model_file = str(model_path() / "lr.pth")
        self.model = LRModel(dim=self.dataset.feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.learning_rate = 0.01

    def score(self, user_id="", item_ids=[]):
        with torch.no_grad():
            user_features = self.dataset.user_feature_by_id(user_id)
            batch_features = []
            for item_id in item_ids:
                item_features = self.dataset.item_feature_by_id(item_id)
                batch_features.append(torch.cat(
                    (
                        torch.tensor(user_features, dtype=torch.float32),
                        torch.tensor(item_features, dtype=torch.float32)
                    ),
                    dim=0
                ))
            score = self.model(torch.stack(batch_features))
        return score.squeeze().tolist()

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

    def save(self):
        torch.save(self.model.state_dict(), self.model_file)

    def load(self):
        self.model.load_state_dict(torch.load(self.model_file))
