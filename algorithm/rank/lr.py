from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pandas import DataFrame
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, Subset

from algorithm.feature.feature_space import FeatureSpace
from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.user_feature import UserFeature
from algorithm.rank.model import RecModel
from algorithm.utils.file_util import DEFAULT_SCENE, feature_path, rank_model_path

CLICK = "click"
EXPOSE = "expose"
LABELLED_EVENTS = (CLICK, EXPOSE)

MODEL_FILENAME = "lr.pth"
FEATURE_FILENAME = "lr.features.json"


class EventDataSet(Dataset):
    """
    (user vector, item vector, clicked) triples over the labelled events.

    Encoding is delegated to a `FeatureSpace` so the column layout can be persisted with the
    checkpoint and reproduced by the rank engine, instead of being re-derived from whatever frame
    each side happens to hold.
    """

    def __init__(self, user_feature: UserFeature = None, item_feature: ItemFeature = None,
                 events: DataFrame = None, feature_space: FeatureSpace = None):
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.raw_events = events
        self.events = None
        self.space = None
        self.user_feature_map = None
        self.item_feature_map = None
        self.labels = None
        self.dim = 1
        self.preprocess(feature_space)

    @property
    def feature_dim(self):
        return self.dim

    @property
    def feature_space(self):
        return self.space

    def preprocess(self, feature_space=None):
        space = feature_space if feature_space is not None else FeatureSpace()
        if not space.fitted:
            space.fit(users=self.user_feature.users, items=self.item_feature.items)
        self._bind(space)

        # Keep only labelled events whose user and item we can actually encode. This used to be an
        # inner merge against both frames, which did the same filtering but also multiplied rows
        # whenever an id repeated, and left id_x/id_y columns behind.
        events = self.raw_events
        # A clicked impression normally has both expose and click events. Its expose is not a
        # negative label: keep it in behavioural history but remove it from supervised labels.
        labelled = events[events["type"].isin(LABELLED_EVENTS)].copy()
        identity = (["trace_id"] if "trace_id" in labelled.columns
                    and labelled["trace_id"].fillna("").astype(str).ne("").any()
                    else ["user_id", "item_id"])
        clicked = labelled[labelled["type"] == CLICK][identity].drop_duplicates()
        if not clicked.empty:
            clicked["_clicked_impression"] = True
            labelled = labelled.merge(clicked, how="left", on=identity)
            labelled = labelled[~((labelled["type"] == EXPOSE)
                                  & labelled["_clicked_impression"].eq(True))]
            labelled = labelled.drop(columns=["_clicked_impression"])
        events = labelled
        keep = (
            events["user_id"].isin(self.user_feature_map.keys())
            & events["item_id"].isin(self.item_feature_map.keys())
        )
        self.events = events[keep].copy()
        if "time" in self.events.columns:
            # Stable chronological order is also the train/validation boundary used by _split.
            self.events = self.events.sort_values("time", kind="mergesort")
        self.events = self.events.reset_index(drop=True)
        self.labels = (self.events["type"] == CLICK).astype(np.float32)

        # plain numpy for __getitem__: a DataFrame.iloc lookup per sample dominated data loading
        self._user_ids = self.events["user_id"].to_numpy()
        self._item_ids = self.events["item_id"].to_numpy()
        self._label_values = self.labels.to_numpy()

    def _bind(self, space):
        self.space = space
        user_map, item_map = space.build_maps(users=self.user_feature.users,
                                              items=self.item_feature.items)
        self.user_feature_map = {k: v.astype(np.float32) for k, v in user_map.items()}
        self.item_feature_map = {k: v.astype(np.float32) for k, v in item_map.items()}
        self.dim = space.dim

    def rebind_space(self, space):
        """Re-encode against an already fitted space, e.g. the one saved with a checkpoint."""
        self.preprocess(space)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        user_feature = torch.from_numpy(self.user_feature_map[self._user_ids[idx]])
        item_feature = torch.from_numpy(self.item_feature_map[self._item_ids[idx]])
        label = torch.tensor(self._label_values[idx], dtype=torch.float32)
        return user_feature, item_feature, label

    @property
    def positive_rate(self):
        """Share of clicks among the labelled events. 0.0 or 1.0 means there is nothing to learn."""
        if not len(self._label_values):
            return 0.0
        return float(self._label_values.mean())

    @property
    def user_feature_width(self):
        """Width of one user vector, needed to stand in for a user we have no features for."""
        return self.space.user_width if self.space else 0

    def user_feature_by_id(self, user_id):
        """None when unknown — scoring an id absent from the training data is expected, not fatal."""
        return self.user_feature_map.get(user_id)

    def item_feature_by_id(self, item_id):
        return self.item_feature_map.get(item_id)


class LRModel(nn.Module):
    def __init__(self, dim=10):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(in_features=dim, out_features=1)

    def forward(self, x):
        # a probability, not a logit: the rank engine POSTs to /model/score and uses this output
        # directly as the score, so switching to BCEWithLogitsLoss would change that contract
        pred = torch.sigmoid(self.linear(x))
        return pred


class LRRecModel(RecModel):

    def __init__(self, user_feature=None, item_feature=None, events=None, feature_space=None,
                 scene=DEFAULT_SCENE, model_file=None, feature_file=None, model_type="lr"):
        """
        Artifacts are filed per scene in the shared model store — `model/rank/{scene}/lr.pth` and
        `model/feature/{scene}/lr.features.json` — so a trained model survives across runs and does
        not collide with the pre-trained Douban checkpoint at the root of `model/rank`.

        The feature space is kept out of the .pth deliberately: the rank engine loads that file with
        a bare `load_state_dict(torch.load(...))`, so burying extra keys in it would break serving.
        """
        super().__init__()
        self.scene = scene
        self.model_file = str(model_file) if model_file else str(rank_model_path(scene) / MODEL_FILENAME)
        self.feature_file = (str(feature_file) if feature_file
                             else str(feature_path(scene) / FEATURE_FILENAME))

        if feature_space is None and Path(self.feature_file).exists():
            # reuse the persisted vocabulary rather than re-fitting encoders over the whole frame
            feature_space = FeatureSpace.load(self.feature_file)
        if feature_space is None:
            feature_space = FeatureSpace.for_model(model_type)

        self.dataset = EventDataSet(user_feature=user_feature, item_feature=item_feature,
                                    events=events, feature_space=feature_space)
        self.model = LRModel(dim=self.dataset.feature_dim)

    def exists(self):
        """True when both artifacts are already on disk, so training can be skipped."""
        return Path(self.model_file).exists() and Path(self.feature_file).exists()

    def load_or_train(self, force=False, **train_kwargs):
        """
        Load the persisted model when there is one, otherwise train and persist it.

        Returns True if it trained. `force=True` retrains and overwrites regardless.
        """
        if not force and self.exists():
            self.load()
            print(f"loaded {self.model_file} (dim {self.model.dim}); skipping training")
            return False
        self.train(**train_kwargs)
        self.save()
        return True

    def score(self, user_id="", item_ids=None):
        """
        Scores in the order given. An unknown user falls back to a zero vector and an unknown item
        scores 0.0 instead of raising KeyError — the same degradation the online rank engine applies,
        so offline and online agree on what happens to ids outside the training data.
        """
        if not item_ids:
            return []

        self.model.eval()
        user_features = self.dataset.user_feature_by_id(user_id)
        if user_features is None:
            user_features = np.zeros(self.dataset.user_feature_width, dtype=np.float32)
        user_tensor = torch.tensor(user_features, dtype=torch.float32)

        scores = {}
        batch_features, scored_ids = [], []
        for item_id in item_ids:
            item_features = self.dataset.item_feature_by_id(item_id)
            if item_features is None:
                scores[item_id] = 0.0
                continue
            batch_features.append(
                torch.cat((user_tensor, torch.tensor(item_features, dtype=torch.float32)), dim=0))
            scored_ids.append(item_id)

        if batch_features:
            with torch.no_grad():
                # reshape rather than squeeze: squeeze collapses a single-item batch to a 0-dim
                # tensor, whose tolist() hands back a bare float instead of a list
                predictions = self.model(torch.stack(batch_features)).reshape(-1).tolist()
            scores.update(zip(scored_ids, predictions))

        return [scores[item_id] for item_id in item_ids]

    def train(self, epoch_num=10, batch_size=100, shuffle=True, learning_rate=0.01,
              val_ratio=0.2, seed=42):
        """
        `shuffle` defaults to True: events arrive in whatever order the source data had, so
        consecutive batches were strongly correlated.

        The newest `val_ratio` slice is held out and scored with AUC each epoch. A temporal holdout
        avoids evaluating older events with a model trained on newer ones.
        """
        if not len(self.dataset):
            print("no labelled events to train on — is the event data empty, or do its "
                  "user_id/item_id values not appear in the user/item frames?")
            return

        positive_rate = self.dataset.positive_rate
        if positive_rate in (0.0, 1.0):
            # say it out loud: BCE will fall to ~0 against a constant predictor and AUC is
            # undefined, so the run looks healthy while the model has learned nothing
            print(f"warning: every labelled event carries the same label (click rate "
                  f"{positive_rate:.0%}) — loss will collapse against a constant predictor and AUC "
                  f"is undefined. Check that the event data holds both '{CLICK}' and '{EXPOSE}'.")

        train_set, val_set = self._split(val_ratio=val_ratio, seed=seed)
        losser = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
        best_auc, best_state = None, None

        for epoch in range(epoch_num):
            self.model.train()
            epoch_loss, batches = 0.0, 0
            for user, item, label in dataloader:
                x = torch.cat((user, item), dim=1)
                y_pred = self.model(x)
                loss = losser(y_pred.squeeze(), label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            # the mean over the epoch, not whatever the last batch happened to be; and no
            # NameError when the dataset yields nothing
            if not batches:
                print(f"epoch {epoch + 1}/{epoch_num}, no batches — is the event data empty?")
                continue
            message = f"epoch {epoch + 1}/{epoch_num}, loss:{epoch_loss / batches:.4f}"
            auc = self.evaluate(val_set, batch_size=batch_size)
            if auc is not None:
                message += f", val auc:{auc:.4f}"
                if best_auc is None or auc > best_auc:
                    best_auc = auc
                    best_state = {name: value.detach().clone()
                                  for name, value in self.model.state_dict().items()}
            print(message)
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"restored best validation checkpoint (auc:{best_auc:.4f})")

    def _split(self, val_ratio=0.2, seed=42):
        total = len(self.dataset)
        val_size = int(total * val_ratio) if val_ratio else 0
        if val_size <= 0 or val_size >= total:
            return self.dataset, None
        boundary = total - val_size
        return (Subset(self.dataset, range(0, boundary)),
                Subset(self.dataset, range(boundary, total)))

    def evaluate(self, dataset=None, batch_size=100):
        """AUC over `dataset`, or None when it is empty or single-class (AUC is undefined then)."""
        if dataset is None or not len(dataset):
            return None
        self.model.eval()
        predictions, labels = [], []
        with torch.no_grad():
            for user, item, label in DataLoader(dataset=dataset, batch_size=batch_size):
                predictions.extend(self.model(torch.cat((user, item), dim=1)).reshape(-1).tolist())
                labels.extend(label.tolist())
        if len(set(labels)) < 2:
            return None
        return roc_auc_score(labels, predictions)

    def save(self):
        torch.save(self.model.state_dict(), self.model_file)
        # without the feature space the checkpoint is unusable: nothing else records what its
        # columns mean, which is how the repo ended up with three different guesses at `dim`
        self.dataset.feature_space.save(self.feature_file)
        print(f"saved {self.model_file} and {self.feature_file} (dim {self.model.dim})")

    def load(self):
        feature_file = Path(self.feature_file)
        if feature_file.exists():
            # re-encode with the vocabulary the model was trained on, not one re-fitted on whatever
            # frames this process happens to hold
            self.dataset.rebind_space(FeatureSpace.load(feature_file))
            if self.model.dim != self.dataset.feature_dim:
                self.model = LRModel(dim=self.dataset.feature_dim)

        state = torch.load(self.model_file, map_location="cpu")
        checkpoint_dim = state["linear.weight"].shape[-1] if "linear.weight" in state else None
        if checkpoint_dim is not None and checkpoint_dim != self.model.dim:
            raise ValueError(
                f"{self.model_file} was trained with dim={checkpoint_dim}, but the current feature "
                f"space yields dim={self.model.dim}. Retrain, or point feature_file at the "
                f"{FEATURE_FILENAME} this checkpoint was saved with.")
        self.model.load_state_dict(state)
        self.model.eval()
