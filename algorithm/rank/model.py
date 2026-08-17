import abc


class RecModel(abc.ABC):
    """
    A ranking model's offline contract.

    `save`/`load` are part of it deliberately: the whole offline-to-online handoff is "train here,
    write an artifact, read it in the rank engine", so a model that cannot persist itself does not
    satisfy the interface.
    """

    @abc.abstractmethod
    def score(self, user_id="", item_ids=None):
        """Scores for `item_ids`, in the order given."""

    @abc.abstractmethod
    def train(self, epoch_num=10, batch_size=100, shuffle=True):
        pass

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def load(self):
        pass
