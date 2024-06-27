import abc


class RecModel(abc.ABC):

    @abc.abstractmethod
    def score(self, user_id="", items=""):
        pass

    @abc.abstractmethod
    def train(self, epoch_num=10, batch_size=5, shuffle=True):
        pass
