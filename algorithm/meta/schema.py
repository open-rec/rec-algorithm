from __future__ import annotations

import abc


class Schema(abc.ABC):

    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def columns(self):
        pass
