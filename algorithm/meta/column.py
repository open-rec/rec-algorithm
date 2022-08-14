import abc

from algorithm.meta.column_type import ColumnType


class Column(abc.ABC):

    def __init__(self, _name=None, _type=str):
        self._name = _name
        self._type = _type

    def name(self):
        return self._name

    def type(self):
        return self._type

    def is_int(self):
        return self._type is ColumnType.INT

    def is_str(self):
        return self._type is ColumnType.STRING

    def is_float(self):
        return self._type is ColumnType.FLOAT

    def is_bool(self):
        return self._type is ColumnType.BOOL

    def is_json(self):
        return self._type is ColumnType.JSON
