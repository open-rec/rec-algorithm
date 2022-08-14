from algorithm.meta.schema import Schema


class Table(Schema):

    def __init__(self, name='', pk=None, columns=[]):
        self._name = name
        self._pk = pk
        self._columns = columns

    def columns(self):
        return self._columns

    def pk(self):
        return self._pk

    def name(self):
        return self._name
