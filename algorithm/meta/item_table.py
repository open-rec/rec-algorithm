from algorithm.meta.column import Column
from algorithm.meta.column_type import ColumnType
from algorithm.meta.table import Table


class ItemTable(Table):

    def __init__(self):
        self._name = 'item'
        self._pk = Column('id')
        self._columns = [
            Column('id', ColumnType.STRING),
            Column('title', ColumnType.STRING),
            Column('category', ColumnType.STRING),
            Column('tags', ColumnType.STRING),
            Column('scene', ColumnType.STRING),
            Column('pub_time', ColumnType.INT),
            Column('modify_time', ColumnType.INT),
            Column('expire_time', ColumnType.INT),
            Column('status', ColumnType.BOOL),
            Column('weight', ColumnType.INT),
            Column('ext_fields', ColumnType.JSON),
        ]
