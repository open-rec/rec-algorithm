from algorithm.meta.column import Column
from algorithm.meta.column_type import ColumnType
from algorithm.meta.table import Table


class EventTable(Table):

    def __init__(self):
        self._name = 'event'
        self._pk = Column('id')
        self._columns = [
            Column('id', ColumnType.STRING),
            Column('user_id', ColumnType.STRING),
            Column('item_id', ColumnType.STRING),
            Column('trace_id', ColumnType.STRING),
            Column('scene', ColumnType.STRING),
            Column('type', ColumnType.STRING),
            Column('value', ColumnType.STRING),
            Column('time', ColumnType.INT),
            Column('is_login', ColumnType.BOOL),
            Column('ext_fields', ColumnType.JSON),
        ]
