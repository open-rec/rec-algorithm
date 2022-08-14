from algorithm.meta.column import Column
from algorithm.meta.column_type import ColumnType
from algorithm.meta.table import Table


class UserTable(Table):

    def __init__(self):
        self._name = 'user'
        self._pk = Column('id')
        self._columns = [
            Column('id', ColumnType.STRING),
            Column('device_number', ColumnType.STRING),
            Column('name', ColumnType.STRING),
            Column('gender', ColumnType.BOOL),
            Column('age', ColumnType.INT),
            Column('country', ColumnType.STRING),
            Column('city', ColumnType.STRING),
            Column('phone', ColumnType.INT),
            Column('tags', ColumnType.STRING),
            Column('register_time', ColumnType.INT),
            Column('login_time', ColumnType.INT),
            Column('ext_fields', ColumnType.JSON),
        ]
