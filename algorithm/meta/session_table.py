from algorithm.meta.column import Column
from algorithm.meta.column_type import ColumnType
from algorithm.meta.table import Table


class SessionTable(Table):
    def __init__(self):
        self._name = "session"
        self._pk = Column("id")
        self._columns = [
            Column("id", ColumnType.STRING),
            Column("user_id", ColumnType.STRING),
            Column("device_id", ColumnType.STRING),
            Column("scene", ColumnType.STRING),
            Column("start_time", ColumnType.INT),
            Column("last_event_time", ColumnType.INT),
            Column("ext_fields", ColumnType.JSON),
        ]
