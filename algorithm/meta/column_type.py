from enum import Enum


class ColumnType(Enum):
    INT = int
    STRING = str
    FLOAT = float
    BOOL = bool
    JSON = dict
