import csv
import random
import time

from algorithm.meta.user_table import UserTable
from algorithm.meta.item_table import ItemTable
from algorithm.meta.behavior_table import BehaviorTable

item_ids = []
user_ids = []


def gen_relation_id(ids=[]):
    return random.choice(ids)


def gen_string(prefix="test", id=0):
    return prefix + '_' + str(id)


def gen_time():
    return int(time.time()) - random.randint(0, 7 * 24 * 60 * 60)


def gen_int():
    return random.randint(0, 100)


def gen_behavior():
    return 'click'


def gen_trace():
    return 'open-rec' if random.randint(0, 1) else 'self-rec'


def gen_bool():
    return random.randint(0, 1)


def gen_json():
    return {}


def gen_record(table, columns, id):
    record = []
    for column in columns:
        if column.is_str():
            if table.name() == 'behavior':
                if column.name() == 'item_id':
                    record.append(gen_relation_id(item_ids))
                elif column.name() == 'user_id':
                    record.append(gen_relation_id(user_ids))
                elif column.name() == 'type':
                    record.append(gen_behavior())
                elif column.name() == 'trace':
                    record.append(gen_trace())
                else:
                    record.append(gen_string(column.name(), id))
            else:
                if column.name() == 'id':
                    record.append(gen_string(table.name(), id))
                else:
                    record.append(gen_string(column.name(), id))
        elif column.is_int():
            if 'time' in column.name():
                record.append(gen_time())
            else:
                record.append(gen_int())
        elif column.is_bool():
            record.append(gen_bool())
        elif column.is_json():
            record.append(gen_json())
    return record


def write_columns_data(table, columns, count, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(table.headers())
        for i in range(count):
            record = gen_record(table, columns, i)
            writer.writerow(record)

            if isinstance(table, UserTable):
                user_ids.append(record[0])

            if isinstance(table, ItemTable):
                item_ids.append(record[0])


def gen_user_data(count=100000, filename='user.csv'):
    table = UserTable()
    columns = table.columns()
    write_columns_data(table, columns, count, filename)


def gen_item_data(count=10000, filename='item.csv'):
    table = ItemTable()
    columns = table.columns()
    write_columns_data(table, columns, count, filename)


def gen_behavior_data(count=1000000, filename='behavior.csv'):
    table = BehaviorTable()
    columns = table.columns()
    write_columns_data(table, columns, count, filename)


if __name__ == "__main__":
    gen_user_data(count=10000, filename='../data/user.csv')
    gen_item_data(count=10000, filename='../data/item.csv')
    gen_behavior_data(count=100000, filename='../data/behavior.csv')
