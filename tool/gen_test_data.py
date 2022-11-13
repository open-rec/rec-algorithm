import csv
import random
import time
import uuid

from algorithm.meta.event_table import BehaviorTable
from algorithm.meta.item_table import ItemTable
from algorithm.meta.user_table import UserTable


def gen_relation_id(ids=[]):
    return random.choice(ids)


def gen_string(prefix="test", id=0):
    return prefix + '_' + str(id)


def gen_time():
    return int(time.time()) - random.randint(0, 7 * 24 * 60 * 60)


def gen_int(start, end):
    return random.randint(start, end)


def gen_bool():
    return random.randint(0, 1)


def gen_uuid():
    return str(uuid.uuid1(random.randint(0, 10000), random.randint(0, 100)))


def gen_json():
    return {}


def gen_scene():
    return gen_string("test", random.randint(0, 3))


user_ids = []


def gen_one_user_record(table, id):
    record = []
    for column in table.columns():
        if column.name() == 'id':
            record.append(gen_string(table.name(), id))
            user_ids.append(record[0])
        elif column.name() == 'device_id':
            record.append(gen_uuid())
        elif column.name() == 'name':
            record.append(gen_string(column.name(), id))
        elif column.name() == 'gender':
            record.append(gen_bool())
        elif column.name() == 'age':
            record.append(gen_int(1, 100))
        elif column.name() == 'country':
            record.append(gen_string(column.name(), gen_int(0, 0)))
        elif column.name() == 'city':
            record.append(gen_string(column.name(), gen_int(0, 9)))
        elif column.name() == 'phone':
            record.append(gen_string(column.name(), id))
        elif column.name() == 'tags':
            record.append(gen_string(column.name(), gen_int(0, 100)))
        elif column.name() == 'register_time':
            record.append(gen_time())
        elif column.name() == 'login_time':
            record.append(gen_time())
        elif column.name() == 'ext_fields':
            record.append(gen_json())
    return record


item_ids = []


def gen_one_item_record(table, id):
    record = []
    for column in table.columns():
        if column.name() == 'id':
            record.append(gen_string(table.name(), id))
            item_ids.append(record[0])
        elif column.name() == 'title':
            record.append(gen_string(column.name(), id))
        elif column.name() == 'category':
            record.append(gen_string(column.name(), gen_int(0, 100)))
        elif column.name() == 'tags':
            record.append(gen_string(column.name(), gen_int(0, 100)))
        elif column.name() == 'scene':
            record.append(gen_string(column.name(), gen_int(0, 2)))
        elif column.name() == 'pub_time':
            record.append(gen_time())
        elif column.name() == 'modify_time':
            record.append(gen_time())
        elif column.name() == 'expire_time':
            record.append(gen_time())
        elif column.name() == 'status':
            record.append(1)
        elif column.name() == 'weight':
            record.append(random.choice([1, 5, 10]))
        elif column.name() == 'ext_fields':
            record.append(gen_json())
    return record


def gen_one_event_record(table, id):
    record = []
    for column in table.columns():
        if column.name() == 'id':
            record.append(gen_string(table.name(), id))
        elif column.name() == 'user_id':
            record.append(gen_relation_id(user_ids))
        elif column.name() == 'item_id':
            record.append(gen_relation_id(item_ids))
        elif column.name() == 'trace_id':
            record.append('open-rec')
        elif column.name() == 'scene':
            record.append(gen_string(column.name(), gen_int(0, 2)))
        elif column.name() == 'type':
            record.append('click')
        elif column.name() == 'value':
            record.append(1)
        elif column.name() == 'time':
            record.append(gen_time())
        elif column.name() == 'is_login':
            record.append(1)
        elif column.name() == 'ext_fields':
            record.append(gen_json())
    return record


def write_columns_data(table, count, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(table.headers())
        for i in range(count):
            record = None
            if table.name() == 'user':
                record = gen_one_user_record(table, i)
            elif table.name() == 'item':
                record = gen_one_item_record(table, i)
            elif table.name() == 'event':
                record = gen_one_event_record(table, i)
            if record:
                writer.writerow(record)


def gen_user_data(count=200000, filename='user.csv'):
    table = UserTable()
    write_columns_data(table, count, filename)


def gen_item_data(count=10000, filename='item.csv'):
    table = ItemTable()
    write_columns_data(table, count, filename)


def gen_behavior_data(count=1000000, filename='event.csv'):
    table = BehaviorTable()
    write_columns_data(table, count, filename)


if __name__ == "__main__":
    gen_user_data(count=10000, filename='../data/test/user.csv')
    gen_item_data(count=10000, filename='../data/test/item.csv')
    gen_behavior_data(count=100000, filename='../data/test/event.csv')
