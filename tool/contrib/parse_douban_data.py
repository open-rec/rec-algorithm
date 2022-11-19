import csv
import random
import time
import uuid
import datetime

from algorithm.meta.event_table import EventTable
from algorithm.meta.item_table import ItemTable
from algorithm.meta.user_table import UserTable


def gen_string(prefix="test", id=0):
    return prefix + '_' + str(id)


def gen_time():
    return 0


def gen_int(start, end):
    return random.randint(start, end)


def gen_bool():
    return random.randint(0, 1)


def gen_uuid():
    return str(uuid.uuid1(random.randint(0, 10000), random.randint(0, 100)))


def gen_json():
    return {}


def parse_one_user_record(table, row):
    record = []
    for column in table.columns():
        if column.name() == 'id':
            record.append(row['USER_MD5'])
        elif column.name() == 'device_id':
            record.append(gen_uuid())
        elif column.name() == 'name':
            record.append(row['USER_NICKNAME'])
        elif column.name() == 'gender':
            record.append(gen_bool())
        elif column.name() == 'age':
            record.append(gen_int(18, 60))
        elif column.name() == 'country':
            record.append(gen_string(column.name(), gen_int(0, 0)))
        elif column.name() == 'city':
            record.append(gen_string(column.name(), gen_int(0, 0)))
        elif column.name() == 'phone':
            record.append(gen_string(column.name(), gen_int(137999999999, 158999999999)))
        elif column.name() == 'tags':
            record.append(gen_string(column.name(), gen_int(0, 0)))
        elif column.name() == 'register_time':
            record.append(gen_time())
        elif column.name() == 'login_time':
            record.append(gen_time())
        elif column.name() == 'ext_fields':
            record.append(gen_json())
    return record


def parse_one_item_record(table, row):
    record = []
    for column in table.columns():
        if column.name() == 'id':
            record.append(row['MOVIE_ID'])
        elif column.name() == 'title':
            record.append(row['NAME'])
        elif column.name() == 'category':
            record.append(row['GENRES'])
        elif column.name() == 'tags':
            record.append(row['TAGS'])
        elif column.name() == 'scene':
            record.append('douban_movie')
        elif column.name() == 'pub_time':
            record.append(gen_time())
        elif column.name() == 'modify_time':
            record.append(gen_time())
        elif column.name() == 'expire_time':
            record.append(gen_time())
        elif column.name() == 'status':
            record.append(1)
        elif column.name() == 'weight':
            record.append(row['DOUBAN_SCORE'])
        elif column.name() == 'ext_fields':
            record.append(gen_json())
    return record


def parse_one_event_record(table, row):
    record = []
    for column in table.columns():
        if column.name() == 'id':
            record.append(row['RATING_ID'])
        elif column.name() == 'user_id':
            record.append(row['USER_MD5'])
        elif column.name() == 'item_id':
            record.append(row['MOVIE_ID'])
        elif column.name() == 'trace_id':
            record.append('open-rec')
        elif column.name() == 'scene':
            record.append('douban_movie')
        elif column.name() == 'type':
            record.append('click' if int(row['RATING']) > 3 else 'expose')
        elif column.name() == 'value':
            record.append(1)
        elif column.name() == 'time':
            record.append(time.mktime(datetime.datetime.strptime(row['RATING_TIME'], '%Y-%m-%d %H:%M:%S').timetuple()))
        elif column.name() == 'is_login':
            record.append(1)
        elif column.name() == 'ext_fields':
            record.append(gen_json())
    return record


def write_columns_data(table, from_filename, to_filename):
    with open(to_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(table.headers())
        with open(from_filename, 'r') as rf:
            reader = csv.DictReader(rf)
            next(reader)
            for row in reader:
                record = None
                if table.name() == 'user':
                    record = parse_one_user_record(table, row)
                elif table.name() == 'item':
                    record = parse_one_item_record(table, row)
                elif table.name() == 'event':
                    record = parse_one_event_record(table, row)
                if record:
                    writer.writerow(record)


def parse_user_data(from_filename='user.csv', to_filename='user.csv'):
    table = UserTable()
    write_columns_data(table, from_filename, to_filename)


def parse_item_data(from_filename='item.csv', to_filename='item.csv'):
    table = ItemTable()
    write_columns_data(table, from_filename, to_filename)


def parse_event_data(from_filename='event.csv', to_filename='event.csv'):
    table = EventTable()
    write_columns_data(table, from_filename, to_filename)


if __name__ == "__main__":
    parse_user_data(from_filename='../../data/douban/users.csv', to_filename='../../example/data/douban/user.csv')
    parse_item_data(from_filename='../../data/douban/movies.csv', to_filename='../../example/data/douban/item.csv')
    parse_event_data(from_filename='../../data/douban/ratings.csv', to_filename='../../example/data/douban/event.csv')
