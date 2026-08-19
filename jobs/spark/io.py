"""Hive entity readers and stable output schemas for Spark jobs."""

import re

from pyspark.sql import functions as F


EVENT_FIELDS = {
    "id": "string", "user_id": "string", "item_id": "string", "trace_id": "string",
    "scene": "string", "type": "string", "value": "string", "time": "long",
}
ITEM_FIELDS = {
    "id": "string", "title": "string", "category": "string", "tags": "string",
    "scene": "string", "pub_time": "long", "modify_time": "long", "expire_time": "long",
    "status": "boolean", "weight": "double", "ext_fields": "string",
}
USER_FIELDS = {
    "id": "string", "device_id": "string", "name": "string", "gender": "string",
    "age": "double", "country": "string", "city": "string", "phone": "string",
    "tags": "string", "register_time": "long", "login_time": "long", "ext_fields": "string",
}

JSON_NAMES = {
    "user_id": "userId", "item_id": "itemId", "trace_id": "traceId",
    "pub_time": "pubTime", "modify_time": "modifyTime", "expire_time": "expireTime",
    "device_id": "deviceId", "register_time": "registerTime", "login_time": "loginTime",
    "ext_fields": "extFields",
}


def _daily_partition(spark, table, date):
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date or ""):
        raise ValueError("date must use YYYY-MM-DD")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", table):
        raise ValueError("invalid Hive table name: %s" % table)
    spark.sql("ALTER TABLE %s ADD IF NOT EXISTS PARTITION (dt='%s')" % (table, date))


def read_entity(spark, table, fields, date=None):
    """Read either a typed Hive table or the ODS one-column JSON external table."""
    if date:
        _daily_partition(spark, table, date)
    frame = spark.table(table)
    if date:
        if "dt" not in frame.columns:
            raise ValueError("table %s is not partitioned by dt" % table)
        frame = frame.filter(F.col("dt") == date)
    if "json" not in frame.columns:
        missing = set(fields) - set(frame.columns)
        if missing:
            raise ValueError("table %s misses columns: %s" % (table, sorted(missing)))
        return frame.select(*fields)
    return frame.select(*[
        F.get_json_object("json", "$.%s" % JSON_NAMES.get(name, name)).cast(dtype).alias(name)
        for name, dtype in fields.items()
    ])


def read_events(spark, table="openrec.event_entity", date=None):
    return read_entity(spark, table, EVENT_FIELDS, date)


def read_items(spark, table="openrec.item_entity", date=None):
    return read_entity(spark, table, ITEM_FIELDS, date)


def read_users(spark, table="openrec.user_entity", date=None):
    return read_entity(spark, table, USER_FIELDS, date)


def write_result(frame, table=None, path=None, mode="overwrite", partition_by=("dt",)):
    if not table and not path:
        raise ValueError("one of table or path is required")
    writer = frame.write.mode(mode)
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    if table and path:
        writer = writer.option("path", path)
    if table:
        writer.format("parquet").saveAsTable(table)
    else:
        writer.parquet(path)
