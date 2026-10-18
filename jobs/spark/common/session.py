"""Spark session creation shared by the job entry points."""


def get_spark(app_name, spark=None, hive=False):
    """Reuse caller-owned sessions; create a session only when needed."""
    if spark is not None:
        return spark
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    if hive:
        builder = builder.enableHiveSupport()
    return builder.getOrCreate()
