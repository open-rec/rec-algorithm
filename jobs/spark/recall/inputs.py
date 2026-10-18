"""Lazy cumulative snapshots shared by recall handlers."""

from functools import cached_property

from jobs.spark.common.io import keep_active_item_events, read_events, read_items, read_users


class RecallInputs:
    """Resolve each snapshot once; events always exclude deleted items."""

    def __init__(self, spark, args):
        self.spark = spark
        self.args = args

    @cached_property
    def items(self):
        return read_items(self.spark, self.args.item_table, self.args.date,
                          cumulative=True, path=self.args.item_path)

    @cached_property
    def events(self):
        events = read_events(self.spark, self.args.event_table, self.args.date,
                             cumulative=True, path=self.args.event_path)
        return keep_active_item_events(events, self.items)

    @cached_property
    def users(self):
        return read_users(self.spark, self.args.user_table, self.args.date,
                          cumulative=True, path=self.args.user_path)

    def users_by_scene(self, event_type=None):
        from pyspark.sql import functions as F

        events = self.events
        if event_type:
            events = events.filter(F.col("type") == event_type)
        scenes = events.select("scene", F.col("user_id").alias("id")).dropDuplicates()
        return self.users.join(scenes, "id")
