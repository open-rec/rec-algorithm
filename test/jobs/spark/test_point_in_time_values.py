"""Run with pytest or spark-submit; requires a local Spark runtime."""
import unittest

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None


@unittest.skipIf(SparkSession is None, "requires pyspark")
class PointInTimeValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession.builder.master("local[2]")
                     .appName("point-in-time-values")
                     .config("spark.sql.ansi.enabled", "true")
                     .config("spark.sql.shuffle.partitions", "2").getOrCreate())

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_non_numeric_events_keep_counts_and_contribute_zero_value(self):
        from jobs.spark.point_in_time import _behavior, _commerce
        labels = self.spark.createDataFrame([("sample", "u", 100)],
            "_sample_id string, user_id string, _label_time long")
        history = self.spark.createDataFrame([
            (str(i), "u", "item", "scene", "stay", value, 50 + i, 50 + i,
             "INSERT", "day", str(i))
            for i, value in enumerate(['{"id":"item_1779"}', None, "", "2.5", "-1"])
        ], "event_id string, user_id string, item_id string, scene string, type string, "
           "value string, time long, _effective_time long, _operation string, dt string, trace_id string")
        row = _behavior(labels, history, "user_id", "item_id").first()
        self.assertEqual(row.event_count, 5)
        self.assertEqual(row.event_value_sum, 1.5)
        self.assertAlmostEqual(row.event_value_mean, 0.3)
        self.assertEqual(row.event_value_sum_5m, 1.5)
        items = self.spark.createDataFrame([("item", "category", 10.0, 1, "INSERT", "day")],
            "id string, category string, price double, _effective_time long, _operation string, dt string")
        commerce = _commerce(labels, history, items).first()
        self.assertEqual(commerce.preferred_categories, "category")
        self.assertEqual(commerce.event_price_mean, 10.0)


if __name__ == "__main__":
    unittest.main()
