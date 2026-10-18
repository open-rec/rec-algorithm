"""Run with spark-submit; checks item/user text documents and scene membership."""
from unittest.mock import patch
from pyspark.sql import SparkSession
from jobs.spark.recall.job import parser, run


def main():
    spark = SparkSession.builder.master("local[2]").appName("text-search-test").getOrCreate()
    items = spark.createDataFrame([("s", "i", "blue train", "jazz", "music")],
                                  ["scene", "id", "title", "category", "tags"])
    users = spark.createDataFrame([("u", "Alice", "jazz"), ("v", "Bob", "rock")],
                                  ["id", "name", "tags"])
    events = spark.createDataFrame([("s", "u", "i", "click"), ("s", "v", "i", "expose")],
                                   ["scene", "user_id", "item_id", "type"])
    try:
        with patch("jobs.spark.recall.inputs.read_items", return_value=items), \
                patch("jobs.spark.recall.inputs.read_users", return_value=users), \
                patch("jobs.spark.recall.inputs.read_events", return_value=events), \
                patch("jobs.spark.recall.output.publish_recall") as publish:
            for entity, identifier, text in (("item", "i", "blue train jazz music"),
                                              ("user", "u", "Alice jazz")):
                args = parser().parse_args(["text_search", "--target-type", entity,
                                            "--date", "2026-10-17", "--publish"])
                rows = run(args, spark).collect()
                assert len(rows) == 1, rows
                assert rows[0][entity] == identifier
                assert rows[0].text == text
                assert publish.call_args.args[1] == entity + "-search"
        print("item/user text search Spark checks passed")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
