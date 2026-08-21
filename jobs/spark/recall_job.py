"""CLI for scheduled Hive/Spark recall computation."""

import argparse
from datetime import datetime, timedelta, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from jobs.spark.io import keep_active_item_events, read_events, read_items, write_result
from jobs.spark.recall import embedding, hot, i2i, new
from publisher.spark import publish_embedding, publish_recall


def parser():
    result = argparse.ArgumentParser(description="OpenRec distributed recall job")
    result.add_argument("algorithm", choices=("hot", "new", "i2i", "embedding"))
    result.add_argument("--event-table", default="openrec.event_entity")
    result.add_argument("--item-table", default="openrec.item_entity")
    result.add_argument("--event-path")
    result.add_argument("--item-path")
    result.add_argument("--output-table")
    result.add_argument("--output-path")
    result.add_argument("--mode", default="overwrite", choices=("overwrite", "append"))
    result.add_argument("--date", default=(datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat(),
                        help="UTC business date, defaults to yesterday (YYYY-MM-DD)")
    result.add_argument("--size", type=int, default=1000)
    result.add_argument("--event-type", default="click")
    result.add_argument("--vector-size", type=int, default=10)
    result.add_argument("--min-count", type=int, default=5)
    result.add_argument("--publish", action="store_true")
    result.add_argument("--redis-host", default="redis")
    result.add_argument("--redis-port", type=int, default=6379)
    result.add_argument("--es-host", default="https://elasticsearch:9200")
    result.add_argument("--es-user", default="elastic")
    result.add_argument("--es-password")
    result.add_argument("--es-ca-certs")
    result.add_argument("--revision", default="r001")
    result.add_argument("--max-index-versions", "--retain-versions",
                        dest="max_index_versions", type=int, default=2)
    result.add_argument("--console-url", default="http://rec-console:8095")
    return result


def run(args, spark=None):
    spark = spark or SparkSession.builder.appName("openrec-%s" % args.algorithm) \
        .enableHiveSupport().getOrCreate()
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    # Build one as-of item snapshot for every recall algorithm. read_items resolves the latest
    # mutation per id and removes DELETE tombstones before any expensive recall computation.
    items = read_items(spark, args.item_table, args.date, cumulative=True,
                       path=args.item_path)
    if args.algorithm == "new":
        output = new(items, args.size)
    else:
        events = read_events(spark, args.event_table, args.date, cumulative=True,
                             path=args.event_path)
        events = keep_active_item_events(events, items)
        if args.algorithm == "hot":
            output = hot(events, args.size, args.event_type)
        elif args.algorithm == "i2i":
            output = i2i(events, args.size, args.event_type)
        else:
            output = embedding(events, args.vector_size, args.min_count,
                               event_type=args.event_type)
    output = output.withColumn("dt", F.lit(args.date))
    if args.output_table or args.output_path:
        write_result(output, args.output_table, args.output_path, args.mode)
    if args.publish:
        if args.algorithm == "embedding":
            publish_embedding(output, [args.es_host], args.es_user, args.es_password,
                              ca_certs=args.es_ca_certs,
                              verify_certs=bool(args.es_ca_certs),
                              business_date=args.date, revision=args.revision,
                              max_index_versions=args.max_index_versions)
        else:
            publish_recall(output, args.algorithm, args.date, args.revision,
                           [args.es_host], args.es_user, args.es_password,
                           ca_certs=args.es_ca_certs,
                           verify_certs=bool(args.es_ca_certs),
                           console_url=args.console_url,
                           max_index_versions=args.max_index_versions)
    return output


def main():
    args = parser().parse_args()
    if not args.publish and not args.output_table and not args.output_path:
        raise SystemExit("configure an output table/path or --publish")
    run(args)


if __name__ == "__main__":
    main()
