"""Scheduled recall pipeline: load inputs, compute, write and stage publication."""

import argparse
from datetime import datetime, timedelta, timezone

from pyspark import StorageLevel
from pyspark.sql import functions as F

from jobs.spark.common.io import write_result
from jobs.spark.common.session import get_spark
from jobs.spark.recall.catalog import RECALL_JOBS, get_recall_spec
from jobs.spark.recall.inputs import RecallInputs
from jobs.spark.recall.output import publish_result


def parser():
    result = argparse.ArgumentParser(description="OpenRec distributed recall job")
    result.add_argument("algorithm", choices=tuple(RECALL_JOBS))
    result.add_argument("--event-table", default="openrec.event_entity")
    result.add_argument("--item-table", default="openrec.item_entity")
    result.add_argument("--event-path")
    result.add_argument("--item-path")
    result.add_argument("--user-table", default="openrec.user_entity")
    result.add_argument("--user-path")
    result.add_argument("--output-table")
    result.add_argument("--output-path")
    result.add_argument("--mode", default="overwrite", choices=("overwrite", "append"))
    result.add_argument("--date", default=(datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat(),
                        help="UTC business date, defaults to yesterday (YYYY-MM-DD)")
    result.add_argument("--size", type=int, default=1000)
    result.add_argument("--event-type", default="click")
    result.add_argument("--vector-size", type=int, default=10)
    result.add_argument("--min-count", type=int, default=5)
    result.add_argument("--neighbour-size", type=int, default=50)
    result.add_argument("--target-type", choices=("item", "user"), default="item")
    result.add_argument("--text-columns")
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
    spec = get_recall_spec(args.algorithm)
    spark = get_spark("openrec-%s" % args.algorithm, spark, hive=True)
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    output = spec.compute(RecallInputs(spark, args), args).withColumn("dt", F.lit(args.date))
    if spec.persist:
        output = output.persist(StorageLevel.DISK_ONLY)
    try:
        if args.output_table or args.output_path:
            write_result(output, args.output_table, args.output_path, args.mode)
        if args.publish:
            publish_result(output, spec, args)
    finally:
        if spec.persist:
            output.unpersist()
    return output


def main():
    args = parser().parse_args()
    if not args.publish and not args.output_table and not args.output_path:
        raise SystemExit("configure an output table/path or --publish")
    run(args)


if __name__ == "__main__":
    main()
