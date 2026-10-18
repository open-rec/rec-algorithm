"""Shared recall output and staging-index publication policy."""

from publisher.spark import publish_embedding, publish_recall


def publish_result(output, spec, args):
    options = dict(
        ca_certs=args.es_ca_certs, verify_certs=bool(args.es_ca_certs),
        max_index_versions=args.max_index_versions,
    )
    if spec.publisher == "embedding":
        publish_embedding(output, [args.es_host], args.es_user, args.es_password,
                          business_date=args.date, revision=args.revision, **options)
    elif spec.publisher == "recall":
        publish_recall(output, spec.serving_table.format(target_type=args.target_type),
                       args.date, args.revision, [args.es_host], args.es_user, args.es_password,
                       console_url=args.console_url, **options)
    else:
        raise ValueError("unsupported recall publisher: %s" % spec.publisher)
