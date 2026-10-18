"""Recall extension catalog. Metadata must not import Spark or model libraries."""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class RecallSpec:
    # A handler accepts (RecallInputs, argparse.Namespace) and returns a DataFrame.
    handler: str
    serving_table: str
    publisher: str = "recall"
    persist: bool = False
    default_size: int = 1000
    command_options: tuple = ()

    def compute(self, inputs, args):
        module, name = self.handler.split(":")
        return getattr(import_module(module), name)(inputs, args)


def _handler(name):
    return "jobs.spark.recall.handlers:" + name


RECALL_JOBS = {
    "hot": RecallSpec(_handler("hot"), "hot"),
    "new": RecallSpec(_handler("new"), "new"),
    "item_cf_i2i": RecallSpec(_handler("item_cf_i2i"), "item-cf-i2i", default_size=20),
    "item_seq_emb": RecallSpec(_handler("item_seq_emb"), "", publisher="embedding",
                               command_options=(("vector_size", 10), ("min_count", 1))),
    "user_cf_u2i": RecallSpec(_handler("user_cf_u2i"), "user-cf-u2i",
                              command_options=(("neighbour_size", 50),)),
    "content_i2i": RecallSpec(_handler("content_i2i"), "content-i2i", default_size=20),
    "user_cf_u2u": RecallSpec(_handler("user_cf_u2u"), "user-cf-u2u",
                              command_options=(("neighbour_size", 50),)),
    "content_u2u": RecallSpec(_handler("content_u2u"), "content-u2u"),
    "user_als_u2u": RecallSpec(_handler("user_als_u2u"), "user-als-u2u",
                               command_options=(("vector_size", 10), ("min_count", 1))),
    "text_search": RecallSpec(_handler("text_search"), "{target_type}-search"),
    "query_title_emb": RecallSpec(_handler("query_title_emb"), "query-title-emb", persist=True),
}
ALGORITHMS = tuple(RECALL_JOBS)
PUBLISHABLE_ALGORITHMS = ALGORITHMS
SERVING_TABLES = {
    name: spec.serving_table.format(target_type="item")
    for name, spec in RECALL_JOBS.items() if spec.publisher == "recall"
}


def get_recall_spec(name):
    try:
        return RECALL_JOBS[name]
    except KeyError:
        raise ValueError("unsupported recall algorithm: %s" % name) from None
