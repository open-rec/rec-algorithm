"""Adapters from shared job inputs/options to independent recall formulas."""

import os

from . import collaborative, content, embedding, popularity


def hot(inputs, args):
    return popularity.hot(inputs.events, args.size, args.event_type)


def new(inputs, args):
    return popularity.new(inputs.items, args.size)


def item_cf_i2i(inputs, args):
    return collaborative.item_cf_i2i(inputs.events, args.size, args.event_type)


def user_cf_u2i(inputs, args):
    return collaborative.user_cf_u2i(inputs.events, args.size, args.neighbour_size, args.event_type)


def user_cf_u2u(inputs, args):
    return collaborative.user_cf_u2u(inputs.events, args.size, args.event_type)


def content_i2i(inputs, args):
    return content.content_i2i(inputs.items, args.size)


def content_u2u(inputs, args):
    return content.content_u2u(inputs.users_by_scene(), args.size)


def user_als_u2u(inputs, args):
    return embedding.user_als_u2u(inputs.events, args.size, args.vector_size,
                                  event_type=args.event_type)


def item_seq_emb(inputs, args):
    return embedding.item_seq_emb(inputs.events, args.vector_size, args.min_count,
                                  event_type=args.event_type)


def text_search(inputs, args):
    target = args.target_type
    defaults = "name,gender,country,city,tags" if target == "user" else "title,category,tags"
    columns = tuple(value.strip() for value in (args.text_columns or defaults).split(",") if value.strip())
    documents = inputs.users_by_scene(args.event_type) if target == "user" else inputs.items
    return content.text_search_documents(documents, columns, target)


def query_title_emb(inputs, args):
    from .title_embedding import title_embeddings

    return title_embeddings(
        inputs.items, os.getenv("EMBEDDING_ENDPOINT", "http://vllm:8000/v1/embeddings"),
        os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        int(os.getenv("EMBEDDING_DIMENSIONS", "1024")), os.getenv("EMBEDDING_API_KEY", ""),
        int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        int(os.getenv("EMBEDDING_PARTITIONS", "1")),
        float(os.getenv("EMBEDDING_BATCH_TIMEOUT_SECONDS", "60")))
