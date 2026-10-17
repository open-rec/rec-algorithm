"""Generate every deployable recall table from OpenRec raw item/event CSVs."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from tool.gen_text_embedding_data import generate as generate_text_embeddings

from algorithm.recall.content_u2u import ContentBasedU2U
from algorithm.recall.user_cf_u2u import UserBasedU2U
from algorithm.recall.user_als_u2u import UserALSRecall
from algorithm.recall.hot import Hot
from algorithm.recall.item_cf_i2i import ItemBasedI2I
from algorithm.recall.new import New
from algorithm.recall.content_i2i import ContentBasedI2I
from algorithm.recall.user_cf_u2i import UserBasedCF


def gen_i2i_data(items, events, i2i_size, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'left_item', 'right_item', 'score'])
        #scene_items_map = {scene: scene_items['id'].tolist() for scene, scene_items in items.groupby('scene')}
        for scene, scene_events in events.groupby('scene'):
            scene_i2i = ItemBasedI2I(events=scene_events, recall_size=i2i_size)
            i2i_items = scene_i2i.dump_i2i(i2i_size)
            for left_item in i2i_items:
                for relate_item, score in i2i_items[left_item]:
                    writer.writerow([scene, left_item, relate_item, score])


def gen_content_i2i_data(items, i2i_size, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'left_item', 'right_item', 'score'])
        for scene, scene_items in items.groupby('scene'):
            table = ContentBasedI2I(items=scene_items, cut_size=i2i_size).dump_i2i(i2i_size)
            for left_item, related in table.items():
                for right_item, score in related:
                    writer.writerow([scene, left_item, right_item, score])


def gen_user_cf_data(events, recall_size, neighbour_size, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'user', 'item', 'score'])
        for scene, scene_events in events.groupby('scene'):
            table = UserBasedCF(events=scene_events, recall_size=recall_size,
                                neighbour_size=neighbour_size).dump_user_recall()
            for user, candidates in table.items():
                for item, score in candidates:
                    writer.writerow([scene, user, item, score])



def gen_embedding_data(items, events, embedding_dim, filename):
    popularity = events.groupby(['scene', 'item_id']).size()
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'item', 'vector'])
        for row in items.itertuples(index=False):
            # Stable semantic hashing keeps same-category/tag items close without requiring a
            # heavyweight Word2Vec dependency during bootstrap generation.
            vector = np.zeros(embedding_dim, dtype=float)
            tokens = [str(getattr(row, 'category', ''))]
            tokens.extend(str(getattr(row, 'tags', '')).replace('/', ',').split(','))
            for token in filter(None, tokens):
                digest = hashlib.sha256(token.encode()).digest()
                for index in range(embedding_dim):
                    vector[index] += (digest[index] / 127.5) - 1.0
            vector[0] += np.log1p(popularity.get((row.scene, row.id), 0))
            norm = np.linalg.norm(vector)
            if norm:
                vector /= norm
            writer.writerow([row.scene, row.id, vector.tolist()])


def gen_hot_data(events, hot_size, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'item', 'score'])
        for scene, scene_events in events.groupby('scene'):
            scene_hot = Hot(events=scene_events, recall_size=hot_size)
            recall_items = scene_hot.recall()
            for score_item in recall_items:
                writer.writerow([scene, score_item.item, score_item.score])


def gen_new_data(items, new_size, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'item', 'score'])
        for scene, scene_items in items.groupby('scene'):
            scene_new = New(items=scene_items, recall_size=new_size)
            recall_items = scene_new.recall()
            for score_item in recall_items:
                writer.writerow([scene, score_item.item, score_item.score])


def gen_text_search_data(entities, scene, columns, entity_column):
    """Export the document fields used by BM25; never serialize missing values as nan."""
    return [{"scene": scene, entity_column: row["id"],
             "text": " ".join(str(row[column]) for column in columns
                              if column in row and pd.notna(row[column]))}
            for _, row in entities.iterrows()]


def gen_user_data(users, events, output, embedding_dim, neighbour_size):
    tables = {name: [] for name in ("user_cf_u2u", "content_u2u", "user_als_u2u", "user_als_emb", "text_search")}
    for scene, scene_events in events.groupby("scene"):
        scene_users = users[users["id"].isin(scene_events["user_id"])]
        tables["user_cf_u2u"].extend(UserBasedU2U(scene_events, neighbour_size).rows(scene))
        tables["content_u2u"].extend(ContentBasedU2U(scene_users, neighbour_size).rows(scene))
        embedding = UserALSRecall(scene_events, embedding_dim, neighbour_size)
        tables["user_als_u2u"].extend(embedding.rows(scene))
        vectors = embedding.dump_vectors()
        for user, vector in vectors.items():
            vector = np.pad(vector, (0, embedding_dim - len(vector)))
            norm = np.linalg.norm(vector)
            if norm:
                vector = vector / norm
            tables["user_als_emb"].append({"scene": scene, "user": user,
                                           "vector": vector.tolist()})
        tables["text_search"].extend(gen_text_search_data(
            scene_users, scene, ("name", "gender", "country", "city", "tags"), "user"))
    for name, rows in tables.items():
        columns = (["scene", "user", "vector"] if name == "user_als_emb" else
                   ["scene", "user", "text"] if name == "text_search" else
                   ["scene", "left_user", "right_user", "score"])
        pd.DataFrame(rows, columns=columns).to_csv(output / (name + ".csv"), index=False)


def generate(item_file, event_file, output_dir, i2i_size=20, recall_size=2000,
             embedding_dim=10, neighbour_size=50, user_file=None, include_text_embeddings=False):
    if not 1 <= embedding_dim <= 32:
        raise ValueError("embedding_dim must be between 1 and 32")
    users = pd.read_csv(user_file or Path(item_file).with_name("user.csv"))
    items = pd.read_csv(item_file, header=0)
    events = pd.read_csv(event_file, header=0)
    positive = events[events['type'].isin(['click', 'collect', 'buy'])].copy()
    if positive.empty:
        raise ValueError("recall generation requires click, collect, or buy events")
    output = Path(output_dir) / "item"
    output.mkdir(parents=True, exist_ok=True)
    gen_i2i_data(items, positive, i2i_size, str(output / 'item_cf_i2i.csv'))
    gen_content_i2i_data(items, i2i_size, str(output / 'content_i2i.csv'))
    gen_user_cf_data(positive, recall_size, neighbour_size,
                     str(output / 'user_cf_u2i.csv'))
    gen_embedding_data(items, positive, embedding_dim, str(output / 'item_seq_emb.csv'))
    if include_text_embeddings:
        generate_text_embeddings(item_file, output)
    gen_hot_data(positive, recall_size, str(output / 'hot.csv'))
    gen_new_data(items, recall_size, str(output / 'new.csv'))
    documents = [row for scene, group in items.groupby("scene")
                 for row in gen_text_search_data(group, scene, ("title", "category", "tags"), "item")]
    pd.DataFrame(documents, columns=["scene", "item", "text"]).to_csv(
        output / "text_search.csv", index=False)
    user_output = Path(output_dir) / "user"
    user_output.mkdir(parents=True, exist_ok=True)
    gen_user_data(users, positive, user_output, embedding_dim, neighbour_size)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--item', required=True)
    parser.add_argument('--user', help='defaults to user.csv beside item.csv')
    parser.add_argument('--event', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--i2i-size', type=int, default=20)
    parser.add_argument('--recall-size', type=int, default=2000)
    parser.add_argument('--embedding-dim', type=int, default=10)
    parser.add_argument('--neighbour-size', type=int, default=50)
    parser.add_argument("--include-text-embeddings", action="store_true",
                        help="generate Qwen title vectors; requires the text extra and model weights")
    args = parser.parse_args()
    generate(args.item, args.event, args.output, args.i2i_size, args.recall_size,
             args.embedding_dim, args.neighbour_size, args.user, args.include_text_embeddings)
