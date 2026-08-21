# rec-algorithm

The offline side of open-rec: computes the recall tables and trains the rank model that the online
service serves. It supports both one-machine development (pandas / gensim / torch) and scheduled
cluster execution (Hive / Spark), with the same recall formulas and serving schemas.

It plays two roles:

- **a batch tool** — generate recall tables as CSV, which [example/init](https://github.com/open-rec/example/tree/master/init) loads into Redis and Elasticsearch
- **a library** — [rank-engine](https://github.com/open-rec/rank-engine) imports `LRModel`, `UserFeature` and `ItemFeature` from it at serving time, pinned as `rec-algorithm==0.0.1`

## install

```shell
pip install -r requirements.txt
pip install -e .
```

Install only what the selected execution mode needs:

```shell
pip install -e .                 # local library development
pip install -e ".[spark]"       # Hive/Spark jobs
pip install -e ".[publish]"     # Redis/Elasticsearch publishing
pip install -e ".[cluster]"     # complete distributed runtime
```

Two directories are required at runtime and both are gitignored, so create them yourself:

```shell
mkdir -p model data/test
cp ../example/data/test/*.csv data/test/     # if you have the example repo checked out
```

`model/` receives trained checkpoints, `data/` holds the CSV datasets.

Build the wheel that `rank-engine` depends on:

```shell
bash package.sh                              # -> dist/rec_algorithm-0.0.1-*.whl
```

## layout

| Path | Contents |
|---|---|
| `algorithm/recall/` | `i2i`, `hot`, `new`, `embedding` — all subclass `Recall` |
| `algorithm/rank/` | `LRModel` / `LRRecModel`, subclassing `RecModel` |
| `algorithm/feature/` | feature encoders for users and items |
| `algorithm/meta/` | table and column definitions — the source of truth for CSV headers |
| `algorithm/structure/` | `ScoreItem`, JSON helpers |
| `tool/` | dataset generation and recall dumping scripts |
| `test/` | pytest suites, doubling as usage examples |
| `jobs/spark/` | distributed Hive readers, recall formulas and scheduled job CLI |
| `publisher/` | Redis and Elasticsearch serving-layer publication |
| `conf/` | cluster configuration examples |

## execution modes

Local mode remains the fast feedback path: existing classes under `algorithm/` accept pandas
frames and produce `ScoreItem` objects or model artifacts. Spark mode is an execution layer, not a
second algorithm library: it reads the Hive entity contract, applies the same deduplication and
scoring rules, writes versionable Parquet/Hive results, and can publish them to the online stores.

```text
data-processor -> Hive ODS/DWD -> jobs.spark -> Hive/Parquet artifacts
                                            -> versioned Elasticsearch hot/new/i2i
                                            -> Elasticsearch embeddings
```

Run with the cluster's Spark installation:

```shell
spark-submit --master spark://spark-master:7077 \
  --py-files dist/rec_algorithm-0.0.1-py3-none-any.whl \
  jobs/spark/recall_job.py hot \
  --event-table openrec.event_entity \
  --date 2026-08-18 --output-table openrec.recall_hot --size 2000 --publish

spark-submit --master spark://spark-master:7077 \
  --py-files dist/rec_algorithm-0.0.1-py3-none-any.whl \
  jobs/spark/recall_job.py i2i \
  --event-table openrec.event_entity \
  --date 2026-08-18 --output-table openrec.recall_i2i --size 20 --publish

spark-submit --master spark://spark-master:7077 \
  --py-files dist/rec_algorithm-0.0.1-py3-none-any.whl \
  jobs/spark/recall_job.py embedding \
  --event-table openrec.event_entity \
  --date 2026-08-18 --output-table openrec.recall_embedding --vector-size 64 --publish \
  --es-password "$ELASTIC_PASSWORD" --es-ca-certs /path/to/ca.crt
```

Use `new` with `--item-table openrec.item_entity`. Output schemas are stable:

- hot/new: `scene, item, score`
- new additionally carries `publish_time` for the online time-window filter
- i2i: `scene, left_item, right_item, score`
- embedding: `scene, item, vector`

All source and result tables are partitioned by UTC `dt`. `--date` defaults to yesterday, so a
daily scheduler can invoke the same command without calculating a date; passing it explicitly is
recommended for backfills and reproducibility. Source reads use the cumulative warehouse snapshot
through `--date`, not just that day's partition: events are de-duplicated and accumulated, while
items are collapsed to their latest mutation and latest `DELETE` tombstones are excluded. Every
recall algorithm then semi-joins its events with this active item snapshot before scoring, so
deleted items do not consume hot/i2i/embedding resources or get published online.
In cluster mode the runner reads the daily ODS directories directly with `--event-path` and
`--item-path`. This preserves Hive-style partition discovery while avoiding the incompatible Hive 4
metastore API in Spark 3.5. Result data is written beneath `--output-path`, remains partitioned by
the requested day, and uses dynamic partition overwrite so rerunning one day does not rewrite other
result partitions. Table-based reads and writes remain available for compatible metastores.

With `--publish`, hot/new/i2i are staged as
`openrec-recall-{algorithm}-{YYYYMMDD}-{revision}`. `rec-console` creates the staging index before
Spark writes it, then verifies the document count, atomically moves
`openrec-recall-{algorithm}-active`, and removes versions beyond the configured retention after the
switch succeeds. The active version plus one previous physical index are retained by default. Use
`--revision r002` for a rerun that must coexist with r001, and `--max-index-versions` to change the
maximum number of loaded physical indexes. Scene remains a document field and query condition; it
is never part of the physical index name.

Embedding publishes one versioned physical index per scene as
`{scene}-item-vector-index-{YYYYMMDD}-{revision}` and atomically moves the legacy serving name
`{scene}-item-vector-index` as an alias after document-count validation. The previous version is
retained for rollback instead of deleting the active vector index before a replacement is ready.

Publishing the same date and revision again is idempotent when its document count matches. An
active physical index is never deleted or overwritten; use the next revision when recomputation
changes its contents. Index creation, activation, retention and rollback belong to `rec-console`;
this project only computes recall data and writes documents into the staging index authorized by
the console. In cluster mode Airflow calls the internal `rec-algorithm-runner` service, which owns
the Spark submission command while Airflow remains Docker-socket-free. Enable the
`openrec_daily_recall` DAG for the `02:00 UTC` schedule, or trigger it with
`{"revision":"r002"}` for a corrected daily release.
Each submitted runner job is capped at four Spark cores by default; set `RECALL_SPARK_CORES` for a
different deployment policy. Worker capacity itself is not artificially capped.

Emergency rollback does not rerun Spark or reload documents. POST
`{"algorithm":"i2i","target_index":"openrec-recall-i2i-20260819-r001"}` to the
internal `rec-console` endpoint `/api/recall/releases/rollback`; it atomically moves only the
active alias. When the target is omitted, the newest retained non-active index is selected.
The same operation is exposed in the Airflow UI as the manual
`openrec_recall_rollback` DAG. Trigger it with the same `algorithm` and optional `target_index`
configuration.

The cluster runner also exposes internal `POST /jobs/rank/train`. Its Spark job reads cumulative
event, item, and user partitions through `date`, joins interactions to the active item snapshot,
and hands the bounded prepared dataset to rank-engine for PyTorch training and evaluation. Rank
submissions default to four total executor cores (`RANK_SPARK_CORES=4`) and emit a version manifest
for the Airflow `openrec_rank_model` publish task.

`jobs.spark.rank.labelled_interactions` performs the large join against the same active item
snapshot, excluding deleted-item samples before deterministic train/validation splitting.
PyTorch/FeatureSpace training remains the model contract so its checkpoint is still
loadable by rank-engine; distributed sample preparation does not introduce an incompatible Spark
ML model format.

## generate data

Both scripts write to `../data/...` and must be run from `tool/`:

```shell
cd tool
python gen_test_data.py       # synthetic user / item / event CSVs
python gen_recall_data.py     # recall tables from those CSVs
```

`gen_recall_data.py` has the hot / new / embedding generators **commented out** — only i2i runs by
default. Uncomment the ones you need, and set the scene at the bottom of the file (`gen_scene_recall`
defaults to `douban`).

## recall

Every algorithm implements `recall(user_triggers, item_triggers)`; the i2i and embedding ones also
implement a `dump_*` method used to write the offline tables.

| Algorithm | Class | Method |
|---|---|---|
| i2i | `ItemBasedI2I` | item co-occurrence within a user's sequence, damped by `1/log(len+1)`, normalized by `sqrt(count_i * count_j)` |
| embedding | `EventEmbedding` | word2vec (gensim) over per-user item sequences; `dump_vectors` exports 10-dim vectors |
| hot | `Hot` | click counts normalized by the maximum |
| new | `New` | freshness min-max normalized over the observed `pub_time` range, raised to `power` (31) |

`ItemEntityEmbedding` (content-based item embeddings) is not implemented; its methods raise
`NotImplementedError`.

All of them are computed **per scene** — group events by `scene` before constructing them, as
`gen_recall_data.py` does. i2i and embedding require triggers; hot and new do not.

Two behaviours worth knowing:

- **i2i merges across triggers.** An item reachable from several triggers is emitted once, keeping its best score; triggers themselves are never recalled back; the result is truncated to `recall_size`. The similarity matrix is quadratic in sequence length, so it is computed once per instance and reused by `recall()` and `dump_i2i()`.
- **embedding tolerates unknown triggers.** `min_count=5` keeps rare items out of the word2vec vocabulary, so a trigger may be absent; those are skipped, and a request where none are known returns an empty list rather than raising.

## rank

`LRRecModel` wraps a torch logistic regression over concatenated user and item features.

```python
user_feature = UserFeature(users=users, events=events)
item_feature = ItemFeature(items=items, events=events)

lr_model = LRRecModel(user_feature=user_feature, item_feature=item_feature, events=events)
lr_model.train(epoch_num=10, batch_size=256, learning_rate=0.003)
lr_model.save()                    # -> model/lr.pth

lr_model.load()
lr_model.score("user_0", ["item_0", "item_1"])
```

Labels come from the event type: `click` is 1, `expose` is 0; other types are dropped.

Features used are deliberately a subset — country, city, gender, age and tags for users; category,
scene and weight for items — plus the event snapshot statistics described below. Raw ids, names and titles are excluded because one-hot encoding them
explodes the tensor size. The categorical input dimension still depends on the training vocabulary,
so `FeatureSpace` is saved beside every checkpoint and must be loaded by `rank-engine`.

### materialize online features

Create flat user/item snapshots that can be imported into the online `user:*` and `item:*` tables:

```shell
python tool/gen_feature_data.py \
  --user ../example/data/test/user.csv \
  --item ../example/data/test/item.csv \
  --event ../example/data/test/event.csv \
  --output data/feature/test
```

The output keeps all entity columns and adds event count/value statistics, active days, distinct
scene and counterpart counts, first/last time, recency, 1/7/30-day counts, fixed event-type counts
(`click`, `expose`, `buy`, `collect`, `stay`) and click rate. The default snapshot time is the newest
event; production jobs should pass `--as-of-time` explicitly. Use a snapshot before the model label
period to avoid target leakage.

Serve a trained checkpoint with [rank-engine](https://github.com/open-rec/rank-engine); pre-trained
Douban artifacts live in [model](https://github.com/open-rec/model).

## test

The suites read their CSVs by paths **relative to the current working directory**, not to the test
file, so they only pass when run from the test's own directory:

```shell
cd test/algorithm/recall && pytest test_i2i.py
cd test/algorithm/rank   && pytest test_lr.py::test_train      # writes model/lr.pth
cd test/algorithm/rank   && pytest test_lr.py::test_inference  # reads it back
```

They expect the dataset at `data/test/` relative to the repo root (see install above).

`test_i2i_merge.py` is the exception: it stubs out the only pandas-backed method and asserts the
recall contract (truncation, dedup across triggers, ordering, caching) on fixed sequences, so it
needs neither the CSVs nor a particular working directory:

```shell
pytest test/algorithm/recall/test_i2i_merge.py
```

# data structure

## item

| name        | type   | required | description                          |
|-------------|--------|----------|--------------------------------------|  
| id          | string | yes      | item uniq id                         |  
| title       | string | yes      |                                      |
| category    | string | yes      | single value                         |  
| tags        | string | no       | multi value                          |  
| scene       | string | yes      | relation recommend or guess you like |  
| pub_time    | int    | yes      |                                      |  
| modify_time | int    | no       | update time                          |  
| expire_time | int    | no       |                                      |  
| status      | bool   | yes      | could be recommend                   |  
| weight      | int    | no       |                                      |  
| ext_fields  | json   | no       |                                      |  

## user

| name          | type   | required | description    |
|---------------|--------|----------|----------------|  
| id            | string | yes      | user uniq id   |  
| device_id     | string | yes      | user device id |
| name          | string | no       | fake name      |  
| gender        | string | no       |                |  
| age           | int    | no       |                |  
| country       | string | no       |                |  
| city          | string | no       | update time    |  
| phone         | long   | no       |                |  
| tags          | string | no       | multi value    |  
| register_time | int    | no       |                |
| login_time    | int    | no       |                |
| ext_fields    | json   | no       |                |

## event

| name       | type   | required | description                            |
|------------|--------|----------|----------------------------------------|  
| id         | string | yes      | event uniq id                          |  
| user_id    | string | yes      |                                        |
| item_id    | string | yes      | single value                           |  
| trace_id   | string | yes      | eg: openrec                            |  
| scene      | string | yes      | relation recommend or guess you like   |  
| type       | string | yes      | eg: click, expose, buy, collect, stay  |  
| value      | string | yes      | the value of the type                  |  
| time       | int    | yes      |                                        |  
| is_login   | bool   | no       | could be recommend                     |  
| ext_fields | json   | no       |                                        |  
