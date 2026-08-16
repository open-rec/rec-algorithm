# rec-algorithm

The offline side of open-rec: computes the recall tables and trains the rank model that the online
service serves. Runs on one machine (pandas / gensim / torch); a distributed version is future work.

It plays two roles:

- **a batch tool** — generate recall tables as CSV, which [example/init](https://github.com/open-rec/example/tree/master/init) loads into Redis and Elasticsearch
- **a library** — [rank-engine](https://github.com/open-rec/rank-engine) imports `LRModel`, `UserFeature` and `ItemFeature` from it at serving time, pinned as `rec-algorithm==0.0.1`

## install

```shell
pip install -r requirements.txt
pip install -e .
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
| new | `New` | `(pub_time / max_pub_time) ** 31`, so scores decay steeply with age |

`ItemEntityEmbedding` (content-based item embeddings) is stubbed out and not implemented.

All of them are computed **per scene** — group events by `scene` before constructing them, as
`gen_recall_data.py` does. i2i and embedding require triggers; hot and new do not.

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
scene and weight for items. Raw ids, names and titles are excluded because one-hot encoding them
explodes the tensor size. The consequence: **the input dimension depends on the cardinality of your
dataset**, so a checkpoint is only loadable against the data it was trained on. `rank-engine` requires
you to pass that `dim` explicitly when loading.

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

