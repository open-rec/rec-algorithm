# rec-algorithm v0.1.0

Released: 2026-10-05

Offline algorithms and training. First coordinated OpenRec source release.

## Features

- Local and Spark recall jobs for collaborative filtering, content similarity, popularity, recency, embeddings and sparse retrieval.
- Canonical feature definitions, fitted model sidecars, point-in-time training samples and temporal validation.
- LR, FM and LightGBM training, evaluation gates, selected feature sets and versioned artifacts.
- Cumulative Hive/HDFS mutation readers, distributed analytics and a Spark submission API for Airflow.
- Staged Elasticsearch publication and validated default model/recall bundle generation.

## Installation and compatibility

The Python distribution is `rec-algorithm==0.1.0` (previous development builds used `0.0.1`). Rebuild consumers such as rank-engine and the offline runner together. Spark uses 4.0.4/Scala 2.13 with Java 21.

Training does not activate releases; rec-console owns activation. Research encoders and Transformer experiments do not imply support in the production training/serving pipeline. Bootstrap training does not guarantee byte-identical checkpoints across runs.

## Validation and known boundaries

See this repository's README for build/test commands and deployment requirements. The coordinated release's [validation record](https://github.com/open-rec/openrec/blob/v0.1.0/release/VALIDATION.md) distinguishes checks executed for this release from historical integration evidence.

This initial release establishes a versioned source baseline. Source archives and checksums are published; external package registries and container registries are not populated by the source-release workflow. Upgrade the complete compatible distribution, retain data/checkpoints/artifacts, and preserve prior component refs for rollback.
