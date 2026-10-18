# rec-algorithm v0.2.0

Released: 2026-10-18

Part of the coordinated OpenRec v0.2.0 source release.

## Changes since v0.1.0

- Reorganize Spark jobs and shared infrastructure while retaining cumulative mutation and feature contracts.
- Introduce bootstrap build 19 with model-family directories, explicit ALS edge/vector names and text_search tables.
- Make language-vector generation opt-in and support scheduled title embedding publication.

## Installation and compatibility

Install rec-algorithm==0.2.0 from source or build its wheel. Rebuild rank-engine and offline runners together. Spark uses 4.0.4/Scala 2.13 and Java 21. Activation remains owned by rec-console.

See the [distribution upgrade and rollback instructions](https://github.com/open-rec/openrec/blob/v0.2.0/RELEASE_NOTES.md) for the complete compatible component set and persisted configuration migrations.

## Validation and publication

The [validation record](https://github.com/open-rec/openrec/blob/v0.2.0/release/VALIDATION.md) distinguishes release checks from maintainer-reported prior CI and integration results. GitHub Actions publishes source archives and SHA-256 checksums; Maven Central, PyPI and versioned container registries are not populated by this workflow.
