ARG SPARK_IMAGE=openrec/spark:3.5.3
ARG TRAINING_BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
FROM ${TRAINING_BASE_IMAGE} AS training
ARG TRAINING_PIP_INDEX_URL=https://pypi.org/simple
COPY requirements-training.txt /tmp/requirements-training.txt
RUN python -m pip install --no-cache-dir --index-url ${TRAINING_PIP_INDEX_URL} \
    -r /tmp/requirements-training.txt

FROM ${SPARK_IMAGE}

USER root
# Keep Spark's driver/executor Python unchanged. The Spark job starts its
# CPU trainer with this independent interpreter on the offline driver host.
COPY --from=training /opt/conda /opt/conda
ENV RANK_TRAINING_PYTHON=/opt/conda/bin/python \
    RANK_TRAINING_THREADS=2 \
    OPENREC_MODEL_HOME=/models
WORKDIR /opt/openrec
COPY . /opt/openrec
RUN python3 -c "import shutil; shutil.make_archive('/tmp/rec-algorithm', 'zip', root_dir='/opt/openrec', base_dir='.')" \
    && mv /tmp/rec-algorithm.zip /opt/openrec/rec-algorithm.zip \
    && chmod -R a+rX /opt/openrec

USER spark
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3", "-m", "jobs.spark.runner"]
