ARG SPARK_IMAGE=openrec/spark:3.5.3
FROM ${SPARK_IMAGE}

USER root
WORKDIR /opt/openrec
COPY . /opt/openrec
RUN python3 -c "import shutil; shutil.make_archive('/tmp/rec-algorithm', 'zip', root_dir='/opt/openrec', base_dir='.')" \
    && mv /tmp/rec-algorithm.zip /opt/openrec/rec-algorithm.zip \
    && chmod -R a+rX /opt/openrec

USER spark
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3", "-m", "jobs.spark.runner"]
