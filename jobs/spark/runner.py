"""Internal HTTP gateway that lets Airflow submit OpenRec PySpark recall jobs."""

import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ALGORITHMS = ("hot", "new", "i2i", "embedding")
JOB_LOCK = threading.Lock()


def recall_command(payload):
    algorithm = payload.get("algorithm")
    business_date = payload.get("date")
    revision = payload.get("revision", "r001")
    if algorithm not in ALGORITHMS:
        raise ValueError("algorithm must be hot, new, i2i or embedding")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", business_date or ""):
        raise ValueError("date must use YYYY-MM-DD")
    if not re.match(r"^r\d{3,}$", revision):
        raise ValueError("revision must look like r001")
    output_table = payload.get("output_table", "openrec.recall_%s" % algorithm)
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", output_table):
        raise ValueError("invalid output table")
    command = [
        "/opt/spark/bin/spark-submit",
        "--master", os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077"),
        "--deploy-mode", "client",
        "--total-executor-cores", os.environ.get("RECALL_SPARK_CORES", "4"),
        "--py-files", "/opt/openrec/rec-algorithm.zip",
        "/opt/openrec/jobs/spark/recall_job.py", algorithm,
        "--date", business_date,
        "--revision", revision,
        "--event-path", os.environ.get(
            "OPENREC_EVENT_PATH", "hdfs://namenode:8020/openrec/hive/event"),
        "--item-path", os.environ.get(
            "OPENREC_ITEM_PATH", "hdfs://namenode:8020/openrec/hive/item"),
        "--output-path", os.environ.get(
            "OPENREC_RECALL_PATH", "hdfs://namenode:8020/openrec/hive/recall") + "/" + algorithm,
        "--publish",
        "--es-host", "https://elasticsearch:9200",
        "--es-user", "elastic",
        "--es-password", os.environ.get("ELASTIC_PASSWORD", "openrec-es-password"),
        "--console-url", os.environ.get("REC_CONSOLE_URL", "http://rec-console:8095"),
        "--max-index-versions", str(payload.get("max_index_versions", 2)),
    ]
    if algorithm == "embedding":
        command.extend(["--vector-size", str(payload.get("vector_size", 10)),
                        "--min-count", str(payload.get("min_count", 1))])
    if algorithm == "i2i":
        command.extend(["--size", str(payload.get("size", 20))])
    else:
        command.extend(["--size", str(payload.get("size", 1000))])
    return command


class Handler(BaseHTTPRequestHandler):
    def _write(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self._write(404, {"error": "not found"})
            return
        self._write(200, {"status": "ok", "busy": JOB_LOCK.locked()})

    def do_POST(self):
        if self.path != "/jobs/recall":
            self._write(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            command = recall_command(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._write(400, {"error": str(error)})
            return
        if not JOB_LOCK.acquire(blocking=False):
            self._write(409, {"error": "another recall job is running"})
            return
        try:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, timeout=7200)
            if process.returncode != 0:
                self._write(500, {"error": "spark job failed", "output": process.stdout[-20000:]})
                return
            self._write(200, {"status": "success", "output": process.stdout[-20000:]})
        except subprocess.TimeoutExpired as error:
            self._write(504, {"error": "spark job timed out", "output": (error.stdout or "")[-20000:]})
        except ValueError as error:
            self._write(400, {"error": str(error)})
        except Exception as error:
            self._write(500, {"error": "recall operation failed: %s" % error})
        finally:
            JOB_LOCK.release()

    def log_message(self, message, *args):
        print("%s - %s" % (self.address_string(), message % args), flush=True)


def main():
    port = int(os.environ.get("RECALL_RUNNER_PORT", "8090"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
