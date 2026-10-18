"""
Offline gateway for feature capabilities and OpenRec Spark jobs.
"""

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jobs.spark.recall.command import recall_command
from jobs.spark.ranking.command import rank_command
from jobs.spark.analytics.command import analytics_command


JOB_LOCK = threading.Lock()


# command builder, stdout marker, response key, missing-result message
JOB_ROUTES = {
    "/jobs/recall": (recall_command, None, None, None),
    "/jobs/rank/train": (rank_command, "OPENREC_MODEL_MANIFEST=", "manifest",
                         "training manifest is missing"),
    "/jobs/analytics": (analytics_command, "OPENREC_ANALYTICS=", "analytics",
                        "analytics result is missing"),
}


def job_result(path, stdout):
    result = {"status": "success", "output": stdout[-20000:]}
    _, marker, key, missing = JOB_ROUTES[path]
    if marker:
        lines = [line for line in stdout.splitlines() if line.startswith(marker)]
        if not lines:
            raise RuntimeError(missing)
        result[key] = json.loads(lines[-1][len(marker):])
    return result


class Handler(BaseHTTPRequestHandler):
    def _write(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/features":
            from algorithm.feature.feature_catalog import feature_catalog

            self._write(200, {"code": 0, "data": feature_catalog()})
            return
        if self.path != "/health":
            self._write(404, {"error": "not found"})
            return
        self._write(200, {"status": "ok", "busy": JOB_LOCK.locked()})

    def do_POST(self):
        if self.path == "/features/validate":
            from algorithm.feature.feature_catalog import select_training_features

            try:
                length = int(self.headers.get("Content-Length", "0"))
                info = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(info, dict):
                    raise ValueError("configuration must be an object")
                selection = select_training_features(
                    info.get("model_type", "lr"),
                    info.get("target_type", "item"),
                    info.get("feature_selection"),
                )
                self._write(200, {"code": 0, "data": selection})
            except (ValueError, TypeError) as error:
                self._write(422, {"detail": str(error)})
            return
        if self.path not in JOB_ROUTES:
            self._write(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("configuration must be an object")
            command = JOB_ROUTES[self.path][0](payload)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._write(400, {"error": str(error)})
            return
        if not JOB_LOCK.acquire(blocking=False):
            self._write(409, {"error": "another recall job is running"})
            return
        try:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=7200,
            )
            if process.returncode != 0:
                self._write(
                    500,
                    {
                        "error": "spark job failed",
                        "output": process.stdout[-20000:],
                    },
                )
                return
            try:
                result = job_result(self.path, process.stdout)
            except RuntimeError as error:
                self._write(500, {"error": str(error)})
                return
            self._write(200, result)
        except subprocess.TimeoutExpired as error:
            self._write(
                504,
                {
                    "error": "spark job timed out",
                    "output": (error.stdout or "")[-20000:],
                },
            )
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
