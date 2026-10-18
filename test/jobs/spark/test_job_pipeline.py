"""Run with pytest or spark-submit; Spark imports required, no live services."""

import unittest
from unittest.mock import MagicMock, patch

try:
    import pyspark
except ImportError:
    pyspark = None


@unittest.skipIf(pyspark is None, "requires pyspark")
class RecallPipelineTests(unittest.TestCase):
    def test_extension_uses_shared_outputs(self):
        from jobs.spark.recall import job
        from jobs.spark.recall.catalog import RecallSpec

        spec = RecallSpec("unused:compute", "custom")
        args = job.parser().parse_args(["hot", "--date", "2026-10-18", "--publish",
                                       "--output-path", "/tmp/custom"])
        args.algorithm = "custom"
        frame = MagicMock()
        with patch.dict(job.RECALL_JOBS, custom=spec), \
                patch.object(RecallSpec, "compute", return_value=frame) as compute, \
                patch.object(job.F, "lit", return_value="business-date"), \
                patch.object(job, "write_result") as write, \
                patch.object(job, "publish_result") as publish:
            result = job.run(args, MagicMock())
        self.assertIs(result, frame.withColumn.return_value)
        self.assertEqual(compute.call_count, 1)
        write.assert_called_once_with(result, None, "/tmp/custom", "overwrite")
        publish.assert_called_once_with(result, spec, args)

    def test_cached_output_released_when_publication_fails(self):
        from jobs.spark.recall import job
        from jobs.spark.recall.catalog import RecallSpec

        args = job.parser().parse_args(["query_title_emb", "--publish"])
        frame = MagicMock()
        with patch.object(RecallSpec, "compute", return_value=frame), \
                patch.object(job.F, "lit"), \
                patch.object(job, "publish_result", side_effect=RuntimeError("publish failed")):
            with self.assertRaisesRegex(RuntimeError, "publish failed"):
                job.run(args, MagicMock())
        frame.withColumn.return_value.persist.return_value.unpersist.assert_called_once()

    def test_inputs_load_once_and_filter_deleted_items(self):
        from jobs.spark.recall.inputs import RecallInputs
        from jobs.spark.recall.job import parser

        with patch("jobs.spark.recall.inputs.read_items") as items, \
                patch("jobs.spark.recall.inputs.read_events") as events, \
                patch("jobs.spark.recall.inputs.read_users") as users, \
                patch("jobs.spark.recall.inputs.keep_active_item_events") as active:
            inputs = RecallInputs(MagicMock(), parser().parse_args(["hot"]))
            self.assertIs(inputs.events, inputs.events)
            self.assertIs(inputs.items, items.return_value)
            self.assertEqual(users.call_count, 0)
            items.assert_called_once()
            events.assert_called_once()
            active.assert_called_once_with(events.return_value, items.return_value)

    def test_all_catalog_handlers_resolve(self):
        from importlib import import_module
        from jobs.spark.recall.catalog import RECALL_JOBS

        for spec in RECALL_JOBS.values():
            module, name = spec.handler.split(":")
            self.assertTrue(callable(getattr(import_module(module), name)))

    def test_runner_arguments_parse_at_actual_job_entry_points(self):
        from pathlib import Path
        from jobs.spark.recall import job as recall
        from jobs.spark.ranking import job as ranking
        from jobs.spark.analytics import job as analytics
        from jobs.spark.recall.catalog import RECALL_JOBS
        from jobs.spark.runner import recall_command, rank_command, analytics_command

        cases = [
            (recall_command, recall, "recall/job.py",
             {"algorithm": name, "date": "2026-10-18"})
            for name in RECALL_JOBS
        ]
        cases += [
            (rank_command, ranking, "ranking/job.py",
             {"date": "2026-10-18", "model_type": model, "target_type": target})
            for model in ("lr", "fm", "lightgbm") for target in ("item", "user")
        ]
        cases += [
            (recall_command, recall, "recall/job.py",
             {"algorithm": "text_search", "target_type": "user", "date": "2026-10-18"}),
            (analytics_command, analytics, "analytics/job.py",
             {"date_from": "2026-10-17", "date_to": "2026-10-18"}),
        ]
        for command_builder, module, script, payload in cases:
            with self.subTest(script=script, payload=payload):
                command = command_builder(payload)
                script_index = command.index("/opt/openrec/jobs/spark/" + script)
                args = module.parser().parse_args(command[script_index + 1:])
                for key, value in payload.items():
                    self.assertEqual(getattr(args, key), value)
                self.assertEqual(Path(module.__file__).parts[-2:], tuple(script.split("/")))

    def test_job_files_can_be_executed_as_scripts(self):
        import contextlib
        import io
        import runpy
        from jobs.spark.recall import job as recall
        from jobs.spark.ranking import job as ranking
        from jobs.spark.analytics import job as analytics

        for module in (recall, ranking, analytics):
            with self.subTest(module=module.__name__), \
                    patch("sys.argv", [module.__file__, "--help"]), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(module.__file__, run_name="__main__")
                self.assertEqual(result.exception.code, 0)
                self.assertIn("usage:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
