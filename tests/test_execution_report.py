import asyncio
import enum
import inspect
import runpy
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "ComfyUI-DINKIssTyle/dinki_execution_report.py"


class Result(enum.Enum):
    SUCCESS = 0
    PENDING = 1
    FAILURE = 2


class Node:
    def __init__(self, function, *, lazy=False):
        self.function = function
        self.lazy = lazy


class InterruptProcessingException(Exception):
    pass


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.lib = runpy.run_path(str(SOURCE))
        self.env = self.lib["install_execution_report_hooks"].__globals__
        self.now = 100.0
        self.clock = patch.object(self.env["time"], "perf_counter", lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.events = []
        self.server = types.SimpleNamespace(send_sync=lambda *args: self.events.append(args))

    def advance(self, seconds):
        self.now += seconds

    def prompt(self, extra=None, enabled=True, sort_by="Execution order"):
        return {"report": {"class_type": "DINKI_Execution_Report", "inputs": {"enabled": enabled, "sort_by": sort_by}},
                **(extra or {})}

    def fake_core(self, plan):
        """Exercise the same pending-task and lazy-input boundaries as ComfyUI."""
        core = types.SimpleNamespace()
        advance = self.advance

        async def get_output_data(prompt_id, unique_id, obj, input_data_all, **kwargs):
            value = obj.function()
            if inspect.isawaitable(value):
                task = asyncio.create_task(value)
                await asyncio.sleep(0)
                if not task.done():
                    return [task], {}, False, True
                value = task.result()
            return [value], {}, False, False

        async def execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id, pending_async_nodes):
            obj = caches.get(current_item)
            if obj == "cached":
                return Result.SUCCESS, None, None
            if obj.lazy:
                # progress/executing has already been emitted at this point.
                advance(0.1)
                return Result.PENDING, None, None
            output = await core.get_output_data(prompt_id, current_item, obj, {})
            if output[3]:
                pending_async_nodes[current_item] = output[0]
                return Result.PENDING, None, None
            executed.add(current_item)
            return Result.SUCCESS, None, None

        class Executor:
            def __init__(executor, server):
                executor.server = server
                executor.success = True
                executor.status_messages = []
                executor.tasks_seen = []

            def execute(self, prompt, prompt_id, extra_data=None):
                executor = self
                return asyncio.run(executor.execute_async(prompt, prompt_id, extra_data))

            async def execute_async(self, prompt, prompt_id, extra_data=None):
                executor = self
                executor.status_messages = []
                executor.success = True
                executed = set()
                pending = {}
                dynprompt = types.SimpleNamespace(get_node=lambda node_id: prompt.get(node_id, {}))
                for current_item, obj in plan(prompt_id):
                    try:
                        await core.execute(executor.server, dynprompt, {current_item: obj}, current_item,
                                           extra_data, executed, prompt_id, pending)
                    except Exception as error:
                        executor.success = False
                        executor.status_messages.append(("execution_interrupted" if isinstance(error, InterruptProcessingException) else "execution_error", {}))
                        break
                tasks = [task for values in pending.values() for task in values]
                executor.tasks_seen = tasks
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    for task in tasks:
                        if task.cancelled() or task.exception() is not None:
                            executor.success = False
                            executor.status_messages.append(("execution_error", {}))
                if executor.success:
                    executor.status_messages.append(("execution_success", {}))
                executor.history_result = {"outputs": {}, "meta": {}}
                return "native-result"

        core.PromptExecutor = Executor
        core.get_output_data = get_output_data
        core.execute = execute
        self.assertTrue(self.lib["install_execution_report_hooks"](core))
        return core

    def final(self, prompt_id="job"):
        return [payload for event, payload, client in self.events if payload["prompt_id"] == prompt_id][-1]

    def run_core(self, core, prompt=None, prompt_id="job", client="client"):
        executor = core.PromptExecutor(self.server)
        result = executor.execute(prompt or self.prompt(), prompt_id, {"client_id": client})
        self.assertEqual(result, "native-result")
        return executor

    def timed(self, seconds, result="unchanged"):
        def call():
            self.advance(seconds)
            return result
        return Node(call)

    def test_actual_calls_exclude_lazy_wait_and_unused_branches(self):
        branch = Node(lambda: None, lazy=True)
        core = self.fake_core(lambda _: [("branch", branch), ("sampler", self.timed(8)),
                                        ("branch", self.timed(0.2)), ("decode", self.timed(2))])
        prompt = self.prompt({key: {"class_type": key, "inputs": {}} for key in ("branch", "sampler", "decode", "unused")})
        self.run_core(core, prompt)
        final = self.final()
        self.assertEqual(final["status"], "Completed")
        self.assertEqual([row["node_id"] for row in final["rows"]], ["sampler", "branch", "decode"])
        self.assertAlmostEqual(final["total_seconds"], 10.2)
        self.assertAlmostEqual(final["elapsed_seconds"], 10.3)
        self.assertAlmostEqual(final["rows"][1]["seconds"], 0.2)
        self.assertEqual(self.env["_session"].get(), None)
        self.assertEqual(self.env["_node_scope"].get(), None)

    def test_repeated_calls_sum_and_nested_ids_remain_distinct(self):
        core = self.fake_core(lambda _: [("105:8", self.timed(2)), ("106:8", self.timed(3)), ("105:8", self.timed(1))])
        self.run_core(core)
        rows = self.final()["rows"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["calls"], 2)
        self.assertEqual(rows[0]["seconds"], 3)
        self.assertEqual(rows[1]["node_id"], "106:8")

    def test_only_visited_cache_hits_are_reported_without_old_duration(self):
        core = self.fake_core(lambda _: [("cached", "cached"), ("worker", self.timed(4))])
        self.run_core(core, self.prompt({"cached": {"class_type": "Loader"}, "unused_cached": {"class_type": "Other"}}))
        rows = self.final()["rows"]
        self.assertEqual(rows[0]["status"], "Cached")
        self.assertEqual(rows[0]["seconds"], 0)
        self.assertEqual(self.final()["total_seconds"], 4)
        self.assertNotIn("unused_cached", self.final()["reports"][0]["markdown"])

    def test_async_task_identity_and_parallel_scheduling_are_preserved(self):
        tasks = []
        async def work():
            tasks.append(asyncio.current_task())
            await asyncio.sleep(0)
            self.advance(5)
            return "async-output"
        core = self.fake_core(lambda _: [("async", Node(work)), ("sync", self.timed(2))])
        executor = self.run_core(core)
        self.assertIs(executor.tasks_seen[0], tasks[0])
        rows = self.final()["rows"]
        self.assertEqual(rows[0]["seconds"], 7)
        self.assertEqual(rows[1]["seconds"], 2)
        self.assertEqual(self.final()["total_seconds"], 9)
        self.assertEqual(self.final()["elapsed_seconds"], 7)

    def test_failed_async_task_records_partial_duration_and_error(self):
        async def fail():
            await asyncio.sleep(0)
            self.advance(3)
            raise ValueError("native-error")
        core = self.fake_core(lambda _: [("async", Node(fail))])
        self.run_core(core)
        self.assertEqual(self.final()["status"], "Error")
        self.assertEqual(self.final()["rows"][0]["status"], "Error")
        self.assertEqual(self.final()["total_seconds"], 3)

    def test_sync_error_and_interruption_produce_partial_reports(self):
        for error, status in [(ValueError("test"), "Error"), (InterruptProcessingException(), "Interrupted")]:
            def fail():
                self.advance(2)
                raise error
            core = self.fake_core(lambda _: [("done", self.timed(1)), ("failed", Node(fail))])
            self.run_core(core)
            self.assertEqual(self.final()["status"], status)
            self.assertEqual(self.final()["rows"][1]["status"], status)
            self.assertEqual(self.final()["total_seconds"], 3)

    def test_reports_go_only_to_originating_client_and_enter_history(self):
        core = self.fake_core(lambda _: [("worker", self.timed(2))])
        executor = self.run_core(core)
        self.assertEqual(len(self.events), 2)
        self.assertTrue(all(client == "client" for _, _, client in self.events))
        saved = executor.history_result["outputs"]["report"]["dkst_execution_report"][0]
        self.assertEqual(saved["markdown"], self.final()["reports"][0]["markdown"])
        self.assertEqual(saved["prompt_id"], "job")

    def test_separate_runs_and_multiple_report_nodes_keep_independent_state(self):
        core = self.fake_core(lambda job: [("worker", self.timed(1 if job == "a" else 3))])
        prompt = self.prompt({"report2": {"class_type": "DINKI_Execution_Report", "inputs": {"sort_by": "Processing time"}}})
        executor = core.PromptExecutor(self.server)
        executor.execute(prompt, "a", {"client_id": "a-client"})
        executor.execute(prompt, "b", {"client_id": "b-client"})
        self.assertEqual(self.final("a")["total_seconds"], 1)
        self.assertEqual(self.final("b")["total_seconds"], 3)
        self.assertEqual(len(self.final("b")["reports"]), 2)

    def test_disabled_or_missing_report_node_leaves_execution_untimed(self):
        core = self.fake_core(lambda _: [("worker", self.timed(1))])
        self.run_core(core, self.prompt(enabled=False))
        self.run_core(core, {"worker": {"class_type": "worker"}})
        self.assertEqual(self.events, [])
        self.assertIsNone(self.env["_session"].get())

    def test_headless_run_keeps_report_in_history(self):
        core = self.fake_core(lambda _: [("worker", self.timed(1))])
        executor = self.run_core(core, client=None)
        self.assertEqual(self.events, [])
        self.assertIn("report", executor.history_result["outputs"])

    def test_install_is_idempotent_and_preserves_signatures(self):
        core = self.fake_core(lambda _: [("worker", self.timed(1))])
        first = core.get_output_data
        self.assertTrue(self.lib["install_execution_report_hooks"](core))
        self.assertIs(first, core.get_output_data)
        self.assertIn("obj", inspect.signature(first).parameters)
        self.run_core(core)
        self.assertEqual(self.final()["total_seconds"], 1)

    def test_unsupported_api_does_not_install_partial_wrappers(self):
        core = types.SimpleNamespace(PromptExecutor=types.SimpleNamespace(execute=lambda wrong: None),
                                     execute=lambda wrong: None, get_output_data=lambda wrong: None)
        original = core.get_output_data
        with self.assertLogs(level="WARNING"):
            self.assertFalse(self.lib["install_execution_report_hooks"](core))
        self.assertIs(core.get_output_data, original)

    def test_markdown_titles_are_escaped_sorted_and_zero_total_is_defined(self):
        rows = [{"node_id": "1", "name": "<script>|\nname", "seconds": 1, "status": "Completed"},
                {"node_id": "2", "name": "Long", "seconds": 9, "status": "Completed"}]
        markdown = self.lib["report_markdown"](rows, 10, 12, "Completed", "Processing time")
        self.assertLess(markdown.index("Long"), markdown.index("name"))
        self.assertIn("90.00%", markdown)
        self.assertIn("&lt;script&gt;\\| name", markdown)
        empty = self.lib["report_markdown"]([], 0, 0, "Completed", "Execution order")
        self.assertNotIn("nan", empty)
        self.assertNotIn("100.00%", empty)

    def test_closed_session_ignores_late_task_callbacks(self):
        async def run():
            session = self.lib["TimingSession"](self.prompt(), "job", self.server, "client", {"report": "Execution order"})
            call = session.begin("worker")
            future = asyncio.get_running_loop().create_future()
            self.lib["_observe_result"](session, call, ([future], {}, False, True))
            self.advance(2)
            session.close("Interrupted")
            snapshot = session.rows["worker"]["seconds"]
            self.advance(10)
            future.set_result("done")
            await asyncio.sleep(0)
            self.assertEqual(session.rows["worker"]["seconds"], snapshot)
        asyncio.run(run())

    def test_send_failure_does_not_change_native_result(self):
        self.server.send_sync = lambda *args: (_ for _ in ()).throw(RuntimeError("socket closed"))
        core = self.fake_core(lambda _: [("worker", self.timed(2))])
        self.run_core(core)

    def test_synchronous_executor_and_old_output_signature_are_supported(self):
        core = types.SimpleNamespace()
        advance = self.advance
        server = self.server
        def get_output_data(obj, input_data_all):
            advance(2)
            return [["unchanged-output"]], {}, False
        def execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id):
            output = core.get_output_data(object(), {})
            executed.add(current_item)
            return Result.SUCCESS, output, None
        class Executor:
            def __init__(self):
                self.server = server
                self.success = True
            def execute(self, prompt, prompt_id, extra_data=None):
                result = core.execute(self.server, None, None, "worker", extra_data, set(), prompt_id)
                self.history_result = {"outputs": {}, "meta": {}}
                return result
        core.PromptExecutor, core.execute, core.get_output_data = Executor, execute, get_output_data
        self.assertTrue(self.lib["install_execution_report_hooks"](core))
        result = core.PromptExecutor().execute(self.prompt(), "job", {"client_id": "client"})
        self.assertEqual(result[1][0], [["unchanged-output"]])
        self.assertEqual(self.final()["total_seconds"], 2)

    def test_escaped_workflow_exception_keeps_original_error_and_previous_history(self):
        core = self.fake_core(lambda _: [])
        error = RuntimeError("original failure")
        async def execute_async(self, prompt, prompt_id, extra_data=None):
            raise error
        core.PromptExecutor.execute_async = self.lib["_workflow_wrapper"](execute_async)
        executor = core.PromptExecutor(self.server)
        previous = {"outputs": {"previous": "leave alone"}, "meta": {}}
        executor.history_result = previous
        with self.assertRaises(RuntimeError) as caught:
            executor.execute(self.prompt(), "job", {"client_id": "client"})
        self.assertIs(caught.exception, error)
        self.assertEqual(previous["outputs"], {"previous": "leave alone"})
        self.assertEqual(self.final()["status"], "Error")
        self.assertIsNone(self.env["_session"].get())

    def test_only_cached_workflow_has_zero_total_without_division_error(self):
        core = self.fake_core(lambda _: [("cached", "cached"), ("report", "cached")])
        self.run_core(core)
        self.assertEqual(self.final()["total_seconds"], 0)
        self.assertEqual(len(self.final()["rows"]), 1)
        self.assertIn("0.00%", self.final()["reports"][0]["markdown"])

    def test_node_title_takes_precedence_over_default_display_name(self):
        core = self.fake_core(lambda _: [("worker", self.timed(1))])
        prompt = self.prompt({"worker": {"class_type": "worker", "_meta": {"title": "내 디코더"}}})
        self.run_core(core, prompt)
        self.assertEqual(self.final()["rows"][0]["name"], "내 디코더")

    def test_pending_async_cancellation_is_recorded_as_interrupted(self):
        async def run():
            session = self.lib["TimingSession"](self.prompt(), "job", self.server, "client", {"report": "Execution order"})
            call = session.begin("worker")
            task = asyncio.get_running_loop().create_future()
            self.lib["_observe_result"](session, call, ([task], {}, False, True))
            self.advance(1)
            task.cancel()
            await asyncio.sleep(0)
            self.assertEqual(session.rows["worker"]["status"], "Interrupted")
            self.assertEqual(session.rows["worker"]["seconds"], 1)
        asyncio.run(run())

    def test_silently_blocked_nodes_are_omitted_while_mixed_batches_are_timed(self):
        class Block: pass
        check = self.lib["_has_processing_inputs"]
        self.assertFalse(check(object(), {"a": [Block()], "b": [1, 2]}, Block))
        self.assertTrue(check(object(), {"a": [Block(), 3], "b": [1]}, Block))
        self.assertFalse(check(types.SimpleNamespace(INPUT_IS_LIST=True), {"a": [1, Block()]}, Block))
        async def run():
            async def get_output_data(obj, input_data_all):
                return [[Block()]], {}, False, False
            session = self.lib["TimingSession"](self.prompt(), "job", self.server, "client", {"report": "Execution order"})
            token = self.env["_session"].set(session)
            scope = self.env["_node_scope"].set((session, "blocked", None))
            try:
                wrapped = self.lib["_output_wrapper"](get_output_data, Block)
                result = await wrapped(object(), {"a": [Block()]})
                self.assertIsInstance(result[0][0][0], Block)
                self.assertEqual(session.rows, {})
            finally:
                self.env["_node_scope"].reset(scope)
                self.env["_session"].reset(token)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
