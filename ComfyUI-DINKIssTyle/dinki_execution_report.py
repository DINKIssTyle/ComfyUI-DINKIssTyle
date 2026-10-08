"""Opt-in node latency reports. Observe execution without changing its scheduling.

get_output_data runs after lazy inputs resolve, unlike progress start events.
Pending async Tasks are observed with callbacks, never awaited by this extension.
No ComfyUI files, node classes, node return values, or progress events are modified.
"""

import asyncio
import contextvars
import functools
import html
import inspect
import logging
import re
import time


NODE_TYPE = "DINKI_Execution_Report"
EVENT = "dkst.execution_report"
_session = contextvars.ContextVar("dkst_execution_report_session", default=None)
_node_scope = contextvars.ContextVar("dkst_execution_report_node", default=None)
HOOKS_AVAILABLE = False
logger = logging.getLogger(__name__)


def _cell(value):
    # Titles are user data: keep them inside a single Markdown table cell.
    text = html.escape(str(value), quote=False).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\|`*_{}\[\]()#!])", r"\\\1", text)


def report_markdown(rows, total, elapsed, status, sort_by):
    ordered = sorted(rows, key=lambda row: -row["seconds"]) if sort_by == "Processing time" else rows
    lines = ["## Execution Report", "", f"Status: **{status}**", "",
             "| Node | Processing time | Share |", "| :--- | ---: | ---: |"]
    for row in ordered:
        label = _cell(f'{row["name"]} · #{row["node_id"]}')
        if row["status"] == "Cached":
            duration, share = "Cached", "—"
        else:
            duration = f'{row["seconds"]:.3f} s'
            if row["status"] != "Completed":
                duration += f' ({row["status"]})'
            share = f'{row["seconds"] / total * 100:.2f}%' if total > 0 else "0.00%"
        lines.append(f"| {label} | {duration} | {share} |")
    lines.extend([f'| **Total** | **{total:.3f} s** | **{("100.00%" if total > 0 else "0.00%")}** |',
                  "", f"**Node processing time sum: {total:.3f} s**  ",
                  f"**Workflow elapsed time: {elapsed:.3f} s**", "",
                  "Share = node processing time / node processing time sum. "
                  "Parallel work can make the sum exceed elapsed time. "
                  "Cached nodes and unexecuted branches add no processing time."])
    return "\n".join(lines)


class TimingSession:
    def __init__(self, prompt, prompt_id, server, client_id, reporters):
        self.prompt = prompt
        self.prompt_id = prompt_id
        self.server = server
        self.client_id = client_id
        self.reporters = reporters
        self.started = time.perf_counter()
        self.rows = {}
        self.active = []
        self.closed = False

    def send(self, payload):
        if self.client_id is None:
            return
        try:
            self.server.send_sync(EVENT, {"prompt_id": self.prompt_id, **payload}, self.client_id)
        except Exception:
            logger.debug("DKST execution report could not be sent", exc_info=True)

    def name_for(self, node_id, dynprompt=None):
        node = self.prompt.get(node_id, {})
        if dynprompt is not None:
            try:
                node = dynprompt.get_node(node_id)
            except (KeyError, AttributeError):
                pass
        title = node.get("_meta", {}).get("title")
        class_type = node.get("class_type", "Unknown node")
        try:
            import nodes
            default_name = nodes.NODE_DISPLAY_NAME_MAPPINGS.get(class_type, class_type)
        except (ImportError, AttributeError):
            default_name = class_type
        return title or default_name

    def begin(self, node_id, dynprompt=None):
        node_id = str(node_id)
        if self.closed or node_id in self.reporters or self.prompt.get(node_id, {}).get("class_type") == NODE_TYPE:
            return None
        row = self.rows.setdefault(node_id, {
            "node_id": node_id, "name": self.name_for(node_id, dynprompt),
            "seconds": 0.0, "calls": 0, "status": "Completed",
        })
        row["status"] = "Completed"
        row["calls"] += 1
        call = {"row": row, "started": time.perf_counter(), "finished": False}
        self.active.append(call)
        return call

    def finish(self, call, status="Completed", ended=None):
        if call is None or call["finished"] or self.closed:
            return
        call["finished"] = True
        call["row"]["seconds"] += max(0.0, (time.perf_counter() if ended is None else ended) - call["started"])
        if status != "Completed":
            call["row"]["status"] = status
        self.active.remove(call)

    def cached(self, node_id, dynprompt=None):
        node_id = str(node_id)
        if not self.closed and node_id not in self.reporters and node_id not in self.rows:
            self.rows[node_id] = {"node_id": node_id, "name": self.name_for(node_id, dynprompt),
                                  "seconds": 0.0, "calls": 0, "status": "Cached"}

    def close(self, status, history_result=None):
        ended = time.perf_counter()
        for call in list(self.active):
            self.finish(call, "Incomplete" if status == "Completed" else status, ended)
        self.closed = True
        rows = list(self.rows.values())
        total = sum(row["seconds"] for row in rows)
        elapsed = max(0.0, ended - self.started)
        reports = [{"node_id": node_id, "markdown": report_markdown(rows, total, elapsed, status, sort_by)}
                   for node_id, sort_by in self.reporters.items()]
        if isinstance(history_result, dict):
            for report in reports:
                node_id = report["node_id"]
                history_result.setdefault("outputs", {}).setdefault(node_id, {})["dkst_execution_report"] = [
                    {"prompt_id": self.prompt_id, "status": status, "markdown": report["markdown"]}]
                history_result.setdefault("meta", {}).setdefault(node_id, {
                    "node_id": node_id, "display_node": node_id, "parent_node": None,
                    "real_node_id": node_id,
                })
        self.send({"status": status, "total_seconds": total, "elapsed_seconds": elapsed,
                   "rows": rows, "reports": reports})


def _error_status(error):
    return "Interrupted" if isinstance(error, asyncio.CancelledError) or type(error).__name__ == "InterruptProcessingException" else "Error"


def _observe_result(session, call, result):
    if call is None:
        return
    # Modern ComfyUI returns its actual async node Tasks before they finish.
    # Retain exactly those Tasks and let the native executor schedule them.
    tasks = list({item for item in result[0] if isinstance(item, asyncio.Future)}) if len(result) >= 4 and result[3] else []
    if not tasks:
        session.finish(call)
        return
    remaining = set(tasks)
    status = "Completed"

    def done(task):
        nonlocal status
        if session.closed:
            return
        if task.cancelled():
            status = "Interrupted"
        elif task.exception() is not None:
            status = _error_status(task.exception())
        remaining.discard(task)
        if not remaining:
            session.finish(call, status)

    for task in tasks:
        if task.done():
            done(task)
        else:
            task.add_done_callback(done)


def _bind(signature, args, kwargs):
    return signature.bind_partial(*args, **kwargs).arguments


def _safe_finish(session, call, result=None, error=None):
    try:
        if error is not None:
            session.finish(call, _error_status(error))
        else:
            _observe_result(session, call, result)
    except Exception:
        logger.debug("DKST node timing could not be recorded", exc_info=True)


def _has_processing_inputs(obj, inputs, blocker_type):
    if blocker_type is None or not inputs:
        return True
    if getattr(obj, "INPUT_IS_LIST", False):
        return not any(isinstance(value, blocker_type) for values in inputs.values() for value in values)
    length = max(len(values) for values in inputs.values())
    if length == 0:
        return True
    # Mirror native list mapping solely to detect when every call is blocked.
    # A mixed batch still has real work and must retain its timing.
    return any(not any(isinstance(values[min(index, len(values) - 1)], blocker_type)
                       for values in inputs.values()) for index in range(length))


def _output_wrapper(original, blocker_type=None):
    signature = inspect.signature(original)

    def begin(args, kwargs):
        session = _session.get()
        scope = _node_scope.get()
        if session is None or scope is None or scope[0] is not session:
            return None, None
        try:
            params = _bind(signature, args, kwargs)
            if params.get("prompt_id", session.prompt_id) != session.prompt_id:
                return None, None
            if not _has_processing_inputs(params.get("obj"), params.get("input_data_all", {}), blocker_type):
                return None, None
            return session, session.begin(params.get("unique_id", scope[1]), scope[2])
        except Exception:
            logger.debug("DKST node timing could not be started", exc_info=True)
            return None, None

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapped(*args, **kwargs):
            session, call = begin(args, kwargs)
            try:
                result = await original(*args, **kwargs)
            except BaseException as error:
                if session is not None:
                    _safe_finish(session, call, error=error)
                raise
            if session is not None:
                _safe_finish(session, call, result=result)
            return result
    else:
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            session, call = begin(args, kwargs)
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                if session is not None:
                    _safe_finish(session, call, error=error)
                raise
            if session is not None:
                _safe_finish(session, call, result=result)
            return result
    wrapped._dkst_execution_report_hook = True
    return wrapped


def _node_wrapper(original):
    signature = inspect.signature(original)

    def start(args, kwargs):
        session = _session.get()
        if session is None:
            return None, None
        params = _bind(signature, args, kwargs)
        if params.get("prompt_id") != session.prompt_id:
            return None, None
        return params, _node_scope.set((session, str(params["current_item"]), params.get("dynprompt")))

    def finish(params, token, result):
        try:
            session = _session.get()
            if params is not None and result and getattr(result[0], "name", None) == "SUCCESS":
                node_id = str(params["current_item"])
                # Cached nodes never call get_output_data or enter executed.
                # Observe only cache hits actually visited by this run; the
                # execution_cached message also contains unused branch caches.
                if node_id not in params.get("executed", set()):
                    session.cached(node_id, params.get("dynprompt"))
        except Exception:
            logger.debug("DKST cached node could not be recorded", exc_info=True)
        finally:
            if token is not None:
                _node_scope.reset(token)

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapped(*args, **kwargs):
            params, token = start(args, kwargs)
            result = None
            try:
                result = await original(*args, **kwargs)
                return result
            finally:
                finish(params, token, result)
    else:
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            params, token = start(args, kwargs)
            result = None
            try:
                result = original(*args, **kwargs)
                return result
            finally:
                finish(params, token, result)
    wrapped._dkst_execution_report_hook = True
    return wrapped


def _workflow_wrapper(original):
    signature = inspect.signature(original)

    def start(args, kwargs):
        params = _bind(signature, args, kwargs)
        prompt = params.get("prompt", {})
        reporters = {str(node_id): node.get("inputs", {}).get("sort_by", "Execution order")
                     for node_id, node in prompt.items()
                     if node.get("class_type") == NODE_TYPE and node.get("inputs", {}).get("enabled", True) is not False}
        if not reporters:
            return None, _session.set(None)
        executor = params["self"]
        session = TimingSession(prompt, params["prompt_id"], executor.server,
                                params.get("extra_data", {}).get("client_id"), reporters)
        session.previous_history = getattr(executor, "history_result", None)
        token = _session.set(session)
        session.send({"status": "Running", "reports": [{"node_id": node_id} for node_id in reporters]})
        return session, token

    def finish(session, token, executor, error):
        if session is None:
            _session.reset(token)
            return
        try:
            messages = [name for name, _data in getattr(executor, "status_messages", [])]
            if error is not None:
                status = _error_status(error)
            elif "execution_interrupted" in messages:
                status = "Interrupted"
            elif "execution_error" in messages or getattr(executor, "success", True) is False:
                status = "Error"
            else:
                status = "Completed"
            history = getattr(executor, "history_result", None)
            session.close(status, history if history is not session.previous_history else None)
        except Exception:
            logger.debug("DKST execution report could not be finalized", exc_info=True)
        finally:
            _session.reset(token)

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapped(*args, **kwargs):
            session, token = start(args, kwargs)
            error = None
            try:
                return await original(*args, **kwargs)
            except BaseException as caught:
                error = caught
                raise
            finally:
                finish(session, token, args[0] if args else kwargs["self"], error)
    else:
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            session, token = start(args, kwargs)
            error = None
            try:
                return original(*args, **kwargs)
            except BaseException as caught:
                error = caught
                raise
            finally:
                finish(session, token, args[0] if args else kwargs["self"], error)
    wrapped._dkst_execution_report_hook = True
    return wrapped


def install_execution_report_hooks(execution_module=None):
    """Attach once, preserving existing extensions' wrappers and signatures."""
    global HOOKS_AVAILABLE
    try:
        if execution_module is None:
            import execution as execution_module
        executor = execution_module.PromptExecutor
        method = "execute_async" if hasattr(executor, "execute_async") else "execute"
        originals = [getattr(executor, method), execution_module.execute, execution_module.get_output_data]
        required = [{"self", "prompt", "prompt_id"}, {"current_item", "prompt_id", "executed"}, {"obj"}]
        if not all(names <= set(inspect.signature(fn).parameters) for fn, names in zip(originals, required)):
            raise ValueError("Unsupported ComfyUI execution signatures")
        if not all(getattr(fn, "_dkst_execution_report_hook", False) for fn in originals):
            if any(getattr(fn, "_dkst_execution_report_hook", False) for fn in originals):
                raise ValueError("Execution report hooks were partially replaced")
            # Prepare every wrapper before installing any of them.
            workflow, node, output = (_workflow_wrapper(originals[0]), _node_wrapper(originals[1]),
                                      _output_wrapper(originals[2], getattr(execution_module, "ExecutionBlocker", None)))
            setattr(executor, method, workflow)
            execution_module.execute = node
            execution_module.get_output_data = output
        HOOKS_AVAILABLE = True
    except (ImportError, AttributeError, TypeError, ValueError):
        HOOKS_AVAILABLE = False
        logger.warning("DKST Execution Report is unavailable with this ComfyUI execution API.")
    return HOOKS_AVAILABLE


class DINKI_Execution_Report:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enabled": ("BOOLEAN", {"default": True}),
            "sort_by": (["Execution order", "Processing time"],),
        }}

    RETURN_TYPES = ()
    FUNCTION = "report"
    CATEGORY = "DINKIssTyle/Util"
    OUTPUT_NODE = True
    DESCRIPTION = "Add without connections. Shows a Markdown timing report after the entire workflow finishes."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def report(self, enabled=True, sort_by="Execution order"):
        if enabled and not HOOKS_AVAILABLE:
            return {"ui": {"execution_report_notice": ["Timing is unavailable with this ComfyUI version. Check the server log."]}}
        return ()
