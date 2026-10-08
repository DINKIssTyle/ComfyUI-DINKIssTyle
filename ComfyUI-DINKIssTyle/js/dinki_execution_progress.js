import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const SETTING = "DKST.Progress.ActualNode";

// ComfyUI starts progress before calling check_lazy_status. A pending switch
// therefore stays 'running' alongside its upstream workers. The queue overlay
// chooses the first running node, which can be that waiting switch for the
// entire sampling/decode operation. Only backend-confirmed waits are corrected.
export function branchProgressSnapshot(detail, waiting) {
    if (!detail?.nodes || !waiting.size) return detail;
    let changed = false;
    const nodes = Object.fromEntries(Object.entries(detail.nodes).map(([id, state]) => {
        if (state?.state !== "running" || !waiting.has(String(state.node_id ?? id))) {
            return [id, state];
        }
        changed = true;
        return [id, { ...state, state: "pending" }];
    }));
    return changed ? { ...detail, nodes } : detail;
}

export function installBranchProgress(api, initiallyEnabled = true) {
    const jobs = new Map();
    const ended = new Set();
    const correctedEvents = new WeakSet();
    const listeners = [];
    let enabled = initiallyEnabled;
    let disposed = false;

    const jobFor = promptId => {
        if (!promptId || ended.has(promptId)) return null;
        if (!jobs.has(promptId)) jobs.set(promptId, { waiting: new Set(), snapshot: null, revision: 0 });
        return jobs.get(promptId);
    };
    const publish = (promptId, force = false) => {
        const job = jobs.get(promptId);
        if (!job?.snapshot) return;
        const revision = ++job.revision;
        // Wait until every native listener has handled the original event.
        // Native RAF coalescers then consume the corrected snapshot last,
        // regardless of whether extensions or native listeners registered first.
        queueMicrotask(() => {
            if (disposed || jobs.get(promptId) !== job || job.revision !== revision) return;
            const snapshot = enabled ? branchProgressSnapshot(job.snapshot, job.waiting) : job.snapshot;
            if (!force && snapshot === job.snapshot) return;
            const event = new CustomEvent("progress_state", { detail: snapshot });
            correctedEvents.add(event);
            api.dispatchEvent(event);
        });
    };
    const listen = (name, handler) => {
        api.addEventListener(name, handler);
        listeners.push([name, handler]);
    };

    listen("progress_state", event => {
        if (correctedEvents.has(event)) return;
        const detail = event.detail;
        if (!detail?.nodes) return;
        const job = jobFor(detail.prompt_id);
        if (!job) return;
        job.snapshot = detail;
        for (const [id, state] of Object.entries(detail.nodes)) {
            if (state.state === "finished" || state.state === "error") {
                job.waiting.delete(String(state.node_id ?? id));
            }
        }
        publish(detail.prompt_id);
    });
    listen("dkst.branch_status", ({ detail }) => {
        if (detail?.node_id == null || typeof detail.waiting !== "boolean") return;
        const job = jobFor(detail.prompt_id);
        if (!job) return;
        const id = String(detail.node_id);
        if (detail.waiting) {
            job.waiting.add(id);
            publish(detail.prompt_id);
        } else {
            job.waiting.delete(id);
            // Do not replay an old snapshot as 'running' when the switch resumes.
            // Its next native start/finish event supplies its authoritative state.
            ++job.revision;
        }
    });
    listen("execution_start", ({ detail }) => {
        const id = detail?.prompt_id;
        if (!id) return;
        ended.delete(id);
        jobs.set(id, { waiting: new Set(), snapshot: null, revision: 0 });
    });
    const end = ({ detail }) => {
        const id = detail?.prompt_id;
        if (!id) return;
        jobs.delete(id);
        ended.add(id);
        // Suppress delayed messages from completed runs without growing forever.
        if (ended.size > 64) ended.delete(ended.values().next().value);
    };
    for (const name of ["execution_success", "execution_error", "execution_interrupted"]) listen(name, end);

    return {
        setEnabled(value) {
            enabled = Boolean(value);
            for (const id of jobs.keys()) publish(id, true);
        },
        destroy() {
            disposed = true;
            for (const [name, handler] of listeners) api.removeEventListener(name, handler);
            jobs.clear();
            ended.clear();
        },
    };
}

let controller;
app.registerExtension({
    name: "DINKI.ExecutionProgress.ActualNode",
    settings: [{
        id: SETTING,
        name: "Show actual processing node in queue progress",
        type: "boolean",
        defaultValue: true,
        category: ["DKST", "Progress", "Actual processing node"],
        tooltip: "Exclude waiting DKST If/Else nodes from Current node so the queue progress follows the processing node.",
        onChange: value => controller?.setEnabled(value),
    }],
    setup() {
        if (controller) return;
        const enabled = app.extensionManager?.setting?.get(SETTING)
            ?? app.ui?.settings?.getSettingValue(SETTING, true) ?? true;
        controller = installBranchProgress(api, enabled);
        window.addEventListener("pagehide", () => {
            controller.destroy();
            controller = undefined;
        }, { once: true });
    },
});
