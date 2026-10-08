import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { renderNoteMarkdown, installNoteMarkdownStyles } from "./dinki_note_markdown.js";

const NODE_TYPE = "DINKI_Execution_Report";

export function hideZeroTimeRows(markdown) {
    // Match the displayed duration column, including rounded sub-millisecond
    // timings. Keep Cached rows, the bold total row, and all report summaries.
    return markdown.split("\n").filter(line =>
        !/\| 0\.000 s(?: \([^|]*\))? \| [\d.]+% \|$/.test(line)
    ).join("\n");
}

export function reportNodes(graph) {
    const result = new Map();
    const visiting = new Set();
    const visit = (graph, path = []) => {
        if (!graph || visiting.has(graph)) return;
        visiting.add(graph);
        for (const node of graph.nodes ?? graph._nodes ?? []) {
            const id = [...path, String(node.id)].join(":");
            if (node.comfyClass === NODE_TYPE) result.set(id, { node, graph });
            if (node.subgraph) visit(node.subgraph, [...path, String(node.id)]);
        }
        visiting.delete(graph);
    };
    visit(graph);
    return result;
}

export function installExecutionReports(app, api) {
    const exports = new WeakMap();
    const jobs = new Map();
    const buffered = new Map();
    let disposed = false;
    const graphToPrompt = app.graphToPrompt;
    const queuePrompt = api.queuePrompt;
    const apply = detail => {
        const bindings = jobs.get(detail.prompt_id);
        if (!bindings) return;
        for (const report of detail.reports ?? []) {
            const binding = bindings.get(String(report.node_id));
            // Do not put an old job's report on a new workflow with the same IDs.
            if (binding?.node.graph === binding?.graph) {
                binding?.node.dkstApplyExecutionReport?.(report, detail.status, detail.prompt_id);
            }
        }
        if (detail.status !== "Running") jobs.delete(detail.prompt_id);
    };
    const receive = ({ detail }) => {
        if (!detail?.prompt_id || !Array.isArray(detail.reports)) return;
        if (jobs.has(detail.prompt_id)) apply(detail);
        else {
            // A short workflow can finish before its HTTP queue response arrives.
            buffered.set(detail.prompt_id, detail);
            if (buffered.size > 32) buffered.delete(buffered.keys().next().value);
        }
    };
    api.addEventListener("dkst.execution_report", receive);

    const wrappedExport = async function(...args) {
        const bindings = reportNodes(args[0] ?? app.rootGraph ?? app.graph);
        const result = await graphToPrompt.apply(this, args);
        if (!disposed && result && typeof result === "object") exports.set(result, bindings);
        return result;
    };
    const wrappedQueue = async function(...args) {
        if (disposed) return queuePrompt.apply(this, args);
        const data = args[1];
        const bindings = exports.get(data) ?? reportNodes(app.rootGraph ?? app.graph);
        const reporters = new Map([...bindings].filter(([id]) => data?.output?.[id]?.class_type === NODE_TYPE));
        const result = await queuePrompt.apply(this, args);
        if (!disposed && result?.prompt_id && reporters.size) {
            jobs.set(result.prompt_id, reporters);
            for (const [id, bindings] of jobs) {
                if ([...bindings.values()].every(({ node, graph }) => node.graph !== graph)) jobs.delete(id);
            }
            const early = buffered.get(result.prompt_id);
            if (early) {
                buffered.delete(result.prompt_id);
                apply(early);
            }
        }
        return result;
    };
    if (typeof graphToPrompt === "function") app.graphToPrompt = wrappedExport;
    if (typeof queuePrompt === "function") api.queuePrompt = wrappedQueue;
    return {
        destroy() {
            disposed = true;
            api.removeEventListener("dkst.execution_report", receive);
            if (app.graphToPrompt === wrappedExport) app.graphToPrompt = graphToPrompt;
            if (api.queuePrompt === wrappedQueue) api.queuePrompt = queuePrompt;
            jobs.clear();
            buffered.clear();
        },
    };
}

async function copyMarkdown(text) {
    if (globalThis.isSecureContext && globalThis.navigator?.clipboard?.writeText) {
        try { await navigator.clipboard.writeText(text); return; } catch {}
    }
    const field = document.createElement("textarea");
    field.value = text;
    Object.assign(field.style, { position: "fixed", left: "-10000px", top: "0", opacity: "0" });
    const focused = document.activeElement;
    document.body.appendChild(field);
    try {
        field.focus();
        field.select();
        if (!document.execCommand?.("copy")) throw new Error("Browser blocked copying. Select the report and use Ctrl+C or Cmd+C.");
    } finally {
        field.remove();
        focused?.focus?.();
    }
}

function validSize(size) {
    return size?.length === 2 && size.every(value => Number.isFinite(value) && value > 0);
}

let controller;
app.registerExtension({
    name: "DINKI.ExecutionReport",
    setup() {
        if (controller) return;
        controller = installExecutionReports(app, api);
        window.addEventListener("pagehide", () => {
            controller.destroy();
            controller = undefined;
        }, { once: true });
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments);
            installNoteMarkdownStyles();
            const root = document.createElement("div");
            const toolbar = document.createElement("div");
            const copy = document.createElement("button");
            const status = document.createElement("span");
            const hideZeroLabel = document.createElement("label");
            const hideZero = document.createElement("input");
            const hideZeroText = document.createElement("span");
            const preview = document.createElement("div");
            Object.assign(root.style, {
                width: "100%", height: "100%", minHeight: "0", display: "flex", flexDirection: "column",
                overflow: "hidden", background: "#202020", borderRadius: "6px", color: "#eee",
            });
            Object.assign(toolbar.style, { display: "flex", alignItems: "center", flexWrap: "wrap", gap: "10px", padding: "6px", borderBottom: "1px solid #444" });
            copy.type = "button";
            copy.textContent = "Copy Markdown";
            copy.disabled = true;
            Object.assign(copy.style, { border: "1px solid #555", borderRadius: "4px", padding: "4px 10px", background: "#333", color: "#eee", cursor: "pointer" });
            status.setAttribute("aria-live", "polite");
            hideZero.type = "checkbox";
            hideZero.checked = this.properties?.dkstExecutionReportHideZero === true;
            hideZeroText.textContent = "Hide 0.000 s";
            hideZeroLabel.title = "Hide rows displaying 0.000 s. Totals and percentages stay unchanged.";
            Object.assign(hideZeroLabel.style, { display: "inline-flex", alignItems: "center", gap: "4px", cursor: "pointer", font: "13px sans-serif" });
            hideZeroLabel.append(hideZero, hideZeroText);
            preview.className = "dkst-note-preview";
            preview.tabIndex = 0;
            preview.setAttribute("role", "region");
            preview.setAttribute("aria-label", "Execution timing report");
            Object.assign(preview.style, { flex: "1 1 0", minHeight: "0", padding: "10px", font: "14px/1.5 sans-serif" });
            for (const name of ["pointerdown", "mousedown", "dblclick", "wheel", "keydown"]) root.addEventListener(name, event => event.stopPropagation());
            toolbar.append(copy, status, hideZeroLabel);
            root.append(toolbar, preview);
            const widget = this.addDOMWidget("dkst_execution_report", "DKST_EXECUTION_REPORT", root, {
                hideOnZoom: false, getMinHeight: () => 180, getMaxHeight: () => 10000, getHeight: () => 320,
            });
            widget.serialize = false;
            widget.options ??= {};
            widget.options.serialize = false;
            let markdown = "";
            let generation = 0;
            let removed = false;
            let feedback;
            let latestReport;
            let latestState;
            let latestPromptId;
            this.dkstApplyExecutionReport = (report, state = "Completed", promptId = "") => {
                if (removed) return;
                latestReport = report;
                latestState = state;
                latestPromptId = promptId;
                const current = ++generation;
                clearTimeout(feedback);
                const source = typeof report?.markdown === "string" ? report.markdown : "";
                markdown = hideZero.checked ? hideZeroTimeRows(source) : source;
                copy.textContent = "Copy Markdown";
                copy.removeAttribute("title");
                copy.disabled = !markdown;
                status.textContent = state === "Running" ? "Recording…" : state;
                if (markdown) {
                    this.properties ??= {};
                    this.properties.dkstExecutionReport = { markdown: source, status: state, prompt_id: promptId };
                    const isCurrent = () => !removed && generation === current;
                    Promise.resolve().then(() => renderNoteMarkdown(preview, markdown, isCurrent)).catch(() => {
                        if (isCurrent()) { preview.textContent = markdown; preview.style.whiteSpace = "pre-wrap"; }
                    });
                } else {
                    preview.textContent = state === "Running" ? "Recording node processing times. The report appears when the workflow finishes." :
                        (report?.notice ?? "Add this node without connections and run the workflow to record processing times.");
                }
                this.setDirtyCanvas?.(true, true);
            };
            this.dkstRestoreExecutionReport = () => {
                hideZero.checked = this.properties?.dkstExecutionReportHideZero === true;
                const saved = this.properties?.dkstExecutionReport;
                this.dkstApplyExecutionReport(saved, saved?.status ?? "Ready", saved?.prompt_id);
            };
            this.dkstRestoreExecutionReport();
            hideZero.addEventListener("change", () => {
                this.properties ??= {};
                this.properties.dkstExecutionReportHideZero = hideZero.checked;
                this.dkstApplyExecutionReport(latestReport, latestState, latestPromptId);
                this.graph?.incrementVersion?.();
            });
            copy.addEventListener("click", async () => {
                const current = generation;
                copy.disabled = true;
                try {
                    await copyMarkdown(markdown);
                    if (generation === current && !removed) copy.textContent = "Copied!";
                } catch (error) {
                    if (generation === current && !removed) { copy.textContent = "Copy failed"; copy.title = error.message; }
                } finally {
                    if (generation === current && !removed) {
                        copy.disabled = !markdown;
                        feedback = setTimeout(() => { copy.textContent = "Copy Markdown"; }, 1800);
                    }
                }
            });
            const configured = this.onConfigure;
            this.onConfigure = function(info) {
                const result = configured?.apply(this, arguments);
                const size = validSize(info?.size) ? [...info.size] : null;
                queueMicrotask(() => {
                    if (removed) return;
                    if (size) this.setSize?.(size);
                    this.dkstRestoreExecutionReport();
                });
                return result;
            };
            const executed = this.onExecuted;
            this.onExecuted = function(message) {
                const result = executed?.apply(this, arguments);
                const saved = message?.dkst_execution_report?.[0];
                if (saved) this.dkstApplyExecutionReport(saved, saved.status, saved.prompt_id);
                else if (message?.execution_report_notice?.[0]) this.dkstApplyExecutionReport({ notice: message.execution_report_notice[0] }, "Unavailable");
                return result;
            };
            const serialized = this.onSerialize;
            this.onSerialize = function(info) {
                const result = serialized?.apply(this, arguments);
                // Nodes 2.0 may hold the resized geometry in its layout store.
                try { void this.renderingSize; } catch {}
                if (validSize(this.size)) info.size = [...this.size];
                return result;
            };
            const onRemoved = this.onRemoved;
            this.onRemoved = function() {
                removed = true;
                generation++;
                clearTimeout(feedback);
                return onRemoved?.apply(this, arguments);
            };
            if (this.size) {
                this.size[0] = Math.max(this.size[0], 620);
                this.size[1] = Math.max(this.size[1], 420);
            }
            return result;
        };
    },
    loadedGraphNode(node) {
        if (node.comfyClass === NODE_TYPE) node.dkstRestoreExecutionReport?.();
    },
});
