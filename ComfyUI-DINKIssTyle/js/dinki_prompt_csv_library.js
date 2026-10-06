import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "DINKI_PromptCsvLibrary";
const NONE = "-- None --";
let activeGraph;
const promotedValues = new WeakMap();

async function copyPromptText(text) {
    if (globalThis.navigator?.clipboard?.writeText) {
        try {
            await navigator.clipboard.writeText(text);
            return;
        } catch { /* Fall back for browsers that deny the Clipboard API. */ }
    }
    const field = document.createElement("textarea");
    field.value = text;
    Object.assign(field.style, { position: "fixed", left: "-10000px", top: "0" });
    const focused = document.activeElement;
    document.body.appendChild(field);
    try {
        field.focus();
        field.select();
        if (!document.execCommand?.("copy")) throw new Error("Clipboard access was denied.");
    } finally {
        field.remove();
        focused?.focus?.();
    }
}

function addCopyPromptButton(node, promptWidget) {
    let feedbackTimer;
    const feedback = label => {
        button.label = label;
        node.graph?.incrementVersion?.();
        node.setDirtyCanvas?.(true, true);
    };
    const button = node.addWidget("button", "Copy Prompt", null, async () => {
        try {
            await copyPromptText(String(promptWidget.value ?? ""));
            feedback("Copied!");
        } catch (error) {
            feedback("Copy failed");
            console.error("Unable to copy prompt:", error);
        } finally {
            clearTimeout(feedbackTimer);
            feedbackTimer = setTimeout(() => feedback("Copy Prompt"), 1800);
        }
    }, { serialize: false });
    button.serialize = false;
    const removed = node.onRemoved;
    node.onRemoved = function() {
        clearTimeout(feedbackTimer);
        return removed?.apply(this, arguments);
    };
}

function writeWidgetValue(node, widget, value) {
    widget.value = value;
    promotedValues.set(widget, value);
    widget.options?.setValue?.(value);
    const element = widget.element ?? widget.inputEl;
    if (element && "value" in element && element.value !== value) element.value = value;
    node.graph?.incrementVersion?.();
    node.setDirtyCanvas?.(true, true);
}

function updateWidgetOptions(widget, changes) {
    // Promoted widgets can expose options through a getter without a setter.
    // Preserve that object (and the frontend's visibility/value-store facade).
    if (widget.options) Object.assign(widget.options, changes);
    else widget.options = { ...changes };
    if (widget._state?.options) {
        Object.assign(widget._state.options, changes);
    }
}

function writeComboValues(widget, values) {
    updateWidgetOptions(widget, { values });
}

// Follow actual promoted-input links, since boundary labels can be renamed.
// Visit children first so updates also reach every enclosing subgraph.
function syncPromotedWidgets(sourceNode, sourceWidget, update) {
    const hosts = [];
    const changed = new Map([[sourceNode, new Set([sourceWidget.name])]]);
    const visiting = new Set();
    const visit = graph => {
        if (!graph || visiting.has(graph)) return;
        visiting.add(graph);
        for (const host of graph.nodes ?? graph._nodes ?? []) {
            if (!host.subgraph) continue;
            visit(host.subgraph);
            const bindings = [];
            const widgets = host.widgets ?? [];
            for (const [index, widget] of widgets.entries()) {
                const overlay = widget._overlay;
                const [id, name] = overlay?.isProxyWidget
                    ? [overlay.nodeId, overlay.widgetName]
                    : (host.properties?.proxyWidgets?.[index] ?? []);
                if (id == null || String(id) === "-1" || !name) continue;
                const target = (host.subgraph.nodes ?? host.subgraph._nodes ?? [])
                    .find(node => String(node.id) === String(id));
                bindings.push({ widget, target, name });
            }
            for (const input of host.inputs ?? []) {
                const widget = host.getWidgetFromSlot?.(input) ?? input._widget ??
                    widgets.find(item => item.name === (input.widget?.name ?? input.name));
                if (!widget) continue;
                const slot = host.subgraph.inputNode?.slots?.find(item => item.name === input.name);
                for (const id of slot?.linkIds ?? []) {
                    const link = host.subgraph.getLink?.(id) ?? host.subgraph.links?.[id];
                    if (!link) continue;
                    const resolved = link.resolve?.(host.subgraph);
                    const target = resolved?.inputNode ?? host.subgraph.getNodeById?.(link.target_id);
                    const targetInput = resolved?.input ?? target?.inputs?.[link.target_slot];
                    const name = target?.subgraph ? targetInput?.name :
                        (target?.getWidgetFromSlot?.(targetInput)?.name ?? targetInput?.widget?.name);
                    bindings.push({ widget, target, name });
                }
            }
            for (const { widget, target, name } of bindings) {
                if (!changed.get(target)?.has(name)) continue;
                hosts.push({ node: host, widget });
                if ("values" in update) writeComboValues(widget, update.values);
                if ("value" in update) writeWidgetValue(host, widget, update.value);
                if (!changed.has(host)) changed.set(host, new Set());
                changed.get(host).add(widget.name);
                // A renamed graph input may differ from its widget's name.
                for (const input of host.inputs ?? []) {
                    if ((host.getWidgetFromSlot?.(input) ?? input._widget) === widget) {
                        changed.get(host).add(input.name);
                    }
                }
                if (Object.keys(update).length) {
                    host.graph?.incrementVersion?.();
                    host.setDirtyCanvas?.(true, true);
                }
            }
        }
        visiting.delete(graph);
    };
    visit(app.rootGraph ?? app.graph ?? sourceNode.graph);
    return hosts;
}

function setWidgetValue(node, widget, value) {
    writeWidgetValue(node, widget, value);
    syncPromotedWidgets(node, widget, { value });
}

function setComboValues(node, widget, values) {
    writeComboValues(widget, values);
    syncPromotedWidgets(node, widget, { values });
    node.graph?.incrementVersion?.();
    node.setDirtyCanvas?.(true, true);
}

function visitLibraryNodes(graph, callback, visited = new Set()) {
    if (!graph || visited.has(graph)) return;
    visited.add(graph);
    for (const node of graph.nodes ?? graph._nodes ?? []) {
        if (node.comfyClass === NODE_CLASS) callback(node);
        visitLibraryNodes(node.subgraph, callback, visited);
    }
}

function syncExternalSelections() {
    if (app.configuringGraph) return;
    visitLibraryNodes(app.rootGraph ?? app.graph, node => {
        for (const name of ["csv_file", "section", "title", "prompt"]) {
            const source = node.widgets?.find(widget => widget.name === name);
            if (!source) continue;
            const hosts = syncPromotedWidgets(node, source, {}).reverse();
            const changed = hosts.find(({ widget }) => promotedValues.has(widget) &&
                promotedValues.get(widget) !== widget.value);
            for (const { widget } of hosts) {
                if (!promotedValues.has(widget)) promotedValues.set(widget, widget.value);
            }
            if (!changed) continue;
            const value = changed.widget.value;
            // Record before callbacks: they can synchronously reset dependent inputs.
            promotedValues.set(changed.widget, value);
            if (name === "prompt") setWidgetValue(node, source, value);
            else source.callback?.call(source, value);
        }
    });
}

function refreshActiveGraph() {
    if (app.configuringGraph) return;
    const graph = app.canvas?.graph ?? app.rootGraph ?? app.graph;
    if (!graph || graph === activeGraph) return;
    activeGraph = graph;
    visitLibraryNodes(graph, node => node.dkstSchedulePromptLibraryRestore?.());
}

app.registerExtension({
    name: "DINKI.PromptCsvLibrary",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = originalCreated?.apply(this, arguments);
            const node = this;
            const get = name => node.widgets?.find(widget => widget.name === name);
            const fileWidget = get("csv_file");
            const sectionWidget = get("section");
            const titleWidget = get("title");
            const promptWidget = get("prompt");
            if (!fileWidget || !sectionWidget || !titleWidget || !promptWidget) return result;

            addCopyPromptButton(node, promptWidget);

            let generation = 0;
            let sections = {};
            const redraw = () => {
                node.graph?.incrementVersion?.();
                node.setDirtyCanvas?.(true, true);
            };
            const setPrompt = value => {
                setWidgetValue(node, promptWidget, value);
                redraw();
            };
            const clearChildren = () => {
                setWidgetValue(node, sectionWidget, NONE);
                setWidgetValue(node, titleWidget, NONE);
                setComboValues(node, sectionWidget, [NONE]);
                setComboValues(node, titleWidget, [NONE]);
                setPrompt("");
            };
            const load = async({ preservePrompt = false } = {}) => {
                const current = ++generation;
                const filesResponse = await api.fetchApi("/dinki/prompt-library/files");
                if (!filesResponse.ok) throw new Error(`Unable to list CSV files (${filesResponse.status})`);
                const fileData = await filesResponse.json();
                if (current !== generation) return;

                const files = [NONE, ...(fileData.files || [])];
                setComboValues(node, fileWidget, files);
                if (!files.includes(fileWidget.value)) setWidgetValue(node, fileWidget, fileData.default || NONE);
                const filename = fileWidget.value;
                if (!filename || filename === NONE) {
                    sections = {};
                    clearChildren();
                    return;
                }

                const response = await api.fetchApi(`/dinki/prompt-library/entries?${new URLSearchParams({ file: filename })}`);
                if (!response.ok) throw new Error(`Unable to load CSV prompts (${response.status})`);
                const data = await response.json();
                if (current !== generation || filename !== fileWidget.value) return;
                sections = data.sections || {};

                const sectionChoices = [NONE, ...Object.keys(sections)];
                setComboValues(node, sectionWidget, sectionChoices);
                if (!sectionChoices.includes(sectionWidget.value)) setWidgetValue(node, sectionWidget, NONE);

                const titleChoices = [NONE, ...Object.keys(sections[sectionWidget.value] || {})];
                setComboValues(node, titleWidget, titleChoices);
                if (!titleChoices.includes(titleWidget.value)) setWidgetValue(node, titleWidget, NONE);

                const selectedPrompt = sections[sectionWidget.value]?.[titleWidget.value] || "";
                if (!preservePrompt) {
                    setPrompt(selectedPrompt);
                } else {
                    redraw();
                }
            };

            const originalFileCallback = fileWidget.callback;
            fileWidget.callback = function(value) {
                originalFileCallback?.apply(this, arguments);
                setWidgetValue(node, fileWidget, value);
                sections = {};
                clearChildren();
                load().catch(console.error);
            };

            const originalSectionCallback = sectionWidget.callback;
            sectionWidget.callback = function(value) {
                originalSectionCallback?.apply(this, arguments);
                setWidgetValue(node, sectionWidget, value);
                setWidgetValue(node, titleWidget, NONE);
                setComboValues(node, titleWidget,
                    [NONE, ...Object.keys(sections[value] || {})]);
                setPrompt("");
            };

            const originalTitleCallback = titleWidget.callback;
            titleWidget.callback = function(value) {
                originalTitleCallback?.apply(this, arguments);
                setWidgetValue(node, titleWidget, value);
                setPrompt(sections[sectionWidget.value]?.[value] || "");
            };

            const clear = node.addWidget("button", "Clear", null, () => {
                ++generation;
                setWidgetValue(node, sectionWidget, NONE);
                setWidgetValue(node, titleWidget, NONE);
                setComboValues(node, titleWidget, [NONE]);
                redraw();
            });
            clear.serialize = false;
            const refresh = node.addWidget("button", "Refresh", null, () => {
                load().catch(console.error);
            });
            refresh.serialize = false;

            for (const widget of [fileWidget, clear, refresh]) {
                // LiteGraph uses widget.advanced; Nodes 2.0 uses the options.
                widget.advanced = true;
                updateWidgetOptions(widget, { advanced: true });
            }

            let restoreScheduled = false;
            node.dkstSchedulePromptLibraryRestore = () => {
                if (restoreScheduled) return;
                restoreScheduled = true;
                requestAnimationFrame(() => setTimeout(() => {
                    restoreScheduled = false;
                    load({ preservePrompt: true }).catch(console.error);
                }, 0));
            };
            node.dkstSchedulePromptLibraryRestore();
            node.size[0] = Math.max(node.size[0], 390);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = originalConfigure?.apply(this, arguments);
            this.dkstSchedulePromptLibraryRestore?.();
            return result;
        };
    },
    loadedGraphNode(node) {
        if (node.comfyClass === NODE_CLASS) node.dkstSchedulePromptLibraryRestore?.();
    },
    afterConfigureGraph() {
        visitLibraryNodes(app.rootGraph ?? app.graph,
            node => node.dkstSchedulePromptLibraryRestore?.());
    },
    setup() {
        setInterval(() => {
            syncExternalSelections();
            refreshActiveGraph();
        }, 250);
        syncExternalSelections();
        refreshActiveGraph();
    },
});
