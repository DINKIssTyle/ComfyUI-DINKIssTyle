import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "DINKI_PromptCsvLibrary";
const NONE = "-- None --";
let activeGraph;

function setComboValues(node, widget, values) {
    const options = { ...(widget.options ?? {}), values };
    widget.options = options;
    // Nodes 2.0 renders the registered widget state, which keeps an options
    // snapshot separate from the LiteGraph widget object.
    if (widget._state?.options) {
        widget._state.options = { ...widget._state.options, values };
    }
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

            let generation = 0;
            let sections = {};
            const redraw = () => {
                node.graph?.incrementVersion?.();
                node.setDirtyCanvas?.(true, true);
            };
            const setPrompt = value => {
                promptWidget.value = value;
                redraw();
            };
            const clearChildren = () => {
                sectionWidget.value = NONE;
                titleWidget.value = NONE;
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
                if (!files.includes(fileWidget.value)) fileWidget.value = fileData.default || NONE;
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
                if (!sectionChoices.includes(sectionWidget.value)) sectionWidget.value = NONE;

                const titleChoices = [NONE, ...Object.keys(sections[sectionWidget.value] || {})];
                setComboValues(node, titleWidget, titleChoices);
                if (!titleChoices.includes(titleWidget.value)) titleWidget.value = NONE;

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
                fileWidget.value = value;
                sections = {};
                clearChildren();
                load().catch(console.error);
            };

            const originalSectionCallback = sectionWidget.callback;
            sectionWidget.callback = function(value) {
                originalSectionCallback?.apply(this, arguments);
                sectionWidget.value = value;
                titleWidget.value = NONE;
                setComboValues(node, titleWidget,
                    [NONE, ...Object.keys(sections[value] || {})]);
                setPrompt("");
            };

            const originalTitleCallback = titleWidget.callback;
            titleWidget.callback = function(value) {
                originalTitleCallback?.apply(this, arguments);
                titleWidget.value = value;
                setPrompt(sections[sectionWidget.value]?.[value] || "");
            };

            const clear = node.addWidget("button", "Clear", null, () => {
                ++generation;
                sectionWidget.value = NONE;
                titleWidget.value = NONE;
                setComboValues(node, titleWidget, [NONE]);
                redraw();
            });
            clear.serialize = false;
            const refresh = node.addWidget("button", "Refresh", null, () => {
                load().catch(console.error);
            });
            refresh.serialize = false;

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
        setInterval(refreshActiveGraph, 250);
        refreshActiveGraph();
    },
});
