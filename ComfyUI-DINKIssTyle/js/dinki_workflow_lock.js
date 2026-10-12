import { app } from "../../scripts/app.js";

const NODE_CLASS = "DINKI_Workflow_Lock";
const STATE_KEY = "dkstWorkflowLock";
let syncingWidgets = false;

function rootOf(node) {
    return node.graph?.rootGraph ?? node.graph ?? app.rootGraph;
}

function* workflowNodes(root) {
    const visited = new Set();
    function* visit(graph, path) {
        if (!graph || visited.has(graph)) return;
        visited.add(graph);
        for (const node of graph.nodes ?? graph._nodes ?? []) {
            const key = JSON.stringify([path, String(node.id)]);
            yield { node, key };
            if (node.subgraph) yield* visit(node.subgraph, [...path, String(node.id)]);
        }
    }
    yield* visit(root, []);
}

function lockWidget(node) {
    return node.widgets?.find(widget => widget.name === "lock");
}

function pinned(node) {
    return Boolean(node.pinned ?? node.flags?.pinned);
}

function setPinned(node, value) {
    if (pinned(node) === value) return;
    if (typeof node.pin === "function") {
        node.pin(value);
    } else {
        node.flags ??= {};
        node.flags.pinned = value || undefined;
        node.resizable = !value;
    }
    node.setDirtyCanvas?.(true, true);
}

function originalState(node) {
    const state = { pinned: pinned(node) };
    if (node.resizable !== undefined) state.resizable = node.resizable;
    return state;
}

function synchronizeWidgets(root, value) {
    syncingWidgets = true;
    try {
        for (const { node } of workflowNodes(root)) {
            if (node.comfyClass !== NODE_CLASS) continue;
            const widget = lockWidget(node);
            if (widget && widget.value !== value) widget.value = value;
        }
    } finally {
        syncingWidgets = false;
    }
}

function lockWorkflow(root) {
    const previous = root.extra?.[STATE_KEY];
    const snapshot = previous?.snapshot && typeof previous.snapshot === "object"
        ? previous.snapshot : {};
    const changes = [];
    let captured = false;
    const currentKeys = new Set();
    for (const entry of workflowNodes(root)) {
        currentKeys.add(entry.key);
        if (!Object.hasOwn(snapshot, entry.key)) {
            snapshot[entry.key] = originalState(entry.node);
            captured = true;
        }
        if (!pinned(entry.node)) changes.push(entry.node);
    }
    for (const key of Object.keys(snapshot)) {
        if (currentKeys.has(key)) continue;
        delete snapshot[key];
        captured = true;
    }
    if (!previous || captured || changes.length) {
        root.beforeChange?.();
        root.extra ??= {};
        root.extra[STATE_KEY] = { version: 1, snapshot };
        root.incrementVersion?.();
        for (const node of changes) setPinned(node, true);
        root.afterChange?.();
        root.setDirtyCanvas?.(true, true);
    }
    synchronizeWidgets(root, true);
}

function unlockWorkflow(root) {
    const state = root.extra?.[STATE_KEY];
    if (!state) {
        synchronizeWidgets(root, false);
        return;
    }
    root.beforeChange?.();
    for (const { node, key } of workflowNodes(root)) {
        const original = state.snapshot?.[key];
        if (!original) continue;
        setPinned(node, original.pinned === true);
        if (Object.hasOwn(original, "resizable")) node.resizable = original.resizable;
        else delete node.resizable;
    }
    delete root.extra[STATE_KEY];
    root.incrementVersion?.();
    root.afterChange?.();
    root.setDirtyCanvas?.(true, true);
    synchronizeWidgets(root, false);
}

function requestLock(node, value) {
    if (syncingWidgets || app.configuringGraph || !node.graph || typeof value !== "boolean") return;
    const root = rootOf(node);
    if (value) lockWorkflow(root);
    else unlockWorkflow(root);
}

function synchronizeGraph(root) {
    if (!root || app.configuringGraph) return;
    const controllers = [...workflowNodes(root)]
        .map(entry => entry.node)
        .filter(node => node.comfyClass === NODE_CLASS);
    if (!controllers.length) {
        if (root.extra?.[STATE_KEY]) unlockWorkflow(root);
        return;
    }
    if (root.extra?.[STATE_KEY] || controllers.some(node => lockWidget(node)?.value === true)) {
        lockWorkflow(root);
    } else {
        synchronizeWidgets(root, false);
    }
}

app.registerExtension({
    name: "DINKI.WorkflowLock",
    nodeCreated(node) {
        if (node.__dkstWorkflowLockHooked) return;
        node.__dkstWorkflowLockHooked = true;

        // New or pasted nodes join the active lock after they enter the graph.
        const added = node.onAdded;
        node.onAdded = function () {
            const result = added?.apply(this, arguments);
            const root = rootOf(this);
            queueMicrotask(() => synchronizeGraph(root));
            return result;
        };

        const configured = node.onConfigure;
        node.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            const root = rootOf(this);
            queueMicrotask(() => synchronizeGraph(root));
            return result;
        };

        const removed = node.onRemoved;
        node.onRemoved = function () {
            const root = rootOf(this);
            const result = removed?.apply(this, arguments);
            queueMicrotask(() => synchronizeGraph(root));
            return result;
        };

        queueMicrotask(() => synchronizeGraph(rootOf(node)));
        if (node.comfyClass !== NODE_CLASS) return;

        const changed = node.onWidgetChanged;
        node.onWidgetChanged = function (name, value) {
            const result = changed?.apply(this, arguments);
            if (name === "lock") requestLock(this, value);
            return result;
        };

        const widget = lockWidget(node);
        if (widget) {
            const callback = widget.callback;
            widget.callback = function (value) {
                const result = callback?.apply(this, arguments);
                requestLock(node, value);
                return result;
            };
        }

    },
    afterConfigureGraph() {
        synchronizeGraph(app.rootGraph ?? app.graph);
    },
});
