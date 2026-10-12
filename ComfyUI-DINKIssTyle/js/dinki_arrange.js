import { app } from "../../scripts/app.js";

const NODE_CLASS = "DINKI_Arrange";
const GRID_GAP = 40;
const panels = new Set();
const ACTIONS = [
    ["Align", [
        ["Left", "left"], ["Center", "center"], ["Right", "right"],
        ["Top", "top"], ["Middle", "middle"], ["Bottom", "bottom"],
    ]],
    ["Distribute", [
        ["Horizontally", "horizontal"], ["Vertically", "vertical"], ["Evenly", "evenly"],
    ]],
];

function displayedGraph() {
    return app.canvas?.graph ?? app.graph;
}

function nodeBox(node) {
    const box = node.getBounding?.() ?? [node.pos?.[0], node.pos?.[1], node.size?.[0], node.size?.[1]];
    if (!box || ![box[0], box[1], box[2], box[3], node.pos?.[0], node.pos?.[1]].every(Number.isFinite) ||
        box[2] <= 0 || box[3] <= 0) return null;
    return { node, x: box[0], y: box[1], w: box[2], h: box[3] };
}

function selectedBoxes(controller) {
    const graph = displayedGraph();
    if (!graph || controller.graph !== graph) return [];
    const nodes = new Set(graph.nodes ?? graph._nodes ?? []);
    const selected = app.canvas?.selectedItems ?? Object.values(app.canvas?.selected_nodes ?? {});
    return [...new Set(selected)].filter(node =>
        nodes.has(node) && node.comfyClass !== NODE_CLASS && node.type !== NODE_CLASS &&
        !(node.pinned ?? node.flags?.pinned))
        .map(nodeBox).filter(Boolean);
}

function limits(boxes) {
    return {
        left: Math.min(...boxes.map(box => box.x)),
        right: Math.max(...boxes.map(box => box.x + box.w)),
        top: Math.min(...boxes.map(box => box.y)),
        bottom: Math.max(...boxes.map(box => box.y + box.h)),
    };
}

function arrangePositions(boxes, action) {
    const bounds = limits(boxes);
    if (action === "horizontal" || action === "vertical") {
        const horizontal = action === "horizontal";
        const axis = horizontal ? "x" : "y";
        const length = horizontal ? "w" : "h";
        const sorted = [...boxes].sort((a, b) => a[axis] - b[axis]);
        const start = horizontal ? bounds.left : bounds.top;
        const end = horizontal ? bounds.right : bounds.bottom;
        const total = sorted.reduce((sum, box) => sum + box[length], 0);
        // Expand an overlapping selection instead of producing negative gaps.
        const gap = Math.max(0, (end - start - total) / (sorted.length - 1));
        let position = start;
        return sorted.map(box => {
            const target = { ...box, [axis]: position };
            position += box[length] + gap;
            return target;
        });
    }
    if (action === "evenly") {
        const columns = Math.ceil(Math.sqrt(boxes.length));
        const cellWidth = Math.max(...boxes.map(box => box.w)) + GRID_GAP;
        const cellHeight = Math.max(...boxes.map(box => box.h)) + GRID_GAP;
        return [...boxes].sort((a, b) => a.y - b.y || a.x - b.x).map((box, index) => ({
            ...box,
            x: bounds.left + (index % columns) * cellWidth,
            y: bounds.top + Math.floor(index / columns) * cellHeight,
        }));
    }
    return boxes.map(box => {
        let { x, y } = box;
        switch (action) {
            case "left": x = bounds.left; break;
            case "center": x = (bounds.left + bounds.right - box.w) / 2; break;
            case "right": x = bounds.right - box.w; break;
            case "top": y = bounds.top; break;
            case "middle": y = (bounds.top + bounds.bottom - box.h) / 2; break;
            case "bottom": y = bounds.bottom - box.h; break;
        }
        return { ...box, x, y };
    });
}

function minimumSelection(action) {
    return action === "horizontal" || action === "vertical" ? 3 : 2;
}

function applyArrangement(controller, action) {
    if (app.configuringGraph) return "Wait for the workflow to finish loading.";
    const boxes = selectedBoxes(controller);
    if (boxes.length < minimumSelection(action)) {
        return `Select at least ${minimumSelection(action)} unpinned nodes.`;
    }
    const positions = arrangePositions(boxes, action);
    const changes = positions.map(target => {
        const original = boxes.find(box => box.node === target.node);
        return { node: target.node, dx: target.x - original.x, dy: target.y - original.y };
    }).filter(change => Math.abs(change.dx) > 0.0001 || Math.abs(change.dy) > 0.0001);
    if (!changes.length) return "Already arranged.";
    const graph = controller.graph;
    const historyGraph = graph.rootGraph ?? graph;
    historyGraph.beforeChange?.();
    try {
        for (const { node, dx, dy } of changes) {
            // Assign through the position setter so Nodes 2.0 receives the move.
            node.pos = [node.pos[0] + dx, node.pos[1] + dy];
        }
        graph.incrementVersion?.();
    } finally {
        historyGraph.afterChange?.();
        app.canvas?.setDirty?.(true, true);
        graph.setDirtyCanvas?.(true, true);
    }
    return `Arranged ${boxes.length} nodes.`;
}

function refreshPanels() {
    for (const panel of panels) panel.update();
}

app.registerExtension({
    name: "DINKI.Arrange",
    setup() {
        const canvas = app.canvas;
        if (!canvas || canvas.__dkstArrangeAttached) return;
        canvas.__dkstArrangeAttached = true;
        let pending = false;
        const schedule = () => {
            if (pending) return;
            pending = true;
            queueMicrotask(() => { pending = false; refreshPanels(); });
        };
        // Vue nodes select directly; classic nodes use onSelectionChange.
        for (const name of ["select", "deselect", "deselectAll", "onSelectionChange"]) {
            const original = canvas[name];
            if (name !== "onSelectionChange" && typeof original !== "function") continue;
            canvas[name] = function (...args) {
                const result = original?.apply(this, args);
                schedule();
                return result;
            };
        }
    },
    nodeCreated(node) {
        if ((node.comfyClass ?? node.type) !== NODE_CLASS || node.dkstArrangePanel) return;
        const root = document.createElement("div");
        root.style.cssText = "display:flex;flex-direction:column;gap:12px;width:100%;height:100%;min-height:206px;padding:8px;box-sizing:border-box;overflow:hidden;font:12px sans-serif;color:var(--fg-color,#bbb)";
        // Operating this panel must not replace the current multi-selection.
        for (const event of ["pointerdown", "pointermove", "pointerup", "click", "dblclick"]) {
            root.addEventListener(event, e => e.stopPropagation());
        }
        const buttons = [];
        for (const [heading, actions] of ACTIONS) {
            const section = document.createElement("div");
            const label = document.createElement("div");
            label.textContent = heading;
            label.style.cssText = "font-weight:600;margin-bottom:6px";
            const grid = document.createElement("div");
            grid.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px";
            for (const [title, action] of actions) {
                const button = document.createElement("button");
                button.type = "button";
                button.textContent = title;
                button.style.cssText = "min-width:0;height:30px;padding:0 4px;border:1px solid var(--border-color,#555);border-radius:5px;background:var(--comfy-input-bg,#32343a);color:var(--input-text,#eee);font:inherit;cursor:pointer";
                button.title = action === "evenly" ? "Arrange in a grid with 40 px between cells" :
                    action === "horizontal" || action === "vertical" ? "Equal gaps between nodes; select at least 3" :
                        `Align ${title.toLowerCase()} within the selection; select at least 2`;
                button.addEventListener("click", () => {
                    const message = applyArrangement(node, action);
                    panel.update(message);
                });
                grid.append(button);
                buttons.push({ button, action });
            }
            section.append(label, grid);
            root.append(section);
        }
        const status = document.createElement("div");
        status.setAttribute("role", "status");
        status.style.cssText = "font-size:11px;line-height:16px;min-height:16px";
        root.append(status);
        const widget = node.addDOMWidget("__dkst_arrange", "dkst-arrange", root, {
            hideOnZoom: false, selectOn: [],
            getMinHeight: () => 206, getMaxHeight: () => 206, getHeight: () => 206,
        });
        widget.serialize = false;
        widget.options.serialize = false;
        const panel = { node, update(message) {
            const count = selectedBoxes(node).length;
            status.textContent = message ?? (count >= 2 ? `${count} nodes selected` : "Select at least 2 unpinned nodes.");
            for (const { button, action } of buttons) {
                button.disabled = count < minimumSelection(action);
                button.style.opacity = button.disabled ? "0.4" : "1";
                button.style.cursor = button.disabled ? "default" : "pointer";
            }
        } };
        node.dkstArrangePanel = panel;
        panels.add(panel);
        panel.update();
        const removed = node.onRemoved;
        node.onRemoved = function (...args) {
            panels.delete(panel);
            delete node.dkstArrangePanel;
            return removed?.apply(this, args);
        };
        node.size[0] = Math.max(node.size[0], 320);
        node.expandToFitContent?.();
    },
    afterConfigureGraph() {
        refreshPanels();
    },
});
