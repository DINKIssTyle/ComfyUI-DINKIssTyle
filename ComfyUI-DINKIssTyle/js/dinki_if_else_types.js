import { app } from "/scripts/app.js";

const PAIR_COUNTS = {
    DINKI_IfElseSwitch: 10,
    DINKI_IfElseImageSwitch: 10,
    DINKI_IfElseBranch: 4,
};

function connectedType(node, input) {
    const index = node.inputs?.indexOf(input) ?? -1;
    if (index < 0) return null;
    const link = node.getInputLink?.(index) ?? node.graph?.getLink?.(input.link)
        ?? node.graph?.links?.[input.link];
    if (!link) return null;
    const source = link.resolve?.(node.graph)?.output
        ?? node.graph?.getNodeById?.(link.origin_id)?.outputs?.[link.origin_slot];
    const type = source?.type;
    return type && type !== "*" ? type : null;
}

function updateNodeTypes(node) {
    const count = PAIR_COUNTS[node.comfyClass ?? node.type];
    if (!count || !node.inputs || !node.outputs) return false;
    let changed = false;
    for (let number = 1; number <= count; number++) {
        const onFalse = node.inputs.find(input => input.name === `on_false_${number}`);
        const onTrue = node.inputs.find(input => input.name === `on_true_${number}`);
        const output = node.outputs[number - 1];
        if (!onFalse || !onTrue || !output) continue;

        const falseType = connectedType(node, onFalse);
        const trueType = connectedType(node, onTrue);
        const commonType = falseType && trueType && falseType !== trueType
            ? "*" : falseType ?? trueType ?? "*";
        for (const slot of [onFalse, onTrue, output]) {
            if (slot.type !== commonType) {
                slot.type = commonType;
                changed = true;
            }
        }
        for (const linkId of output.links ?? []) {
            const link = typeof linkId === "object" ? linkId
                : node.graph?.getLink?.(linkId) ?? node.graph?.links?.[linkId];
            if (link && link.type !== commonType) {
                link.type = commonType;
                changed = true;
            }
        }
    }
    return changed;
}

function updateGraphTypes(node) {
    const graph = node.graph;
    const nodes = Array.isArray(graph?._nodes) ? graph._nodes : [node];
    // Revisit later branch nodes when an upstream switch changes its output type.
    for (let pass = 0; pass < Math.min(nodes.length, 16); pass++) {
        let changed = false;
        for (const candidate of nodes) {
            changed = updateNodeTypes(candidate) || changed;
        }
        if (!changed) break;
    }
    graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "DINKI.IfElseSwitch.SocketTypes",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!(nodeData.name in PAIR_COUNTS)) return;
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(...args) {
            const result = onConnectionsChange?.apply(this, args);
            Promise.resolve().then(() => updateGraphTypes(this));
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(...args) {
            const result = onConfigure?.apply(this, args);
            Promise.resolve().then(() => updateGraphTypes(this));
            return result;
        };
    },
});
