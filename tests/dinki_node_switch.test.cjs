const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

function fixture({ ids = '2', active = true, targets = [{ id: 2, mode: 0 }] } = {}) {
    let extension;
    const app = { registerExtension(ext) { if (ext.name === 'DINKI.NodeSwitch') extension = ext; } };
    // Load the actual extension file; unrelated extensions are registered but not invoked.
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), { app, api: {}, queueMicrotask });
    const graph = { _nodes: targets, change() {}, setDirtyCanvas() {} };
    app.graph = app.rootGraph = graph;
    const node = { id: 1, comfyClass: 'DINKI_Node_Switch', graph, mode: 0,
        widgets: [{ name: 'node_ids', value: ids }, { name: 'active', value: active }] };
    graph._nodes.push(node);
    return { app, graph, node, extension, targets };
}

test('Nodes 2.0 notification alone toggles targets, even before widget value is committed', () => {
    const { node, extension, targets } = fixture();
    extension.nodeCreated(node);
    node.onWidgetChanged('active', false);
    assert.equal(targets[0].mode, 4);
    node.widgets[1].value = false;
    node.onWidgetChanged('active', true);
    assert.equal(targets[0].mode, 0);
});

test('classic callbacks preserve existing callback arguments and return value', () => {
    const { node, extension, targets } = fixture();
    let seen;
    const widget = node.widgets[1];
    widget.callback = function (...args) { seen = [this, ...args]; return 42; };
    extension.nodeCreated(node);
    assert.equal(widget.callback(false, 'extra'), 42);
    assert.deepEqual(seen, [widget, false, 'extra']);
    assert.equal(targets[0].mode, 4);
});

test('changed ID text is used immediately; IDs match exactly across number/string storage', () => {
    const { node, extension, targets } = fixture({ active: false, targets: [
        { id: '2', mode: 0 }, { id: 3, mode: 0 }, { id: 'uuid-node', mode: 0 }
    ] });
    extension.nodeCreated(node);
    node.widgets[0].callback(' 3, uuid-node, 2junk, , ');
    assert.deepEqual(targets.slice(0, 3).map(n => n.mode), [0, 4, 4]);
});

test('targets are resolved in the owning graph, not the displayed graph', () => {
    const { app, node, extension, targets } = fixture();
    const unrelated = { id: 2, mode: 0 };
    app.graph = { _nodes: [unrelated] };
    extension.nodeCreated(node);
    node.onWidgetChanged('active', false);
    assert.equal(targets[0].mode, 4);
    assert.equal(unrelated.mode, 0);
});

test('workflow load synchronizes root and nested subgraphs after configuration', () => {
    const { app, graph, node, extension, targets } = fixture({ active: false });
    const nestedTarget = { id: 2, mode: 0 };
    const subgraph = { nodes: [nestedTarget] };
    subgraph.nodes.push({ ...node, graph: subgraph });
    graph._nodes.push({ id: 10, subgraph });
    app.configuringGraph = true;
    extension.nodeCreated(node);
    node.onWidgetChanged('active', false);
    assert.equal(targets[0].mode, 0);
    app.configuringGraph = false;
    extension.afterConfigureGraph();
    assert.equal(targets[0].mode, 4);
    assert.equal(nestedTarget.mode, 4);
});

test('newly added and pasted nodes synchronize after lifecycle hook completes', async () => {
    for (const hook of ['onAdded', 'onConfigure']) {
        const { node, extension, targets } = fixture({ active: false });
        extension.nodeCreated(node);
        node[hook]();
        await Promise.resolve();
        assert.equal(targets[0].mode, 4);
    }
});

test('does not bypass itself or unmute a manually muted node when enabled', () => {
    const { node, extension, targets } = fixture({ ids: '1,2', targets: [{ id: 2, mode: 2 }] });
    extension.nodeCreated(node);
    node.onWidgetChanged('active', true);
    assert.equal(targets[0].mode, 2);
    node.onWidgetChanged('active', false);
    assert.equal(node.mode, 0);
});

test('preserves node notification and ignores unrelated widgets or detached nodes', () => {
    const { node, extension, targets } = fixture();
    node.onWidgetChanged = () => 99;
    extension.nodeCreated(node);
    assert.equal(node.onWidgetChanged('other', false), 99);
    assert.equal(targets[0].mode, 0);
    node.graph = null;
    node.onWidgetChanged('active', false);
    assert.equal(targets[0].mode, 0);
});
