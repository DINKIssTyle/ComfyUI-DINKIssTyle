const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

function fixture({ first = '2, 3', second = '4, uuid-node', active = true } = {}) {
    let extension, changes = 0, tick;
    const app = { registerExtension(ext) { if (ext.name === 'DINKI.NodeChange') extension = ext; } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: {}, queueMicrotask, setInterval(fn) { tick = fn; return 1; }
    });
    const targets = [{ id: 2, mode: 4 }, { id: '3', mode: 2 }, { id: 4, mode: 0 },
        { id: 'uuid-node', mode: 0 }, { id: 5, mode: 2 }];
    const graph = { _nodes: [...targets], change() { changes++; }, setDirtyCanvas() {} };
    const node = { id: 1, mode: 0, comfyClass: 'DINKI_Node_Change', graph,
        widgets: [{ name: 'node_ids_1', value: first }, { name: 'node_ids_2', value: second },
            { name: 'active', value: active }] };
    graph._nodes.push(node);
    app.graph = app.rootGraph = graph;
    return { app, graph, node, targets, extension, changes: () => changes, tick: () => tick() };
}

test('Nodes 2.0 switches both groups before the toggle value is committed', () => {
    const { node, targets, extension, changes } = fixture();
    extension.nodeCreated(node);
    node.onWidgetChanged('active', true);
    assert.deepEqual(targets.map(n => n.mode), [0, 0, 4, 4, 2]);
    node.onWidgetChanged('active', false);
    assert.deepEqual(targets.map(n => n.mode), [4, 4, 0, 0, 2]);
    node.onWidgetChanged('active', false);
    assert.equal(changes(), 2);
});

test('classic callback preserves context, arguments and return value', () => {
    const { node, targets, extension } = fixture();
    let seen;
    const widget = node.widgets[2];
    widget.callback = function(...args) { seen = [this, ...args]; return 42; };
    extension.nodeCreated(node);
    assert.equal(widget.callback(false, 'extra'), 42);
    assert.deepEqual(seen, [widget, false, 'extra']);
    assert.deepEqual(targets.map(n => n.mode), [4, 4, 0, 0, 2]);
});

test('editing either ID field applies its new value immediately with exact ID matching', () => {
    const { node, targets, extension } = fixture({ first: '', second: '' });
    extension.nodeCreated(node);
    node.onWidgetChanged('node_ids_1', ' 2, 3, 2, 5junk, , missing ');
    assert.deepEqual(targets.map(n => n.mode), [0, 0, 0, 0, 2]);
    node.widgets[1].callback('4, uuid-node');
    assert.deepEqual(targets.map(n => n.mode), [0, 0, 4, 4, 2]);
});

test('shared IDs stay enabled, own ID is ignored, and empty groups work', () => {
    const { node, targets, extension } = fixture({ first: '1, 2', second: '1, 2, 4' });
    extension.nodeCreated(node);
    node.onWidgetChanged('active', true);
    assert.equal(node.mode, 0);
    assert.equal(targets[0].mode, 0);
    assert.equal(targets[2].mode, 4);
    node.onWidgetChanged('active', false);
    assert.equal(node.mode, 0);
    assert.equal(targets[0].mode, 0);
    node.widgets[1].value = '';
    node.onWidgetChanged('active', false);
    assert.equal(targets[0].mode, 4);
});

test('resolves the owning graph and synchronizes nested graphs after workflow load', () => {
    const { app, graph, node, targets, extension } = fixture({ active: false });
    extension.nodeCreated(node);
    app.configuringGraph = true;
    node.onWidgetChanged('active', false);
    assert.equal(targets[2].mode, 0);
    const nested = fixture({ active: true });
    graph._nodes.push({ id: 20, subgraph: nested.graph });
    app.graph = { _nodes: [{ id: 2, mode: 2 }] };
    app.configuringGraph = false;
    extension.afterConfigureGraph();
    assert.deepEqual(targets.map(n => n.mode), [4, 4, 0, 0, 2]);
    assert.deepEqual(nested.targets.map(n => n.mode), [0, 0, 4, 4, 2]);
    assert.equal(app.graph._nodes[0].mode, 2);
});

test('added and configured nodes synchronize after lifecycle completion', async () => {
    for (const hook of ['onAdded', 'onConfigure']) {
        const { node, targets, extension } = fixture({ active: false });
        node[hook] = () => 99;
        extension.nodeCreated(node);
        assert.equal(node[hook](), 99);
        await Promise.resolve();
        assert.deepEqual(targets.map(n => n.mode), [4, 4, 0, 0, 2]);
    }
});

test('preserves notification and ignores unrelated edits or detached nodes', () => {
    const { node, targets, extension, changes } = fixture();
    node.onWidgetChanged = () => 42;
    extension.nodeCreated(node);
    assert.equal(node.onWidgetChanged('unrelated', true), 42);
    node.graph = null;
    node.onWidgetChanged('active', true);
    assert.deepEqual(targets.map(n => n.mode), [4, 2, 0, 0, 2]);
    assert.equal(changes(), 0);
});

function promote(inner, node, widgetName, hostName, value) {
    const input = { name: widgetName, widget: { name: widgetName } };
    node.inputs ??= [];
    node.inputs.push(input);
    inner.inputNode ??= { slots: [] };
    inner.links ??= {};
    const id = Object.keys(inner.links).length + 1;
    inner.links[id] = { resolve: () => ({ inputNode: node, input }) };
    inner.inputNode.slots.push({ name: hostName, linkIds: [id] });
    const widget = { name: hostName, value };
    return { subgraph: inner, inputs: [{ name: hostName, _widget: widget }], widgets: [widget] };
}

test('host-only promoted values drive inner modes without changing the inner widget', () => {
    const f = fixture();
    const host = promote(f.graph, f.node, 'active', 'Choose a group', true);
    f.app.rootGraph = f.app.graph = { _nodes: [host] };
    f.extension.setup();
    f.tick();
    host.widgets[0].value = false; // No callback, setter hook or inner notification.
    f.tick();
    assert.deepEqual(f.targets.map(n => n.mode), [4, 4, 0, 0, 2]);
    assert.equal(f.node.widgets[2].value, true, 'do not overwrite framework source values');
    const count = f.changes();
    f.tick();
    assert.equal(f.changes(), count);
    host.widgets[0].value = true;
    f.tick();
    assert.deepEqual(f.targets.map(n => n.mode), [0, 0, 4, 4, 2]);
});

test('nested promotions forward outer overrides through renamed inputs', () => {
    const f = fixture();
    const innerHost = promote(f.graph, f.node, 'active', 'inner group', true);
    const middle = { _nodes: [innerHost] };
    const outerHost = promote(middle, innerHost, 'inner group', 'outer group', false);
    f.app.rootGraph = { _nodes: [outerHost] };
    f.extension.setup();
    f.tick();
    assert.deepEqual(f.targets.map(n => n.mode), [4, 4, 0, 0, 2]);
});

test('promoted ID fields and load synchronization use current host values', () => {
    const f = fixture();
    const host = promote(f.graph, f.node, 'node_ids_1', 'Enabled IDs', '5');
    f.app.rootGraph = { _nodes: [host] };
    f.extension.afterConfigureGraph();
    assert.equal(f.targets[4].mode, 0);
    f.extension.setup();
    host.widgets[0].value = '2';
    f.tick();
    assert.equal(f.targets[0].mode, 0);
});

test('legacy direct writes are observed and removed subgraphs are not retained', () => {
    const f = fixture();
    const root = { _nodes: [{ subgraph: f.graph }] };
    f.app.rootGraph = root;
    f.extension.setup();
    f.tick();
    f.node.widgets[2].value = false;
    f.tick();
    assert.deepEqual(f.targets.map(n => n.mode), [4, 4, 0, 0, 2]);
    root._nodes = [];
    const count = f.changes();
    f.node.widgets[2].value = true;
    f.tick();
    assert.equal(f.changes(), count);
});

test('subgraph controls with repeated IDs remain isolated and pause during load', () => {
    const f = fixture();
    const other = fixture();
    const host = promote(f.graph, f.node, 'active', 'group', false);
    const otherHost = promote(other.graph, other.node, 'active', 'group', true);
    f.app.rootGraph = { _nodes: [host, otherHost] };
    f.extension.setup();
    f.app.configuringGraph = true;
    f.tick();
    assert.equal(f.changes(), 0);
    f.app.configuringGraph = false;
    f.tick();
    assert.deepEqual(f.targets.map(n => n.mode), [4, 4, 0, 0, 2]);
    assert.deepEqual(other.targets.map(n => n.mode), [0, 0, 4, 4, 2]);
});

test('legacy proxy metadata resolves host-only values without matching display labels', () => {
    for (const overlay of [false, true]) {
        const f = fixture();
        const widget = { name: 'renamed control', value: false };
        if (overlay) widget._overlay = { isProxyWidget: true, nodeId: String(f.node.id), widgetName: 'active' };
        const host = { subgraph: f.graph, widgets: [widget],
            properties: overlay ? {} : { proxyWidgets: [[String(f.node.id), 'active']] } };
        f.app.rootGraph = { _nodes: [host] };
        f.extension.setup();
        f.tick();
        assert.deepEqual(f.targets.map(n => n.mode), [4, 4, 0, 0, 2]);
    }
});
