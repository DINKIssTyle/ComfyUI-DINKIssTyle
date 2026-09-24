const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

function fixture({ first = '2, 3', second = '4, uuid-node', active = true, disableMode = 'Bypass',
    firstLabel = 'Group 1', secondLabel = 'Group 2' } = {}) {
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
            { name: 'active', value: active }, { name: 'disable_mode', value: disableMode },
            { name: 'group_1_label', value: firstLabel }, { name: 'group_2_label', value: secondLabel }] };
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
    assert.equal(f.node.widgets[2].value, true);
});

test('returning to a tab reapplies the selected group even when its widget value is unchanged', () => {
    const f = fixture();
    const host = promote(f.graph, f.node, 'active', 'Choose a group', false);
    const root = { _nodes: [host] };
    f.app.rootGraph = f.app.graph = root;
    f.extension.setup();
    f.tick();
    assert.deepEqual(f.targets.map(n => n.mode), [4, 4, 0, 0, 2]);

    f.app.rootGraph = { _nodes: [] };
    f.tick();
    f.targets[0].mode = 0;
    f.targets[2].mode = 4;
    f.app.rootGraph = root;
    f.tick();
    assert.deepEqual(f.targets.map(n => n.mode), [4, 4, 0, 0, 2]);
    assert.equal(host.widgets[0].value, false);
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
    f.app.configuringGraph = true;
    f.extension.setup();
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

test('Mute blocks the inactive group and reactivates the other group normally', () => {
    const f = fixture({ disableMode: 'Mute' });
    f.extension.nodeCreated(f.node);
    f.node.onWidgetChanged('active', true);
    assert.deepEqual(f.targets.map(n => n.mode), [0, 0, 2, 2, 2]);
    f.node.onWidgetChanged('active', false);
    assert.deepEqual(f.targets.map(n => n.mode), [2, 2, 0, 0, 2]);
    f.node.widgets[2].value = false;
    f.node.widgets[3].callback('Bypass');
    assert.deepEqual(f.targets.map(n => n.mode), [4, 4, 0, 0, 2]);
});

test('custom labels update both widget renderers without changing group modes', () => {
    const f = fixture({ firstLabel: 'Positive', secondLabel: 'Negative' });
    f.extension.nodeCreated(f.node);
    f.extension.afterConfigureGraph();
    const toggle = f.node.widgets[2];
    assert.equal(toggle.options.on, 'Positive');
    assert.equal(toggle.options.off, 'Negative');
    const changes = f.changes();
    f.node.onWidgetChanged('group_1_label', ' Portrait ');
    assert.equal(toggle.options.on, 'Portrait');
    f.node.widgets[5].callback('');
    assert.equal(toggle.options.off, 'Group 2');
    assert.equal(f.changes(), changes);
});

test('old nodes without new widgets retain Bypass and default labels', () => {
    const f = fixture();
    f.node.widgets.splice(3);
    f.extension.afterConfigureGraph();
    assert.deepEqual(f.targets.map(n => n.mode), [0, 0, 4, 4, 2]);
    assert.equal(f.node.widgets[2].options.on, 'Group 1');
});

test('Mute and custom labels propagate to nested outer controls', () => {
    const f = fixture({ disableMode: 'Mute', firstLabel: 'A text', secondLabel: 'B text' });
    const innerHost = promote(f.graph, f.node, 'active', 'inner', true);
    const middle = { _nodes: [innerHost] };
    const outerHost = promote(middle, innerHost, 'inner', 'outer', false);
    f.app.rootGraph = { _nodes: [outerHost] };
    f.extension.setup();
    f.tick();
    assert.deepEqual(f.targets.map(n => n.mode), [2, 2, 0, 0, 2]);
    assert.equal(outerHost.widgets[0].options.on, 'A text');
    assert.equal(outerHost.widgets[0].options.off, 'B text');
    f.node.widgets[4].value = 'Updated';
    f.tick();
    assert.equal(outerHost.widgets[0].options.on, 'Updated');
});

test('disable mode promoted outside a subgraph applies without a callback', () => {
    const f = fixture();
    const host = promote(f.graph, f.node, 'disable_mode', 'Mode', 'Mute');
    f.app.rootGraph = { _nodes: [host] };
    f.extension.setup();
    f.tick();
    assert.deepEqual(f.targets.map(n => n.mode), [0, 0, 2, 2, 2]);
    host.widgets[0].value = 'Bypass';
    f.tick();
    assert.deepEqual(f.targets.map(n => n.mode), [0, 0, 4, 4, 2]);
});

test('root Nodes 2.0 text writes update labels without callbacks or mode changes', () => {
    const f = fixture();
    const options = { on: 'Group 1', off: 'Group 2' };
    f.node.widgets[2].options = options;
    let revisions = 0;
    f.graph.incrementVersion = () => revisions++;
    f.extension.setup();
    f.node.widgets[4].value = 'LLM';
    f.node.widgets[5].value = 'Qwen3.5 9B T2I';
    f.tick();
    assert.equal(options.on, 'LLM', 'update the object retained by the renderer');
    assert.equal(options.off, 'Qwen3.5 9B T2I');
    assert.equal(f.node.widgets[2].options.on, 'LLM');
    assert.equal(revisions, 1);
    assert.equal(f.changes(), 0, 'label refresh must not change execution modes');
    f.tick();
    assert.equal(revisions, 1, 'unchanged labels must not repeatedly invalidate the UI');
    f.node.widgets[5].value = 'PE';
    f.tick();
    assert.equal(options.off, 'PE');
    f.node.widgets[5].value = 'Test';
    f.tick();
    assert.equal(options.off, 'Test', 'the original renderer reference must survive repeated edits');
    assert.equal(f.node.widgets[2].options, options);
});

test('inner labels and nested outer controls keep updating after successive edits', async () => {
    const f = fixture({ firstLabel: 'LLM', secondLabel: 'PE' });
    const originalOptions = { on: 'Group 1', off: 'Group 2' };
    f.node.widgets[2].options = originalOptions;
    const inner = promote(f.graph, f.node, 'active', 'inner group', true);
    const outer = promote({ _nodes: [inner] }, inner, 'inner group', 'outer group', true);
    f.app.rootGraph = { _nodes: [outer] };
    f.extension.nodeCreated(f.node);
    f.extension.setup();
    assert.equal(originalOptions.off, 'PE');
    for (const label of ['Test', 'Text', 'PE']) {
        f.node.widgets[5].callback(label);
        f.node.widgets[5].value = label;
        f.tick();
        assert.equal(originalOptions.off, label);
        assert.equal(inner.widgets[0].options.off, label);
        assert.equal(outer.widgets[0].options.off, label);
    }
});

test('outer label callback synchronizes inner labels before the next polling tick', async () => {
    const f = fixture({ secondLabel: 'PE' });
    const host = promote(f.graph, f.node, 'group_2_label', 'Second label', 'PE');
    host.widgets[0].callback = function(value) { this.value = value; return 42; };
    f.app.rootGraph = { _nodes: [host] };
    f.extension.setup();
    assert.equal(host.widgets[0].callback('Test'), 42);
    await Promise.resolve();
    assert.equal(f.node.widgets[2].options.off, 'Test');
});

test('promoted widgets with getter-only options accept label changes', () => {
    const f = fixture({ firstLabel: 'LLM', secondLabel: 'Image' });
    const host = promote(f.graph, f.node, 'active', 'group', true);
    const options = { on: 'Group 1', off: 'Group 2' };
    Object.defineProperty(host.widgets[0], 'options', { get: () => options });
    f.app.rootGraph = { _nodes: [host] };
    f.extension.setup();
    f.tick();
    assert.equal(options.on, 'LLM');
    assert.equal(options.off, 'Image');
});

test('store-backed options setter receives a new reference after in-place refresh', () => {
    const f = fixture({ firstLabel: 'LLM' });
    const original = { on: 'Group 1', off: 'Group 2' };
    let current = original, writes = 0;
    Object.defineProperty(f.node.widgets[2], 'options', {
        get: () => current, set: value => { current = value; writes++; }
    });
    f.extension.setup();
    f.tick();
    assert.equal(original.on, 'LLM');
    assert.equal(current.on, 'LLM');
    assert.notEqual(current, original);
    assert.equal(writes, 1);
});
