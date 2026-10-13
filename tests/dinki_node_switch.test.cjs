const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

function fixture({ ids = '2', active = true, disableMode = 'Bypass', targets = [{ id: 2, mode: 0 }] } = {}) {
    let extension, tick;
    const app = { registerExtension(ext) { if (ext.name === 'DINKI.NodeSwitch') extension = ext; } };
    // Load the actual extension file; unrelated extensions are registered but not invoked.
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: {}, queueMicrotask, setInterval(fn) { tick = fn; return 1; }
    });
    const graph = { _nodes: targets, change() {}, setDirtyCanvas() {} };
    app.graph = app.rootGraph = graph;
    const node = { id: 1, comfyClass: 'DINKI_Node_Switch', graph, mode: 0,
        widgets: [{ name: 'node_ids', value: ids }, { name: 'active', value: active }] };
    if (disableMode !== null) node.widgets.push({ name: 'disable_mode', value: disableMode });
    graph._nodes.push(node);
    return { app, graph, node, extension, targets, tick: () => tick() };
}

test('Mute disables targets and On reactivates them without muting the controller', () => {
    const { node, extension, targets } = fixture({ ids: '1,2', disableMode: 'Mute' });
    extension.nodeCreated(node);
    node.onWidgetChanged('active', false);
    assert.equal(targets[0].mode, 2);
    assert.equal(node.mode, 0);
    node.widgets[1].value = false;
    node.onWidgetChanged('active', true);
    assert.equal(targets[0].mode, 0);
});

test('changing disable mode while Off updates targets before the value is committed', () => {
    const { node, extension, targets } = fixture({ active: false });
    extension.nodeCreated(node);
    extension.afterConfigureGraph();
    assert.equal(targets[0].mode, 4);
    node.onWidgetChanged('disable_mode', 'Mute');
    assert.equal(targets[0].mode, 2);
    node.widgets[2].value = 'Mute';
    node.widgets[2].callback('Bypass');
    assert.equal(targets[0].mode, 4);
});

test('legacy workflows without a disable mode retain Bypass behavior', () => {
    const { node, extension, targets } = fixture({ active: false, disableMode: null });
    extension.nodeCreated(node);
    extension.afterConfigureGraph();
    assert.equal(targets[0].mode, 4);
    node.onWidgetChanged('active', true);
    assert.equal(targets[0].mode, 0);
});

test('workflow load restores Mute mode in root and nested graphs', () => {
    const f = fixture({ active: false, disableMode: 'Mute' });
    const nested = fixture({ active: false, disableMode: 'Mute' });
    f.graph._nodes.push({ id: 10, subgraph: nested.graph });
    f.extension.afterConfigureGraph();
    assert.equal(f.targets[0].mode, 2);
    assert.equal(nested.targets[0].mode, 2);
});

test('promoted disable mode follows host-only value changes inside a subgraph', () => {
    const f = fixture({ active: false });
    const widget = { name: 'Disable with', value: 'Bypass',
        _overlay: { isProxyWidget: true, nodeId: f.node.id, widgetName: 'disable_mode' } };
    f.app.rootGraph = { _nodes: [{ subgraph: f.graph, widgets: [widget] }] };
    f.extension.setup();
    assert.equal(f.targets[0].mode, 4);
    widget.value = 'Mute';
    f.tick();
    assert.equal(f.targets[0].mode, 2);
    widget.value = 'Bypass';
    f.tick();
    assert.equal(f.targets[0].mode, 4);
    assert.equal(f.node.widgets[2].value, 'Bypass');
});

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

function sizedFixture() {
    const f = fixture(), node = f.node;
    node.properties = {};
    node.size = new Float32Array([225, 110]);
    node.showAdvanced = false;
    // Nodes 2.0 includes the hidden ID/mode widgets in its minimum height.
    node.computeSize = () => new Float32Array([210, node.showAdvanced ? 260 : 178]);
    node.setSize = function(size) {
        const minimum = this.computeSize();
        this.size = new Float32Array([Math.max(size[0], minimum[0]), Math.max(size[1], minimum[1])]);
        this.onResize?.(this.size);
    };
    node.toggleAdvanced = function() {
        this.showAdvanced = !this.showAdvanced;
        this.setSize(this.size);
    };
    node.onConfigure = function(info) {
        this.properties = { ...this.properties, ...info.properties };
        if (info.showAdvanced !== undefined) this.showAdvanced = info.showAdvanced;
        if (info.size) this.setSize(info.size);
        return 42;
    };
    node.onSerialize = () => 99;
    f.extension.nodeCreated(node);
    return f;
}

test('collapsed Node Switch ignores hidden widget height and restores saved manual size on reload', () => {
    const { node } = sizedFixture();
    assert.deepEqual(Array.from(node.computeSize()), [210, 90]);
    node.setSize([240, 105]);
    const saved = { properties: {}, size: Array.from(node.size), showAdvanced: false };
    assert.equal(node.onSerialize(saved), 99);
    assert.deepEqual(Array.from(saved.properties.dkstNodeSwitchCompactSize), [240, 105]);
    const restored = sizedFixture().node;
    restored.size = new Float32Array([240, 178]);
    assert.equal(restored.onConfigure(saved), 42);
    assert.deepEqual(Array.from(restored.size), [240, 105]);
    assert.equal(restored.outputs, undefined, 'sizing must not add Node Change outputs');
});

test('Node Switch advanced toggle restores the latest manually sized collapsed geometry', () => {
    const { node } = sizedFixture();
    node.toggleAdvanced();
    assert.deepEqual(Array.from(node.size), [225, 260]);
    node.toggleAdvanced();
    assert.deepEqual(Array.from(node.size), [225, 110]);
    node.setSize([280, 100]);
    node.toggleAdvanced();
    node.toggleAdvanced();
    assert.deepEqual(Array.from(node.size), [280, 100]);
});

test('Nodes 2.0 direct layout writes preserve collapsed Node Switch size through serialization', () => {
    const f = sizedFixture(), node = f.node;
    f.extension.setup();
    // The layout store can update without a LiteGraph onResize callback.
    Object.defineProperty(node, 'renderingSize', { get() {
        this.size = new Float32Array([250, 100]);
        return this.size;
    } });
    f.tick();
    assert.deepEqual(Array.from(node.properties.dkstNodeSwitchCompactSize), [250, 100]);
    const saved = { properties: {}, size: [250, 178] };
    node.onSerialize(saved);
    assert.deepEqual(Array.from(saved.size), [250, 100]);
    assert.deepEqual(Array.from(saved.properties.dkstNodeSwitchCompactSize), [250, 100]);
});

test('legacy Node Switch without compact metadata keeps its saved size and direct advanced changes restore it', () => {
    const f = sizedFixture(), node = f.node;
    node.onConfigure({ properties: {}, size: [300, 120], showAdvanced: false });
    f.extension.setup();
    node.showAdvanced = true;
    node.setSize([300, 260]);
    f.tick();
    const saved = { properties: {}, size: Array.from(node.size), showAdvanced: true };
    node.onSerialize(saved);
    const restored = sizedFixture().node;
    restored.onConfigure(saved);
    assert.deepEqual(Array.from(restored.size), [300, 260]);
    restored.toggleAdvanced();
    assert.deepEqual(Array.from(restored.size), [300, 120]);
    node.showAdvanced = false;
    f.tick();
    assert.deepEqual(Array.from(node.size), [300, 120]);
});
