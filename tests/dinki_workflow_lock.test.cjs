const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_workflow_lock.js'), 'utf8');

function fixture() {
    let extension;
    const app = { registerExtension(value) { extension = value; }, configuringGraph: false };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), { app, queueMicrotask });
    const graph = {
        _nodes: [], extra: {}, changes: 0,
        beforeChange() { this.changes++; }, afterChange() { this.changes++; },
        setDirtyCanvas() {},
    };
    graph.rootGraph = graph;
    app.rootGraph = graph;
    return { app, extension, graph };
}

function makeNode(graph, id, options = {}) {
    const node = {
        id, graph, comfyClass: options.controller ? 'DINKI_Workflow_Lock' : 'Other',
        flags: { pinned: options.pinned || undefined },
        resizable: options.resizable ?? true,
        widgets: options.controller ? [{ name: 'lock', value: options.locked || false }] : [],
        setDirtyCanvas() {},
        get pinned() { return !!this.flags.pinned; },
        pin(value) { this.flags.pinned = value || undefined; this.resizable = !value; },
    };
    graph._nodes.push(node);
    return node;
}

test('Lock pins all nodes and Unlock restores preexisting Pin and resize state', () => {
    const { extension, graph } = fixture();
    const normal = makeNode(graph, 1);
    const alreadyPinned = makeNode(graph, 2, { pinned: true, resizable: false });
    const nonResizable = makeNode(graph, 3, { resizable: false });
    const control = makeNode(graph, 4, { controller: true });
    extension.nodeCreated(control);

    control.widgets[0].callback(true);
    assert.deepEqual(graph._nodes.map(node => node.pinned), [true, true, true, true]);
    assert.equal(control.widgets[0].value, true);
    assert.ok(graph.extra.dkstWorkflowLock.snapshot);

    control.onWidgetChanged('lock', false);
    assert.deepEqual(graph._nodes.map(node => node.pinned), [false, true, false, false]);
    assert.equal(normal.resizable, true);
    assert.equal(nonResizable.resizable, false);
    assert.equal(control.widgets[0].value, false);
    assert.equal(graph.extra.dkstWorkflowLock, undefined);
});

test('saved locked workflow restores original states after reopening', () => {
    const first = fixture();
    makeNode(first.graph, 1);
    makeNode(first.graph, 2, { pinned: true, resizable: false });
    const control = makeNode(first.graph, 3, { controller: true });
    first.extension.nodeCreated(control);
    control.widgets[0].callback(true);
    const saved = JSON.parse(JSON.stringify(first.graph.extra));

    const reopened = fixture();
    reopened.graph.extra = saved;
    makeNode(reopened.graph, 1, { pinned: true, resizable: false });
    makeNode(reopened.graph, 2, { pinned: true, resizable: false });
    const restoredControl = makeNode(reopened.graph, 3, { controller: true, pinned: true, locked: true, resizable: false });
    reopened.extension.nodeCreated(restoredControl);
    reopened.extension.afterConfigureGraph();
    restoredControl.onWidgetChanged('lock', false);
    assert.deepEqual(reopened.graph._nodes.map(node => node.pinned), [false, true, false]);
});

test('new nodes and nested subgraph nodes join an active lock', async () => {
    const { extension, graph } = fixture();
    const subgraph = { _nodes: [], rootGraph: graph };
    const host = makeNode(graph, 1);
    host.subgraph = subgraph;
    const inner = makeNode(subgraph, 2);
    const control = makeNode(graph, 3, { controller: true });
    extension.nodeCreated(control);
    control.widgets[0].callback(true);
    assert.equal(inner.pinned, true);

    const added = makeNode(subgraph, 4);
    extension.nodeCreated(added);
    added.onAdded();
    await Promise.resolve();
    assert.equal(added.pinned, true);
    control.widgets[0].callback(false);
    assert.equal(added.pinned, false);
    assert.equal(inner.pinned, false);
});

test('multiple controllers mirror the same lock state', () => {
    const { extension, graph } = fixture();
    const target = makeNode(graph, 1);
    const first = makeNode(graph, 2, { controller: true });
    const second = makeNode(graph, 3, { controller: true });
    extension.nodeCreated(first);
    extension.nodeCreated(second);
    first.widgets[0].callback(true);
    assert.equal(second.widgets[0].value, true);
    second.widgets[0].callback(false);
    assert.equal(target.pinned, false);
    assert.equal(first.widgets[0].value, false);
});

test('removing the last controller restores unlocked state', async () => {
    const { extension, graph } = fixture();
    const target = makeNode(graph, 1);
    const control = makeNode(graph, 2, { controller: true });
    extension.nodeCreated(control);
    control.widgets[0].callback(true);
    graph._nodes.splice(graph._nodes.indexOf(control), 1);
    control.onRemoved();
    await Promise.resolve();
    assert.equal(target.pinned, false);
    assert.equal(graph.extra.dkstWorkflowLock, undefined);
});

test('removing a target discards its saved Pin state', async () => {
    const { extension, graph } = fixture();
    const target = makeNode(graph, 1);
    const control = makeNode(graph, 2, { controller: true });
    extension.nodeCreated(target);
    extension.nodeCreated(control);
    control.widgets[0].callback(true);
    graph._nodes.splice(graph._nodes.indexOf(target), 1);
    target.onRemoved();
    await Promise.resolve();
    const keys = Object.keys(graph.extra.dkstWorkflowLock.snapshot);
    assert.equal(keys.length, 1);
    assert.equal(JSON.parse(keys[0])[1], '2');
});
