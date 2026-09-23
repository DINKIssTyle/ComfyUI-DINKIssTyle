const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

async function fixture({ classic = false } = {}) {
    let extension;
    const updates = [];
    const widget = { name: 'selected_node_id', value: 'None' };
    const check = { id: 1, type: 'DINKI_Node_Check', widgets: [
        { name: 'unrelated', value: 'unchanged' }, widget
    ], onWidgetChanged(...args) { updates.push(args); }, setDirtyCanvas() {} };
    const first = { id: 20 }, second = { id: 3 };
    const graph = { _nodes: [check, first, second] };
    const canvas = { graph, selected_nodes: {},
        select(item) { this.selectedItems.add(item); return 'selected'; },
        deselect(item) { this.selectedItems.delete(item); },
        deselectAll() { this.selectedItems.clear(); }
    };
    if (!classic) canvas.selectedItems = new Set();
    else for (const name of ['select', 'deselect', 'deselectAll']) delete canvas[name];
    const app = { canvas, graph, registerExtension(ext) {
        if (ext.name === 'Dinki.NodeCheck') extension = ext;
    } };
    // No LGraphCanvas global is needed by the extension.
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), { app, api: {}, queueMicrotask });
    await extension.setup();
    return { app, canvas, graph, extension, check, widget, first, second, updates };
}

test('Nodes 2.0 direct selection updates the named widget and notifies the UI', async () => {
    const { canvas, widget, first, updates, check } = await fixture();
    let stored, callback;
    widget.options = { setValue(value) { stored = value; } };
    widget.callback = function(...args) { callback = [this, ...args]; };
    assert.equal(canvas.select(first), 'selected');
    await Promise.resolve();
    assert.equal(widget.value, '20');
    assert.equal(stored, '20');
    assert.deepEqual(callback, [widget, '20', canvas, check]);
    assert.deepEqual(updates, [['selected_node_id', '20', 'None', widget]]);
    assert.equal(check.widgets[0].value, 'unchanged');
});

test('selection order, deselection and clearing work without the legacy callback', async () => {
    const { canvas, widget, first, second } = await fixture();
    canvas.select(first);
    canvas.select(second);
    await Promise.resolve();
    assert.equal(widget.value, '3');
    canvas.deselect(second);
    await Promise.resolve();
    assert.equal(widget.value, '20');
    canvas.deselectAll();
    await Promise.resolve();
    assert.equal(widget.value, 'None');
});

test('deselectAll plus select is batched and unchanged values do not notify again', async () => {
    const { canvas, first, second, updates } = await fixture();
    canvas.select(first);
    await Promise.resolve();
    canvas.deselectAll();
    canvas.select(second);
    canvas.onSelectionChange();
    await Promise.resolve();
    assert.equal(updates.length, 2);
    assert.equal(updates[1][2], '20');
    canvas.select(second);
    await Promise.resolve();
    assert.equal(updates.length, 2);
});

test('groups, reroutes and stale nodes are ignored even when IDs overlap', async () => {
    const { canvas, first, widget } = await fixture();
    canvas.select(first);
    canvas.select({ id: first.id });
    await Promise.resolve();
    assert.equal(widget.value, '20');
    canvas.deselect(first);
    await Promise.resolve();
    assert.equal(widget.value, 'None');
});

test('uses the displayed subgraph and updates every check node in it', async () => {
    const { app, graph, canvas, check, first, widget } = await fixture();
    const other = { ...check, id: 8, widgets: [{ name: 'selected_node_id', value: 'None' }] };
    graph._nodes.push(other);
    app.graph = { _nodes: [] };
    canvas.select(first);
    await Promise.resolve();
    assert.equal(widget.value, '20');
    assert.equal(other.widgets[0].value, '20');
});

test('classic selected_nodes works and repeated setup does not wrap twice', async () => {
    const { canvas, extension, first, widget } = await fixture({ classic: true });
    const wrapped = canvas.onSelectionChange;
    await extension.setup();
    assert.equal(canvas.onSelectionChange, wrapped);
    canvas.selected_nodes = { 20: first };
    canvas.onSelectionChange(canvas.selected_nodes);
    await Promise.resolve();
    assert.equal(widget.value, '20');
    canvas.selected_nodes = {};
    canvas.onSelectionChange({});
    await Promise.resolve();
    assert.equal(widget.value, 'None');
});

test('preserves existing selection callback context, arguments and return value', async () => {
    const { canvas, extension, first, widget } = await fixture({ classic: true });
    delete canvas.__dinki_node_check_attached;
    let seen;
    canvas.onSelectionChange = function(...args) { seen = [this, ...args]; return 42; };
    await extension.setup();
    canvas.selected_nodes = { 20: first };
    assert.equal(canvas.onSelectionChange(canvas.selected_nodes, 'extra'), 42);
    await Promise.resolve();
    assert.deepEqual(seen, [canvas, canvas.selected_nodes, 'extra']);
    assert.equal(widget.value, '20');
});
