const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

async function fixture({ classic = false } = {}) {
    let extension, keydown;
    const frames = [];
    const changes = [];
    const enabled = { name: 'enable', value: true, options: { setValue(value) { this.last = value; } } };
    const focus = { id: 1, comfyClass: 'DINKI_Auto_Focus', widgets: [
        { name: 'smoothness', value: 1 },
        { name: 'shortcut_key', value: 'a' },
        enabled,
        { name: 'zoom_level', value: 2 },
    ], onWidgetChanged(...args) { changes.push(args); }, setDirtyCanvas() {} };
    const target = { id: 2, pos: [300, 200], size: [100, 100] };
    const graph = { _nodes: [focus, target] };
    const canvas = {
        graph, ds: { offset: [0, 0], scale: 1 },
        canvas: { getBoundingClientRect: () => ({ width: 800, height: 600 }) },
        selected_nodes: {}, setDirty() {},
        onSelectionChange(...args) { this.selectionArgs = args; return 42; },
        select(node) { this.selectedItems.add(node); return 'selected'; },
        deselect(node) { this.selectedItems.delete(node); },
        deselectAll() { this.selectedItems.clear(); },
    };
    if (!classic) canvas.selectedItems = new Set();
    else for (const name of ['select', 'deselect', 'deselectAll']) delete canvas[name];
    const app = { canvas, graph: { _nodes: [] }, registerExtension(value) {
        if (value.name === 'Dinki.AutoFocus') extension = value;
    } };
    const document = { activeElement: { tagName: 'BODY', matches: () => false } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: {}, document,
        window: { addEventListener(name, callback) { if (name === 'keydown') keydown = callback; } },
        queueMicrotask, requestAnimationFrame(callback) { frames.push(callback); },
    });
    extension.setup();
    return { app, canvas, graph, focus, target, enabled, changes, document, extension,
        keydown: event => keydown(event),
        finishAnimation() {
            for (let i = 0; frames.length && i < 100; i++) frames.shift()();
            assert.equal(frames.length, 0);
        }, frames,
    };
}

test('Nodes 2.0 selection focuses the selected node without a classic callback', async () => {
    const f = await fixture();
    assert.equal(f.canvas.select(f.target), 'selected');
    await Promise.resolve();
    assert.equal(f.frames.length, 1);
    f.finishAnimation();
    assert.deepEqual(Array.from(f.canvas.ds.offset), [-150, -100]);
    assert.equal(f.canvas.ds.scale, 2);
});

test('classic selection and repeated setup keep the original callback', async () => {
    const f = await fixture({ classic: true });
    const wrapped = f.canvas.onSelectionChange;
    f.extension.setup();
    assert.equal(f.canvas.onSelectionChange, wrapped);
    f.canvas.selected_nodes = { 2: f.target };
    assert.equal(f.canvas.onSelectionChange('selection'), 42);
    assert.deepEqual(f.canvas.selectionArgs, ['selection']);
    await Promise.resolve();
    f.finishAnimation();
    assert.deepEqual(Array.from(f.canvas.ds.offset), [-150, -100]);
});

test('shortcut updates the named enable widget and ignores text input', async () => {
    const f = await fixture();
    let callbackValue;
    f.enabled.callback = value => { callbackValue = value; };
    f.keydown({ key: 'A' });
    assert.equal(f.enabled.value, false);
    assert.equal(f.enabled.options.last, false);
    assert.equal(callbackValue, false);
    assert.equal(f.changes[0][0], 'enable');
    f.canvas.select(f.target);
    await Promise.resolve();
    assert.equal(f.frames.length, 0);
    f.document.activeElement = { matches: () => true, isContentEditable: false };
    f.keydown({ key: 'a' });
    assert.equal(f.enabled.value, false);
});

test('selected Auto Focus control does not move the canvas', async () => {
    const f = await fixture();
    f.canvas.select(f.focus);
    await Promise.resolve();
    assert.equal(f.frames.length, 0);
});

test('switching the displayed graph stops a pending move', async () => {
    const f = await fixture();
    f.canvas.select(f.target);
    await Promise.resolve();
    f.canvas.graph = { _nodes: [] };
    f.finishAnimation();
    assert.deepEqual(Array.from(f.canvas.ds.offset), [0, 0]);
    assert.equal(f.canvas.ds.scale, 1);
});
