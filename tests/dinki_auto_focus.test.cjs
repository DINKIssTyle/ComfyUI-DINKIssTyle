const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

async function fixture({ classic = false, fit = false, restore = false, targetSize = [100, 100] } = {}) {
    let extension, keydown;
    const frames = [];
    const changes = [];
    const enabled = { name: 'enable', value: true, options: { setValue(value) { this.last = value; } } };
    const focus = { id: 1, comfyClass: 'DINKI_Auto_Focus', widgets: [
        { name: 'smoothness', value: 1 },
        { name: 'shortcut_key', value: 'a' },
        enabled,
        { name: 'zoom_level', value: 2 },
        { name: 'fit', value: fit },
        { name: 'restore_on_deselect', value: restore },
    ], onWidgetChanged(...args) { changes.push(args); }, setDirtyCanvas() {} };
    const target = { id: 2, pos: [300, 200], size: targetSize };
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

test('Fit sizes a large node to the visible canvas with a margin', async () => {
    const f = await fixture({ fit: true, targetSize: [1200, 600] });
    f.canvas.select(f.target);
    await Promise.resolve();
    f.finishAnimation();
    assert.ok(Math.abs(f.canvas.ds.scale - 0.6) < 1e-9);
    assert.ok(f.target.size[0] * f.canvas.ds.scale <= 800 * 0.9);
    assert.ok(f.target.size[1] * f.canvas.ds.scale <= 600 * 0.9);
});

test('Fit caps magnification and leaves manual zoom unchanged when disabled', async () => {
    const f = await fixture({ fit: true, targetSize: [10, 10] });
    f.canvas.ds.max_scale = 2.5;
    f.canvas.select(f.target);
    await Promise.resolve();
    f.finishAnimation();
    assert.equal(f.canvas.ds.scale, 2.5);

    const manual = await fixture({ fit: false, targetSize: [1200, 600] });
    manual.canvas.select(manual.target);
    await Promise.resolve();
    manual.finishAnimation();
    assert.equal(manual.canvas.ds.scale, 2);
});

test('deselecting restores the zoom and position from before the first focus', async () => {
    const f = await fixture({ restore: true });
    f.canvas.ds.offset = [25, -12];
    f.canvas.ds.scale = 1.25;
    f.canvas.select(f.target);
    await Promise.resolve();
    f.finishAnimation();
    assert.equal(f.canvas.ds.scale, 2);
    f.canvas.deselectAll();
    await Promise.resolve();
    f.finishAnimation();
    assert.deepEqual(Array.from(f.canvas.ds.offset), [25, -12]);
    assert.equal(f.canvas.ds.scale, 1.25);
});

test('switching selected nodes keeps the original view until selection is empty', async () => {
    const f = await fixture({ restore: true });
    const another = { id: 3, pos: [50, 60], size: [120, 80] };
    f.graph._nodes.push(another);
    f.canvas.ds.offset = [17, 23];
    f.canvas.select(f.target);
    await Promise.resolve();
    f.finishAnimation();
    f.canvas.deselectAll();
    f.canvas.select(another);
    await Promise.resolve();
    f.finishAnimation();
    assert.equal(f.canvas.ds.scale, 2);
    f.canvas.deselectAll();
    await Promise.resolve();
    f.finishAnimation();
    assert.deepEqual(Array.from(f.canvas.ds.offset), [17, 23]);
    assert.equal(f.canvas.ds.scale, 1);
});

test('restore stays off by default and can cancel a move still in progress', async () => {
    const off = await fixture();
    off.canvas.select(off.target);
    await Promise.resolve();
    off.finishAnimation();
    off.canvas.deselectAll();
    await Promise.resolve();
    assert.equal(off.frames.length, 0);
    assert.equal(off.canvas.ds.scale, 2);

    const on = await fixture({ restore: true });
    on.canvas.select(on.target);
    await Promise.resolve();
    on.canvas.deselectAll();
    await Promise.resolve();
    on.finishAnimation();
    assert.deepEqual(Array.from(on.canvas.ds.offset), [0, 0]);
    assert.equal(on.canvas.ds.scale, 1);
});
