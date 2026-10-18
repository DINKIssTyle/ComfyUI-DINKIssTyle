const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');
const widget = (node, name) => node.widgets.find(item => item.name === name);
const view = canvas => ({ scale: canvas.ds.scale, offset: Array.from(canvas.ds.offset) });

function fixture({ classic = false } = {}) {
    const extensions = new Map(), listeners = [], frames = [], warnings = [];
    const graph = { _nodes: [], findNodesByType(type) {
        return this._nodes.filter(node => (node.comfyClass ?? node.type) === type);
    } };
    const canvas = {
        graph, ds: { scale: 1.25, offset: new Float32Array([17, -23]), max_scale: 3 },
        canvas: { getBoundingClientRect: () => ({ width: 800, height: 600 }) },
        selected_nodes: {}, selectedItems: new Set(), setDirty() {},
        setGraph(graph) { this.graph = graph; return 'changed'; },
        select(node) { this.selectedItems.add(node); },
        deselect(node) { this.selectedItems.delete(node); },
        deselectAll() { this.selectedItems.clear(); },
        onSelectionChange() {},
    };
    if (classic) delete canvas.selectedItems;
    const app = { canvas, graph, registerExtension(extension) { extensions.set(extension.name, extension); },
        extensionManager: { toast: { add(message) { warnings.push(message); } } } };
    const document = { activeElement: { tagName: 'BODY', matches: () => false } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: {}, document, queueMicrotask,
        window: { addEventListener(type, callback, capture) { listeners.push({ type, callback, capture }); } },
        requestAnimationFrame(callback) { frames.push(callback); },
    });
    const extension = extensions.get('Dinki.FitNode');
    extension.setup();
    const makeControl = (id, key, targetId) => {
        const node = { id, comfyClass: 'DINKI_Fit_Node', widgets: [
            { name: 'smoothness', value: 1 }, { name: 'target_node_id', value: String(targetId) },
            { name: 'shortcut_key', value: key },
        ] };
        graph._nodes.push(node);
        return node;
    };
    const control = makeControl(1, 'f', 2);
    const target = { id: 2, pos: [300, 200], size: [1200, 600],
        getBounding() { return [this.pos[0], this.pos[1] - 30, this.size[0], this.size[1] + 30]; } };
    graph._nodes.push(target);
    return { app, graph, canvas, control, target, extension, extensions, warnings, frames, listeners, document,
        makeControl,
        key(key = 'f', options = {}) {
            const event = { key, defaultPrevented: false, ...options,
                preventDefault() { this.defaultPrevented = true; },
                stopImmediatePropagation() { this.stopped = true; } };
            for (const listener of [...listeners.filter(item => item.capture),
                ...listeners.filter(item => !item.capture)]) {
                if (listener.type === 'keydown') listener.callback(event);
                if (event.stopped) break;
            }
            return event;
        },
        finish() {
            for (let i = 0; frames.length && i < 2000; i++) frames.shift()();
            assert.equal(frames.length, 0, 'viewport animation must settle');
        },
    };
}

function assertFitted(canvas, box) {
    const rect = canvas.canvas.getBoundingClientRect();
    const [x, y, width, height] = box, { scale, offset } = canvas.ds;
    assert.ok(width * scale <= rect.width * 0.9 + 1e-8);
    assert.ok(height * scale <= rect.height * 0.9 + 1e-8);
    assert.ok(Math.abs((x + width / 2 + offset[0]) * scale - rect.width / 2) < 1e-8);
    assert.ok(Math.abs((y + height / 2 + offset[1]) * scale - rect.height / 2) < 1e-8);
}

for (const classic of [false, true]) {
    test(`${classic ? 'classic' : 'Nodes 2.0'} fits full visible bounds and restores exact position and zoom`, () => {
        const f = fixture({ classic });
        const original = view(f.canvas);
        const selected = classic ? f.canvas.selected_nodes : f.canvas.selectedItems;
        const key = f.key('F');
        assert.equal(key.defaultPrevented, true);
        f.finish();
        assertFitted(f.canvas, f.target.getBounding());
        assert.equal(f.canvas.ds.scale, 0.6);
        assert.equal(classic ? f.canvas.selected_nodes : f.canvas.selectedItems, selected);
        f.key(); f.finish();
        assert.deepEqual(view(f.canvas), original);
    });
}

test('snapshot copies offset and survives manual pan/zoom and target deletion', () => {
    const f = fixture(), original = view(f.canvas), originalOffset = f.canvas.ds.offset;
    f.key(); f.finish();
    originalOffset[0] = 999;
    f.canvas.ds.offset[0] = -450;
    f.canvas.ds.offset[1] = 200;
    f.canvas.ds.scale = 2.7;
    f.graph._nodes.splice(f.graph._nodes.indexOf(f.target), 1);
    f.key(); f.finish();
    assert.deepEqual(view(f.canvas), original);
    f.graph._nodes.push(f.target);
    f.canvas.ds.offset = [80, 90];
    const fresh = view(f.canvas);
    f.key(); f.finish(); f.key(); f.finish();
    assert.deepEqual(view(f.canvas), fresh);
});

test('a second press redirects an unfinished animation back to the original view', () => {
    const f = fixture(), original = view(f.canvas);
    widget(f.control, 'smoothness').value = 0.2;
    f.key();
    f.frames.shift()();
    assert.notDeepEqual(view(f.canvas), original);
    f.key(); f.finish();
    assert.deepEqual(view(f.canvas), original);
});

test('switching A to B retains the first return view and uses B current geometry', () => {
    const f = fixture(), original = view(f.canvas);
    const b = f.makeControl(3, 'g', 4);
    const targetB = { id: 4, pos: [-400, 500], size: [100, 120] };
    f.graph._nodes.push(targetB);
    f.key(); f.finish();
    targetB.pos = [-700, 150];
    targetB.size = [500, 1000];
    f.key('g'); f.finish();
    assertFitted(f.canvas, [-700, 120, 500, 1030]);
    widget(b, 'target_node_id').value = 'deleted';
    f.key('g'); f.finish();
    assert.deepEqual(view(f.canvas), original);
});

test('collapsed bounds and fallback geometry fit what is visible', () => {
    const f = fixture();
    f.target.flags = { collapsed: true };
    f.target.getBounding = () => [300, 170, 150, 30];
    f.key(); f.finish();
    assertFitted(f.canvas, [300, 170, 150, 30]);
    f.key(); f.finish();
    delete f.target.getBounding;
    f.target._collapsed_width = 90;
    f.key(); f.finish();
    assertFitted(f.canvas, [300, 170, 90, 30]);
});

test('Fit respects max zoom and adapts to a resized viewport', () => {
    const f = fixture();
    f.target.size = [10, 10];
    f.canvas.ds.max_scale = 2.5;
    f.key(); f.finish();
    assert.equal(f.canvas.ds.scale, 2.5);
    f.key(); f.finish();
    delete f.canvas.ds.max_scale;
    f.target.size = [1000, 1500];
    f.canvas.canvas.getBoundingClientRect = () => ({ width: 400, height: 1200 });
    f.key(); f.finish();
    assertFitted(f.canvas, f.target.getBounding());
});

test('missing IDs, malformed bounds and zero-size viewports leave view and session untouched', () => {
    for (const invalidate of [
        f => { widget(f.control, 'target_node_id').value = ''; },
        f => { widget(f.control, 'target_node_id').value = '999'; },
        f => { f.target.getBounding = () => [NaN, 0, 50, 50]; },
        f => { f.target.getBounding = () => [0, 0, 50, 0]; },
        f => { f.canvas.canvas.getBoundingClientRect = () => ({ width: 0, height: 0 }); },
        f => { widget(f.control, 'smoothness').value = 0; },
    ]) {
        const f = fixture(), original = view(f.canvas);
        invalidate(f);
        f.key(); f.finish();
        assert.deepEqual(view(f.canvas), original);
        assert.equal(f.frames.length, 0);
    }
    const f = fixture(), original = view(f.canvas);
    f.makeControl(3, 'g', 999);
    f.key(); f.finish();
    const fitted = view(f.canvas);
    f.key('g'); f.finish();
    assert.deepEqual(view(f.canvas), fitted);
    f.key(); f.finish();
    assert.deepEqual(view(f.canvas), original);
});

test('typing, composition, modified keys and handled events do not trigger Fit', () => {
    const f = fixture(), original = view(f.canvas);
    for (const options of [{ ctrlKey: true }, { metaKey: true }, { altKey: true },
        { isComposing: true }, { defaultPrevented: true }]) {
        f.key('f', options);
    }
    f.document.activeElement = { matches: () => true };
    f.key();
    f.document.activeElement = { isContentEditable: true };
    f.key();
    f.finish();
    assert.deepEqual(view(f.canvas), original);
});

test('held-key repeats cannot toggle back, and blank keys do not claim other shortcuts', () => {
    const f = fixture();
    f.key(); f.finish();
    const fitted = view(f.canvas);
    assert.equal(f.key('f', { repeat: true }).defaultPrevented, true);
    f.finish();
    assert.deepEqual(view(f.canvas), fitted);
    widget(f.control, 'shortcut_key').value = '';
    assert.equal(f.key('x').defaultPrevented, false);
});

for (const type of ['DINKI_Fit_Node', 'DINKI_Anchor', 'DINKI_Auto_Focus']) {
    test(`a duplicate ${type} key warns before either viewport control runs`, () => {
        const f = fixture(), original = view(f.canvas);
        const other = { id: 5, comfyClass: type, widgets: [
            { name: 'shortcut_key', value: 'F' }, { name: 'zoom_levels', value: '200' },
            { name: 'enable', value: true },
        ] };
        f.graph._nodes.push(other);
        // Register actual existing listeners as well, so event ordering is tested.
        f.extensions.get('Dinki.Anchor').setup();
        f.extensions.get('Dinki.AutoFocus').setup();
        f.key(); f.finish();
        assert.deepEqual(view(f.canvas), original);
        assert.equal(widget(other, 'enable').value, true);
        assert.equal(f.warnings.length, 1);
        assert.match(f.warnings[0].detail, /unique key/);
        f.key('f', { repeat: true });
        assert.equal(f.warnings.length, 1);
    });
}

test('repeated setup retains one listener and the original setGraph result', () => {
    const f = fixture(), wrapped = f.canvas.setGraph;
    f.extension.setup();
    assert.equal(f.listeners.length, 1);
    assert.equal(f.canvas.setGraph, wrapped);
    assert.equal(f.canvas.setGraph(f.graph), 'changed');
});

test('leaving and returning to the same graph discards the old return view', () => {
    const f = fixture();
    f.key(); f.finish();
    f.canvas.setGraph({ _nodes: [] });
    f.canvas.setGraph(f.graph);
    f.canvas.ds.offset = [200, 300]; f.canvas.ds.scale = 0.75;
    const returned = view(f.canvas);
    f.key(); f.finish(); f.key(); f.finish();
    assert.deepEqual(view(f.canvas), returned);
});

test('workflow reconfiguration resets the session even when the graph object is reused', () => {
    for (const hook of ['beforeConfigureGraph', 'afterConfigureGraph']) {
        const f = fixture();
        f.key(); f.frames.shift()();
        f.extension[hook]();
        f.canvas.ds.offset = [80, 90]; f.canvas.ds.scale = 0.8;
        const loaded = view(f.canvas);
        f.finish();
        assert.deepEqual(view(f.canvas), loaded);
        f.key(); f.finish(); f.key(); f.finish();
        assert.deepEqual(view(f.canvas), loaded);
    }
});

test('graph changes cancel outgoing Fit and return animations', () => {
    for (const returning of [false, true]) {
        const f = fixture();
        f.key(); f.finish();
        if (returning) f.key();
        else { f.key(); f.finish(); f.key(); }
        f.canvas.setGraph({ _nodes: [] });
        f.canvas.ds.offset = [9, 8]; f.canvas.ds.scale = 0.7;
        const other = view(f.canvas);
        f.finish();
        assert.deepEqual(view(f.canvas), other);
    }
});

test('targets and controls resolve in the displayed subgraph with local IDs', () => {
    const f = fixture();
    f.app.graph = { _nodes: [{ id: 2, pos: [10000, 10000], size: [10, 10] }] };
    delete f.graph._nodes;
    f.graph.nodes = [f.control, f.target];
    f.key(); f.finish();
    assertFitted(f.canvas, f.target.getBounding());
    f.app.graph._nodes.push({ comfyClass: 'DINKI_Anchor', widgets: [{ name: 'shortcut_key', value: 'f' }] });
    const fitted = view(f.canvas);
    f.key(); f.finish();
    assert.equal(f.warnings.length, 1);
    assert.deepEqual(view(f.canvas), fitted);
});

test('Fit does not select its target or trigger Auto Focus', async () => {
    const f = fixture();
    const autofocus = { comfyClass: 'DINKI_Auto_Focus', widgets: [
        { name: 'shortcut_key', value: 'a' }, { name: 'enable', value: true },
        { name: 'smoothness', value: 1 }, { name: 'zoom_level', value: 2 },
    ] };
    f.graph._nodes.push(autofocus);
    f.canvas.selectedItems.add(f.control);
    f.extensions.get('Dinki.AutoFocus').setup();
    f.key(); await Promise.resolve(); f.finish();
    assertFitted(f.canvas, f.target.getBounding());
    assert.deepEqual([...f.canvas.selectedItems], [f.control]);
    assert.equal(widget(autofocus, 'enable').value, true);
});
