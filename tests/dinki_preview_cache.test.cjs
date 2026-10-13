const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_preview_cache.js'), 'utf8');
const file = letter => `${letter.repeat(32)}.npy`;

function fixture() {
    let extension, changes = 0, rootChanges = 0, dirty = 0;
    const rootGraph = { change() { rootChanges++; } };
    const app = { rootGraph, registerExtension(value) { extension = value; } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), { app });
    class Cache {
        constructor(filename = '') {
            this.widgets = [{ name: 'cache_mode', value: 'Capture once' },
                { name: 'cache_file', value: filename, options: {} }];
            this.graph = { change() { changes++; } };
            this.onNodeCreated();
        }
        onNodeCreated() { return 11; }
        onExecuted(message, extra) { this.preview = message.dkst_images; this.extra = extra; return 42; }
        onWidgetChanged(...args) { this.widgetChanged = args; }
        setDirtyCanvas() { dirty++; }
        serialize() { return { widgets_values: this.widgets.map(widget => widget.value) }; }
    }
    extension.beforeRegisterNodeDef(Cache, { name: 'DINKI_Preview_Image_Cache' });
    return { Cache, extension, rootGraph, changes: () => changes, rootChanges: () => rootChanges, dirty: () => dirty };
}

test('managed reference is hidden from both UIs and remains in saved workflow widgets', () => {
    const { Cache } = fixture();
    const node = new Cache(file('a'));
    const widget = node.widgets[1];
    assert.equal(widget.hidden, true);
    assert.equal(widget.type, 'converted-widget');
    assert.deepEqual(Array.from(widget.computeSize()), [0, -4]);
    assert.notEqual(widget.serialize, false);
    assert.equal(node.serialize().widgets_values[1], file('a'));
});

test('capture updates saved reference and keeps the existing preview handler', () => {
    const f = fixture(), node = new f.Cache();
    const message = { dkst_cache_file: [file('a')], dkst_images: [{ filename: 'depth.png' }] };
    assert.equal(node.onExecuted(message, 'extra'), 42);
    assert.deepEqual(node.preview, message.dkst_images);
    assert.equal(node.extra, 'extra');
    assert.deepEqual(node.widgetChanged, ['cache_file', file('a'), '']);
    assert.equal(node.serialize().widgets_values[1], file('a'));
    assert.equal(f.changes(), 1);
    assert.equal(f.rootChanges(), 1);
    assert.equal(f.dirty(), 1);
});

test('reuse results and malformed references cannot replace a saved snapshot', () => {
    const f = fixture(), node = new f.Cache(file('a'));
    for (const message of [{}, { dkst_cache_file: [] }, { dkst_cache_file: ['', null, '../outside.npy'] },
        { dkst_cache_file: file('b') }]) node.onExecuted(message);
    assert.equal(node.widgets[1].value, file('a'));
    assert.equal(f.changes(), 0);
});

test('refresh picks the last valid snapshot and root graphs are dirtied once', () => {
    const f = fixture(), node = new f.Cache(file('a'));
    node.graph = f.rootGraph;
    node.onExecuted({ dkst_cache_file: [file('a'), file('b'), null] });
    assert.equal(node.widgets[1].value, file('b'));
    assert.equal(f.rootChanges(), 1);
    node.onExecuted({ dkst_cache_file: [file('b')] });
    assert.equal(f.rootChanges(), 1);
});

test('other node types keep their lifecycle', () => {
    const f = fixture();
    class Other { onExecuted() { return 99; } }
    const handler = Other.prototype.onExecuted;
    f.extension.beforeRegisterNodeDef(Other, { name: 'DINKI_Preview_Image' });
    assert.equal(Other.prototype.onExecuted, handler);
});

test('a serialized capture survives node recreation before an empty-input run', () => {
    const { Cache } = fixture();
    let node = new Cache();
    node.onExecuted({ dkst_cache_file: [file('c')], dkst_images: [{ filename: 'depth.png' }] });
    const workflow = JSON.parse(JSON.stringify(node.serialize()));
    node = new Cache(workflow.widgets_values[1]);
    node.onExecuted({ dkst_images: [{ filename: 'depth.png' }] });
    assert.equal(node.serialize().widgets_values[1], file('c'));
    assert.equal(node.widgets[0].value, 'Capture once');
});
