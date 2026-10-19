const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');
const names = ['filename_prefix', 'format', 'bit_depth', 'input_color_space', 'always_save', 'meta_save'];

async function fixture({ available = true, readonly = false, nodeClass = 'DINKI_Preview_Image' } = {}) {
    let extension;
    const pending = [], sources = [];
    const graph = { _nodes: [], add(node) { node.id = 100 + sources.length; sources.push(node); },
        remove(node) { sources.splice(sources.indexOf(node), 1); }, change() {} };
    const app = { rootGraph: graph, nodeOutputs: {}, registerExtension(ext) {
        if (ext.name === 'DINKI.PreviewImage.Resolution') extension = ext;
    } };
    const createElement = () => ({ style: {}, append() {}, replaceChildren() {},
        addEventListener() {}, removeAttribute() {} });
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: { apiURL: url => url }, URLSearchParams, document: { createElement },
        queueMicrotask: fn => pending.push(fn),
        LiteGraph: { createNode(name) {
            assert.equal(name, 'DINKI_Text_Multiline');
            return available ? { widgets: [{ name: 'text' }], size: [300, 200],
                connect(slot, node, input) { assert.equal(slot, 0); node.inputs[input].link = this.id; } } : null;
        } },
    });
    class Viewer {
        constructor() {
            this.graph = graph; this.comfyClass = nodeClass; this.pos = [800, 40];
            this.widgets = names.map(name => ({ name, options: {} }));
            if (readonly) for (const widget of this.widgets) {
                const options = widget.options;
                Object.defineProperty(widget, 'options', { get: () => options });
            }
            this.inputs = [{ name: 'images' }, { name: 'metadata_text', link: null }];
            this.properties = {}; this.size = new Float32Array([360, 280]);
            this.showAdvanced = false;
            graph._nodes.push(this);
            this.onNodeCreated();
        }
        addDOMWidget(name, type, container, options) {
            const widget = { name, type, options }; this.widgets.push(widget); return widget;
        }
        computeSize() { return new Float32Array([300, this.showAdvanced ? 600 : 400]); }
        setSize(size) {
            const min = this.computeSize();
            this.size = new Float32Array([Math.max(size[0], min[0]), Math.max(size[1], min[1])]);
            this.onResize?.(this.size);
        }
        toggleAdvanced() { this.showAdvanced = !this.showAdvanced; this.setSize(this.size); }
        setDirtyCanvas() {}
        onConfigure(info) {
            Object.assign(this.properties, info.properties);
            if (info.showAdvanced !== undefined) this.showAdvanced = info.showAdvanced;
            info.widgets_values?.forEach((value, index) => { this.widgets[index].value = value; });
            if (info.size) this.setSize(info.size);
            return 42;
        }
        onSerialize(info) {
            info.properties = { ...this.properties };
            info.widgets_values = this.widgets.filter(widget => widget.serialize !== false).map(widget => widget.value);
            return 99;
        }
    }
    await extension.beforeRegisterNodeDef(Viewer, { name: nodeClass });
    return { app, Viewer, extension, sources, flush() { while (pending.length) pending.shift()(); } };
}

test('only the four file settings are advanced in both widget renderers', async () => {
    const f = await fixture({ readonly: true }), node = new f.Viewer();
    for (const name of names) {
        const widget = node.widgets.find(widget => widget.name === name);
        const advanced = names.indexOf(name) < 4;
        assert.equal(widget.advanced === true, advanced);
        assert.equal(widget.options.advanced === true, advanced);
    }
    assert.equal(node.widgets.some(widget => widget.name === 'metadata_text'), false);
});

test('advanced toggles restore manual preview space and survive typed-array workflow geometry', async () => {
    const f = await fixture(), node = new f.Viewer();
    assert.deepEqual(Array.from(node.computeSize()), [300, 280]);
    node.toggleAdvanced();
    assert.equal(node.size[1], 600);
    node.toggleAdvanced();
    assert.equal(node.size[1], 280);
    node.setSize([420, 330]);
    node.toggleAdvanced(); node.toggleAdvanced();
    assert.deepEqual(Array.from(node.size), [420, 330]);
    const saved = { size: Array.from(node.size), showAdvanced: false };
    assert.equal(node.onSerialize(saved), 99);
    const restored = new f.Viewer();
    assert.equal(restored.onConfigure(saved), 42);
    restored.toggleAdvanced(); restored.toggleAdvanced();
    assert.deepEqual(Array.from(restored.size), [420, 330]);
    f.flush();
});

test('legacy inline text becomes a separate connected source without shifting file settings', async () => {
    const f = await fixture(), node = new f.Viewer();
    const values = ['saved/name', 'png', '16bit', 'sRGB', true, true];
    const text = '  한글 prompt\nline two  ';
    f.app.configuringGraph = true;
    node.onConfigure({ properties: {}, widgets_values: [...values, text], size: [360, 740] });
    f.flush();
    assert.equal(f.sources.length, 0);
    assert.deepEqual(node.widgets.slice(0, 6).map(widget => widget.value), values);
    assert.equal(node.size[1], 280);
    f.app.configuringGraph = false;
    f.extension.afterConfigureGraph();
    assert.equal(f.sources.length, 1);
    assert.equal(f.sources[0].widgets[0].value, text);
    const textSource = f.sources[0];
    textSource.widgets[0].options.dynamicPrompts = true;
    textSource.comfyClass = 'DINKI_Text_Multiline';
    f.extension.loadedGraphNode(textSource);
    assert.equal(textSource.widgets[0].options.dynamicPrompts, false, 'legacy prompt must stay literal on reload');
    assert.equal(node.inputs[1].link, f.sources[0].id);
    assert.equal(node.properties.dkstLegacyMetadataText, undefined);
    f.extension.afterConfigureGraph();
    assert.equal(f.sources.length, 1);
    const saved = {}; node.onSerialize(saved);
    assert.deepEqual(saved.widgets_values, values);
});

test('existing text links are retained and stale widget markers are removed', async () => {
    const f = await fixture(), node = new f.Viewer();
    node.inputs[1] = { name: 'metadata_text', link: 25, widget: { name: 'metadata_text' } };
    node.onConfigure({ properties: {}, widgets_values: ['', 'png', '8bit', 'sRGB', false, true, 'unused'] });
    f.flush();
    assert.equal(node.inputs[1].link, 25);
    assert.equal(node.inputs[1].widget, undefined);
    assert.equal(f.sources.length, 0);
});

test('unavailable text node preserves pending legacy text for workflow saving and retry', async () => {
    const f = await fixture({ available: false }), node = new f.Viewer();
    node.onConfigure({ widgets_values: ['', 'png', '8bit', 'sRGB', false, true, 'keep this'] });
    f.flush();
    const saved = {}; node.onSerialize(saved);
    assert.equal(saved.properties.dkstLegacyMetadataText, 'keep this');
    assert.equal(f.sources.length, 0);
});

test('Preview Cache keeps its independent controls and sizing', async () => {
    const f = await fixture({ nodeClass: 'DINKI_Preview_Image_Cache' }), node = new f.Viewer();
    assert.equal(node.dkstMigrateMetadata, undefined);
    assert.equal(node.widgets[0].advanced, undefined);
    assert.equal(node.computeSize()[1], 400);
});
