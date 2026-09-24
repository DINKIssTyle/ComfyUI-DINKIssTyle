const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

function loadExtension(promptMap) {
    let extension, tick;
    const requested = [];
    const app = {
        registerExtension(ext) {
            if (ext.name === 'DINKI.PromptSelectorLive.Attach.v2') extension = ext;
        }
    };
    const api = {
        addEventListener() {},
        async fetchApi(path) {
            requested.push(path);
            return {
                ok: true,
                async json() { return path === '/dinki/prompts' ? promptMap : Object.keys(promptMap); }
            };
        }
    };
    const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api, console, queueMicrotask,
        requestAnimationFrame: fn => fn(),
        setTimeout: fn => { fn(); return 0; },
        clearTimeout() {},
        setInterval(fn) { tick = fn; return 1; }
    });
    return { extension, requested, app, api, tick: () => tick() };
}

async function subgraphFixture({ mode = 'replace', text = '', nested = false } = {}) {
    const f = loadExtension({ A: 'alpha', B: 'beta' });
    class Node {
        constructor() {
            this.id = 1;
            this.widgets = ['title', 'text', 'mode', 'separator'].map((name, index) => ({
                name, value: ['-- None --', '', 'replace', '\\n'][index], options: {}
            }));
            this.inputs = this.widgets.map(w => ({ name: w.name, widget: { name: w.name } }));
        }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget); return widget;
        }
        setDirtyCanvas() {}
    }
    await f.extension.beforeRegisterNodeDef(Node, { name: 'DINKI_PromptSelectorLive' });
    const node = new Node();
    function hostFor(graph, target, sourceNames) {
        const slots = sourceNames.map((name, index) => ({ name: `exposed_${index}`, linkIds: [index] }));
        graph.inputNode = { slots };
        graph.getLink = index => ({ resolve: () => ({ inputNode: target,
            input: target.inputs.find(input => input.name === sourceNames[index]) }) });
        const widgets = slots.map((slot, index) => ({ name: slot.name,
            value: ['-- None --', text, mode, ' | '][index], options: {} }));
        return { subgraph: graph, widgets, inputs: widgets.map(widget => ({ name: widget.name, _widget: widget })) };
    }
    const graph = { _nodes: [node], incrementVersion() {} };
    node.graph = graph;
    const inner = hostFor(graph, node, ['title', 'text', 'mode', 'separator']);
    let outer = inner;
    if (nested) {
        outer = hostFor({ _nodes: [inner] }, inner, inner.inputs.map(input => input.name));
        // Inner promoted inputs are themselves connected to the outer boundary.
        inner.inputs.forEach((input, index) => { input.link = 100 + index; });
    }
    f.app.rootGraph = f.app.graph = { _nodes: [outer] };
    node.onNodeCreated();
    f.extension.setup();
    f.tick();
    return { ...f, node, inner, outer };
}

const flush = () => new Promise(resolve => setImmediate(resolve));

test('host-only title selection writes both original text and renamed outer text', async () => {
    const f = await subgraphFixture();
    let stored;
    f.outer.widgets[1].options.setValue = value => { stored = value; };
    f.outer.widgets[0].value = 'A';
    f.tick();
    await flush();
    assert.equal(f.node.widgets[1].value, 'alpha');
    assert.equal(f.outer.widgets[1].value, 'alpha');
    assert.equal(stored, 'alpha');
    f.tick(); await flush();
    assert.deepEqual(f.requested, ['/dinki/prompts']);
});

test('nested promotions append using the outer text, mode and separator', async () => {
    const f = await subgraphFixture({ nested: true, mode: 'append', text: 'prefix' });
    f.outer.widgets[0].value = 'A';
    f.tick(); await flush();
    assert.equal(f.outer.widgets[1].value, 'prefix | alpha');
    assert.equal(f.inner.widgets[1].value, 'prefix | alpha');
    assert.equal(f.node.widgets[1].value, 'prefix | alpha');
    f.outer.widgets[0].value = 'B';
    f.tick(); await flush();
    assert.equal(f.outer.widgets[1].value, 'prefix | alpha | beta');
    assert.equal(f.requested.length, 2);
});

test('loading a saved append workflow preserves existing text without another append', async () => {
    const f = await subgraphFixture({ mode: 'append', text: 'saved alpha' });
    f.outer.widgets[0].value = 'A';
    f.extension.afterConfigureGraph();
    f.tick(); await flush();
    assert.equal(f.outer.widgets[1].value, 'saved alpha');
    assert.equal(f.requested.length, 0);
});

test('none mode does not alter either text value', async () => {
    const f = await subgraphFixture({ mode: 'none', text: 'manual' });
    f.outer.widgets[0].value = 'A';
    f.tick(); await flush();
    assert.equal(f.outer.widgets[1].value, 'manual');
    assert.equal(f.node.widgets[1].value, '');
});

test('slow responses and removed subgraphs cannot overwrite the latest text', async () => {
    const f = await subgraphFixture();
    const pending = [];
    f.api.fetchApi = () => new Promise(resolve => pending.push(resolve));
    f.outer.widgets[0].value = 'A'; f.tick();
    f.outer.widgets[0].value = 'B'; f.tick();
    pending[1]({ ok: true, json: async() => ({ A: 'alpha', B: 'beta' }) });
    await flush();
    pending[0]({ ok: true, json: async() => ({ A: 'alpha', B: 'beta' }) });
    await flush();
    assert.equal(f.outer.widgets[1].value, 'beta');
    f.outer.widgets[0].value = 'A'; f.tick();
    f.app.rootGraph._nodes = []; f.tick();
    pending[2]({ ok: true, json: async() => ({ A: 'alpha' }) });
    await flush();
    assert.equal(f.outer.widgets[1].value, 'beta');
});

test('a legacy callback plus observer does not append the same selection twice', async () => {
    const f = await subgraphFixture({ mode: 'append', text: 'prefix' });
    f.outer.widgets[0].value = 'A';
    const loading = f.node.widgets[0].callback('A');
    f.tick();
    await loading;
    f.tick(); await flush();
    assert.equal(f.outer.widgets[1].value, 'prefix | alpha');
    assert.equal(f.requested.length, 1);
});

test('two hosts sharing an inner graph update only their own promoted text', async () => {
    const f = await subgraphFixture({ mode: 'append', text: 'first' });
    const widgets = f.outer.widgets.map(widget => ({ ...widget, options: {} }));
    widgets[1].value = 'second';
    const second = { subgraph: f.outer.subgraph, widgets,
        inputs: widgets.map(widget => ({ name: widget.name, _widget: widget })) };
    f.app.rootGraph._nodes.push(second);
    f.tick();
    f.outer.widgets[0].value = 'A';
    second.widgets[0].value = 'B';
    f.tick(); await flush();
    assert.equal(f.outer.widgets[1].value, 'first | alpha');
    assert.equal(second.widgets[1].value, 'second | beta');
});

test('legacy proxy promotions synchronize the host text store', async () => {
    const f = await subgraphFixture();
    f.outer.inputs = [];
    f.outer.properties = { proxyWidgets: ['title', 'text', 'mode', 'separator'].map(name => ['1', name]) };
    f.outer.widgets[0].value = 'A';
    f.tick(); await flush();
    assert.equal(f.outer.widgets[1].value, 'alpha');
});

test('an unknown selection preserves the existing text', async () => {
    const f = await subgraphFixture({ text: 'keep me' });
    f.outer.widgets[0].value = 'Missing title';
    f.tick(); await flush();
    assert.equal(f.outer.widgets[1].value, 'keep me');
});

test('live prompt selector loads the selected preset through the ComfyUI API helper', async () => {
    const { extension, requested } = loadExtension({ Portrait: 'cinematic portrait' });

    class Node {
        constructor() {
            this.widgets = [
                { name: 'title', value: '-- None --', options: { values: ['-- None --', 'Portrait'] }, callback() { throw new Error('legacy callback'); } },
                { name: 'text', value: '' },
                { name: 'mode', value: 'replace' },
                { name: 'separator', value: '\\n' }
            ];
        }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        }
        setDirtyCanvas() {}
    }

    await extension.beforeRegisterNodeDef(Node, { name: 'DINKI_PromptSelectorLive' });
    const node = new Node();
    node.onNodeCreated();
    await node.widgets.find(widget => widget.name === 'title').callback('Portrait');

    assert.equal(node.widgets.find(widget => widget.name === 'text').value, 'cinematic portrait');
    assert.deepEqual(requested, ['/dinki/prompts']);
});

test('live prompt selector handles a Nodes 2.0 widget notification without a classic callback', async () => {
    const { extension, requested } = loadExtension({ Landscape: 'wide mountain vista' });

    class Node {
        constructor() {
            this.widgets = [
                { name: 'title', value: '-- None --', options: { values: ['-- None --', 'Landscape'] } },
                { name: 'text', value: '' },
                { name: 'mode', value: 'replace' },
                { name: 'separator', value: '\\n' }
            ];
        }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        }
        setDirtyCanvas() {}
    }

    await extension.beforeRegisterNodeDef(Node, { name: 'DINKI_PromptSelectorLive' });
    const node = new Node();
    node.onNodeCreated();
    node.onWidgetChanged('title', 'Landscape');
    await new Promise(resolve => setImmediate(resolve));

    assert.equal(node.widgets.find(widget => widget.name === 'text').value, 'wide mountain vista');
    assert.deepEqual(requested, ['/dinki/prompts']);
});

test('live prompt selector synchronizes a Nodes 2.0 multiline value store', async () => {
    const { extension } = loadExtension({ Studio: 'soft studio light' });
    let storedText = '';

    class Node {
        constructor() {
            this.widgets = [
                { name: 'title', value: '-- None --', options: { values: ['-- None --', 'Studio'] } },
                {
                    name: 'text', value: '',
                    options: { setValue(value) { storedText = value; } }
                },
                { name: 'mode', value: 'replace' },
                { name: 'separator', value: '\\n' }
            ];
        }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        }
        setDirtyCanvas() {}
    }

    await extension.beforeRegisterNodeDef(Node, { name: 'DINKI_PromptSelectorLive' });
    const node = new Node();
    node.onNodeCreated();
    node.widgets.find(widget => widget.name === 'title').value = 'Studio';
    node.onDrawForeground();
    await new Promise(resolve => setImmediate(resolve));

    assert.equal(storedText, 'soft studio light');
    assert.equal(node.widgets.find(widget => widget.name === 'text').value, 'soft studio light');
});
