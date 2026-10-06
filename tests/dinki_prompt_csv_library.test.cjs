const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname,
    '../ComfyUI-DINKIssTyle/js/dinki_prompt_csv_library.js'), 'utf8');

function getterOptions(widget) {
    const options = widget.options ?? {};
    Object.defineProperty(widget, 'options', { get: () => options, configurable: true });
    if (widget._state?.options) {
        const stateOptions = widget._state.options;
        Object.defineProperty(widget._state, 'options', { get: () => stateOptions });
    }
    return widget;
}

function fixture() {
    const copied = [];
    let extension, graphTick;
    const scheduled = [];
    const graph = { _nodes: [] };
    const files = ['Cinema_Prompt.csv', 'Other.csv'];
    const entries = {
        'Cinema_Prompt.csv': { Camera: { Zoom: 'slow zoom', Dolly: 'slow dolly' }, Light: { Day: 'daylight' } },
        'Other.csv': { Style: { Noir: 'dark shadows' } },
    };
    const app = { canvas: { graph }, registerExtension(value) { extension = value; } };
    const api = { async fetchApi(url) {
        return { ok: true, async json() {
            if (url.endsWith('/files')) return { files, default: 'Cinema_Prompt.csv' };
            const name = new URL(`https://example.test${url}`).searchParams.get('file');
            return { sections: entries[name] };
        } };
    } };
    vm.runInNewContext("'use strict';\n" + source.replace(/^import .*;\r?\n/gm, ''), {
        app, api, URLSearchParams,
        navigator: { clipboard: { async writeText(text) { copied.push(text); } } },
        clearTimeout() {},
        requestAnimationFrame: callback => callback(),
        setTimeout: callback => scheduled.push(callback),
        setInterval: callback => { graphTick = callback; return 1; },
        console,
    });
    const combo = (name, value) => ({ name, value,
        options: { values: [] }, _state: { options: { values: [] } } });
    class Node {
        constructor() {
            this.comfyClass = 'DINKI_PromptCsvLibrary';
            this.graph = graph;
            this.size = [300, 300];
            this.widgets = [combo('csv_file', 'Cinema_Prompt.csv'), combo('section', '-- None --'),
                combo('title', '-- None --'), { name: 'prompt', type: 'customtext', value: '',
                    element: { tagName: 'TEXTAREA' }, options: { multiline: true } }];
            this.widgets.forEach(getterOptions);
            this.onNodeCreated();
            graph._nodes.push(this);
        }
        addWidget(type, name, value, callback, options = {}) {
            const widget = getterOptions({ type, name, value, callback, options });
            this.widgets.push(widget);
            return widget;
        }
        setDirtyCanvas() {}
        onConfigure(saved) {
            for (const [name, value] of Object.entries(saved || {})) {
                this.widgets.find(widget => widget.name === name).value = value;
            }
        }
    }
    extension.beforeRegisterNodeDef(Node, { name: 'DINKI_PromptCsvLibrary' });
    const drain = async() => {
        while (scheduled.length) scheduled.shift()();
        await new Promise(resolve => setImmediate(resolve));
    };
    return { Node, extension, app, entries, drain, copied, getTick: () => graphTick };
}

test('file, section and title selections populate editable prompt and Clear only clears selections', async() => {
    const { Node, drain, entries } = fixture();
    const node = new Node();
    const get = name => node.widgets.find(widget => widget.name === name);
    await drain();
    assert.deepEqual(Array.from(get('section').options.values), ['-- None --', 'Camera', 'Light']);
    assert.deepEqual(Array.from(get('section')._state.options.values), ['-- None --', 'Camera', 'Light']);
    get('section').callback('Camera');
    assert.deepEqual(Array.from(get('title')._state.options.values), ['-- None --', 'Zoom', 'Dolly']);
    get('title').callback('Zoom');
    assert.equal(get('prompt').value, 'slow zoom');
    get('prompt').value = 'manual edit';
    entries['Cinema_Prompt.csv'].Camera.Zoom = 'new zoom';
    get('Refresh').callback();
    await drain();
    assert.equal(get('prompt').value, 'new zoom');
    get('Clear').callback();
    assert.equal(get('csv_file').value, 'Cinema_Prompt.csv');
    assert.equal(get('section').value, '-- None --');
    assert.equal(get('title').value, '-- None --');
    assert.equal(get('prompt').value, 'new zoom');
    assert.deepEqual(Array.from(get('section')._state.options.values), ['-- None --', 'Camera', 'Light']);
    assert.deepEqual(Array.from(get('title')._state.options.values), ['-- None --']);
    node.onConfigure({ csv_file: 'Cinema_Prompt.csv', section: '-- None --',
        title: '-- None --', prompt: 'new zoom' });
    await drain();
    assert.equal(get('prompt').value, 'new zoom');
});

test('workflow restoration keeps a manually edited prompt and reselects file-specific titles', async() => {
    const { Node, extension, app, drain, getTick } = fixture();
    const node = new Node();
    node.onConfigure({ csv_file: 'Other.csv', section: 'Style', title: 'Noir',
        prompt: 'my saved override' });
    extension.loadedGraphNode(node);
    await drain();
    const get = name => node.widgets.find(widget => widget.name === name);
    assert.deepEqual(Array.from(get('title')._state.options.values), ['-- None --', 'Noir']);
    assert.equal(get('prompt').value, 'my saved override');

    extension.setup();
    app.canvas.graph = { _nodes: [] };
    getTick()();
    app.canvas.graph = node.graph;
    getTick()();
    await drain();
    assert.equal(get('prompt').value, 'my saved override');
    assert.deepEqual(Array.from(get('title').options.values), ['-- None --', 'Noir']);
});


test('native prompt remains promotable and Copy Prompt follows edits and restoration', async() => {
    const { Node, drain, copied } = fixture();
    const node = new Node();
    await drain();
    const get = name => node.widgets.find(widget => widget.name === name);
    const prompt = get('prompt');
    const button = get('Copy Prompt');
    assert.equal(prompt.type, 'customtext');
    assert.equal(prompt.element.tagName, 'TEXTAREA');
    assert.notEqual(prompt.hidden, true);
    assert.notEqual(prompt.options.hidden, true);
    assert.equal(node.widgets[node.widgets.indexOf(prompt) + 1], button);
    assert.equal(button.serialize, false);
    assert.equal(button.options.serialize, false);
    assert.equal(node.widgets.some(widget => widget.name === 'dkst_prompt_editor'), false);
    get('section').callback('Camera');
    get('title').callback('Zoom');
    await button.callback();
    assert.equal(copied[0], 'slow zoom');
    prompt.value = '수정한 프롬프트\nsecond line';
    await button.callback();
    assert.equal(copied[1], '수정한 프롬프트\nsecond line');
    node.onConfigure({ prompt: 'saved prompt' });
    await drain();
    await button.callback();
    assert.equal(copied[2], 'saved prompt');
    for (const name of ['csv_file', 'Clear', 'Refresh']) {
        assert.equal(get(name).advanced, true);
        assert.equal(get(name).options.advanced, true);
    }
    assert.notEqual(button.advanced, true);
});

function promote(node, app, { nested = false, legacy = false } = {}) {
    node.id = 17;
    node.inputs = ['section', 'title', 'prompt'].map(name => ({ name, widget: { name } }));
    const makeHost = (graph, target, sourceNames) => {
        const widgets = sourceNames.map((name, index) => ({ name: `renamed_${index}`,
            value: index === 2 ? 'old text' : 'old selection', options: { values: ['old choice'] },
            _state: { options: { values: ['old choice'] } } }));
        widgets.forEach(getterOptions);
        const inputs = widgets.map(widget => ({ name: widget.name, _widget: widget }));
        graph.inputNode = { slots: inputs.map((input, index) => ({ name: input.name, linkIds: [index] })) };
        graph.getLink = index => ({ resolve: () => ({ inputNode: target,
            input: target.inputs.find(input => input.name === sourceNames[index]) }) });
        const host = { id: 21, subgraph: graph, widgets, inputs, setDirtyCanvas() {} };
        if (legacy) {
            host.inputs = [];
            host.properties = { proxyWidgets: sourceNames.map(name => [String(target.id), name]) };
        }
        return host;
    };
    const inner = makeHost(node.graph, node, node.inputs.map(input => input.name));
    let outer = inner;
    if (nested) {
        inner.graph = { _nodes: [inner], incrementVersion() {} };
        outer = makeHost(inner.graph, inner, inner.inputs.map(input => input.name));
    }
    app.rootGraph = { _nodes: [outer], incrementVersion() {} };
    outer.graph = app.rootGraph;
    return { inner, outer };
}

for (const variant of [{}, { nested: true }, { legacy: true }]) {
    test(`CSV changes propagate renamed boundary menus and prompt stores ${JSON.stringify(variant)}`, async() => {
        const { Node, app, drain } = fixture();
        const node = new Node();
        const { inner, outer } = promote(node, app, variant);
        const stored = {};
        outer.widgets.forEach(widget => { widget.options.setValue = value => { stored[widget.name] = value; }; });
        const get = name => node.widgets.find(widget => widget.name === name);
        await drain();
        get('section').callback('Camera');
        get('title').callback('Zoom');
        assert.equal(outer.widgets[2].value, 'slow zoom');
        assert.equal(stored.renamed_2, 'slow zoom');
        get('csv_file').callback('Other.csv');
        await drain();
        for (const host of new Set([inner, outer])) {
            assert.deepEqual(Array.from(host.widgets[0].options.values), ['-- None --', 'Style']);
            assert.deepEqual(Array.from(host.widgets[0]._state.options.values), ['-- None --', 'Style']);
            assert.equal(host.widgets[0].value, '-- None --');
            assert.equal(host.widgets[1].value, '-- None --');
            assert.deepEqual(Array.from(host.widgets[1].options.values), ['-- None --']);
            assert.equal(host.widgets[2].value, '');
        }
        get('section').callback('Style');
        assert.deepEqual(Array.from(outer.widgets[1].options.values), ['-- None --', 'Noir']);
        get('title').callback('Noir');
        assert.equal(outer.widgets[2].value, 'dark shadows');
        assert.equal(stored.renamed_2, 'dark shadows');
        outer.widgets[2].value = 'saved outer edit';
        node.onConfigure({ csv_file: 'Other.csv', section: 'Style', title: 'Noir', prompt: 'saved inner edit' });
        await drain();
        assert.deepEqual(Array.from(outer.widgets[0].options.values), ['-- None --', 'Style']);
        assert.deepEqual(Array.from(outer.widgets[1].options.values), ['-- None --', 'Noir']);
        assert.equal(outer.widgets[2].value, 'saved outer edit');
    });
}

for (const variant of [{}, { nested: true }, { legacy: true }]) {
    test(`outer section/title selections load CSV text ${JSON.stringify(variant)}`, async() => {
        const { Node, app, drain, extension, getTick } = fixture();
        const node = new Node();
        const { outer, inner } = promote(node, app, variant);
        await drain();
        extension.setup();
        await drain();
        outer.widgets[0].value = 'Camera';
        getTick()();
        await drain();
        assert.deepEqual(Array.from(outer.widgets[1].options.values), ['-- None --', 'Zoom', 'Dolly']);
        assert.equal(node.widgets.find(widget => widget.name === 'section').value, 'Camera');
        outer.widgets[1].value = 'Dolly';
        getTick()();
        await drain();
        assert.equal(outer.widgets[2].value, 'slow dolly');
        assert.equal(inner.widgets[2].value, 'slow dolly');
        outer.widgets[2].value = 'manual outer text';
        getTick()();
        assert.equal(node.widgets.find(widget => widget.name === 'prompt').value, 'manual outer text');
        getTick()();
        assert.equal(outer.widgets[2].value, 'manual outer text');
    });
}
