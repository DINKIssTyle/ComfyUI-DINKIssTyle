const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname,
    '../ComfyUI-DINKIssTyle/js/dinki_prompt_csv_library.js'), 'utf8');

function fixture() {
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
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api, URLSearchParams,
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
                combo('title', '-- None --'), { name: 'prompt', value: '' }];
            this.onNodeCreated();
            graph._nodes.push(this);
        }
        addWidget(type, name, value, callback) {
            const widget = { type, name, value, callback };
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
    return { Node, extension, app, entries, drain, getTick: () => graphTick };
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
