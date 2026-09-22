const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

function loadExtension(promptMap) {
    let extension;
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
        clearTimeout() {}
    });
    return { extension, requested };
}

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
