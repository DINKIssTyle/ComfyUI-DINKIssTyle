const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');
const start = source.indexOf('// 6-2. Preview XMP Node Logic');
const end = source.indexOf('// 7. DINKI Video Player', start);
const section = source.slice(start, end);

async function fixture() {
    let extension;
    const handlers = new Set();
    const requests = [];
    const revoked = [];
    const app = {
        registerExtension(value) { extension = value; },
        graph: { setDirtyCanvas() {} },
    };
    const api = {
        addEventListener(name, handler) { if (name === 'executed') handlers.add(handler); },
        removeEventListener(name, handler) { if (name === 'executed') handlers.delete(handler); },
        async fetchApi(path, options) {
            requests.push({ path, data: JSON.parse(options.body) });
            return { status: 200, blob: async () => ({}) };
        },
    };
    const URL = {
        createObjectURL() { return `blob:${requests.length}`; },
        revokeObjectURL(url) { revoked.push(url); },
    };
    class Image {
        set src(value) { this._src = value; this.onload?.(); }
        get src() { return this._src; }
    }
    vm.runInNewContext(section, { app, api, URL, Image, console });
    class Node {
        constructor(id) {
            this.id = id;
            this.widgets = [
                { name: 'xmp_file', value: 'preset.xmp' },
                { name: 'strength', value: 0.75 },
                { name: 'grain_seed', value: 42 },
            ];
            this.size = [200, 240];
        }
        addWidget() {}
    }
    await extension.beforeRegisterNodeDef(Node, { name: 'DINKI_Adobe_XMP_Preview' }, app);
    const nodes = [new Node(1), new Node(2)];
    nodes.forEach(node => node.onNodeCreated());
    return { handlers, requests, revoked, nodes };
}

test('XMP preview requests use the token from their own node execution', async () => {
    const { handlers, requests, nodes } = await fixture();
    assert.equal(handlers.size, 2);
    for (const handler of handlers) handler({ detail: { node: 1, output: { preview_token: ['first-token'] } } });
    for (const handler of handlers) handler({ detail: { node: 2, output: { preview_token: ['second-token'] } } });
    await new Promise(setImmediate);
    assert.deepEqual(requests.map(request => request.data.preview_token), ['first-token', 'second-token']);
    assert.equal(requests[0].data.grain_seed, 42);
    nodes[0].widgets[1].value = 0.5;
    nodes[0].widgets[1].callback();
    await new Promise(setImmediate);
    assert.equal(requests.at(-1).data.preview_token, 'first-token');
    assert.equal(requests.at(-1).data.strength, 0.5);
});

test('XMP preview releases its event listener and object URL when removed', async () => {
    const { handlers, revoked, nodes } = await fixture();
    for (const handler of handlers) handler({ detail: { node: 1, output: { preview_token: ['first-token'] } } });
    await new Promise(setImmediate);
    nodes[0].onRemoved();
    assert.equal(handlers.size, 1);
    assert.deepEqual(revoked, ['blob:1']);
});
