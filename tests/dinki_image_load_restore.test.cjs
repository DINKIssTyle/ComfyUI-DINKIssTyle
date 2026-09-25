const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

test('pasted Image Load preview retains its temp source after tab reconstruction', async () => {
    let extension;
    const scheduled = [];
    const app = { registerExtension(value) {
        if (value.name === 'DINKI.ImageLoad') extension = value;
    } };
    function element(tag) {
        return {
            tag, style: {}, children: [],
            append(...children) { this.children.push(...children); },
            addEventListener() {}, removeAttribute(name) { delete this[name]; },
        };
    }
    class LoadedImage {
        set src(value) {
            this._src = value;
            this.naturalWidth = 640;
            this.naturalHeight = 480;
            this.onload?.();
        }
        get src() { return this._src; }
    }
    const api = {
        apiURL: url => url,
        async fetchApi(url) {
            return { ok: true, async json() {
                return url.endsWith('/categories') ? { categories: ['', 'folder'] }
                    : { files: ['ordinary.png'] };
            } };
        },
    };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api, document: { createElement: element }, Image: LoadedImage,
        URLSearchParams, requestAnimationFrame: fn => fn(),
        setTimeout: fn => { scheduled.push(fn); }, console,
    });
    class Node {
        constructor(properties) {
            this.properties = properties;
            this.widgets = [
                { name: 'category', value: 'folder' },
                { name: 'filename', value: 'DKST_Paste_saved.png' },
                { name: 'source_type', value: 'input' },
            ];
            this.onNodeCreated();
        }
        addDOMWidget(name, type, preview) { this.preview = preview; return {}; }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Node, { name: 'DINKI_Image_Load' });
    const node = new Node({ dkstImageLoad: {
        category: 'folder', filename: 'DKST_Paste_saved.png', source_type: 'temp',
    } });
    node.onConfigure();
    while (scheduled.length) scheduled.shift()();
    await new Promise(resolve => setImmediate(resolve));

    assert.equal(node.widgets[2].value, 'temp');
    assert.equal(node.widgets[1].value, 'DKST_Paste_saved.png');
    assert.deepEqual(Array.from(node.widgets[1].options.values),
        ['DKST_Paste_saved.png', 'ordinary.png']);
    assert.match(node.preview.children[0].src, /filename=DKST_Paste_saved.png/);
    assert.match(node.preview.children[0].src, /type=temp/);
    assert.equal(node.preview.style.display, 'flex');
    assert.equal(node.dkstImageResolution, '640 × 480');
});
