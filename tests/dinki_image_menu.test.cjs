const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');

test('image menu opens original and passes temp source to mask editor; saved reference updates loader', async () => {
    const elements = [];
    function element(tag) {
        const el = { tag, style: {}, children: [], events: {},
            append(...items) { this.children.push(...items); },
            appendChild(item) { this.children.push(item); },
            addEventListener(name, fn) { this.events[name] = fn; },
            setAttribute() {}, removeAttribute() {}, remove() {}, focus() {},
            contains() { return false; }, querySelector() { return this.children[0]; }
        };
        elements.push(el);
        return el;
    }
    let extension, opened, editorOpened = false;
    const requests = [];
    const app = { registerExtension(ext) { if (ext.name === 'DINKI.ImageLoad') extension = ext; } };
    const ComfyApp = {
        copyToClipspace() { this.clipspace = {}; },
        open_maskeditor() { editorOpened = true; }
    };
    const api = {
        apiURL: (url) => url,
        async fetchApi(url) {
            requests.push(url);
            return { ok: true, async json() {
                if (url === '/upload/image') return { name: 'DKST_Paste_new.png' };
                return url.endsWith('categories') ? { categories: ['', 'clipspace'] } : { files: ['edited.png'] };
            } };
        }
    };
    vm.runInNewContext(readFileSync(require('node:path').join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8').replace(/^import .*;\r?\n/gm, ''), {
        app, api, ComfyApp, URLSearchParams, console,
        File: class {}, FormData: class { append() {} },
        document: { createElement: element, body: element('body'), addEventListener() {}, removeEventListener() {} },
        window: { innerWidth: 1000, innerHeight: 800, open: (...args) => { opened = args; } },
        Image: class {}, requestAnimationFrame() {}, alert: (message) => { throw new Error(message); }
    });
    class Node {
        constructor() {
            this.widgets = ['category', 'filename', 'source_type'].map((name, index) => ({ name, value: ['folder', 'DKST_Paste_1.png', 'temp'][index] }));
        }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget); return widget;
        }
        addDOMWidget() { return {}; }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Node, { name: 'DINKI_Image_Load' });
    const node = new Node(); node.onNodeCreated();
    node.dkstLoadedImage = { src: '/view?filename=DKST_Paste_1.png&type=temp' };
    const img = elements.find(el => el.tag === 'img');
    const context = () => img.events.contextmenu({ clientX: 100, clientY: 100, preventDefault() {}, stopPropagation() {} });
    context();
    elements.filter(el => el.tag === 'button').at(-2).onclick();
    assert.equal(opened[0], node.dkstLoadedImage.src);
    context();
    elements.filter(el => el.tag === 'button').at(-1).onclick();
    assert.ok(editorOpened);
    assert.equal(ComfyApp.clipspace.images[0].type, 'temp');
    assert.equal(ComfyApp.clipspace.images[0].subfolder, '');
    assert.equal(ComfyApp.clipspace_return_node, node);
    node.widgets.find(w => w.name === 'image').value = 'clipspace/edited.png [input]';
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(node.widgets[0].value, 'clipspace');
    assert.equal(node.widgets[1].value, 'edited.png');
    assert.equal(node.widgets[2].value, 'input');
    assert.ok(requests.includes('/dinki/image-load/delete-temp'));
    await node.dkstUploadClipboardImage({ type: 'image/png' });
    assert.deepEqual(Array.from(node.widgets[1].options.values), ['DKST_Paste_new.png', 'edited.png']);
    assert.equal(node.widgets[2].value, 'temp');
    node.widgets[1].value = 'edited.png';
    await node.widgets[1].callback('edited.png');
    assert.equal(node.widgets[2].value, 'input');
    assert.equal(node.widgets[1].value, 'edited.png');
});
