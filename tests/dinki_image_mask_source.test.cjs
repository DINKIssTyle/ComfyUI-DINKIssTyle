const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));

async function createLoader(nodeClass) {
    let extension, opened;
    const images = [];
    const files = ['original.png', 'other.png', 'edited.png', 'uploaded.png'];
    const element = () => ({ style: {}, append() {}, addEventListener() {}, removeAttribute() {} });
    class Image {
        constructor() { images.push(this); }
        load() { this.naturalWidth = 640; this.naturalHeight = 480; this.onload?.(); }
    }
    const app = { registerExtension(ext) {
        if (ext.name === 'DINKI.ImageLoad') extension = ext;
    } };
    const ComfyApp = {
        copyToClipspace(node) { this.clipspace = { imgs: [...node.imgs] }; },
        open_maskeditor() {
            // Native editors prefer the file descriptor and image widget,
            // even when the custom crop canvas already displays a new image.
            const node = this.clipspace_return_node;
            opened = { descriptor: { ...node.images[0] },
                widget: node.widgets.find(w => w.name === 'image').value,
                preview: node.imgs[0].src };
        },
    };
    const api = {
        apiURL: value => value,
        async fetchApi(url, options) {
            return { ok: true, async json() {
                if (url === '/upload/image') {
                    return { name: options.body.type === 'temp' ? 'DKST_Paste_new.png' : 'uploaded.png' };
                }
                return url.endsWith('categories') ? { categories: ['', 'folder', 'clipspace'] } : { files };
            } };
        },
    };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, ComfyApp, api, Image, URLSearchParams, console,
        document: { createElement: element }, requestAnimationFrame() {},
        File: class {}, FormData: class { append(key, value) { this[key] = value; } },
        alert(message) { throw new Error(message); },
    });
    class Node {
        constructor() {
            this.images = [];
            this.imgs = [];
            this.widgets = ['category', 'filename', 'source_type'].map((name, i) =>
                ({ name, value: ['folder', 'original.png', 'input'][i] }));
        }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget); return widget;
        }
        addDOMWidget() { return {}; }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Node, { name: nodeClass });
    const node = new Node(); node.onNodeCreated();
    const widget = name => node.widgets.find(w => w.name === name);
    const select = async filename => {
        widget('filename').value = filename;
        await widget('filename').callback(filename);
        return images.at(-1);
    };
    const open = () => {
        const menu = []; node.getExtraMenuOptions(null, menu);
        // Load has its action on the preview menu; Load & Crop also exposes it
        // on the node menu. The shared source is used by native commands too.
        const action = menu.find(item => item.content === 'Open Mask Editor');
        if (action) action.callback();
        else {
            ComfyApp.copyToClipspace(node);
            ComfyApp.clipspace_return_node = node;
            ComfyApp.open_maskeditor();
        }
        return opened;
    };
    return { node, widget, images, select, open };
}

for (const nodeClass of ['DINKI_Image_Load', 'DINKI_Image_Load_Crop']) {
    test(`${nodeClass}: changing a masked image updates native editor references`, async () => {
        const { node, widget, images, select, open } = await createLoader(nodeClass);
        (await select('original.png')).load();

        // Saving a mask leaves both the native descriptor and widget on the
        // edited file. Reopening that same selection should keep its mask.
        node.images = [{ filename: 'edited.png', subfolder: 'clipspace', type: 'input' }];
        node.imgs = [{ src: '/view?filename=edited.png&subfolder=clipspace&type=input' }];
        widget('image').value = 'clipspace/edited.png [input]';
        await settle();
        images.at(-1).load();
        assert.equal(open().descriptor.filename, 'edited.png');
        assert.equal(open().widget, 'clipspace/edited.png [input]');

        const replacement = await select('other.png');
        assert.equal(node.dkstLoadedImage, null, 'loading must not expose the old masked preview');
        assert.equal(node.imgs.length, 0);
        assert.equal(node.images.length, 0);
        assert.equal(widget('image').value, 'clipspace/other.png [input]');
        replacement.load();
        assert.deepEqual(open().descriptor, { filename: 'other.png', subfolder: 'clipspace', type: 'input' });
        assert.match(open().preview, /filename=other.png/);

        // The same filename in another category is a different source.
        widget('category').value = 'folder';
        await widget('category').callback('folder');
        await settle();
        images.at(-1).load();
        assert.equal(open().descriptor.subfolder, 'folder');
        assert.equal(open().widget, 'folder/other.png [input]');

        await node.dkstUploadClipboardImage({ type: 'image/png' });
        images.at(-1).load();
        assert.deepEqual(open().descriptor, { filename: 'DKST_Paste_new.png', subfolder: '', type: 'temp' });
        assert.equal(open().widget, 'DKST_Paste_new.png [temp]');

        await node.dkstUploadDroppedImage({ name: 'photo.png', type: 'image/png' });
        images.at(-1).load();
        assert.equal(open().descriptor.filename, 'uploaded.png');
        assert.equal(open().descriptor.type, 'input');
        assert.equal(open().widget, 'uploaded.png [input]');
    });

    test(`${nodeClass}: stale or failed loads cannot restore the previous mask source`, async () => {
        const { node, widget, select } = await createLoader(nodeClass);
        const first = await select('edited.png');
        const second = await select('other.png');
        second.load(); first.load();
        assert.equal(node.images[0]?.filename, 'other.png');
        assert.equal(node.imgs[0], second);
        assert.equal(widget('image').value, 'folder/other.png [input]');

        const failed = await select('missing.png');
        failed.onerror(); first.load();
        assert.equal(node.dkstLoadedImage, null);
        assert.equal(node.images.length, 0);
        assert.equal(node.imgs.length, 0);
        assert.equal(widget('image').value, 'folder/missing.png [input]');
        await select('');
        assert.equal(widget('image').value, '');
    });
}
