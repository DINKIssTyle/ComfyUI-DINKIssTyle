const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

function element(tag) {
    return {
        tag, style: {}, children: [], value: '', textContent: '',
        append(...children) { this.children.push(...children); },
        appendChild(child) { this.children.push(child); },
        replaceChildren(...children) { this.children = children; },
        removeAttribute(name) { delete this[name]; },
        addEventListener() {},
    };
}

test('Viewer continues updating after an upstream execution handler throws', async () => {
    let extension;
    const app = { registerExtension(ext) {
        if (ext.name === 'DINKI.PreviewImage.Resolution') extension = ext;
    } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: { apiURL: path => path }, document: { createElement: element },
        queueMicrotask, URLSearchParams, Date: { now: () => 1000 },
    });
    class Viewer {
        constructor() { this.properties = {}; this.onNodeCreated(); }
        onExecuted() { throw new Error('native handler failed'); }
        addDOMWidget(name, type, container) { this.container = container; return { options: {} }; }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Viewer, { name: 'DINKI_Preview_Image' });
    const viewer = new Viewer();
    const message = name => ({ dkst_images: [{ filename: name, type: 'temp' }] });
    assert.throws(() => viewer.onExecuted(message('first.png')), /native handler failed/);
    assert.match(viewer.container.children[0].src, /first.png/);
    assert.throws(() => viewer.onExecuted(message('second.png')), /native handler failed/);
    assert.match(viewer.container.children[0].src, /second.png/);
    const image = viewer.container.children[0];
    image.onerror();
    assert.equal(viewer.container.children[1].textContent, 'Unable to preview image');
    const previousURL = image.src;
    assert.throws(() => viewer.onExecuted(message('second.png')), /native handler failed/);
    assert.notEqual(image.src, previousURL);
    image.naturalWidth = 800; image.naturalHeight = 600; image.onload();
    assert.equal(viewer.container.children[1].textContent, '800 × 600');
});

test('Viewer restores its last image after a tab recreates the node', async () => {
    let extension;
    const app = { nodeOutputs: {}, registerExtension(ext) {
        if (ext.name === 'DINKI.PreviewImage.Resolution') extension = ext;
    } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: { apiURL: path => path }, document: { createElement: element },
        queueMicrotask, URLSearchParams,
    });
    class Viewer {
        constructor(properties = {}) {
            this.id = 7;
            this.comfyClass = 'DINKI_Preview_Image';
            this.properties = properties;
            this.onNodeCreated();
        }
        addDOMWidget(name, type, container) {
            this.container = container;
            return { options: {} };
        }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Viewer, { name: 'DINKI_Preview_Image' });
    const first = new Viewer();
    const output = { dkst_images: [{ filename: 'portrait.png', subfolder: '', type: 'temp' }],
        resolution: ['1200 × 1500'] };
    first.onExecuted(output);
    assert.match(first.container.children[0].src, /portrait\.png/);
    assert.equal(first.properties.dkstPreview.dkst_images[0].filename, 'portrait.png');

    const restored = new Viewer(structuredClone(first.properties));
    app.nodeOutputs[7] = { dkst_images: [{ filename: 'another-tab.png', type: 'temp' }] };
    extension.loadedGraphNode(restored);
    assert.match(restored.container.children[0].src, /portrait\.png/);
    assert.equal(restored.container.children[0].style.display, 'block');
    restored.onConfigure();
    await Promise.resolve();
    assert.match(restored.container.children[0].src, /portrait\.png/);
});
