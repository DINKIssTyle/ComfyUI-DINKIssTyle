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

async function runtime(nodeClass = 'DINKI_Preview_Image') {
    let extension;
    const events = new Map();
    const document = { createElement: element, visibilityState: 'visible',
        addEventListener(name, handler) { events.set(`document:${name}`, handler); } };
    const app = { nodeOutputs: {}, rootGraph: { _nodes: [] }, registerExtension(ext) {
        if (ext.name === 'DINKI.PreviewImage.Resolution') extension = ext;
    } };
    const api = { apiURL: path => path,
        addEventListener(name, handler) { events.set(`api:${name}`, handler); } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api, document, queueMicrotask, URLSearchParams,
        window: { addEventListener(name, handler) { events.set(`window:${name}`, handler); } },
    });
    class Viewer {
        constructor(id = 7, properties = {}, graph = app.rootGraph) {
            this.id = id; this.comfyClass = nodeClass; this.properties = properties;
            this.graph = graph; graph._nodes.push(this); this.onNodeCreated();
        }
        addDOMWidget(name, type, container, options) {
            this.container = container; this.previewWidget = { options }; return this.previewWidget;
        }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Viewer, { name: nodeClass });
    extension.setup?.();
    const emit = (name, detail) => events.get(name)?.({ detail });
    return { app, extension, Viewer, document, emit };
}

const output = filename => ({ dkst_images: [{ filename, type: 'temp' }] });

test('Preview Cache renders and restores the same persistent preview and batch controls as Viewer', async () => {
    const { app, extension, Viewer, emit } = await runtime('DINKI_Preview_Image_Cache');
    const saved = { dkst_images: [
        { filename: 'depth-one.png', subfolder: 'dkst_preview_cache', type: 'output' },
        { filename: 'depth-two.png', subfolder: 'dkst_preview_cache', type: 'output' },
    ], resolution: ['832 × 1248'] };
    const node = new Viewer(7, { dkstPreview: saved });
    extension.loadedGraphNode(node);
    assert.match(node.container.children[0].src, /depth-one.png/);
    assert.equal(node.container.children[1].textContent, '832 × 1248');
    const selector = node.container.children[2];
    selector.value = '1'; selector.onchange();
    assert.match(node.container.children[0].src, /depth-two.png/);
    app.nodeOutputs[7] = output('refreshed-depth.png');
    emit('api:executed', { node: '7', output: app.nodeOutputs[7] });
    assert.match(node.container.children[0].src, /refreshed-depth.png/);
    assert.equal(node.properties.dkstPreview.dkst_images[0].filename, 'refreshed-depth.png');
});

test('returning to a browser tab replaces an old preview with the latest stored output', async () => {
    const { app, Viewer, document, emit } = await runtime();
    const viewer = new Viewer();
    viewer.onExecuted(output('old.png'));
    document.visibilityState = 'hidden';
    // Native output storage can advance without calling the custom DOM handler.
    app.nodeOutputs[7] = output('latest.png');
    emit('document:visibilitychange');
    assert.match(viewer.container.children[0].src, /old.png/);
    document.visibilityState = 'visible';
    emit('document:visibilitychange');
    assert.match(viewer.container.children[0].src, /latest.png/);
    assert.equal(viewer.properties.dkstPreview.dkst_images[0].filename, 'latest.png');
});

test('an offscreen Viewer catches up when drawn without resetting batch selection every frame', async () => {
    const { app, Viewer } = await runtime();
    const viewer = new Viewer();
    viewer.onExecuted(output('old.png'));
    app.nodeOutputs[7] = { dkst_images: [
        { filename: 'new-one.png', type: 'temp' }, { filename: 'new-two.png', type: 'temp' },
    ] };
    viewer.onDrawBackground();
    assert.match(viewer.container.children[0].src, /new-one.png/);
    const selector = viewer.container.children[2];
    selector.value = '1'; selector.onchange();
    const selectedURL = viewer.container.children[0].src;
    viewer.onDrawBackground(); viewer.onDrawBackground();
    assert.equal(viewer.container.children[0].src, selectedURL);
    assert.equal(selector.value, '1');
});

test('direct execution events refresh hidden Viewers once and honor merged batches', async () => {
    const { app, Viewer, document, emit } = await runtime();
    const viewer = new Viewer();
    document.visibilityState = 'hidden';
    const message = output('hidden.png');
    app.nodeOutputs[7] = message;
    emit('api:executed', { node: '7', output: message });
    assert.match(viewer.container.children[0].src, /hidden.png/);
    const next = output('last.png');
    app.nodeOutputs[7] = { dkst_images: [...message.dkst_images, ...next.dkst_images] };
    viewer.onExecuted(next);
    emit('api:executed', { node: '7', output: next, merge: true });
    assert.equal(viewer.dkstPreviewImages.length, 2);
    const final = output('final.png');
    app.nodeOutputs[7] = final;
    viewer.onExecuted(final);
    const finalURL = viewer.container.children[0].src;
    emit('api:executed', { node: '7', output: final });
    assert.equal(viewer.container.children[0].src, finalURL);
});

test('workflow output restoration supersedes saved properties after configuration', async () => {
    const { app, extension, Viewer } = await runtime();
    app.nodeOutputs[7] = output('other-workflow.png');
    const viewer = new Viewer(7, { dkstPreview: output('saved.png') });
    extension.loadedGraphNode(viewer); viewer.onConfigure(); await Promise.resolve();
    viewer.onDrawBackground();
    assert.match(viewer.container.children[0].src, /saved.png/);
    app.nodeOutputs = { 7: output('completed-in-background.png') };
    extension.onNodeOutputsUpdated(app.nodeOutputs);
    assert.match(viewer.container.children[0].src, /completed-in-background.png/);
});

test('restoring native outputs while hidden overrides the configuration baseline', async () => {
    const { app, extension, Viewer, document } = await runtime();
    const completed = output('completed.png');
    app.nodeOutputs[7] = completed;
    const viewer = new Viewer(7, { dkstPreview: output('saved.png') });
    extension.loadedGraphNode(viewer);
    assert.match(viewer.container.children[0].src, /saved.png/);
    document.visibilityState = 'hidden';
    extension.onNodeOutputsUpdated(app.nodeOutputs);
    assert.match(viewer.container.children[0].src, /completed.png/);
    const url = viewer.container.children[0].src;
    extension.onNodeOutputsUpdated(app.nodeOutputs);
    assert.equal(viewer.container.children[0].src, url);
});

test('focus retries a failed image request and removed nodes ignore late callbacks', async () => {
    const { app, Viewer, emit } = await runtime();
    const viewer = new Viewer();
    const message = output('latest.png');
    app.nodeOutputs[7] = message; viewer.onExecuted(message);
    const image = viewer.container.children[0];
    image.complete = true; image.naturalWidth = 0;
    image.onerror();
    const failedURL = image.src;
    emit('window:focus');
    assert.notEqual(image.src, failedURL);
    assert.match(image.src, /latest.png/);
    const currentURL = image.src;
    viewer.onRemoved();
    app.nodeOutputs[7] = output('late.png');
    emit('api:executed', { node: '7', output: app.nodeOutputs[7] });
    emit('window:pageshow'); viewer.onDrawBackground();
    assert.equal(image.src, currentURL);
    assert.equal(image.onload, null);
});

test('subgraph output keys and execution paths do not collide with root node IDs', async () => {
    const { app, Viewer, emit } = await runtime();
    const root = new Viewer();
    const graph = { id: 'subgraph-uuid', _nodes: [] };
    app.rootGraph._nodes.push({ id: 20, subgraph: graph });
    const nested = new Viewer(7, {}, graph);
    app.nodeOutputs[7] = output('root.png');
    app.nodeOutputs['subgraph-uuid:7'] = output('nested.png');
    nested.onDrawBackground(); root.onDrawBackground();
    assert.match(nested.container.children[0].src, /nested.png/);
    assert.match(root.container.children[0].src, /root.png/);
    const message = output('nested-new.png');
    app.nodeOutputs['subgraph-uuid:7'] = message;
    emit('api:executed', { node: '20:7', output: message });
    assert.match(nested.container.children[0].src, /nested-new.png/);
    assert.match(root.container.children[0].src, /root.png/);
});

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
