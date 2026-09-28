const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

function element(tagName) {
    return {
        tagName: tagName.toUpperCase(), style: {}, children: [], events: {},
        get firstChild() { return this.children[0]; },
        append(...children) { this.children.push(...children); },
        appendChild(child) { this.children.push(child); },
        addEventListener(name, callback) { this.events[name] = callback; },
        removeAttribute(name) { delete this[name]; },
        remove() { this.removed = true; },
        contains() { return false; },
        click() { this.clicked = true; },
        pause() { this.paused = true; },
        load() { this.loaded = true; },
    };
}

async function fixture() {
    let extension, opened, fetched;
    const elements = [];
    const createElement = tag => {
        const result = element(tag);
        elements.push(result);
        return result;
    };
    const body = createElement('body');
    const app = { nodeOutputs: {}, registerExtension(value) {
        if (value.name === 'DINKI.VideoViewer') extension = value;
    } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: { apiURL: path => path },
        document: { createElement, body, addEventListener() {}, removeEventListener() {} },
        window: { innerWidth: 1000, innerHeight: 800, open: url => { opened = url; } },
        fetch: async url => { fetched = url; return { ok: true, blob: async() => ({}) }; },
        URL: { createObjectURL: () => 'blob:video', revokeObjectURL() {} },
        URLSearchParams, queueMicrotask, setTimeout() {},
    });
    class Viewer {
        constructor(properties = {}) {
            this.id = 12;
            this.comfyClass = 'DINKI_Video_Viewer';
            this.properties = properties;
            this.size = [420, 320];
            this.widgets = [
                { name: 'format', value: 'auto', options: { values: ['auto', 'mp4', 'mkv', 'webm'] } },
                { name: 'codec', value: 'auto', options: { values: ['auto', 'h264', 'av1'] } },
            ];
            this.onNodeCreated();
        }
        addDOMWidget(name, type, container, options) {
            this.container = container;
            this.widgetOptions = options;
            return { options: {} };
        }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Viewer, { name: 'DINKI_Video_Viewer' });
    return { app, extension, Viewer, elements,
        get opened() { return opened; }, get fetched() { return fetched; } };
}

function parts(node) {
    const [toolbar, viewport] = node.container.children;
    const [fit, actual, resolution] = toolbar.children;
    return { fit, actual, resolution, viewport, video: viewport.firstChild };
}

test('video stays inside the user-sized node and Fit/100% switch its viewport scale', async () => {
    const { Viewer } = await fixture();
    const node = new Viewer();
    const message = {
        dkst_video: [{ filename: 'final.webm', subfolder: '', type: 'output' }],
        dkst_video_preview: [{ filename: 'preview.mp4', subfolder: '', type: 'temp' }],
        resolution: ['1920 × 1080'],
    };
    node.onExecuted(message);
    const { fit, actual, resolution, viewport, video } = parts(node);
    assert.deepEqual(node.size, [420, 320]);
    assert.equal(node.container.style.width, '100%');
    assert.equal(node.widgetOptions.getHeight(), 240);
    assert.equal(resolution.textContent, '1920 × 1080');
    assert.match(video.src, /filename=preview\.mp4/);
    assert.equal(video.style.objectFit, 'contain');
    assert.equal(video.controls, true);
    actual.onclick();
    assert.equal(video.style.width, '1920px');
    assert.equal(video.style.height, '1080px');
    assert.equal(viewport.style.overflow, 'auto');
    assert.deepEqual(node.size, [420, 320]);
    fit.onclick();
    assert.equal(video.style.width, '100%');
    assert.equal(video.style.objectFit, 'contain');
    assert.equal(viewport.style.overflow, 'hidden');
    video.videoWidth = 1280;
    video.videoHeight = 720;
    video.onloadedmetadata();
    assert.equal(resolution.textContent, '1920 × 1080');
});

test('right-click opens and saves the original file, never the playback copy', async () => {
    const context = await fixture();
    const node = new context.Viewer();
    node.onExecuted({
        dkst_video: [{ filename: 'final.mkv', subfolder: 'job', type: 'output' }],
        dkst_video_preview: [{ filename: 'preview.mp4', subfolder: '', type: 'temp' }],
    });
    const event = { button: 2, clientX: 50, clientY: 50,
        preventDefault() { this.prevented = true; },
        stopImmediatePropagation() { this.stopped = true; } };
    node.container.events.contextmenu(event);
    assert.equal(event.prevented, true);
    assert.equal(event.stopped, true);
    const menuButtons = context.elements.filter(item => item.tagName === 'BUTTON' &&
        ['Open Video', 'Save Video'].includes(item.textContent));
    assert.deepEqual(menuButtons.map(item => item.textContent), ['Open Video', 'Save Video']);
    await menuButtons[0].onclick();
    assert.match(context.opened, /filename=final\.mkv/);
    await menuButtons[1].onclick();
    assert.match(context.fetched, /filename=final\.mkv/);
    assert.equal(context.elements.find(item => item.tagName === 'A').download, 'final.mkv');
    const options = [];
    node.getExtraMenuOptions(null, options);
    assert.deepEqual(options.map(item => item.content), ['Open Video', 'Save Video']);
});

test('Nodes 2.0 recreation restores playback and 100% without resizing the node', async () => {
    const { Viewer, extension } = await fixture();
    const first = new Viewer();
    first.onExecuted({
        dkst_video: [{ filename: 'final.webm', subfolder: '', type: 'temp' }],
        dkst_video_preview: [{ filename: 'preview.mp4', subfolder: '', type: 'temp' }],
        resolution: ['640 × 480'],
    });
    parts(first).actual.onclick();
    const restored = new Viewer(structuredClone(first.properties));
    extension.loadedGraphNode(restored);
    assert.match(parts(restored).video.src, /filename=preview\.mp4/);
    assert.equal(parts(restored).video.style.width, '640px');
    assert.deepEqual(restored.size, [420, 320]);
    restored.onConfigure();
    await Promise.resolve();
    assert.equal(parts(restored).video.style.width, '640px');
});

test('WebM hides H.264 in classic and Nodes 2.0 controls', async () => {
    const { Viewer } = await fixture();
    const node = new Viewer();
    const [format, codec] = node.widgets;
    codec.value = 'h264';
    format.callback('webm');
    assert.deepEqual(Array.from(codec.options.values), ['auto', 'av1']);
    assert.equal(codec.value, 'auto');
    node.onWidgetChanged('format', 'mp4');
    assert.deepEqual(Array.from(codec.options.values), ['auto', 'h264', 'av1']);
});
