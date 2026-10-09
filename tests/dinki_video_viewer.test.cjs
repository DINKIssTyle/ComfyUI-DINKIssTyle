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

async function fixture(className = "DINKI_Video_Viewer", nodeData = {}) {
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
            this.comfyClass = className;
            this.properties = properties;
            this.size = [420, 320];
            this.widgets = [
                { name: 'format', value: className === 'DINKI_Video_Combine' ? 'h264-mp4' : 'auto', options: { values: ['auto', 'mp4', 'mkv', 'webm'] } },
                ...(className === 'DINKI_Video_Combine' ? [] : [{ name: 'codec', value: 'auto', options: { values: ['auto', 'h264', 'av1'] } }]),
                { name: 'pixel_format', value: className === 'DINKI_Video_Combine' ? 'yuv420p' : 'auto', options: {} },
                { name: 'bitrate_mbps', value: 8, options: {} },
                { name: 'encoder', value: 'auto', options: {} },
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
    await extension.beforeRegisterNodeDef(Viewer, { name: className, ...nodeData });
    return { app, extension, Viewer, elements,
        get opened() { return opened; }, get fetched() { return fetched; } };
}

function parts(node) {
    const [toolbar, viewport] = node.container.children;
    const [fit, actual, resolution] = toolbar.children;
    return { fit, actual, resolution, viewport, video: viewport.firstChild, image: viewport.children[1] };
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

test('Combine uses the same Fit/100% viewer and restores its saved preview', async () => {
    const { Viewer, extension } = await fixture('DINKI_Video_Combine');
    const node = new Viewer();
    node.onExecuted({ dkst_video: [{ filename: 'combined.mp4', subfolder: '', type: 'temp' }], resolution: ['1280 × 720'] });
    assert.match(parts(node).video.src, /filename=combined\.mp4/);
    parts(node).actual.onclick();
    assert.equal(parts(node).video.style.width, '1280px');
    const restored = new Viewer(structuredClone(node.properties));
    extension.loadedGraphNode(restored);
    assert.equal(parts(restored).video.style.width, '1280px');
    assert.deepEqual(restored.size, [420, 320]);
});

test('GIF/WebP use an animated image preview with Fit/100% and original file actions', async () => {
    const { Viewer } = await fixture('DINKI_Video_Combine');
    const node = new Viewer();
    for (const extension of ['gif', 'webp']) {
        node.onExecuted({ dkst_video: [{ filename: `animation.${extension}`, subfolder: '', type: 'output' }], resolution: ['640 × 480'] });
        const { video, image, fit, actual, viewport, resolution } = parts(node);
        assert.match(image.src, new RegExp(`filename=animation\\.${extension}`));
        assert.equal(image.style.display, 'block');
        assert.equal(video.style.display, 'none');
        assert.equal(video.src, undefined);
        assert.equal(resolution.textContent, '640 × 480');
        actual.onclick();
        assert.equal(image.style.width, '640px');
        assert.equal(viewport.style.overflow, 'auto');
        fit.onclick();
        assert.equal(image.style.objectFit, 'contain');
        assert.equal(node.dkstSavedVideo.filename, `animation.${extension}`);
    }
    node.onRemoved();
    assert.equal(parts(node).image.src, undefined);
});

test('format selection limits pixel formats and disables irrelevant encoding controls', async () => {
    const { Viewer } = await fixture('DINKI_Video_Combine');
    const node = new Viewer();
    const get = name => node.widgets.find(widget => widget.name === name);
    get('format').callback('prores-mov');
    assert.equal(get('pixel_format').value, 'yuv422p10le');
    assert.equal(get('bitrate_mbps').disabled, true);
    assert.deepEqual(Array.from(get('pixel_format').options.values), ['auto', 'yuv422p10le', 'yuv444p10le']);
    get('format').callback('gif');
    assert.equal(get('pixel_format').value, 'auto');
    assert.equal(get('pixel_format').disabled, true);
    get('format').callback('h264-mp4');
    assert.equal(get('pixel_format').disabled, false);
    assert.equal(get('bitrate_mbps').disabled, false);
});

test('extended Player presets choose their codec and preserve resolution for 100% proxy playback', async () => {
    const { Viewer } = await fixture();
    const node = new Viewer();
    node.widgets.find(widget => widget.name === 'format').callback('h265-mp4');
    assert.equal(node.widgets.find(widget => widget.name === 'codec').disabled, true);
    node.onExecuted({ dkst_video: [{ filename: 'hevc.mp4', type: 'output', subfolder: '' }],
        dkst_video_preview: [{ filename: 'proxy.mp4', type: 'temp', subfolder: '' }], resolution: ['1920 × 1080'] });
    parts(node).video.videoWidth = 640; parts(node).video.videoHeight = 360;
    parts(node).video.onloadedmetadata(); parts(node).actual.onclick();
    assert.equal(parts(node).video.style.width, '1920px');
    assert.equal(parts(node).resolution.textContent, '1920 × 1080');
    node.widgets.find(widget => widget.name === 'format').callback('mp4');
    assert.equal(node.widgets.find(widget => widget.name === 'codec').disabled, false);
});

test('old workflow encoding controls migrate to CPU while new nodes default to Auto', async () => {
    for (const type of ['DINKI_Video_Viewer', 'DINKI_Video_Combine']) {
        const { Viewer } = await fixture(type);
        const getEncoder = node => node.widgets.find(widget => widget.name === 'encoder');
        const node = new Viewer();
        assert.equal(getEncoder(node).value, 'auto');
        const legacyProperties = {};
        const oldNode = new Viewer(legacyProperties);
        oldNode.onConfigure({ widgets_values: ['DKST_Video', 'h264-mp4', 24, 'yuv420p', 8, false], properties: legacyProperties });
        assert.equal(getEncoder(oldNode).value, 'cpu', 'Creating a node must not mark shared legacy properties as Auto');
        node.onConfigure({ widgets_values: ['DKST_Video', 'auto', 'auto', false], properties: {} });
        assert.equal(getEncoder(node).value, 'cpu');
        node.onConfigure({ widgets_values: ['DKST_Video', 'mp4', 'auto', false, 'auto', 0, null], properties: {} });
        assert.equal(getEncoder(node).value, 'cpu');
        node.onConfigure({ widgets_values: ['DKST_Video', 'mp4', 'auto', false, 'auto', 0, 'videotoolbox'], properties: {} });
        assert.equal(getEncoder(node).value, 'videotoolbox');
        const restored = new Viewer(structuredClone(node.properties));
        assert.equal(getEncoder(restored).value, 'videotoolbox');
    }
});

test('encoder controls use server-supported pixel formats and persist Nodes 2.0 changes', async () => {
    const { Viewer } = await fixture('DINKI_Video_Combine', { input: { required: { format: [[], {
        dkst_encoder_pixels: { 'h264-mp4': {
            auto: ['auto', 'yuv420p', 'yuv444p10le'], cpu: ['auto', 'yuv420p', 'yuv444p10le'],
            videotoolbox: ['auto', 'yuv420p', 'nv12'], nvenc: ['auto', 'yuv420p', 'p010le'],
        } },
    }] } } });
    const node = new Viewer();
    const get = name => node.widgets.find(widget => widget.name === name);
    get('pixel_format').value = 'yuv444p10le';
    get('encoder').callback('videotoolbox');
    assert.deepEqual(Array.from(get('pixel_format').options.values), ['auto', 'yuv420p', 'nv12']);
    assert.equal(get('pixel_format').value, 'yuv444p10le', 'Changing device must not silently reduce precision');
    node.onWidgetChanged('encoder', 'nvenc');
    assert.deepEqual(Array.from(get('pixel_format').options.values), ['auto', 'yuv420p', 'p010le']);
    assert.equal(node.properties.dkstEncoder, 'nvenc');
    get('format').callback('gif');
    assert.equal(get('encoder').disabled, true);
    assert.equal(get('encoder').value, 'cpu');
});

test('actual encoder, proxy encoder and fallback reason survive preview restoration', async () => {
    const { Viewer, extension } = await fixture('DINKI_Video_Combine');
    const node = new Viewer();
    node.onExecuted({
        dkst_video: [{ filename: 'hevc.mp4', type: 'temp', subfolder: '' }],
        dkst_video_preview: [{ filename: 'preview.mp4', type: 'temp', subfolder: '' }],
        encoding: [{ label: 'HEVC · CPU', bitrate_mbps: 8, reason: 'Device busy' }],
        preview_encoding: [{ label: 'H.264 · VideoToolbox' }], resolution: ['128 × 128'],
    });
    const status = node.container.children[0].children[3];
    assert.match(status.textContent, /HEVC · CPU.*Preview: H.264 · VideoToolbox.*8 Mbps.*CPU fallback/);
    assert.equal(status.title, 'Device busy');
    const restored = new Viewer(structuredClone(node.properties));
    extension.loadedGraphNode(restored);
    assert.equal(restored.container.children[0].children[3].textContent, status.textContent);
});
