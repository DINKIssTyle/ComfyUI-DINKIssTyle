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
        appendChild(child) { this.children.push(child); },
        replaceChildren(...children) { this.children = children; },
        addEventListener(name, callback) { this.events[name] = callback; },
        remove() {}, contains() { return false; },
        click() { this.clicked = true; },
        pause() { this.paused = true; },
    };
}

function fixture() {
    let extension, opened, fetched;
    const elements = [];
    const createElement = tag => {
        const value = element(tag);
        elements.push(value);
        return value;
    };
    const body = createElement('body');
    const app = { nodeOutputs: {}, registerExtension(value) {
        if (value.name === 'DINKI.VideoComparer.Preview') extension = value;
    } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: { apiURL: path => path },
        document: { createElement, body, addEventListener() {}, removeEventListener() {} },
        window: { innerWidth: 1000, innerHeight: 800, open: url => { opened = url; } },
        fetch: async url => { fetched = url; return { ok: true, blob: async() => ({}) }; },
        URL: { createObjectURL: () => 'blob:video', revokeObjectURL() {} },
        URLSearchParams, queueMicrotask, setTimeout() {},
    });
    class Comparer {
        constructor(properties = {}) {
            this.id = 7;
            this.comfyClass = 'DINKI_Image_Comparer_MOV';
            this.properties = properties;
            this.onNodeCreated();
        }
        addDOMWidget(_name, _type, container, options) {
            this.container = container;
            this.widgetOptions = options;
            return { options: {} };
        }
        onExecuted(message) { this.previousMessage = message; }
        getExtraMenuOptions(_canvas, options) { options.push({ content: 'Existing' }); }
        setDirtyCanvas() {}
    }
    extension.beforeRegisterNodeDef(Comparer, { name: 'DINKI_Image_Comparer_MOV' });
    return { app, extension, Comparer, elements, body,
        get opened() { return opened; }, get fetched() { return fetched; } };
}

test('MP4 plays in the comparer and preview right-click shows only video actions', async () => {
    const context = fixture();
    const node = new context.Comparer();
    const descriptor = { filename: 'comparison.mp4', type: 'temp', subfolder: 'previews' };
    node.onExecuted({ video: [descriptor] });
    assert.equal(node.previousMessage.video[0], descriptor);
    const video = node.container.firstChild;
    assert.equal(video.tagName, 'VIDEO');
    assert.equal(video.controls, true);
    assert.equal(video.style.objectFit, 'contain');
    assert.match(video.src, /filename=comparison\.mp4/);
    assert.match(video.src, /type=temp/);
    assert.equal(node.widgetOptions.getMinHeight(), 160);
    assert.ok(!context.body.children.includes(node.container));

    const event = { button: 2, clientX: 50, clientY: 50,
        preventDefault() { this.prevented = true; },
        stopImmediatePropagation() { this.stopped = true; },
        stopPropagation() { this.stopped = true; } };
    node.container.events.pointerdown(event);
    node.container.events.contextmenu(event);
    assert.equal(event.prevented, true);
    assert.equal(event.stopped, true);
    const buttons = context.elements.filter(item => item.tagName === 'BUTTON');
    assert.deepEqual(buttons.map(item => item.textContent), ['Open Video', 'Save Video']);
    await buttons[0].onclick();
    assert.match(context.opened, /filename=comparison\.mp4/);
    await buttons[1].onclick();
    assert.match(context.fetched, /filename=comparison\.mp4/);
    assert.match(context.fetched, /subfolder=previews/);
    assert.equal(context.elements.find(item => item.tagName === 'A').download, 'comparison.mp4');
    const nodeMenu = [];
    node.getExtraMenuOptions(null, nodeMenu);
    assert.deepEqual(nodeMenu.map(item => item.content), ['Existing']);
});

test('GIF and WebP previews retain the current file across node recreation', async () => {
    const { extension, Comparer } = fixture();
    const node = new Comparer();
    node.onExecuted({ video: [{ filename: 'first.gif', type: 'output', subfolder: '' }] });
    const first = node.container.firstChild;
    assert.equal(first.tagName, 'IMG');
    node.onExecuted({ video: [{ filename: 'latest.webp', type: 'output', subfolder: '' }] });
    assert.equal(node.container.firstChild.tagName, 'IMG');
    assert.match(node.container.firstChild.src, /latest\.webp/);

    const restored = new Comparer(structuredClone(node.properties));
    extension.loadedGraphNode(restored);
    assert.equal(restored.container.firstChild.tagName, 'IMG');
    assert.match(restored.container.firstChild.src, /latest\.webp/);
    restored.onConfigure();
    await Promise.resolve();
    assert.equal(restored.container.children.length, 1);
});
