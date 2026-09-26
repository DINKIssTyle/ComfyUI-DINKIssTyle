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
        remove() { this.removed = true; },
        contains() { return false; },
        click() { this.clicked = true; },
        pause() { this.paused = true; },
    };
}

function fixture() {
    let extension;
    let opened, fetched;
    const elements = [];
    const createElement = tag => {
        const value = element(tag);
        elements.push(value);
        return value;
    };
    const body = createElement('body');
    const app = { nodeOutputs: {}, registerExtension(value) {
        if (value.name === 'DINKI.VideoPlayer') extension = value;
    } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: { apiURL: path => path },
        document: { createElement, body, addEventListener() {}, removeEventListener() {} },
        window: { innerWidth: 1000, innerHeight: 800, open: url => { opened = url; } },
        fetch: async url => { fetched = url; return { ok: true, blob: async() => ({}) }; },
        URL: { createObjectURL: () => 'blob:video', revokeObjectURL() {} },
        URLSearchParams, queueMicrotask, setTimeout() {},
    });
    class Player {
        constructor(properties = {}) {
            this.id = 7;
            this.comfyClass = 'DINKI_Video_Player';
            this.properties = properties;
            this.onNodeCreated();
        }
        addDOMWidget(name, type, container, options) {
            this.container = container;
            this.widgetOptions = options;
            return { options: {} };
        }
        setDirtyCanvas() {}
    }
    return extension.beforeRegisterNodeDef(Player, { name: 'DINKI_Video_Player' })
        .then(() => ({ app, extension, Player, body, elements,
            get opened() { return opened; }, get fetched() { return fetched; } }));
}

test('video is contained in the node on first execution and remains sized to the widget', async () => {
    const { Player, body } = await fixture();
    const node = new Player();
    node.onExecuted({ video: [{ filename: 'comparison.mp4', type: 'temp', subfolder: '' }] });
    const video = node.container.firstChild;
    assert.equal(video.tagName, 'VIDEO');
    assert.equal(node.container.style.width, '100%');
    assert.equal(node.container.style.height, '100%');
    assert.equal(video.style.width, '100%');
    assert.equal(video.style.height, '100%');
    assert.equal(video.style.objectFit, 'contain');
    assert.equal(video.controls, true);
    assert.match(video.src, /filename=comparison\.mp4/);
    assert.match(video.src, /type=temp/);
    assert.equal(node.widgetOptions.getMinHeight(), 160);
    assert.ok(!body.children.includes(node.container));

    node.onExecuted({ video: [{ filename: 'comparison.gif', type: 'output', subfolder: '' }] });
    assert.equal(video.paused, true);
    assert.equal(node.container.children.length, 1);
    assert.equal(node.container.firstChild.tagName, 'IMG');
    assert.match(node.container.firstChild.src, /filename=comparison\.gif/);
});

test('video right-click menu opens and saves the current file', async () => {
    const context = await fixture();
    const node = new context.Player();
    node.onExecuted({ video: [{ filename: 'clip.mp4', type: 'temp', subfolder: '' }] });
    const event = { button: 2, clientX: 50, clientY: 50,
        preventDefault() { this.prevented = true; },
        stopPropagation() { this.stopped = true; },
        stopImmediatePropagation() { this.stopped = true; } };
    node.container.events.contextmenu(event);
    assert.equal(event.prevented, true);
    assert.equal(event.stopped, true);
    const buttons = context.elements.filter(item => item.tagName === 'BUTTON');
    assert.deepEqual(buttons.map(item => item.textContent), ['Open Video', 'Save Video']);
    await buttons[0].onclick();
    assert.match(context.opened, /filename=clip\.mp4/);
    await buttons[1].onclick();
    assert.match(context.fetched, /filename=clip\.mp4/);
    assert.match(context.fetched, /type=temp/);
    assert.equal(context.elements.find(item => item.tagName === 'A').download, 'clip.mp4');
    const options = [];
    node.getExtraMenuOptions(null, options);
    assert.deepEqual(options.map(item => item.content), ['Open Video', 'Save Video']);
});

test('player restores its saved descriptor when Nodes 2.0 recreates the node', async () => {
    const { extension, Player } = await fixture();
    const first = new Player();
    first.onExecuted({ video: [{ filename: 'saved.webp', type: 'output', subfolder: '' }] });
    const restored = new Player(structuredClone(first.properties));
    extension.loadedGraphNode(restored);
    assert.equal(restored.container.firstChild.tagName, 'IMG');
    assert.match(restored.container.firstChild.src, /saved\.webp/);
    restored.onConfigure();
    await Promise.resolve();
    assert.equal(restored.container.children.length, 1);
});
