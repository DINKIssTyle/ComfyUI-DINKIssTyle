const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_image_comparison.js'), 'utf8');

function element(tagName = 'div') {
    return {
        tagName: tagName.toUpperCase(), style: {}, children: [], listeners: {},
        append(...children) { this.children.push(...children); },
        appendChild(child) { this.children.push(child); },
        removeAttribute(name) { delete this[name]; },
        addEventListener(name, callback) { this.listeners[name] = callback; },
        remove() { this.removed = true; },
        contains() { return false; },
        click() { this.clicked = true; },
        getBoundingClientRect() { return { left: 20, width: 200 }; },
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
    const app = { nodeOutputs: {}, registerExtension(value) { extension = value; } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: { apiURL: path => path },
        document: { createElement, body, addEventListener() {}, removeEventListener() {} },
        window: { innerWidth: 1000, innerHeight: 800, open: url => { opened = url; } },
        fetch: async url => { fetched = url; return { ok: true, blob: async() => ({}) }; },
        URL: { createObjectURL: () => 'blob:comparison', revokeObjectURL() {} },
        URLSearchParams, queueMicrotask, setTimeout() {},
    });
    class CompareNode {
        constructor(properties = {}) {
            this.id = 7;
            this.comfyClass = 'DINKI_Image_Comparison';
            this.properties = properties;
            this.widgets = [{ name: 'mode', value: 'Slide' }];
            this.onNodeCreated();
        }
        addDOMWidget(name, type, root) {
            this.root = root;
            return { options: {} };
        }
        setDirtyCanvas() {}
    }
    extension.beforeRegisterNodeDef(CompareNode, { name: 'DINKI_Image_Comparison' });
    return { app, extension, CompareNode, elements,
        get opened() { return opened; }, get fetched() { return fetched; } };
}

const output = { dkst_comparison: [
    { filename: 'first.png', type: 'temp' },
    { filename: 'second.png', type: 'temp' },
    { filename: 'difference.png', type: 'temp' },
] };

test('Slide follows pointer position and Difference displays the computed preview', () => {
    const { CompareNode } = fixture();
    const node = new CompareNode();
    node.onExecuted(output);
    const [first, second, difference, divider] = node.root.children;
    assert.match(first.src, /first\.png/);
    assert.equal(second.style.display, 'block');
    node.root.listeners.pointermove({ clientX: 70 });
    assert.equal(second.style.clipPath, 'inset(0 75% 0 0)');
    assert.equal(divider.style.left, '25%');
    node.onWidgetChanged('mode', 'Difference');
    assert.equal(difference.style.display, 'block');
    assert.equal(first.style.display, 'none');
    assert.equal(divider.style.display, 'none');
    node.widgets[0].callback('Slide');
    assert.equal(second.style.display, 'block');
});

test('comparison restores its own images after a tab switch', async () => {
    const { app, extension, CompareNode } = fixture();
    const first = new CompareNode();
    first.onExecuted(output);
    app.nodeOutputs[7] = { dkst_comparison: [
        { filename: 'other-1.png' }, { filename: 'other-2.png' }, { filename: 'other-diff.png' },
    ] };
    const restored = new CompareNode(structuredClone(first.properties));
    extension.loadedGraphNode(restored);
    assert.match(restored.root.children[0].src, /first\.png/);
    restored.widgets[0].value = 'Difference';
    restored.onConfigure();
    await Promise.resolve();
    assert.equal(restored.root.children[2].style.display, 'block');
});

test('right-click menu changes mode and opens or saves the selected comparison images', async () => {
    const context = fixture();
    const node = new context.CompareNode();
    node.onExecuted(output);
    const event = { button: 2, clientX: 40, clientY: 40,
        preventDefault() { this.prevented = true; },
        stopPropagation() { this.stopped = true; },
        stopImmediatePropagation() { this.stopped = true; } };
    node.root.listeners.contextmenu(event);
    assert.equal(event.prevented, true);
    assert.equal(event.stopped, true);
    const buttons = context.elements.filter(item => item.tagName === 'BUTTON');
    assert.deepEqual(buttons.map(item => item.textContent), [
        'Mode: Slide', 'Mode: Difference', 'Open Image 1', 'Save Image 1',
        'Open Image 2', 'Save Image 2',
    ]);
    await buttons[1].onclick();
    assert.equal(node.widgets[0].value, 'Difference');
    assert.equal(node.root.children[2].style.display, 'block');
    await buttons[2].onclick();
    assert.match(context.opened, /filename=first\.png/);
    await buttons[5].onclick();
    assert.match(context.fetched, /filename=second\.png/);
    assert.match(context.fetched, /type=temp/);
    const link = context.elements.find(item => item.tagName === 'A');
    assert.equal(link.download, 'second.png');
    assert.equal(link.clicked, true);
    const options = [];
    node.getExtraMenuOptions(null, options);
    assert.deepEqual(options.map(item => item.content), buttons.map(item => item.textContent));
});
