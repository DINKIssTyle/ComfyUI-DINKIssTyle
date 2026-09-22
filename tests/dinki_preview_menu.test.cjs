const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

test('DOM preview menu uses selected batch image and blocks ordinary menu', async () => {
    let extension, opened;
    const elements = [];
    function createElement(tag) {
        const el = { tag, style: {}, events: {}, children: [], value: '',
            append(...items) { this.children.push(...items); },
            appendChild(item) { this.children.push(item); },
            replaceChildren() { this.children = []; },
            addEventListener(name, callback) { this.events[name] = callback; },
            removeAttribute() {}, remove() {}, contains() { return false; }
        };
        elements.push(el); return el;
    }
    const app = { registerExtension(ext) { if (ext.name === 'DINKI.PreviewImage.Resolution') extension = ext; } };
    vm.runInNewContext(readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8').replace(/^import .*;\r?\n/gm, ''), {
        app, api: { apiURL: url => url }, URLSearchParams,
        document: { createElement, body: createElement('body'), addEventListener() {}, removeEventListener() {} },
        window: { innerWidth: 1000, innerHeight: 1000, open: url => { opened = url; } }
    });
    class Node {
        addDOMWidget(name, type, element, options) { this.preview = element; return { options }; }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Node, { name: 'DINKI_Preview_Image' });
    const node = new Node(); node.onNodeCreated();
    node.onExecuted({ dkst_images: [
        { filename: 'one.png', subfolder: '', type: 'temp' },
        { filename: 'two.png', subfolder: '', type: 'output' }
    ] });
    const image = elements.find(el => el.tag === 'img');
    const select = elements.find(el => el.tag === 'select');
    assert.match(image.src, /one.png/);
    select.value = '1'; select.onchange();
    assert.match(image.src, /two.png/);
    const event = { button: 2, clientX: 10, clientY: 10,
        preventDefault() { this.prevented = true; },
        stopPropagation() { this.stopped = true; },
        stopImmediatePropagation() { this.stopped = true; }
    };
    image.events.pointerdown(event);
    assert.ok(event.stopped);
    image.events.contextmenu(event);
    assert.ok(event.prevented);
    const buttons = elements.filter(el => el.tag === 'button');
    assert.deepEqual(buttons.map(el => el.textContent), ['Open Image', 'Save Image']);
    await buttons[0].onclick();
    assert.match(opened, /two.png/);
});
