const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_text_note.js'), 'utf8');

function fixture(secure = false, copyAllowed = true) {
    let extension, copied, commandCount = 0, changed, feedbackTimer, alertMessage;
    const app = { canvas: {}, registerExtension(value) { extension = value; } };
    const document = { activeElement: null, createElement, execCommand(command) {
        assert.equal(command, 'copy');
        commandCount++;
        copied = selected;
        return copyAllowed;
    } };
    let selected;
    function createElement(tag) {
        return {
            tagName: tag.toUpperCase(), style: {}, children: [], events: {}, attributes: {},
            append(...items) { this.children.push(...items); },
            appendChild(item) { this.children.push(item); },
            addEventListener(name, handler) { this.events[name] = handler; },
            setAttribute(name, value) { this.attributes[name] = value; },
            focus() { document.activeElement = this; },
            select() { selected = this.value; },
            remove() { this.removed = true; },
        };
    }
    document.body = createElement('body');
    const navigator = secure ? { clipboard: { writeText: async text => { copied = text; } } } : {};
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, document, navigator, isSecureContext: secure, queueMicrotask,
        setTimeout(callback) { feedbackTimer = callback; return 1; },
        clearTimeout() { feedbackTimer = undefined; },
        alert(message) { alertMessage = message; },
    });
    class Note {
        constructor(value = '', properties = {}) {
            this.comfyClass = 'DINKI_Text_Note';
            this.widgets = [{ name: 'text', value, options: {}, callback() {} }];
            this.properties = properties;
            this.graph = { incrementVersion() {} };
            this.onNodeCreated();
        }
        addDOMWidget(_name, _type, root, options) {
            this.root = root;
            this.layout = options;
            this.noteWidget = { options: {} };
            return this.noteWidget;
        }
        setSize(size) { this.size = size; }
        setDirtyCanvas() {}
        onWidgetChanged(name, value) { changed = { name, value }; }
    }
    extension.beforeRegisterNodeDef(Note, { name: 'DINKI_Text_Note' });
    return { extension, Note, get copied() { return copied; },
        get commandCount() { return commandCount; }, get changed() { return changed; },
        get alertMessage() { return alertMessage; },
        resetFeedback() { const callback = feedbackTimer; feedbackTimer = undefined; callback?.(); } };
}

test('note text persists, Lock prevents editing, and HTTP Copy uses selection', async () => {
    const context = fixture();
    const node = new context.Note('first line');
    const [toolbar, textarea] = node.root.children;
    const [lock, copy] = toolbar.children;
    assert.deepEqual(toolbar.children.map(button => button.textContent), ['Lock', 'Copy']);
    assert.equal(node.widgets[0].hidden, true);
    assert.equal(node.noteWidget.serialize, false);
    assert.equal(textarea.value, 'first line');
    textarea.value = 'first line\nsecond line';
    textarea.events.input();
    assert.equal(node.widgets[0].value, textarea.value);
    assert.equal(context.changed.value, textarea.value);

    lock.events.click();
    assert.equal(node.properties.dkstNoteLocked, true);
    assert.equal(textarea.readOnly, true);
    assert.equal(lock.attributes['aria-pressed'], 'true');
    textarea.value = 'altered';
    textarea.events.input();
    assert.equal(textarea.value, 'first line\nsecond line');
    await copy.events.click();
    assert.equal(context.copied, 'first line\nsecond line');
    assert.equal(context.commandCount, 1);
    assert.equal(copy.textContent, 'Copied!');
    assert.equal(copy.style.background, '#326547');
    assert.equal(copy.disabled, false);
    context.resetFeedback();
    assert.equal(copy.textContent, 'Copy');

    const restored = new context.Note();
    restored.widgets[0].value = node.widgets[0].value;
    restored.properties = structuredClone(node.properties);
    restored.onConfigure();
    await Promise.resolve();
    context.extension.loadedGraphNode(restored);
    assert.equal(restored.root.children[1].readOnly, true);
    assert.equal(restored.root.children[1].value, 'first line\nsecond line');
    restored.root.children[0].children[0].events.click();
    assert.equal(restored.root.children[1].readOnly, false);
});

test('secure context Copy uses the Clipboard API', async () => {
    const context = fixture(true);
    const node = new context.Note('copy me');
    const copy = node.root.children[0].children[1];
    await copy.events.click();
    assert.equal(context.copied, 'copy me');
    assert.equal(context.commandCount, 0);
    assert.equal(copy.textContent, 'Copied!');
});

test('copy failure is shown on the button and can be retried', async () => {
    const context = fixture(false, false);
    const node = new context.Note('note');
    const copy = node.root.children[0].children[1];
    await copy.events.click();
    assert.equal(copy.textContent, 'Copy failed');
    assert.equal(copy.disabled, false);
    assert.match(context.alertMessage, /blocked copying/);
    context.resetFeedback();
    assert.equal(copy.textContent, 'Copy');
});
