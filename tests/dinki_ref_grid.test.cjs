const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_ref_grid.js'), 'utf8');

class Element {
    constructor(tagName) {
        this.tagName = tagName;
        this.style = {};
        this.children = [];
        this.listeners = {};
        this.attributes = {};
    }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attributes[name] = value; this[name] = value; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    fire(name, event = {}) { if (this.listeners[name]) this.listeners[name](event); }
    querySelectorAll(selector) {
        let results = [];
        for (const child of this.children) {
            if (child.className && selector.includes(child.className)) {
                results.push(child);
            }
            if (child.querySelectorAll) {
                results = results.concat(child.querySelectorAll(selector));
            }
        }
        return results;
    }
}

function makeWidget(initialData = "[]") {
    let extension;
    const app = { registerExtension(val) { extension = val; } };
    const document = {
        createElement: tag => new Element(tag),
    };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app,
        document,
        requestAnimationFrame: cb => cb(),
        console,
    });

    const node = {
        size: [300, 200],
        graph: { version: 0, incrementVersion() { this.version++; } },
        changes: [],
        onWidgetChanged(...args) { this.changes.push(args); },
        addDOMWidget(name, type, element, options) {
            const widget = { name, type, element, options };
            this.widget = widget;
            Object.defineProperty(widget, 'value', {
                get: () => options.getValue(),
                set: val => options.setValue(val),
            });
            return widget;
        },
        setDirtyCanvas() {},
    };

    const customWidget = extension.getCustomWidgets().DKST_IMAGE_REF_GRID(node, "image_gallery", [null, { default: initialData }]);
    return { node, widget: customWidget.widget, root: customWidget.widget.element };
}

test('Image Ref Grid widget initializes with empty items', () => {
    const { widget } = makeWidget();
    assert.equal(widget.value, "[]");
});

test('Image Ref Grid deserializes and serializes items properly', () => {
    const initial = JSON.stringify([{ filename: "cat.png", subfolder: "ref_images", type: "input" }]);
    const { widget } = makeWidget(initial);
    assert.deepEqual(JSON.parse(widget.value), [
        { filename: "cat.png", subfolder: "ref_images", type: "input" }
    ]);
});
