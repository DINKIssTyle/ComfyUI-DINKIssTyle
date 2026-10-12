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

function makeWidget(initialData = "[]", captions = false) {
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

    const customWidget = extension.getCustomWidgets().DKST_IMAGE_REF_GRID(node, "image_gallery", [null, { default: initialData, captions }]);
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

function elements(root, tag) {
    return [...(root.tagName === tag ? [root] : []), ...root.children.flatMap(child => elements(child, tag))];
}

test('Training captions persist without remounting the textarea while typing', () => {
    const initial = JSON.stringify([{ filename: 'face.png', caption: 'side profile' }]);
    const { widget, root, node } = makeWidget(initial, true);
    const input = elements(root, 'textarea')[0];
    assert.equal(input.value, 'side profile');
    input.value = 'side profile, grey coat';
    input.fire('input');
    assert.equal(elements(root, 'textarea')[0], input);
    assert.equal(JSON.parse(widget.value)[0].caption, 'side profile, grey coat');
    assert.equal(node.graph.version, 1);
    widget.value = widget.value;
    assert.equal(elements(root, 'textarea')[0].value, 'side profile, grey coat');
});

test('Training thumbnail resize preserves the caption area', () => {
    const { root } = makeWidget(JSON.stringify([{ filename: 'face.png', caption: 'portrait' }]), true);
    const slider = elements(root, 'input').find(element => element.type === 'range');
    slider.fire('input', { target: { value: '110' } });
    const card = root.querySelectorAll('.dkst-ref-card')[0];
    assert.equal(card.style.height, '172px');
    assert.equal(card.children[0].style.height, '110px');
    assert.equal(elements(root, 'textarea')[0].value, 'portrait');
});

test('Creator keeps its existing serialization and has no caption controls', () => {
    const { root, widget } = makeWidget(JSON.stringify([{ filename: 'face.png', caption: 'unused' }]));
    assert.equal(elements(root, 'textarea').length, 0);
    assert.deepEqual(JSON.parse(widget.value), [{ filename: 'face.png', subfolder: '', type: 'input' }]);
    assert.equal(widget.dkstCaptionTargets, undefined);
});

test('Automatic captions only fill empty photos without remounting editable textareas', () => {
    const { root, widget, node } = makeWidget(JSON.stringify([
        { filename: 'face.png', caption: '' }, { filename: 'pose.png', caption: 'my description' }
    ]), true);
    const inputs = elements(root, 'textarea');
    const targets = widget.dkstCaptionTargets();
    assert.equal(targets.length, 1);
    assert.equal(targets[0].image.filename, 'face.png');
    assert.equal(targets[0].apply('a person facing the camera'), true);
    assert.equal(elements(root, 'textarea')[1], inputs[1]);
    assert.equal(inputs[0].value, 'a person facing the camera');
    assert.equal(JSON.parse(widget.value)[1].caption, 'my description');
    assert.equal(node.graph.version, 1);
    assert.equal(targets[0].apply('stale overwrite'), false);
});

test('Stale LLM results cannot overwrite manual edits or replaced gallery photos', () => {
    const { root, widget } = makeWidget(JSON.stringify([{ filename: 'face.png' }]), true);
    const [target] = widget.dkstCaptionTargets();
    const input = elements(root, 'textarea')[0];
    input.value = 'manual edit while waiting';
    input.fire('input');
    assert.equal(target.apply('LLM text'), false);
    assert.equal(JSON.parse(widget.value)[0].caption, 'manual edit while waiting');
    widget.value = JSON.stringify([{ filename: 'new.png' }]);
    assert.equal(target.apply('LLM text'), false);
    assert.equal(JSON.parse(widget.value)[0].caption, '');
});

test('Gallery and caption wheels preserve native scrolling without reaching the canvas', () => {
    const { root } = makeWidget(JSON.stringify([{ filename: 'face.png', caption: 'long description' }]), true);
    const grid = root.children.at(-1);
    const caption = elements(root, 'textarea')[0];
    for (const element of [grid, caption]) {
        let stopped = false;
        element.fire('wheel', {
            stopPropagation() { stopped = true; },
            preventDefault() { assert.fail('Native scrolling must remain enabled'); },
        });
        assert.equal(stopped, true);
        stopped = false;
        element.fire('pointerdown', { stopPropagation() { stopped = true; } });
        assert.equal(stopped, true);
    }
});
