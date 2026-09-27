const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_multi_lora.js'), 'utf8');

class Element {
    constructor(tagName) {
        this.tagName = tagName;
        this.style = {};
        this.children = [];
        this.listeners = {};
    }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this[name] = value; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    fire(name) { this.listeners[name](); }
}

function descendants(element, match) {
    return [element, ...(element.children || []).flatMap(child => descendants(child, match))]
        .filter(match);
}

function makeWidget() {
    let extension;
    const app = { registerExtension(value) { extension = value; } };
    const document = {
        createElement: tag => new Element(tag),
        createTextNode: text => ({ textContent: text, children: [] }),
    };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, document, requestAnimationFrame: callback => callback(),
    });
    const node = {
        size: [340, 100],
        addDOMWidget(name, type, element, options) {
            const widget = { name, type, element, options };
            this.widget = widget;
            Object.defineProperty(widget, 'value', {
                get: () => options.getValue(), set: value => options.setValue(value),
            });
            return widget;
        },
        computeSize() { return [340, 60 + this.widget.options.getMinHeight()]; },
        setSize(size) { this.size = size; },
        setDirtyCanvas() {},
    };
    const factory = extension.getCustomWidgets().DKST_LORA_STACK;
    const { widget } = factory(node, 'lora_stack', ['DKST_LORA_STACK', {
        lora_names: ['None', 'a.safetensors', 'b.safetensors'],
    }]);
    return { widget, node };
}

test('add, select, toggle, set strength, remove and serialize rows', () => {
    const { widget } = makeWidget();
    let buttons = descendants(widget.element, item => item.tagName === 'button');
    assert.equal(buttons.length, 2);
    buttons.at(-1).fire('click');
    const selects = descendants(widget.element, item => item.tagName === 'select');
    selects[0].value = 'a.safetensors';
    selects[0].fire('change');
    selects[1].value = 'b.safetensors';
    selects[1].fire('change');
    const toggles = descendants(widget.element, item => item.type === 'checkbox');
    toggles[1].checked = false;
    toggles[1].fire('change');
    const strengths = descendants(widget.element, item => item.type === 'number');
    strengths[0].value = '0.75';
    strengths[0].fire('input');
    const sliders = descendants(widget.element, item => item.type === 'range');
    assert.equal(sliders[0].value, '0.75');
    sliders[1].value = '-0.25';
    sliders[1].fire('input');
    assert.equal(strengths[1].value, '-0.25');
    assert.equal(widget.value, JSON.stringify([
        { name: 'a.safetensors', enabled: true, strength_model: 0.75 },
        { name: 'b.safetensors', enabled: false, strength_model: -0.25 },
    ]));
    buttons = descendants(widget.element, item => item.tagName === 'button');
    buttons[0].fire('click');
    assert.equal(JSON.parse(widget.value).length, 1);
    assert.equal(JSON.parse(widget.value)[0].name, 'b.safetensors');
});

test('restores saved rows including unavailable LoRA names', () => {
    const { widget } = makeWidget();
    widget.value = JSON.stringify([{ name: 'missing.safetensors', enabled: false, strength_model: -1.25 }]);
    assert.equal(JSON.parse(widget.value)[0].name, 'missing.safetensors');
    const select = descendants(widget.element, item => item.tagName === 'select')[0];
    assert.equal(select.value, 'missing.safetensors');
    assert.ok(select.children.some(option => option.value === 'missing.safetensors'));
});

test('adding a row preserves user width and extra height', () => {
    const { widget, node } = makeWidget();
    node.size = [510, 480];
    descendants(widget.element, item => item.tagName === 'button').at(-1).fire('click');
    assert.deepEqual(node.size, [510, 480]);
    node.size = [510, 180];
    descendants(widget.element, item => item.tagName === 'button').at(-1).fire('click');
    assert.equal(node.size[0], 510);
    assert.equal(node.size[1], node.computeSize()[1]);
});
