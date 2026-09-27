const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { execFileSync } = require('node:child_process');
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

function moveButton(root, index, direction) {
    return descendants(root, item => item.tagName === 'button' &&
        item['aria-label'] === `Move LoRA ${index} ${direction}`)[0];
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
    assert.equal(buttons.length, 4);
    assert.equal(moveButton(widget.element, 1, 'up').disabled, true);
    assert.equal(moveButton(widget.element, 1, 'down').disabled, true);
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
    strengths[1].value = '-0.25';
    strengths[1].fire('input');
    assert.equal(descendants(widget.element, item => item.type === 'range').length, 0);
    assert.equal(widget.value, JSON.stringify([
        { name: 'a.safetensors', enabled: true, strength_model: 0.75 },
        { name: 'b.safetensors', enabled: false, strength_model: -0.25 },
    ]));
    buttons = descendants(widget.element, item => item.tagName === 'button');
    buttons[0].fire('click');
    assert.equal(JSON.parse(widget.value).length, 1);
    assert.equal(JSON.parse(widget.value)[0].name, 'b.safetensors');
});

test('number input retains the full supported strength range without a slider', () => {
    const { widget } = makeWidget();
    const number = descendants(widget.element, item => item.type === 'number')[0];
    number.value = '4.5';
    number.fire('input');
    assert.equal(JSON.parse(widget.value)[0].strength_model, 4.5);
    number.value = '-100';
    number.fire('change');
    assert.equal(JSON.parse(widget.value)[0].strength_model, -100);
    assert.equal(descendants(widget.element, item => item.type === 'range').length, 0);
});

test('up and down move complete rows, update boundaries, and persist the new order', () => {
    const { widget } = makeWidget();
    widget.value = JSON.stringify([
        { name: 'a.safetensors', enabled: true, strength_model: 0.5 },
        { name: 'b.safetensors', enabled: false, strength_model: -0.25 },
        { name: 'missing.safetensors', enabled: true, strength_model: 4.5 },
    ]);
    let notifications = 0;
    widget.callback = () => { notifications++; };
    assert.equal(moveButton(widget.element, 1, 'up').disabled, true);
    assert.equal(moveButton(widget.element, 1, 'down').disabled, false);
    assert.equal(moveButton(widget.element, 3, 'up').disabled, false);
    assert.equal(moveButton(widget.element, 3, 'down').disabled, true);
    moveButton(widget.element, 3, 'up').fire('click');
    assert.deepEqual(JSON.parse(widget.value).map(row => row.name),
        ['a.safetensors', 'missing.safetensors', 'b.safetensors']);
    moveButton(widget.element, 1, 'down').fire('click');
    assert.deepEqual(JSON.parse(widget.value), [
        { name: 'missing.safetensors', enabled: true, strength_model: 4.5 },
        { name: 'a.safetensors', enabled: true, strength_model: 0.5 },
        { name: 'b.safetensors', enabled: false, strength_model: -0.25 },
    ]);
    assert.equal(notifications, 2);
    assert.equal(moveButton(widget.element, 1, 'up').disabled, true);
    assert.equal(moveButton(widget.element, 3, 'down').disabled, true);
    // Restoring the workflow keeps the same visible and serialized order.
    const restored = makeWidget().widget;
    restored.value = widget.value;
    assert.deepEqual(JSON.parse(restored.value), JSON.parse(widget.value));
    assert.equal(descendants(restored.element, item => item.tagName === 'select')[0].value,
        'missing.safetensors');
});

test('saved order after arrow clicks is the Python loader execution order', () => {
    const { widget } = makeWidget();
    widget.value = JSON.stringify([
        { name: 'a.safetensors', enabled: true, strength_model: 0.5 },
        { name: 'b.safetensors', enabled: true, strength_model: -0.25 },
        { name: 'a.safetensors', enabled: true, strength_model: 1.5 },
    ]);
    moveButton(widget.element, 3, 'up').fire('click');
    moveButton(widget.element, 2, 'up').fire('click');
    const python = `import json, runpy, sys, types
folder_paths = types.ModuleType('folder_paths')
folder_paths.get_filename_list = lambda kind: ['a.safetensors', 'b.safetensors']
nodes = types.ModuleType('nodes')
class Loader:
    def load_lora_model_only(self, model, name, strength):
        return (model + [[name, strength]],)
nodes.LoraLoaderModelOnly = Loader
sys.modules.update({'folder_paths': folder_paths, 'nodes': nodes})
LoaderNode = runpy.run_path(sys.argv[1])['DINKI_Multi_LoRA_Loader']
print(json.dumps(LoaderNode().load_loras([], sys.stdin.read())[0]))`;
    const result = execFileSync('python3', ['-c', python,
        join(__dirname, '../ComfyUI-DINKIssTyle/dinki_multi_lora.py')],
    { input: widget.value, encoding: 'utf8' });
    assert.deepEqual(JSON.parse(result), [
        ['a.safetensors', 1.5],
        ['a.safetensors', 0.5],
        ['b.safetensors', -0.25],
    ]);
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
