const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

test('saved ten-output switches gain one state output without duplicates', () => {
    let extension;
    const app = { registerExtension(value) {
        if (value.name === 'DINKI.IfElseSwitch.StateOutput') extension = value;
    } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), { app, api: {} });

    class Node {
        constructor() {
            this.outputs = Array.from({ length: 10 }, (_, index) => ({ name: `output_${index + 1}` }));
        }
        onConfigure() { return 42; }
        addOutput(name, type) { this.outputs.push({ name, type }); }
    }
    extension.beforeRegisterNodeDef(Node, { name: 'DINKI_IfElseSwitch' });

    const node = new Node();
    assert.equal(node.onConfigure(), 42);
    assert.equal(node.outputs.length, 11);
    assert.deepEqual(node.outputs[10], { name: 'switch', type: 'BOOLEAN' });
    node.onConfigure();
    assert.equal(node.outputs.length, 11);
});

test('saved four-output branches gain a Boolean state output without duplicates', () => {
    let extension;
    const app = { registerExtension(value) {
        if (value.name === 'DINKI.IfElseSwitch.StateOutput') extension = value;
    } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), { app, api: {} });

    class Node {
        constructor() {
            this.outputs = Array.from({ length: 4 }, (_, index) => ({ name: `output_${index + 1}` }));
        }
        onConfigure() { return 42; }
        addOutput(name, type) { this.outputs.push({ name, type }); }
    }
    extension.beforeRegisterNodeDef(Node, { name: 'DINKI_IfElseBranch' });

    const node = new Node();
    assert.equal(node.onConfigure(), 42);
    assert.equal(node.outputs.length, 5);
    assert.deepEqual(node.outputs[4], { name: 'switch', type: 'BOOLEAN' });
    node.onConfigure();
    assert.equal(node.outputs.length, 5);
});

test('saved widget switch becomes a true socket without losing its fallback value', () => {
    let extension;
    const app = { registerExtension(value) {
        if (value.name === 'DINKI.IfElseSwitch.StateOutput') extension = value;
    } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), { app, api: {} });

    class Node {
        constructor() {
            this.inputs = [{ name: 'switch', type: 'BOOLEAN', link: 955, widget: { name: 'switch' } }];
            this.widgets = [{ name: 'default_switch', value: false }];
            this.outputs = [{ name: 'switch', type: 'BOOLEAN' }];
        }
        onConfigure() { return 42; }
    }
    extension.beforeRegisterNodeDef(Node, { name: 'DINKI_IfElseSwitch' });

    const node = new Node();
    assert.equal(node.onConfigure({ widgets_values: [true] }), 42);
    assert.equal(node.widgets[0].value, true);
    assert.equal(node.inputs[0].link, 955);
    assert.equal(node.inputs[0].widget, undefined);
});
