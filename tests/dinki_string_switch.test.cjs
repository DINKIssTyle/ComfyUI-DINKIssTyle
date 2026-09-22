const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

test('slash model IDs remain literal and truncated menu selections recover the full ID', async () => {
    let extension;
    const app = {
        registerExtension(ext) { if (ext.name === 'DINKI.StringSwitchRT') extension = ext; },
        graph: { setDirtyCanvas() {} }
    };
    const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: {}, queueMicrotask, requestAnimationFrame: fn => fn()
    });
    class Node {
        constructor() {
            this.widgets = [
                { name: 'select_string', type: 'text', value: 'qwen/qwen3.8-27b' },
                { name: 'input_text', value: 'gemma\nqwen/qwen3.8-27b' }
            ];
        }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        }
    }
    await extension.beforeRegisterNodeDef(Node, { name: 'DINKI_String_Switch_RT' }, app);
    const node = new Node();
    node.onNodeCreated();
    const combo = node.widgets[0];
    assert.equal(combo.value, 'qwen/qwen3.8-27b');
    assert.equal(combo.options.getOptionLabel(combo.value), 'qwen/qwen3.8-27b');
    combo.callback('qwen3.8-27b');
    assert.equal(combo.value, 'qwen/qwen3.8-27b');
    combo.callback('gemma');
    assert.equal(combo.value, 'gemma');
});
