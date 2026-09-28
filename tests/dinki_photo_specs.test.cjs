const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

function loadNode(values) {
    let extension;
    const app = { registerExtension(value) {
        if (value.name === 'DINKI.PhotoSpecifications.Orientation') extension = value;
    } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), { app, api: {} });
    class Node {
        constructor() {
            this.widgets = ['resolution', 'resolution_multiple', 'megapixels',
                'aspect_ratio', 'orientation'].map(name => ({ name, type: 'combo' }));
        }
        onNodeCreated() {}
        onConfigure(data) {
            this.widgets.forEach((widget, index) => { widget.value = data.widgets_values[index]; });
        }
    }
    extension.beforeRegisterNodeDef(Node, { name: 'DINKI_photo_specifications' });
    const node = new Node();
    node.onNodeCreated();
    node.onConfigure({ widgets_values: values });
    return node;
}

test('old Photo Specs workflow gets Custom and multiple 8 without shifting settings', () => {
    const node = loadNode(['2MP', 'Photo 4:6', 'Portrait']);
    assert.deepEqual(Object.fromEntries(node.widgets.map(widget => [widget.name, widget.value])), {
        resolution: 'Custom', resolution_multiple: 8, megapixels: 2,
        aspect_ratio: 'Photo 4:6', orientation: false,
    });
});

test('Image mode keeps the native selector and custom controls visible', () => {
    const node = loadNode(['Image', '32', '3MP', 'Photo 4:6', true]);
    assert.equal(node.widgets.length, 5);
    assert.equal(node.widgets[0].type, 'combo');
    assert.equal(node.widgets[0].value, 'Image');
    assert.equal(node.widgets[3].value, 'Photo 4:6');
    assert.equal(node.widgets[3].hidden, undefined);
    assert.equal(node.widgets[4].value, true);
    assert.equal(node.widgets[4].hidden, undefined);
});

test('legacy Photo Specs layout restores a fractional megapixel preset', () => {
    const node = loadNode(['0.56MP', 'Basic 1:1', 'Portrait']);
    assert.equal(node.widgets[0].value, 'Custom');
    assert.equal(node.widgets[1].value, 8);
    assert.equal(node.widgets[2].value, 0.56);
});
