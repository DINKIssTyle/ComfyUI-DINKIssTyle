const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_if_else_types.js'), 'utf8');

function fixture() {
    let extension;
    const app = { registerExtension(value) { extension = value; } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), { app });

    class Node {
        constructor(comfyClass, count) {
            this.comfyClass = comfyClass;
            this.inputs = Array.from({ length: count }, (_, index) => [
                { name: `on_false_${index + 1}`, type: '*' },
                { name: `on_true_${index + 1}`, type: '*' },
            ]).flat();
            this.outputs = Array.from({ length: count }, (_, index) => ({ name: `output_${index + 1}`, type: '*' }));
        }
        getInputLink(index) { return this.inputLinks?.[index]; }
    }
    extension.beforeRegisterNodeDef(Node, { name: 'DINKI_IfElseSwitch' });

    const graph = { _nodes: [], links: { 1: { type: '*' } }, setDirtyCanvas() {} };
    const sourceNode = { outputs: [{ type: 'LATENT' }] };
    const first = new Node('DINKI_IfElseSwitch', 10);
    const second = new Node('DINKI_IfElseBranch', 4);
    first.outputs[0].links = [1];
    first.graph = second.graph = graph;
    first.inputLinks = { 0: { resolve: () => ({ output: sourceNode.outputs[0] }) } };
    second.inputLinks = { 0: { resolve: () => ({ output: first.outputs[0] }) } };
    graph._nodes.push(second, first); // Reverse order checks propagation over multiple passes.
    return { extension, Node, first, second, graph };
}

test('connected type propagates through switch and downstream branch without changing data', async () => {
    const { first, second } = fixture();
    first.onConnectionsChange();
    await Promise.resolve();
    assert.equal(first.inputs[0].type, 'LATENT');
    assert.equal(first.inputs[1].type, 'LATENT');
    assert.equal(first.outputs[0].type, 'LATENT');
    assert.equal(first.graph.links[1].type, 'LATENT');
    assert.equal(second.inputs[0].type, 'LATENT');
    assert.equal(second.inputs[1].type, 'LATENT');
    assert.equal(second.outputs[0].type, 'LATENT');
    assert.equal(first.outputs[1].type, '*');
});

test('disconnecting a source restores wildcard socket types', async () => {
    const { first, second } = fixture();
    first.onConfigure();
    await Promise.resolve();
    first.inputLinks = {};
    first.onConnectionsChange();
    await Promise.resolve();
    assert.equal(first.outputs[0].type, '*');
    assert.equal(first.graph.links[1].type, '*');
    assert.equal(second.outputs[0].type, '*');
});

test('a pair with conflicting source types does not claim one concrete type', async () => {
    const { first } = fixture();
    first.inputLinks[1] = { resolve: () => ({ output: { type: 'IMAGE' } }) };
    first.onConfigure();
    await Promise.resolve();
    assert.equal(first.inputs[0].type, '*');
    assert.equal(first.inputs[1].type, '*');
    assert.equal(first.outputs[0].type, '*');
});

test('image-triggered switch colors routed data without changing its image trigger', async () => {
    const { Node, graph } = fixture();
    const node = new Node('DINKI_IfElseImageSwitch', 10);
    node.graph = graph;
    node.inputLinks = { 3: { resolve: () => ({ output: { type: 'IMAGE' } }) } };
    node.inputs.push({ name: 'image', type: 'IMAGE' });
    node.outputs.push({ name: 'switch', type: 'BOOLEAN' });
    graph._nodes.push(node);
    node.onConnectionsChange();
    await Promise.resolve();
    assert.equal(node.inputs[2].type, 'IMAGE');
    assert.equal(node.inputs[3].type, 'IMAGE');
    assert.equal(node.outputs[1].type, 'IMAGE');
    assert.equal(node.inputs[20].type, 'IMAGE');
    assert.equal(node.outputs[10].type, 'BOOLEAN');
});
