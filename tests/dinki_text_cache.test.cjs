const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { spawnSync } = require('node:child_process');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_text_cache.js'), 'utf8');
function fixture() {
    let extension, changes = 0, rootChanges = 0, dirty = 0, callback, notification;
    const root = { change() { rootChanges++; } };
    const app = { rootGraph: root, registerExtension(value) { extension = value; } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), { app });
    class Cache {
        constructor(text = '') {
            this.widgets = [{ name: 'text', value: text, callback(...args) { callback = { receiver: this, args }; } }];
            this.graph = { change() { changes++; } };
        }
        setDirtyCanvas() { dirty++; }
        onWidgetChanged(...args) { notification = args; }
        onExecuted(...args) { this.originalArgs = args; return 42; }
        serialize() { return { widgets_values: this.widgets.map(widget => widget.value) }; }
    }
    extension.beforeRegisterNodeDef(Cache, { name: 'DINKI_Text_Multiline_Cache' });
    return { app, Cache, extension, root,
        changes: () => changes, rootChanges: () => rootChanges, dirty: () => dirty,
        callback: () => callback, notification: () => notification };
}

test('received text updates the visible widget, saved workflow and Nodes 2.0 notification', () => {
    const f = fixture(), node = new f.Cache('old');
    const message = { dkst_cached_text: ['새 텍스트\n  second line  '] };
    assert.equal(node.onExecuted(message, 'extra'), 42);
    assert.deepEqual(node.originalArgs, [message, 'extra']);
    assert.equal(node.widgets[0].value, message.dkst_cached_text[0]);
    assert.equal(node.serialize().widgets_values[0], message.dkst_cached_text[0]);
    assert.equal(f.callback().receiver, node.widgets[0]);
    assert.deepEqual(f.callback().args, [message.dkst_cached_text[0]]);
    assert.deepEqual(f.notification(), ['text', message.dkst_cached_text[0]]);
    assert.equal(f.changes(), 1);
    assert.equal(f.rootChanges(), 1);
    assert.equal(f.dirty(), 1);
});

test('empty or missing updates keep the existing text and do not dirty the graph', () => {
    const f = fixture(), node = new f.Cache('keep this');
    for (const message of [undefined, {}, { dkst_cached_text: [] }, { dkst_cached_text: ['', null] },
        { dkst_cached_text: 'invalid payload' }]) node.onExecuted(message);
    assert.equal(node.widgets[0].value, 'keep this');
    assert.equal(f.changes(), 0);
    assert.equal(f.rootChanges(), 0);
});

test('identical text avoids redraws and root graphs are dirtied only once', () => {
    const f = fixture(), node = new f.Cache('same');
    node.graph = f.root;
    node.onExecuted({ dkst_cached_text: ['same'] });
    assert.equal(f.dirty(), 0);
    node.onExecuted({ dkst_cached_text: ['new'] });
    assert.equal(f.rootChanges(), 1);
    assert.equal(f.dirty(), 1);
});

test('the last nonempty string wins and whitespace is preserved', () => {
    const f = fixture(), node = new f.Cache('initial');
    node.onExecuted({ dkst_cached_text: ['first', 'second\nline', '', null] });
    assert.equal(node.widgets[0].value, 'second\nline');
    node.onExecuted({ dkst_cached_text: [' \n '] });
    assert.equal(node.widgets[0].value, ' \n ');
});

test('other node types and missing widgets retain their original behavior', () => {
    const f = fixture();
    class Other { onExecuted() { return 99; } }
    const original = Other.prototype.onExecuted;
    f.extension.beforeRegisterNodeDef(Other, { name: 'DINKI_Text_Multiline' });
    assert.equal(Other.prototype.onExecuted, original);
    const node = new f.Cache();
    node.widgets = [];
    assert.equal(node.onExecuted({ dkst_cached_text: ['incoming'] }), 42);
    assert.equal(f.changes(), 0);
});

test('real backend results round-trip through the frontend then survive no-text and blocked runs', () => {
    const f = fixture();
    let node = new f.Cache();
    const backend = (text, incoming, blocked = false) => {
        const script = `import json, runpy, sys, types
data = json.loads(sys.argv[2])
node = runpy.run_path(sys.argv[1])["DINKI_Text_Multiline_Cache"]()
outputs = [[object() if data["blocked"] else data["incoming"]]]
execution_list = types.SimpleNamespace(get_cache=lambda source, target: types.SimpleNamespace(outputs=outputs))
print(json.dumps(node.output_text(data["text"], ["source", 0], execution_list, "target")))`;
        const result = spawnSync('python3', ['-c', script,
            join(__dirname, '../ComfyUI-DINKIssTyle/dinki_text.py'), JSON.stringify({ text, incoming, blocked })], { encoding: 'utf8' });
        assert.equal(result.status, 0, result.stderr);
        return JSON.parse(result.stdout);
    };
    let result = backend(node.widgets[0].value, 'generated\n텍스트');
    assert.deepEqual(result.result, ['generated\n텍스트']);
    node.onExecuted(result.ui);
    // Serialize, recreate the UI node and reload the workflow's stored text.
    node = new f.Cache(node.serialize().widgets_values[0]);
    for (const [incoming, blocked] of [[null, false], ['', false], [null, true]]) {
        result = backend(node.widgets[0].value, incoming, blocked);
        assert.deepEqual(result.result, ['generated\n텍스트']);
        node.onExecuted(result.ui);
        assert.equal(node.widgets[0].value, 'generated\n텍스트');
    }
    result = backend(node.widgets[0].value, 'replacement');
    node.onExecuted(result.ui);
    assert.equal(node.widgets[0].value, 'replacement');
    assert.deepEqual(result.result, ['replacement']);
});
