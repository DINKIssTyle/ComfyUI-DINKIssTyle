const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const { JSDOM } = require('jsdom');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_llm_connections.js'), 'utf8');
const NONE = '-- Select connection --';
const generationValues = [true, 'hello', 'system', 'model/name', 42, 1000, 0.7, 300, false, 0];
// There are ten widgets before the legacy address, not counting seed control.
const names = ['assistant_enabled', 'user_prompt', 'system_prompt', 'model_key', 'seed',
    'max_tokens', 'temperature', 'timeout_seconds', 'auto_unload', 'unload_delay'];
const settle = () => new Promise(resolve => setImmediate(resolve));

async function fixture() {
    const dom = new JSDOM('<!doctype html><body></body>');
    const profiles = new Map([['Local', { alias: 'Local', ip_address: '127.0.0.1', port: 1234, protocol: 'http', api_key: 'private-key' }]]);
    const requests = [], messages = [], apiLoads = [];
    let extension, failMigration = false, migrationGate, failAliases = false;
    const app = {
        registerExtension(value) { extension = value; },
        graph: { _nodes: [], incrementVersion() {}, getNodeById(id) { return this._nodes.find(node => String(node.id) === String(id)); } },
        ui: { dialog: { show(message) { messages.push(message); } } },
        async graphToPrompt() {
            const workflow = { nodes: this.graph._nodes.map(node => node.serialize()) };
            const output = Object.fromEntries(this.graph._nodes.map((node, index) => [String(index), {
                class_type: node.comfyClass,
                inputs: Object.fromEntries(node.widgets.filter(w => w.serialize !== false).map(w => [w.name, w.value])),
            }]));
            return { workflow, output };
        },
        async loadApiJson(data) {
            apiLoads.push(data);
            for (const [id, info] of Object.entries(data)) {
                if (info.class_type !== 'DINKI_LMStudio') continue;
                const node = new Node(); node.id = id;
                for (const [name, value] of Object.entries(info.inputs)) {
                    const widget = node.widgets.find(item => item.name === name);
                    if (widget) widget.value = value;
                }
            }
            return 'loaded';
        },
    };
    const api = { async fetchApi(path, options = {}) {
        const payload = options.body ? JSON.parse(options.body) : undefined;
        requests.push({ path, payload });
        let data;
        if (path.endsWith('/aliases')) {
            if (failAliases) throw new Error('Alias refresh failed');
            data = { aliases: [...profiles.keys()] };
        }
        else if (!payload) data = { connections: [...profiles.values()].map(({ api_key, ...profile }) => ({ ...profile, has_api_key: Boolean(api_key) })) };
        else if (payload.action === 'migrate') {
            if (migrationGate) await migrationGate;
            if (failMigration) return { ok: false, async json() { return { error: 'Migration failed' }; } };
            const alias = 'Migrated LLM 1';
            profiles.set(alias, { ...payload, alias });
            data = { alias };
        } else if (payload.action === 'save') {
            profiles.set(payload.alias, { ...profiles.get(payload.alias), ...payload });
            data = { alias: payload.alias };
        } else if (payload.action === 'delete') {
            profiles.delete(payload.alias); data = { ok: true };
        } else data = { ok: true, message: 'Connected successfully.' };
        return { ok: true, async json() { return data; } };
    } };
    const context = vm.createContext({ document: dom.window.document, app, api, console, structuredClone,
        crypto: { getRandomValues: crypto.getRandomValues } }); // Also covers LAN HTTP without randomUUID.
    const module = new vm.SourceTextModule(source, { context });
    await module.link(async specifier => {
        const key = specifier.includes('app.js') ? 'app' : 'api';
        return new vm.SyntheticModule([key], function() { this.setExport(key, key === 'app' ? app : api); }, { context });
    });
    await module.evaluate();
    class Node {
        constructor(seedControl = true) {
            this.comfyClass = 'DINKI_LMStudio';
            this.properties = {};
            this.inputs = [];
            this.graph = app.graph;
            this.widgets = names.map((name, index) => ({ name, value: generationValues[index], options: {}, _state: { options: {} } }));
            if (seedControl) this.widgets.splice(5, 0, { name: 'control_after_generate', value: 'fixed', options: {} });
            const options = {}, stateOptions = {};
            const widget = { name: 'connection_alias', value: NONE, _state: {} };
            Object.defineProperty(widget, 'options', { get: () => options });
            Object.defineProperty(widget._state, 'options', { get: () => stateOptions });
            this.widgets.push(widget);
            this.widgets.push({ name: 'generation_mode', value: 'Always Generate' },
                { name: 'cached_response', value: '' });
            this.onNodeCreated();
            app.graph._nodes.push(this);
        }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget); return widget;
        }
        configure(info) {
            this.properties = { ...info.properties };
            this.inputs = [...(info.inputs ?? [])];
            if (Array.isArray(info.widgets_values)) info.widgets_values.forEach((value, index) => {
                if (this.widgets[index]) this.widgets[index].value = value;
            });
            this.onConfigure(info);
        }
        serialize() {
            const info = { type: this.comfyClass, properties: { ...this.properties }, inputs: this.inputs,
                widgets_values: this.widgets.filter(w => w.serialize !== false).map(w => w.value) };
            this.onSerialize(info); return info;
        }
        setDirtyCanvas() {}
        onExecuted(...args) { this.executedArgs = args; return 42; }
    }
    extension.beforeRegisterNodeDef(Node, { name: 'DINKI_LMStudio' });
    await extension.setup();
    await settle();
    return { Node, extension, app, profiles, requests, messages, apiLoads, dom, exports: module.namespace,
        failMigration() { failMigration = true; }, failAliases() { failAliases = true; },
        gate(promise) { migrationGate = promise; } };
}

function legacy(seedControl = true, key = true) {
    const prefix = [...generationValues];
    if (seedControl) prefix.splice(5, 0, 'fixed');
    return { type: 'DINKI_LMStudio', inputs: [], properties: {},
        widgets_values: [...prefix, 'private-host', 1234, ...(key ? ['private-key'] : [])] };
}

test('native alias dropdown refreshes and retains an unavailable saved alias', async () => {
    const f = await fixture();
    const node = new f.Node();
    const widget = node.widgets.find(w => w.name === 'connection_alias');
    assert.deepEqual(Array.from(widget.options.values), [NONE, 'Local']);
    widget.value = 'Local';
    f.profiles.delete('Local'); f.profiles.set('Work', { alias: 'Work' });
    await node.widgets.find(w => w.name === 'Refresh connections').callback();
    assert.equal(widget.value, 'Local');
    assert.deepEqual(Array.from(widget.options.values), [NONE, 'Work', 'Local']);
    assert.deepEqual(Array.from(widget._state.options.values), [NONE, 'Work', 'Local']);
    assert.equal(node.widgets.find(w => w.name === 'Refresh connections').serialize, false);
    assert.ok(!JSON.stringify(node.serialize()).includes('private-key'));
});

for (const seedControl of [true, false]) for (const key of [true, false]) {
    test(`legacy workflow migrates credentials and keeps widget positions: seed=${seedControl}, key=${key}`, async () => {
        const f = await fixture();
        const old = legacy(seedControl, key);
        const data = { nodes: [old] };
        await f.extension.beforeConfigureGraph(data);
        assert.ok(!JSON.stringify(data).includes('private-host'));
        assert.ok(!JSON.stringify(data).includes('private-key'));
        assert.equal(old.widgets_values.at(-1), NONE);
        const node = new f.Node(seedControl);
        node.configure(old);
        const prompt = await f.app.graphToPrompt();
        assert.equal(prompt.output['0'].inputs.connection_alias, 'Migrated LLM 1');
        assert.equal(prompt.output['0'].inputs.user_prompt, 'hello');
        assert.equal(prompt.output['0'].inputs.max_tokens, 1000);
        assert.equal(prompt.output['0'].inputs.timeout_seconds, 300);
        assert.ok(!JSON.stringify(prompt).includes('private-key'));
    });
}

test('subgraph definitions are prepared and linked old credentials do not block import', async () => {
    const f = await fixture();
    const old = legacy();
    await f.extension.beforeConfigureGraph({ nodes: [], definitions: { subgraphs: [{ nodes: [old] }] } });
    assert.equal(old.properties.dkstLLMVersion, 2);
    const linked = legacy(); linked.inputs = [{ name: 'api_key', link: 12 }];
    await f.extension.beforeConfigureGraph({ nodes: [linked] });
    const node = new f.Node(); node.configure(linked);
    assert.equal((await f.app.graphToPrompt()).output['0'].inputs.connection_alias, NONE);
    assert.ok(f.messages.some(message => message.includes('links were disconnected')));
});

test('named legacy values restore generation options and leave only an alias', async () => {
    const f = await fixture();
    const info = { type: 'DINKI_LMStudio', properties: {}, widgets_values_named: {
        ip_address: 'private-host', port: 1234, api_key: 'private-key',
        user_prompt: 'named prompt', max_tokens: 4321,
    } };
    await f.extension.beforeConfigureGraph({ nodes: [info] });
    assert.ok(!JSON.stringify(info).includes('private'));
    const node = new f.Node(); node.configure(info);
    const exported = await f.app.graphToPrompt();
    assert.equal(exported.output['0'].inputs.connection_alias, 'Migrated LLM 1');
    assert.equal(exported.output['0'].inputs.user_prompt, 'named prompt');
    assert.equal(exported.output['0'].inputs.max_tokens, 4321);
});

test('a completed old paste migration cannot overwrite a newer configuration', async () => {
    const f = await fixture();
    let release;
    f.gate(new Promise(resolve => { release = resolve; }));
    const node = new f.Node(); node.configure(legacy());
    const saved = node.serialize();
    saved.widgets_values[node.widgets.findIndex(widget => widget.name === 'connection_alias')] = 'Local';
    node.configure(saved);
    release(); await settle();
    assert.equal(node.widgets.find(w => w.name === 'connection_alias').value, 'Local');
});

test('pasted old node is safe immediately and export awaits migration', async () => {
    const f = await fixture();
    let release;
    f.gate(new Promise(resolve => { release = resolve; }));
    const node = new f.Node();
    node.configure(legacy());
    assert.ok(!JSON.stringify(node.serialize()).includes('private'));
    let completed = false;
    const pending = f.app.graphToPrompt().then(result => { completed = true; return result; });
    await settle(); assert.equal(completed, false);
    release();
    const exported = await pending;
    assert.equal(exported.output['0'].inputs.connection_alias, 'Migrated LLM 1');
});

test('failed paste migration does not retain credentials or permanently block manual repair', async () => {
    const f = await fixture(); f.failMigration();
    const node = new f.Node(); node.configure(legacy());
    await settle();
    assert.ok(!JSON.stringify(node.serialize()).includes('private'));
    node.widgets.find(w => w.name === 'connection_alias').value = 'Local';
    assert.equal((await f.app.graphToPrompt()).output['0'].inputs.connection_alias, 'Local');
    assert.ok(f.messages.some(message => message.includes('workflow was opened')));
});

test('API JSON import migrates literal connection fields and does not mutate input data', async () => {
    const f = await fixture();
    const data = { '1': { class_type: 'DINKI_LMStudio', inputs: { ip_address: 'private-host', port: 1234, api_key: 'private-key', user_prompt: 'hello' } } };
    assert.equal(await f.app.loadApiJson(data), 'loaded');
    assert.equal(f.apiLoads[0]['1'].inputs.connection_alias, NONE);
    assert.equal((await f.app.graphToPrompt()).output['0'].inputs.connection_alias, 'Migrated LLM 1');
    assert.ok(!JSON.stringify(f.apiLoads).includes('private-key'));
    assert.equal(data['1'].inputs.api_key, 'private-key');
    data['1'].inputs.api_key = ['2', 0];
    assert.equal(await f.app.loadApiJson(data), 'loaded');
    assert.ok(f.messages.some(message => message.includes('links removed')));
});

test('settings editor saves through server API, keeps keys masked and updates dropdowns', async () => {
    const f = await fixture();
    assert.equal(f.extension.settings[0].id, 'DKST.LLM.Connections');
    const root = f.extension.settings[0].type('unused', () => { throw new Error('Must not save browser settings'); });
    f.dom.window.document.body.append(root);
    await settle();
    const select = root.querySelector('[aria-label="Registered LLM connections"]');
    select.value = 'Local'; select.dispatchEvent(new f.dom.window.Event('change'));
    const key = root.querySelector('input[type=password]');
    assert.equal(key.value, ''); assert.equal(key.placeholder, 'API key is stored');
    assert.ok(!root.innerHTML.includes('private-key'));
    const click = async label => {
        [...root.querySelectorAll('button')].find(button => button.textContent === label).click();
        await settle(); await settle();
    };
    await click('Save');
    let payload = f.requests.filter(r => r.payload?.action === 'save').at(-1).payload;
    assert.equal(payload.overwrite, true); assert.equal('api_key' in payload, false);
    key.value = 'replacement-key'; await click('Save');
    payload = f.requests.filter(r => r.payload?.action === 'save').at(-1).payload;
    assert.equal(payload.api_key, 'replacement-key'); assert.equal(key.value, '');
    root.querySelector('input[type=checkbox]').checked = true; await click('Save');
    assert.equal(f.requests.filter(r => r.payload?.action === 'save').at(-1).payload.api_key, '');
    await click('Test saved connection');
    assert.equal(root.querySelector('[role=status]').textContent, 'Connected successfully.');
    const node = new f.Node();
    node.widgets.find(w => w.name === 'connection_alias').value = 'Local';
    await click('Delete');
    assert.ok(!f.profiles.has('Local'));
    assert.equal(node.widgets.find(w => w.name === 'connection_alias').value, 'Local');
});

test('workflow load and serialize never enumerate runtime LinkMap proxies or follow widget cycles', async () => {
    const f = await fixture();
    let enumerations = 0;
    class LinkMap {}
    const linkMap = new Proxy(new LinkMap(), { ownKeys() {
        enumerations++;
        throw new RangeError('Maximum call stack size exceeded');
    } });
    const old = legacy();
    const runtimeWidget = { graph: { links: linkMap } };
    runtimeWidget.node = old;
    old.inputs = [{ name: 'image', link: null, _widget: runtimeWidget }];
    const cyclic = { api_key: 'private-key' };
    cyclic.self = cyclic;
    old.properties = { linkMap, cyclic, graph: runtimeWidget.graph };
    const graph = { nodes: [old], links: [] };
    assert.doesNotThrow(() => f.extension.beforeConfigureGraph(graph));
    const node = new f.Node();
    assert.doesNotThrow(() => node.configure(old));
    assert.doesNotThrow(() => node.serialize());
    await settle();
    assert.equal(enumerations, 0);
    assert.equal('api_key' in cyclic, false);
    assert.equal(node.widgets.find(w => w.name === 'connection_alias').value, 'Migrated LLM 1');
});

test('workflow normalization completes without waiting for profile registration', async () => {
    const f = await fixture();
    let release;
    f.gate(new Promise(resolve => { release = resolve; }));
    const old = legacy();
    const requestCount = f.requests.length;
    assert.equal(f.extension.beforeConfigureGraph({ nodes: [old] }), undefined);
    assert.equal(f.requests.length, requestCount);
    const node = new f.Node();
    assert.doesNotThrow(() => node.configure(old));
    assert.equal(node.widgets.find(w => w.name === 'connection_alias').value, NONE);
    release(); await settle();
});

test('failed automatic registration leaves an opened workflow and unchanged generation settings', async () => {
    const f = await fixture(); f.failMigration();
    const old = legacy();
    assert.equal(f.extension.beforeConfigureGraph({ nodes: [old] }), undefined);
    const node = new f.Node(); node.configure(old);
    const exported = await f.app.graphToPrompt();
    assert.equal(exported.output['0'].inputs.connection_alias, NONE);
    assert.equal(exported.output['0'].inputs.user_prompt, 'hello');
    assert.equal(exported.output['0'].inputs.model_key, 'model/name');
    assert.ok(!JSON.stringify(exported).includes('private-key'));
    assert.ok(f.messages.some(message => message.includes('workflow was opened')));
});

test('an alias refresh failure does not undo successful registration or affect unrelated workflow loading', async () => {
    const f = await fixture(); f.failAliases();
    assert.equal(f.extension.beforeConfigureGraph({ nodes: [{ type: 'SaveImage', widgets_values: ['image'] }] }), undefined);
    const old = legacy();
    f.extension.beforeConfigureGraph({ nodes: [old] });
    const node = new f.Node(); node.configure(old);
    assert.equal((await f.app.graphToPrompt()).output['0'].inputs.connection_alias, 'Migrated LLM 1');
});

for (const objectLinks of [false, true]) {
    test(`linked old connection inputs are detached without dangling links or moving other connections: objects=${objectLinks}`, async () => {
        const f = await fixture();
        const old = legacy(); old.id = 2;
        old.inputs = [{ name: 'api_key', link: 8 }, { name: 'image', link: 9 }];
        const source = { id: 1, type: 'Source', outputs: [{ links: [8, 9] }] };
        const links = [[8, 1, 0, 2, 0, 'STRING'], [9, 1, 0, 2, 1, 'IMAGE']];
        const graph = { nodes: [source, old], links: objectLinks
            ? Object.fromEntries(links.map(([id, origin_id, origin_slot, target_id, target_slot, type]) => [id, { id, origin_id, origin_slot, target_id, target_slot, type }]))
            : links };
        f.extension.beforeConfigureGraph(graph);
        assert.deepEqual(source.outputs[0].links, [9]);
        assert.deepEqual(old.inputs.map(input => input.name), ['image']);
        if (objectLinks) {
            assert.equal(Object.keys(graph.links).length, 1);
            assert.equal(graph.links[9].target_slot, 0);
        } else {
            assert.equal(graph.links.length, 1); assert.equal(graph.links[0][4], 0);
        }
        const node = new f.Node(); node.configure(old);
        assert.equal((await f.app.graphToPrompt()).output['0'].inputs.connection_alias, NONE);
        assert.equal(f.requests.filter(request => request.payload?.action === 'migrate').length, 0);
    });
}

test('bundled legacy sample with promoted generation inputs opens and preserves those links', async () => {
    const f = await fixture();
    const data = JSON.parse(readFileSync(join(__dirname, '../sample_workflows/Z_Image_Turbo_Sample.json'), 'utf8'));
    let old;
    const find = value => {
        if (!value || typeof value !== 'object') return;
        if (value.type === 'DINKI_LMStudio') old = value;
        for (const child of Object.values(value)) find(child);
    };
    find(data);
    assert.ok(old);
    const savedInputs = structuredClone(old.inputs);
    const previousValues = [...old.widgets_values];
    f.extension.beforeConfigureGraph(data);
    const node = new f.Node(); node.configure(old);
    const exported = await f.app.graphToPrompt();
    assert.deepEqual(node.inputs, savedInputs);
    assert.equal(exported.output['0'].inputs.model_key, previousValues[3]);
    assert.equal(exported.output['0'].inputs.max_tokens, previousValues[6]);
    assert.equal(exported.output['0'].inputs.connection_alias, 'Migrated LLM 1');
});

test('failed and linked API migrations still complete API workflow loading', async () => {
    const f = await fixture(); f.failMigration();
    const data = { '1': { class_type: 'DINKI_LMStudio', inputs: {
        ip_address: 'private-host', port: 1234, api_key: 'private-key', user_prompt: 'hello',
    } } };
    assert.equal(await f.app.loadApiJson(data), 'loaded');
    assert.equal((await f.app.graphToPrompt()).output['0'].inputs.user_prompt, 'hello');
    assert.equal(f.apiLoads[0]['1'].inputs.connection_alias, NONE);
    assert.ok(!JSON.stringify(f.apiLoads).includes('private-key'));
});

function response(text = 'generated answer', scope = 'a'.repeat(32), signature = 'b'.repeat(64)) {
    return JSON.stringify({ version: 1, scope, signature, text });
}

test('generation mode has an English label and the internal cache remains serialized but hidden', async () => {
    const f = await fixture(), node = new f.Node();
    const mode = node.widgets.find(widget => widget.name === 'generation_mode');
    const cache = node.widgets.find(widget => widget.name === 'cached_response');
    assert.equal(mode.label, 'Generation mode');
    assert.equal(mode.value, 'Always Generate');
    assert.equal(cache.hidden, true);
    assert.equal(cache.type, 'converted-widget');
    assert.deepEqual(Array.from(cache.computeSize()), [0, -4]);
    assert.match(JSON.parse(cache.value).scope, /^[0-9a-f]{32}$/);
    assert.ok(node.serialize().widgets_values.includes(cache.value));
    assert.equal((await f.app.graphToPrompt()).output['0'].inputs.cached_response, cache.value);
});

test('successful answer updates the saved cache, including empty answers and Nodes 2.0 notifications', async () => {
    const f = await fixture(), node = new f.Node();
    let changes = 0, dirty = 0, seen;
    node.graph.change = () => { changes++; };
    f.app.rootGraph = node.graph;
    node.setDirtyCanvas = () => { dirty++; };
    node.onWidgetChanged = (...args) => { seen = args; };
    const cache = node.widgets.find(widget => widget.name === 'cached_response');
    const initial = cache.value;
    const message = { dkst_llm_response: [response('')] };
    assert.equal(node.onExecuted(message, 'extra'), 42);
    assert.deepEqual(node.executedArgs, [message, 'extra']);
    assert.deepEqual(seen, ['cached_response', message.dkst_llm_response[0], initial]);
    assert.equal(JSON.parse(cache.value).text, '');
    assert.equal(changes, 1);
    assert.equal(dirty, 1);
    node.onExecuted(message);
    assert.equal(changes, 1);
    const saved = node.serialize();
    const restored = new f.Node(); restored.configure(saved);
    assert.equal(restored.widgets.find(widget => widget.name === 'cached_response').value, cache.value);
});

test('cached answers stay isolated and updates dirty both the owning subgraph and root workflow', async () => {
    const f = await fixture(), first = new f.Node(), second = new f.Node();
    let inner = 0, root = 0;
    f.app.rootGraph = { change() { root++; } };
    first.graph = { change() { inner++; } };
    const secondCache = second.widgets.find(widget => widget.name === 'cached_response');
    const untouched = secondCache.value;
    first.onExecuted({ dkst_llm_response: [response('first answer')] });
    assert.equal(inner, 1);
    assert.equal(root, 1);
    assert.equal(secondCache.value, untouched);
});

test('failed or malformed execution updates preserve the previous successful cache', async () => {
    const f = await fixture(), node = new f.Node();
    node.onExecuted({ dkst_llm_response: [response('keep')] });
    const cache = node.widgets.find(widget => widget.name === 'cached_response');
    const saved = cache.value;
    for (const message of [undefined, {}, { dkst_llm_response: 'wrong shape' },
        { dkst_llm_response: ['bad JSON'] }, { dkst_llm_response: ['null'] },
        { dkst_llm_response: [response(null)] }, { dkst_llm_response: [response('bad', 'wrong')] }]) {
        assert.equal(node.onExecuted(message), 42);
        assert.equal(cache.value, saved);
    }
});

test('legacy paste cannot leave connection fields in newly appended generation and cache widgets', async () => {
    const f = await fixture(), node = new f.Node(false);
    node.configure(legacy(false, true));
    assert.equal(node.widgets.find(widget => widget.name === 'generation_mode').value, 'Always Generate');
    const cache = node.widgets.find(widget => widget.name === 'cached_response');
    assert.equal(JSON.parse(cache.value).version, 1);
    assert.ok(!JSON.stringify(node.serialize()).includes('private-key'));
});

test('real backend result survives frontend serialization and a new backend skips HTTP despite seed changes', async () => {
    const f = await fixture();
    let node = new f.Node();
    node.widgets.find(widget => widget.name === 'generation_mode').value = 'Generate on Change';
    node.widgets.find(widget => widget.name === 'connection_alias').value = 'Local';
    const backend = inputs => {
        const script = `import contextlib, io, json, runpy, sys, types
from unittest.mock import MagicMock, patch
requests = MagicMock()
requests.HTTPError = RuntimeError
requests.post.return_value.status_code = 200
requests.post.return_value.headers = {"Content-Type": "application/json"}
requests.post.return_value.json.return_value = {"choices": [{"message": {"content": "saved answer"}}]}
connections = types.SimpleNamespace(NONE="-- Select connection --", aliases=lambda: ["Local"],
    resolve=lambda alias: {"revision": "stable"}, connection_url=lambda profile: "http://local:1234",
    connection_headers=lambda profile: {})
with patch.dict(sys.modules, {"requests": requests, "numpy": MagicMock(), "PIL": MagicMock(),
    "integration": types.ModuleType("integration"), "integration.dinki_llm_connections": connections}):
    Node = runpy.run_path(sys.argv[1], run_name="integration.dinki_lmstudio")["DINKI_LMStudio"]
inputs = json.loads(sys.argv[2])
inputs.pop("control_after_generate", None)
with contextlib.redirect_stdout(io.StringIO()):
    result = Node().run(**inputs, unique_id="root:2")
print(json.dumps({"output": result, "calls": requests.post.call_count}))`;
        const result = spawnSync('python3', ['-c', script,
            join(__dirname, '../ComfyUI-DINKIssTyle/dinki_lmstudio.py'), JSON.stringify(inputs)], { encoding: 'utf8' });
        assert.equal(result.status, 0, result.stderr);
        return JSON.parse(result.stdout);
    };
    const first = backend((await f.app.graphToPrompt()).output['0'].inputs);
    assert.equal(first.calls, 1);
    node.onExecuted(first.output.ui);
    const saved = node.serialize();
    node = new f.Node(); node.configure(saved);
    node.widgets.find(widget => widget.name === 'seed').value = 9001;
    const second = backend((await f.app.graphToPrompt()).output['1'].inputs);
    assert.equal(second.calls, 0);
    assert.deepEqual(second.output.result, ['saved answer']);
    node.widgets.find(widget => widget.name === 'user_prompt').value = 'changed prompt';
    const third = backend((await f.app.graphToPrompt()).output['1'].inputs);
    assert.equal(third.calls, 1);
});
