const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');
const AUTO = 'DINKI_Auto_Node_Change';
const TOGGLE = 'After run: Toggle 1 / 2';
class DetailEvent extends Event {
    constructor(name, detail) { super(name); this.detail = detail; }
}

function fixture({ autoMode = TOGGLE, active = true, disableMode = 'Mute' } = {}) {
    const extensions = new Map(), ticks = [], calls = [];
    let nextId = 0, queueHandler, unload;
    const api = new EventTarget();
    api.queuePrompt = async function (...args) {
        calls.push({ receiver: this, args });
        return queueHandler ? queueHandler(...args) : { prompt_id: `job-${++nextId}` };
    };
    const graph = { _nodes: [], change() {}, setDirtyCanvas() {} };
    const app = { graph, rootGraph: graph, registerExtension(ext) { extensions.set(ext.name, ext); },
        async graphToPrompt(root = this.rootGraph) {
            const output = {};
            // Simulate frontend export using the effective promoted widget values.
            for (const [id, binding] of context.autoNodeChangeBindings(root)) {
                output[id] = { class_type: AUTO, inputs: { ...binding.values } };
            }
            const walk = (graph, path = []) => {
                for (const node of graph._nodes) {
                    if (node.mode === 2 || node.mode === 4) continue;
                    const id = [...path, node.id].join(':');
                    if (node.comfyClass && node.comfyClass !== AUTO) output[id] = { class_type: node.comfyClass };
                    if (node.subgraph) walk(node.subgraph, [...path, node.id]);
                }
            };
            walk(root);
            return { output, workflow: { selection: [...context.autoNodeChangeBindings(root).values()].map(b => b.values.active) } };
        },
    };
    const context = { app, api, queueMicrotask, setInterval(fn) { ticks.push(fn); return ticks.length; },
        window: { addEventListener(name, fn) { if (name === 'pagehide') unload = fn; } } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), context);
    const controls = extensions.get('DINKI.AutoNodeChange');
    const execution = extensions.get('DINKI.AutoNodeChange.Execution');
    function addNode(owner = graph, id = '1', values = {}) {
        const node = { id, graph: owner, mode: 0, comfyClass: AUTO, properties: {}, outputs: [],
            widgets: Object.entries({ node_ids_1: '2', node_ids_2: '3', active, disable_mode: disableMode,
                group_1_label: 'First', group_2_label: 'Second', auto_mode: autoMode, ...values })
                .map(([name, value]) => ({ name, value })),
            addOutput(name, type) { this.outputs.push({ name, type }); }, setDirtyCanvas() {} };
        owner._nodes.push(node);
        controls.nodeCreated(node);
        return node;
    }
    const node = addNode();
    const targets = ['2', '3'].map(id => ({ id, mode: 0, comfyClass: 'Target' }));
    graph._nodes.push(...targets);
    return { app, api, graph, node, targets, controls, execution, extensions, calls, context, addNode,
        widget(name, target = node) { return target.widgets.find(w => w.name === name); },
        emit(name = 'execution_success', id = 'job-1') { api.dispatchEvent(new DetailEvent(name, { prompt_id: id })); },
        handler(fn) { queueHandler = fn; },
        setup() { controls.setup(); execution.setup(); },
        tick() { for (const tick of ticks) tick(); }, unload() { unload(); },
        async queue(options = {}) { return api.queuePrompt(0, await app.graphToPrompt(), options); },
    };
}

function promote(inner, node, widgetName, hostName, value, id) {
    const input = { name: widgetName, widget: { name: widgetName } };
    node.getWidgetFromSlot = slot => node.widgets.find(w => w.name === slot.widget?.name);
    node.inputs ??= [];
    node.inputs.push(input);
    inner.inputNode ??= { slots: [] };
    inner.links ??= {};
    const linkId = Object.keys(inner.links).length + 1;
    inner.links[linkId] = { resolve: () => ({ inputNode: node, input }) };
    inner.inputNode.slots.push({ name: hostName, linkIds: [linkId] });
    const widget = { name: hostName, value };
    return { id, mode: 0, subgraph: inner, inputs: [{ name: hostName, _widget: widget }], widgets: [widget] };
}

test('sequential successful runs export 1, 2, 1 and only then change the next selection', async () => {
    const f = fixture();
    f.setup();
    for (const [index, active] of [true, false, true].entries()) {
        const response = await f.queue();
        const data = f.calls[index].args[1];
        assert.equal(data.output['1'].inputs.active, active);
        assert.equal(data.workflow.selection[0], active);
        assert.equal(!!data.output['2'], active);
        assert.equal(!!data.output['3'], !active);
        assert.equal(f.widget('active').value, active, 'queue acceptance does not switch');
        f.emit('execution_success', response.prompt_id);
        assert.equal(f.widget('active').value, !active);
        assert.deepEqual(f.targets.map(n => n.mode), active ? [2, 0] : [0, 2]);
        assert.equal(data.output['1'].inputs.active, active, 'queued snapshot remains unchanged');
        f.emit('execution_success', response.prompt_id);
        assert.equal(f.widget('active').value, !active, 'duplicate completion is ignored');
    }
});

test('Group 2 mode changes once and keeps Group 2 selected', async () => {
    const f = fixture({ autoMode: 'After run: Group 2' });
    f.setup();
    for (let i = 0; i < 3; i++) {
        const response = await f.queue();
        assert.equal(f.calls[i].args[1].output['1'].inputs.active, i === 0);
        f.emit('execution_success', response.prompt_id);
        assert.equal(f.widget('active').value, false);
    }
});

test('Manual mode keeps its group, labels and Boolean socket', async () => {
    const f = fixture({ autoMode: 'Manual' });
    f.setup();
    await f.queue();
    f.emit();
    assert.equal(f.widget('active').value, true);
    assert.equal(f.widget('active').options.on, 'First');
    assert.equal(f.widget('active').options.off, 'Second');
    assert.deepEqual(f.node.outputs, [{ name: 'switch', type: 'BOOLEAN' }]);
});

for (const name of ['execution_error', 'execution_interrupted']) {
    test(`${name} preserves selection and releases the next run`, async () => {
        const f = fixture();
        f.setup();
        await f.queue();
        f.emit(name);
        f.emit(); // Delayed success must not revive a failed job.
        assert.equal(f.widget('active').value, true);
        const response = await f.queue();
        f.emit('execution_success', response.prompt_id);
        assert.equal(f.widget('active').value, false);
    });
}

test('queue failure and a response without a prompt ID do not consume a transition', async () => {
    const f = fixture();
    f.setup();
    f.handler(async () => { throw new Error('validation failed'); });
    await assert.rejects(f.queue(), /validation failed/);
    f.handler(async () => ({}));
    await f.queue();
    f.emit();
    assert.equal(f.widget('active').value, true);
    f.handler(async () => ({ prompt_id: 'valid' }));
    await f.queue();
    f.emit('execution_success', 'valid');
    assert.equal(f.widget('active').value, false);
});

test('fully cached completion before HTTP responds still switches exactly once', async () => {
    const f = fixture();
    f.setup();
    f.handler(async () => { f.emit(); f.emit(); return { prompt_id: 'job-1' }; });
    await f.queue();
    assert.equal(f.widget('active').value, false);
    f.emit();
    assert.equal(f.widget('active').value, false);
});

test('an early failure cannot be overwritten by a delayed success before HTTP responds', async () => {
    const f = fixture();
    f.setup();
    f.handler(async () => { f.emit('execution_error'); f.emit(); return { prompt_id: 'job-1' }; });
    await f.queue();
    assert.equal(f.widget('active').value, true);
});

test('missing optional mode uses the backend default after workflow restoration', async () => {
    const f = fixture();
    f.node.widgets = f.node.widgets.filter(widget => widget.name !== 'auto_mode');
    f.setup();
    await f.queue();
    f.emit();
    assert.equal(f.widget('active').value, false);
});

test('completion from an unknown prompt and exporting without queueing never switches', async () => {
    const f = fixture();
    f.setup();
    await f.app.graphToPrompt();
    f.emit();
    await f.queue();
    f.emit('execution_success', 'foreign');
    assert.equal(f.widget('active').value, true);
});

test('manual selection or mode changes during execution are preserved', async () => {
    for (const [name, value] of [['active', false], ['auto_mode', 'Manual']]) {
        const f = fixture();
        f.setup();
        await f.queue();
        f.widget(name).value = value;
        f.widget(name).callback(value);
        f.emit();
        assert.equal(f.widget(name).value, value);
        if (name === 'auto_mode') assert.equal(f.widget('active').value, true);
    }
});

test('switching workflows or reconfiguring the same graph discards stale completions', async () => {
    for (const reconfigure of [false, true]) {
        const f = fixture();
        f.setup();
        await f.queue();
        if (reconfigure) f.execution.afterConfigureGraph();
        else {
            f.app.rootGraph = f.app.graph = { _nodes: [] };
            const replacement = f.addNode(f.app.rootGraph);
            assert.equal(replacement.id, f.node.id);
        }
        f.emit();
        assert.equal(f.widget('active').value, true);
        assert.equal([...f.context.autoNodeChangeBindings(f.app.rootGraph).values()][0].values.active, true);
    }
});

test('removing the original control prevents completion from altering it', async () => {
    const f = fixture();
    f.setup();
    await f.queue();
    f.graph._nodes = f.graph._nodes.filter(node => node !== f.node);
    f.emit();
    assert.equal(f.widget('active').value, true);
});

test('nested promoted controls update all owners and stay changed on the monitor tick', async () => {
    const f = fixture({ active: false });
    const innerHost = promote(f.graph, f.node, 'active', 'inner group', true, '10');
    const middle = { _nodes: [innerHost] };
    innerHost.graph = middle;
    const outerHost = promote(middle, innerHost, 'inner group', 'outer group', true, '20');
    f.app.rootGraph = f.app.graph = { _nodes: [outerHost] };
    outerHost.graph = f.app.rootGraph;
    f.setup();
    await f.queue();
    assert.equal(f.calls[0].args[1].output['20:10:1'].inputs.active, true);
    assert.deepEqual(f.targets.map(n => n.mode), [0, 2]);
    f.emit();
    assert.equal(outerHost.widgets[0].value, false);
    assert.equal(innerHost.widgets[0].value, false);
    assert.equal(f.widget('active').value, false);
    f.tick();
    assert.deepEqual(f.targets.map(n => n.mode), [2, 0]);
    const response = await f.queue();
    assert.equal(f.calls[1].args[1].output['20:10:1'].inputs.active, false);
    f.emit('execution_success', response.prompt_id);
    f.tick();
    assert.equal(outerHost.widgets[0].value, true);
    assert.deepEqual(f.targets.map(n => n.mode), [0, 2]);
});

test('legacy proxy promotions write back to the actual owning widget', async () => {
    for (const overlay of [false, true]) {
        const f = fixture();
        const widget = { name: 'renamed', value: false };
        if (overlay) widget._overlay = { isProxyWidget: true, nodeId: '1', widgetName: 'active' };
        const host = { id: '5', mode: 0, subgraph: f.graph, widgets: [widget],
            properties: overlay ? {} : { proxyWidgets: [['1', 'active']] } };
        f.app.rootGraph = f.app.graph = { _nodes: [host] };
        host.graph = f.app.rootGraph;
        f.setup();
        await f.queue();
        assert.equal(f.calls[0].args[1].output['5:1'].inputs.active, false);
        f.emit();
        f.tick();
        assert.equal(widget.value, true);
        assert.deepEqual(f.targets.map(n => n.mode), [0, 2]);
    }
});

test('overlapping queued jobs suppress auto changes and sequential use recovers', async () => {
    const f = fixture();
    f.setup();
    await f.queue();
    await f.queue();
    f.emit();
    f.emit('execution_success', 'job-2');
    assert.equal(f.widget('active').value, true);
    assert.equal(f.calls[1].args[1].output['1'].inputs.active, true);
    const response = await f.queue();
    f.emit('execution_success', response.prompt_id);
    assert.equal(f.widget('active').value, false);
});

test('separate subgraphs with duplicate local IDs each advance their own control', async () => {
    const f = fixture();
    const second = { _nodes: [], change() {}, setDirtyCanvas() {} };
    const other = f.addNode(second, '1', { active: false });
    f.app.rootGraph = f.app.graph = { _nodes: [{ id: '10', subgraph: f.graph }, { id: '20', subgraph: second }] };
    f.setup();
    await f.queue();
    f.emit();
    assert.equal(f.widget('active').value, false);
    assert.equal(f.widget('active', other).value, true);
});

test('muted or bypassed auto controls and their subgraph hosts never advance', async () => {
    for (const mode of [2, 4]) {
        for (const hostOnly of [false, true]) {
            const f = fixture();
            if (hostOnly) f.app.rootGraph = f.app.graph = { _nodes: [{ id: '10', mode, subgraph: f.graph }] };
            else f.node.mode = mode;
            f.setup();
            await f.queue();
            f.emit();
            assert.equal(f.widget('active').value, true);
        }
    }
});

test('wrappers preserve options, receivers and response identity and detach on page unload', async () => {
    const f = fixture({ disableMode: 'Bypass' });
    const originalQueue = f.api.queuePrompt, originalExport = f.app.graphToPrompt;
    const response = { prompt_id: 'job-1', number: 9 };
    f.handler(async () => response);
    f.setup();
    f.setup();
    const options = { previewMethod: 'taesd' };
    assert.equal(await f.queue(options), response);
    assert.equal(f.calls[0].receiver, f.api);
    assert.equal(f.calls[0].args[2], options);
    assert.deepEqual(f.targets.map(n => n.mode), [0, 4]);
    f.unload();
    assert.equal(f.api.queuePrompt, originalQueue);
    assert.equal(f.app.graphToPrompt, originalExport);
    f.emit();
    assert.equal(f.widget('active').value, true);
});

test('auto switching and execution reports coexist in either wrapper installation order', async () => {
    const reportSource = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_execution_report.js'), 'utf8');
    for (const reportsFirst of [false, true]) {
        const f = fixture();
        vm.runInNewContext(reportSource.replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, ''), f.context);
        const reports = f.extensions.get('DINKI.ExecutionReport');
        let report;
        f.graph._nodes.push({ id: 'report', comfyClass: 'DINKI_Execution_Report', graph: f.graph,
            dkstApplyExecutionReport(value) { report = value; } });
        if (reportsFirst) reports.setup();
        f.setup();
        if (!reportsFirst) reports.setup();
        const response = await f.queue();
        f.api.dispatchEvent(new DetailEvent('dkst.execution_report', { prompt_id: response.prompt_id,
            status: 'Completed', reports: [{ node_id: 'report', markdown: '# done' }] }));
        f.emit('execution_success', response.prompt_id);
        assert.equal(report.markdown, '# done');
        assert.equal(f.widget('active').value, false);
    }
});
