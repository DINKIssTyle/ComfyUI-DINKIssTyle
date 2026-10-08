const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_execution_report.js'), 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));
class DetailEvent extends Event {
    constructor(name, detail) { super(name); this.detail = detail; }
}

function fixture({ secure = false } = {}) {
    let extension, copied, selected, unload, queueHandler;
    const api = new EventTarget();
    const calls = [];
    api.queuePrompt = async function(...args) {
        calls.push({ receiver: this, args });
        return queueHandler ? queueHandler(...args) : { prompt_id: 'job' };
    };
    const graph = { _nodes: [] };
    const app = {
        rootGraph: graph, graph,
        registerExtension(value) { extension = value; },
        async graphToPrompt(graph = this.rootGraph) {
            const output = {};
            const walk = (graph, path = []) => {
                for (const node of graph._nodes ?? []) {
                    const id = [...path, node.id].join(':');
                    if (node.comfyClass === 'DINKI_Execution_Report') output[id] = { class_type: node.comfyClass };
                    if (node.subgraph) walk(node.subgraph, [...path, node.id]);
                }
            };
            walk(graph);
            return { output, workflow: {} };
        },
    };
    const document = { activeElement: null, createElement, execCommand() { copied = selected; return true; } };
    function createElement(tag) {
        return {
            tagName: tag.toUpperCase(), style: {}, children: [], events: {}, attributes: {},
            append(...items) { this.children.push(...items); },
            appendChild(item) { this.children.push(item); },
            addEventListener(name, fn) { this.events[name] = fn; },
            setAttribute(name, value) { this.attributes[name] = value; },
            removeAttribute(name) { delete this.attributes[name]; },
            focus() { document.activeElement = this; }, select() { selected = this.value; }, remove() {},
        };
    }
    document.body = createElement('body');
    const navigator = secure ? { clipboard: { writeText: async text => { copied = text; } } } : {};
    const renders = [];
    const pendingRenders = [];
    const context = { app, api, document, navigator, isSecureContext: secure, queueMicrotask,
        window: { addEventListener(name, fn) { if (name === 'pagehide') unload = fn; } },
        installNoteMarkdownStyles() {},
        async renderNoteMarkdown(preview, markdown, isCurrent) {
            if (pendingRenders.length) await pendingRenders.shift();
            if (isCurrent()) { preview.rendered = markdown; renders.push(markdown); }
        },
        setTimeout() { return 1; }, clearTimeout() {},
    };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, ''), context);
    class Report {
        constructor(id = 1, owner = graph) {
            this.id = id;
            this.comfyClass = 'DINKI_Execution_Report';
            this.widgets = [];
            this.properties = {};
            this.size = [200, 100];
            this.graph = owner;
            this.onNodeCreated();
            owner._nodes.push(this);
        }
        addDOMWidget(_name, _type, root) { this.root = root; return this.domWidget = { options: {} }; }
        setDirtyCanvas() {}
        setSize(size) { this.size = size; }
    }
    extension.beforeRegisterNodeDef(Report, { name: 'DINKI_Execution_Report' });
    return { extension, Report, app, api, graph, calls, context, renders, pendingRenders,
        emit(detail) { api.dispatchEvent(new DetailEvent('dkst.execution_report', detail)); },
        get copied() { return copied; }, queueHandler(fn) { queueHandler = fn; },
        unload() { unload(); }, setup() { extension.setup(); },
        async queue(options = {}) { return api.queuePrompt(0, await app.graphToPrompt(), options); },
    };
}

const final = (prompt_id = 'job', node_id = '1', markdown = '| Node | Time |\n| --- | --- |\n| Sampler | 5 s |') =>
    ({ prompt_id, status: 'Completed', reports: [{ node_id, markdown }] });

test('queued report receives start and final Markdown, copied verbatim over HTTP', async () => {
    const f = fixture();
    f.setup();
    const node = new f.Report();
    const [toolbar, preview] = node.root.children;
    const [copy, status] = toolbar.children;
    assert.equal(copy.disabled, true);
    await f.queue();
    f.emit({ prompt_id: 'job', status: 'Running', reports: [{ node_id: '1' }] });
    assert.equal(status.textContent, 'Recording…');
    f.emit(final());
    await settle();
    assert.equal(status.textContent, 'Completed');
    assert.equal(preview.rendered, final().reports[0].markdown);
    assert.equal(copy.disabled, false);
    await copy.events.click();
    assert.equal(f.copied, final().reports[0].markdown);
    assert.equal(copy.textContent, 'Copied!');
    assert.equal(node.domWidget.serialize, false);
});

test('secure Copy uses Clipboard API and error reports remain copyable', async () => {
    const f = fixture({ secure: true });
    const node = new f.Report();
    node.dkstApplyExecutionReport({ markdown: '# partial' }, 'Error', 'job');
    await node.root.children[0].children[0].events.click();
    assert.equal(f.copied, '# partial');
    assert.equal(node.properties.dkstExecutionReport.status, 'Error');
});

test('a completed job arriving before the HTTP response is retained', async () => {
    const f = fixture();
    f.setup();
    const node = new f.Report();
    f.queueHandler(async () => { f.emit(final()); return { prompt_id: 'job' }; });
    await f.queue();
    await settle();
    assert.equal(node.root.children[1].rendered, final().reports[0].markdown);
});

test('same local report IDs in nested graphs route to their full paths', async () => {
    const f = fixture();
    const a = { _nodes: [] }, b = { _nodes: [] };
    f.graph._nodes.push({ id: 105, subgraph: a }, { id: 106, subgraph: b });
    const nodeA = new f.Report(8, a), nodeB = new f.Report(8, b);
    f.setup();
    await f.queue();
    f.emit({ prompt_id: 'job', status: 'Completed', reports: [
        { node_id: '105:8', markdown: '# A' }, { node_id: '106:8', markdown: '# B' },
    ] });
    await settle();
    assert.equal(nodeA.root.children[1].rendered, '# A');
    assert.equal(nodeB.root.children[1].rendered, '# B');
});

test('queue wrappers preserve receivers, all options, and native response identity', async () => {
    const f = fixture();
    new f.Report();
    const queue = f.api.queuePrompt;
    const response = { prompt_id: 'job', number: 123 };
    f.queueHandler(async () => response);
    f.setup();
    const options = { partialExecutionTargets: ['1'], previewMethod: 'taesd' };
    assert.equal(await f.queue(options), response);
    assert.equal(f.calls[0].receiver, f.api);
    assert.equal(f.calls[0].args[2], options);
    f.unload();
    assert.equal(f.api.queuePrompt, queue);
});

test('switching workflows during export never writes on a new node with the same ID', async () => {
    const f = fixture();
    const oldNode = new f.Report();
    f.setup();
    const exported = await f.app.graphToPrompt();
    oldNode.onRemoved();
    oldNode.graph = null;
    f.app.rootGraph = f.app.graph = { _nodes: [] };
    const newNode = new f.Report(1, f.app.rootGraph);
    await f.api.queuePrompt(0, exported);
    f.emit(final());
    await settle();
    assert.equal(newNode.properties.dkstExecutionReport, undefined);
    assert.equal(oldNode.properties.dkstExecutionReport, undefined);
});

test('unknown prompt events cannot attach a report to the current workflow', async () => {
    const f = fixture();
    const node = new f.Report();
    f.setup();
    f.emit(final('foreign'));
    await settle();
    assert.equal(node.properties.dkstExecutionReport, undefined);
});

test('separate queued jobs preserve each report binding', async () => {
    const f = fixture();
    f.setup();
    const nodeA = new f.Report();
    f.queueHandler(async () => ({ prompt_id: 'a' }));
    await f.queue();
    const graphB = { _nodes: [] };
    f.app.rootGraph = f.app.graph = graphB;
    const nodeB = new f.Report(1, graphB);
    f.queueHandler(async () => ({ prompt_id: 'b' }));
    await f.queue();
    f.emit(final('a', '1', '# A'));
    f.emit(final('b', '1', '# B'));
    await settle();
    assert.equal(nodeA.root.children[1].rendered, '# A');
    assert.equal(nodeB.root.children[1].rendered, '# B');
});

test('saved Markdown, history output and resized node restore correctly', async () => {
    const f = fixture();
    const node = new f.Report();
    node.properties.dkstExecutionReport = { markdown: '# saved', status: 'Completed', prompt_id: 'old' };
    node.onConfigure({ size: [740, 520] });
    await settle();
    assert.deepEqual(Array.from(node.size), [740, 520]);
    const serialized = {};
    node.onSerialize(serialized);
    assert.deepEqual(Array.from(serialized.size), [740, 520]);
    assert.equal(node.root.children[1].rendered, '# saved');
    node.onExecuted({ dkst_execution_report: [{ markdown: '# history', status: 'Interrupted', prompt_id: 'history' }] });
    await settle();
    assert.equal(node.root.children[1].rendered, '# history');
    assert.equal(node.properties.dkstExecutionReport.status, 'Interrupted');
    node.onExecuted({ execution_report_notice: ['Unsupported execution API'] });
    assert.equal(node.root.children[0].children[1].textContent, 'Unavailable');
    assert.equal(node.root.children[1].textContent, 'Unsupported execution API');
});

test('old asynchronous rendering cannot replace a newer report', async () => {
    const f = fixture();
    const node = new f.Report();
    let release;
    f.pendingRenders.push(new Promise(resolve => { release = resolve; }));
    node.dkstApplyExecutionReport({ markdown: '# old' });
    await Promise.resolve();
    node.dkstApplyExecutionReport({ markdown: '# new' });
    await settle();
    release();
    await settle();
    assert.equal(node.root.children[1].rendered, '# new');
});

test('removed nodes and unloaded extension stop receiving events', async () => {
    const f = fixture();
    const node = new f.Report();
    f.setup();
    f.setup();
    await f.queue();
    node.onRemoved();
    f.emit(final());
    await settle();
    assert.equal(node.properties.dkstExecutionReport, undefined);
    f.unload();
    assert.doesNotThrow(() => f.emit(final('another')));
});

test('failed Markdown renderer falls back to readable report source', async () => {
    const f = fixture();
    const node = new f.Report();
    f.context.renderNoteMarkdown = async () => { throw new Error('renderer unavailable'); };
    node.dkstApplyExecutionReport({ markdown: '| Node | Time |' });
    await settle();
    assert.equal(node.root.children[1].textContent, '| Node | Time |');
    assert.equal(node.root.children[1].style.whiteSpace, 'pre-wrap');
    assert.equal(node.root.children[0].children[0].disabled, false);
});

test('queue failure remains a rejection and produces no report on another run', async () => {
    const f = fixture();
    const node = new f.Report();
    const error = new Error('native queue error');
    f.setup();
    f.queueHandler(async () => { throw error; });
    await assert.rejects(() => f.queue(), caught => caught === error);
    f.emit(final('foreign'));
    await settle();
    assert.equal(node.properties.dkstExecutionReport, undefined);
});

test('large queue does not discard the report binding for its first job', async () => {
    const f = fixture();
    const node = new f.Report();
    f.setup();
    let sequence = 0;
    f.queueHandler(async () => ({ prompt_id: `job-${++sequence}` }));
    for (let i = 0; i < 140; i++) await f.queue();
    f.emit(final('job-1', '1', '# first'));
    await settle();
    assert.equal(node.root.children[1].rendered, '# first');
});

const zeroTimesReport = [
    '## Execution Report', '', 'Status: **Completed**', '',
    '| Node | Processing time | Share |', '| :--- | ---: | ---: |',
    '| Instant · #1 | 0.000 s | 0.00% |',
    '| Rounded tiny time · #2 | 0.000 s | 0.01% |',
    '| Interrupted tiny time · #3 | 0.000 s (Interrupted) | 0.00% |',
    '| Worker · #4 | 1.001 s | 99.99% |',
    '| Cached loader · #5 | Cached | — |',
    '| Name contains 0.000 s · #6 | 0.002 s | 0.20% |',
    '| **Total** | **1.003 s** | **100.00%** |', '',
    '**Node processing time sum: 1.003 s**  ', '**Workflow elapsed time: 1.100 s**',
].join('\n');

test('zero-time checkbox updates table and copied Markdown without changing the saved source', async () => {
    const f = fixture();
    const node = new f.Report();
    const [copy, , label] = node.root.children[0].children;
    const check = label.children[0];
    assert.equal(check.type, 'checkbox');
    assert.equal(check.checked, false);
    node.dkstApplyExecutionReport({ markdown: zeroTimesReport }, 'Completed', 'job');
    await settle();
    check.checked = true;
    check.events.change();
    await settle();
    const filtered = node.root.children[1].rendered;
    assert.doesNotMatch(filtered, /Instant|Rounded tiny|Interrupted tiny/);
    assert.match(filtered, /Worker|Cached loader|Name contains 0.000 s/);
    assert.match(filtered, /\*\*Total\*\* \| \*\*1.003 s\*\* \| \*\*100.00%\*\*/);
    assert.match(filtered, /99.99%/);
    await copy.events.click();
    assert.equal(f.copied, filtered);
    assert.equal(node.properties.dkstExecutionReport.markdown, zeroTimesReport);
    assert.equal(node.properties.dkstExecutionReportHideZero, true);
    check.checked = false;
    check.events.change();
    await settle();
    assert.equal(node.root.children[1].rendered, zeroTimesReport);
});

test('saved checkbox restores filtering and keeps an all-zero report total visible', async () => {
    const f = fixture();
    const node = new f.Report();
    const source = '| Tiny · #1 | 0.000 s | 0.00% |\n| **Total** | **0.000 s** | **0.00%** |';
    node.properties.dkstExecutionReportHideZero = true;
    node.properties.dkstExecutionReport = { markdown: source, status: 'Completed', prompt_id: 'old' };
    node.onConfigure({});
    await settle();
    assert.equal(node.root.children[0].children[2].children[0].checked, true);
    assert.equal(node.root.children[1].rendered, '| **Total** | **0.000 s** | **0.00%** |');
    assert.equal(node.properties.dkstExecutionReport.markdown, source);
});

test('toggling while recording keeps the current recording state', async () => {
    const f = fixture();
    const node = new f.Report();
    node.dkstApplyExecutionReport({ markdown: zeroTimesReport }, 'Completed', 'old');
    node.dkstApplyExecutionReport({}, 'Running', 'new');
    const [copy, status, label] = node.root.children[0].children;
    const check = label.children[0];
    check.checked = true;
    check.events.change();
    await settle();
    assert.equal(status.textContent, 'Recording…');
    assert.equal(copy.disabled, true);
    assert.match(node.root.children[1].textContent, /Recording node processing times/);
    assert.equal(node.properties.dkstExecutionReport.prompt_id, 'old');
});
