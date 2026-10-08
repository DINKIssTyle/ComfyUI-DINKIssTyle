const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_execution_progress.js'), 'utf8');
class DetailEvent extends Event {
    constructor(name, options) { super(name); this.detail = options.detail; }
}
const running = (node_id, value = 0, max = 1) => ({ node_id, display_node_id: node_id, state: 'running', value, max });

function fixture({ nativeFirst = true, enabled = true } = {}) {
    const api = new EventTarget();
    const microtasks = [];
    const frames = [];
    let extension, latest, nativeState, deliveries = 0;
    const nativeHandler = ({ detail }) => {
        ++deliveries;
        latest = detail;
        if (!frames.length) frames.push(() => { nativeState = latest; });
    };
    if (nativeFirst) api.addEventListener('progress_state', nativeHandler);
    const app = {
        registerExtension(value) { extension = value; },
        extensionManager: { setting: { get: () => enabled } },
    };
    let unload;
    const context = { app, api, CustomEvent: DetailEvent,
        queueMicrotask: fn => microtasks.push(fn),
        window: { addEventListener(name, fn) { if (name === 'pagehide') unload = fn; } } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, ''), context);
    extension.setup();
    if (!nativeFirst) api.addEventListener('progress_state', nativeHandler);
    const emit = (name, detail) => api.dispatchEvent(new DetailEvent(name, { detail }));
    const flush = () => {
        let limit = 1000;
        while (microtasks.length) { assert.ok(--limit > 0, 'must not redispatch recursively'); microtasks.shift()(); }
        while (frames.length) frames.shift()();
    };
    return { extension, emit, flush, unload: () => unload({}), deliveries: () => deliveries,
        state: () => nativeState,
        current: () => Object.values(nativeState?.nodes ?? {}).find(state => state.state === 'running') };
}

test('last waiting branch does not hide sampler progress in either listener order', () => {
    for (const nativeFirst of [true, false]) {
        const f = fixture({ nativeFirst });
        f.emit('execution_start', { prompt_id: 'job' });
        f.emit('dkst.branch_status', { prompt_id: 'job', node_id: '105:592', waiting: true });
        f.emit('dkst.branch_status', { prompt_id: 'job', node_id: '105:603', waiting: true });
        const sampler = Object.freeze(running('105:14', 2, 4));
        const branch = Object.freeze(running('105:592'));
        const raw = Object.freeze({ prompt_id: 'job', nodes: Object.freeze({
            '105:592': branch, '105:603': Object.freeze(running('105:603')), '105:14': sampler,
        }) });
        f.emit('progress_state', raw);
        f.flush();
        assert.equal(f.current().node_id, '105:14');
        assert.equal(f.current().value / f.current().max, 0.5);
        assert.equal(f.state().nodes['105:592'].state, 'pending');
        assert.equal(f.state().nodes['105:14'], sampler);
        assert.equal(raw.nodes['105:592'].state, 'running');
        assert.equal(f.deliveries(), 2);
    }
});

test('confirmed waiting status also corrects a snapshot that arrived earlier', () => {
    const f = fixture();
    f.emit('progress_state', { prompt_id: 'job', nodes: { '1': running('1'), '2': running('2', 3, 10) } });
    f.flush();
    assert.equal(f.current().node_id, '1');
    f.emit('dkst.branch_status', { prompt_id: 'job', node_id: 1, waiting: true });
    f.flush();
    assert.equal(f.current().node_id, '2');
    assert.equal(f.current().value, 3);
});

test('sampler to decode updates both the selected node and its percent', () => {
    const f = fixture();
    f.emit('dkst.branch_status', { prompt_id: 'job', node_id: '105:592', waiting: true });
    f.emit('progress_state', { prompt_id: 'job', nodes: {
        '105:592': running('105:592'), '105:14': running('105:14', 2, 4),
    } });
    f.flush();
    f.emit('progress_state', { prompt_id: 'job', nodes: {
        '105:592': running('105:592'),
        '105:14': { ...running('105:14', 4, 4), state: 'finished' },
        '105:591': running('105:591', 12, 30),
    } });
    f.flush();
    assert.equal(f.current().node_id, '105:591');
    assert.equal(f.current().value / f.current().max, 0.4);
});

test('same local IDs in nested subgraphs and different jobs remain separate', () => {
    const f = fixture();
    f.emit('dkst.branch_status', { prompt_id: 'a', node_id: '105:8:592', waiting: true });
    f.emit('progress_state', { prompt_id: 'a', nodes: {
        '105:8:592': running('105:8:592'), '106:8:592': running('106:8:592', 1, 4),
    } });
    f.flush();
    assert.equal(f.current().node_id, '106:8:592');
    const other = { prompt_id: 'b', nodes: { '105:8:592': running('105:8:592') } };
    f.emit('progress_state', other);
    f.flush();
    assert.equal(f.state(), other);
});

test('without confirmed waits, unknown nodes and multi-worker progress remain untouched', () => {
    const f = fixture();
    const raw = { prompt_id: 'job', nodes: { '14': running('14', 1, 4), '591': running('591', 2, 30) } };
    f.emit('progress_state', raw);
    f.flush();
    assert.equal(f.state(), raw);
    assert.equal(f.deliveries(), 1);
});

test('resuming or finished switches are not replayed as stale waiting nodes', () => {
    const f = fixture();
    f.emit('dkst.branch_status', { prompt_id: 'job', node_id: '592', waiting: true });
    f.emit('progress_state', { prompt_id: 'job', nodes: { '592': running('592') } });
    f.emit('dkst.branch_status', { prompt_id: 'job', node_id: '592', waiting: false });
    f.flush();
    assert.equal(f.deliveries(), 1);
    f.emit('dkst.branch_status', { prompt_id: 'job', node_id: '592', waiting: true });
    const raw = { prompt_id: 'job', nodes: { '592': { ...running('592', 1, 1), state: 'finished' } } };
    f.emit('progress_state', raw);
    f.flush();
    assert.equal(f.state(), raw);
    assert.equal(f.state().nodes['592'].state, 'finished');
});

test('completion, cancellation and errors cancel queued corrections and late events', () => {
    for (const name of ['execution_success', 'execution_interrupted', 'execution_error']) {
        const f = fixture();
        f.emit('dkst.branch_status', { prompt_id: 'job', node_id: '1', waiting: true });
        f.emit('progress_state', { prompt_id: 'job', nodes: { '1': running('1') } });
        f.emit(name, { prompt_id: 'job' });
        f.flush();
        assert.equal(f.deliveries(), 1);
        f.emit('dkst.branch_status', { prompt_id: 'job', node_id: '1', waiting: true });
        f.emit('progress_state', { prompt_id: 'job', nodes: { '1': running('1') } });
        f.flush();
        assert.equal(f.deliveries(), 2);
        f.emit('execution_start', { prompt_id: 'job' });
        const raw = { prompt_id: 'job', nodes: { '1': running('1') } };
        f.emit('progress_state', raw);
        f.flush();
        assert.equal(f.state(), raw);
    }
});

test('rapid progress events publish only the latest snapshot', () => {
    const f = fixture();
    f.emit('dkst.branch_status', { prompt_id: 'job', node_id: '1', waiting: true });
    for (let value = 0; value < 20; value++) {
        f.emit('progress_state', { prompt_id: 'job', nodes: { '1': running('1'), '2': running('2', value, 20) } });
    }
    f.flush();
    assert.equal(f.current().node_id, '2');
    assert.equal(f.current().value, 19);
    assert.equal(f.deliveries(), 21);
});

test('setting restores the native snapshot and can be re-enabled during a run', () => {
    const f = fixture({ enabled: false });
    f.extension.setup();
    f.emit('dkst.branch_status', { prompt_id: 'job', node_id: '1', waiting: true });
    const raw = { prompt_id: 'job', nodes: { '1': running('1'), '2': running('2', 1, 4) } };
    f.emit('progress_state', raw);
    f.flush();
    assert.equal(f.state(), raw);
    f.extension.settings[0].onChange(true);
    f.flush();
    assert.equal(f.current().node_id, '2');
    f.extension.settings[0].onChange(false);
    f.flush();
    assert.equal(f.state(), raw);
    f.unload();
    f.extension.settings[0].onChange(true);
    f.emit('dkst.branch_status', { prompt_id: 'job', node_id: '1', waiting: true });
    f.emit('progress_state', raw);
    f.flush();
    assert.equal(f.state(), raw);
});
