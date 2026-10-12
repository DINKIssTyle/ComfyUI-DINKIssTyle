const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_qwen_train.js'), 'utf8');
const scrollSource = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_scroll.js'), 'utf8').replace(/^export /gm, '');
const uiSource = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_training_ui.js'), 'utf8')
    .replace(/^export /gm, '').replace(/^import .*;\r?\n/gm, '');

class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.dataset = {}; this.listeners = {}; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, fn) { this.listeners[name] = fn; }
    setAttribute(name, value) { this[name] = value; }
    showModal() { this.open = true; }
    close() { this.open = false; this.listeners.close?.(); }
    remove() { this.removed = true; }
    get firstChild() { return this.children[0]; }
}

function fixture(handler, family = 'qwen') {
    let extension, setting;
    const requests = [], intervals = new Map();
    let cleared = false;
    const app = { ui: { settings: { addSetting(value) { setting = value; } } },
        registerExtension(value) { extension = value; }, queuePrompt: async (...args) => args };
    const api = {
        apiURL: path => '/base' + path,
        async fetchApi(path, options) {
            const body = options?.body ? JSON.parse(options.body) : undefined;
            requests.push({ path, body });
            const result = await handler(path, body);
            return { ok: !result.error, json: async () => result, text: async () => result.text };
        },
    };
    const body = new Element('body');
    const wrapper = family === 'qwen' ? source : readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_minimax_train.js'), 'utf8');
    vm.runInNewContext(scrollSource + '\n' + uiSource + '\n' + wrapper.replace(/^import .*;\r?\n/gm, ''), { app, api,
        document: { createElement: tag => new Element(tag), body },
        setTimeout() {}, setInterval(fn) { intervals.set(1, fn); return 1; },
        clearInterval(id) { intervals.delete(id); cleared = true; }, console,
        requestAnimationFrame: cb => cb(),
    });
    extension.setup();
    const node = { comfyClass: family === 'qwen' ? 'DINKI_Qwen21_LoRA_Trainer' : 'DINKI_Minimax_H3_LoRA_Trainer', properties: {}, size: [400, 700],
        widgets: [{ name: 'project_name', value: 'person' },
            { name: 'image_gallery', value: '[{"filename":"face.png","caption":"side profile"}]' },
            { name: 'trigger_word', value: 'person_xyz' }, { name: 'preset', value: 'Fast' },
            { name: 'epochs', value: 30 },
            { name: 'caption_connection_alias', value: '-- Select connection --', options: {} },
            { name: 'caption_model', value: '' }], graph: { incrementVersion() {} },
        setSize(size) { this.size = size; }, setDirtyCanvas() {},
        addDOMWidget(name, type, root, options) {
            const widget = { name, type, root, options };
            this.widgets.push(widget);
            return widget;
        },
    };
    extension.nodeCreated(node);
    const widget = node.widgets.at(-1);
    const buttons = widget.root.children[0].children;
    return { app, node, widget, buttons, requests, intervals, setting, body, get cleared() { return cleared; } };
}

const state = { id: 'a'.repeat(32), status: 'training', project_name: 'person', num_images: 8,
    output_dir: '/loras/DKST_Qwen21/person', checkpoints: [], samples: [], log: 'working' };

test('Full log opens from its first line and offers worker selection and a download', async () => {
    const text = 'Traceback: error header\n' + 'decoder.weight\n'.repeat(3000);
    const f = fixture(async path => ({ text: path.includes('kind=worker') ? 'Worker error' : text }));
    f.node.properties.dkst_qwen21_job_id = state.id;
    await f.buttons.find(button => button.textContent === 'View Full Log').onclick();
    const dialog = f.body.children.at(-1);
    assert.equal(dialog.open, true);
    const log = dialog.children.find(child => child.tag === 'pre');
    assert.equal(log.textContent, text);
    assert.equal(log.scrollTop, 0);
    assert.equal(dialog.children.find(child => child.tag === 'a').href,
        '/base/dinki/qwen21/log?id=' + state.id + '&kind=train&download=1');
    const select = dialog.children.find(child => child.tag === 'select');
    select.value = 'worker';
    await select.onchange();
    assert.equal(log.textContent, 'Worker error');
    dialog.children.find(child => child.textContent === 'Close').onclick();
    assert.equal(dialog.removed, true);
});

test('Logs and status scroll despite a cancelled canvas wheel and allow text selection', () => {
    const f = fixture(async () => ({}));
    for (const element of f.widget.root.children.filter(child => child.style.overflow === 'auto')) {
        let stopped = false;
        element.scrollTop = 0;
        element.clientHeight = 100;
        element.scrollHeight = 1000;
        element.listeners.wheel({ deltaY: 60, defaultPrevented: true, stopPropagation() { stopped = true; },
            preventDefault() {} });
        assert.equal(stopped, true);
        assert.equal(element.scrollTop, 60);
        assert.equal(element.style.userSelect, 'text');
    }
});

test('Samples support arrows, position slider and wheel; polling preserves the viewed position', async () => {
    let samples = ['one.png', 'two.png'];
    const f = fixture(async () => ({ ...state, samples }));
    f.node.properties.dkst_qwen21_job_id = state.id;
    await f.intervals.get(1)();
    const panel = f.widget.root.children[3];
    const [viewport, controls] = panel.children;
    const [left, slider, right] = controls.children;
    assert.equal(panel.hidden, false);
    viewport.clientWidth = 200;
    viewport.scrollWidth = 800;
    viewport.scrollLeft = 0;
    viewport.listeners.scroll();
    right.listeners.click();
    assert.equal(viewport.scrollLeft, 160);
    viewport.listeners.wheel({ deltaY: 40, stopPropagation() {}, preventDefault() {} });
    assert.equal(viewport.scrollLeft, 200);
    const originalImage = viewport.children[0];
    samples = [...samples, 'three.png'];
    await f.intervals.get(1)();
    assert.equal(viewport.scrollLeft, 200);
    assert.equal(viewport.children[0], originalImage);
    assert.equal(originalImage.style.flex, '0 0 auto');
    slider.value = '1000';
    slider.listeners.input();
    assert.equal(viewport.scrollLeft, 600);
    assert.equal(right.disabled, true);
    left.listeners.click();
    assert.equal(viewport.scrollLeft, 440);
    f.node.properties.dkst_qwen21_job_id = 'b'.repeat(32);
    samples = [];
    await f.intervals.get(1)();
    assert.equal(panel.hidden, true);
});

test('Only Start Training submits a job; its ID survives workflow serialization', async () => {
    const f = fixture(async (path, body) => body?.action ? { ...state } : { jobs: [], active: null });
    assert.equal(f.requests.length, 0);
    assert.equal(f.widget.serialize, false);
    await f.buttons.find(button => button.textContent === 'Start Training').onclick();
    assert.equal(f.requests.length, 1);
    assert.equal(f.requests[0].body.action, 'start');
    assert.equal(JSON.parse(f.requests[0].body.image_gallery)[0].caption, 'side profile');
    assert.equal(f.node.properties.dkst_qwen21_job_id, state.id);
    assert.equal(f.buttons.find(button => button.textContent === 'Start Training').disabled, true);
    assert.equal(f.buttons.find(button => button.textContent === 'Pause').disabled, false);
    const serialized = JSON.parse(JSON.stringify(f.node.properties));
    assert.equal(serialized.dkst_qwen21_job_id, state.id);
    assert.equal(Object.keys(serialized).length, 1);
});

test('Pause and resume use the existing job ID, and polling stops when the node is removed', async () => {
    const f = fixture(async (path, body) => ({ ...state, status: body?.action === 'pause' ? 'paused' : 'training' }));
    f.node.properties.dkst_qwen21_job_id = state.id;
    await f.buttons.find(button => button.textContent === 'Pause').onclick();
    assert.equal(f.requests[0].body.action, 'pause');
    assert.equal(f.requests[0].body.id, state.id);
    await f.buttons.find(button => button.textContent === 'Resume').onclick();
    assert.equal(f.requests[1].body.action, 'resume');
    assert.equal(f.requests[1].body.id, state.id);
    f.node.onRemoved();
    assert.equal(f.cleared, true);
    assert.equal(f.intervals.size, 0);
});

test('Generation is blocked for active training and allowed after it finishes', async () => {
    let active = state.id;
    const f = fixture(async () => ({ jobs: [], active }));
    await assert.rejects(f.app.queuePrompt(0, 1), /Pause or finish/);
    active = null;
    assert.deepEqual(await f.app.queuePrompt(0, 1), [0, 1]);
});

test('Backend errors stay visible and do not persist a fake job ID', async () => {
    const f = fixture(async () => ({ error: 'Configure the training Python first.' }));
    await f.buttons.find(button => button.textContent === 'Start Training').onclick();
    assert.equal(f.node.properties.dkst_qwen21_job_id, undefined);
    assert.match(f.widget.root.children[1].textContent, /Configure the training Python/);
    assert.equal(f.buttons.find(button => button.textContent === 'Start Training').disabled, false);
});

test('Model dropdowns select server paths, detect Python and preserve choices on refresh', async () => {
    const choices = Object.fromEntries(['python', 'dit', 'vae', 'text_encoder', 'training_adapter', 'turbo_lora']
        .map(key => [key, [{ label: `${key} model`, path: `/models/${key}` }]]));
    const f = fixture(async (path, data) => path.endsWith('/models') ? { choices } : data ? { ok: true } : {});
    await f.buttons.find(button => button.textContent === 'Configure Models').onclick();
    const dialog = f.body.firstChild;
    assert.equal(dialog.open, true);
    const rows = dialog.children.filter(child => child.tag === 'label');
    assert.equal(rows.length, 6);
    assert.equal(rows[0].firstChild.value, '/models/python');
    for (const [index, key] of ['python', 'dit', 'vae', 'text_encoder', 'training_adapter', 'turbo_lora'].entries()) {
        const [select, input] = rows[index].children;
        assert.equal(select.tag, 'select');
        select.value = `/models/${key}`;
        select.onchange();
        assert.equal(input.hidden, true);
    }
    await dialog.children.find(child => child.textContent === 'Refresh Models').onclick();
    await dialog.children.find(child => child.textContent === 'Save').onclick();
    const saved = f.requests.at(-1).body;
    assert.equal(saved.dit, '/models/dit');
    assert.equal(saved.python, '/models/python');
    assert.equal(f.node.properties.dit, undefined);
});

test('Saved external models and manual paths remain usable when discovery fails', async () => {
    const f = fixture(async (path, data) => path.endsWith('/models') ? { error: 'Scan unavailable' } : data ? { ok: true } : { dit: '/external/saved.safetensors' });
    await f.buttons.find(button => button.textContent === 'Configure Models').onclick();
    const dialog = f.body.firstChild;
    const rows = dialog.children.filter(child => child.tag === 'label');
    const [select, input] = rows[1].children;
    assert.equal(select.value, '/external/saved.safetensors');
    assert.ok(select.children.some(option => option.value === select.value));
    select.value = '__manual__';
    select.onchange();
    assert.equal(input.hidden, false);
    input.value = '/external/new.safetensors';
    await dialog.children.find(child => child.textContent === 'Save').onclick();
    assert.equal(f.requests.at(-1).body.dit, '/external/new.safetensors');
});

function captionOptions(f) {
    f.node.widgets.find(w => w.name === 'caption_connection_alias').value = 'Home';
    f.node.widgets.find(w => w.name === 'caption_model').value = 'my-vision-model';
}

test('Optional captions need an explicit click and missing LLM configuration never starts requests', async () => {
    const f = fixture(async () => ({}));
    await f.buttons.find(button => button.textContent === 'Auto Caption (Empty)').onclick();
    assert.equal(f.requests.length, 0);
    assert.match(f.widget.root.children.at(-1).textContent, /optional LLM alias/);
    f.node.widgets.find(w => w.name === 'caption_connection_alias').value = 'Home';
    await f.buttons.find(button => button.textContent === 'Auto Caption (Empty)').onclick();
    assert.equal(f.requests.length, 0);
    assert.match(f.widget.root.children.at(-1).textContent, /exact name/);
});

test('Caption alias refresh retains missing saved aliases without switching the selected server', async () => {
    const f = fixture(async path => ({ aliases: ['Home', 'Office'] }));
    const widget = f.node.widgets.find(w => w.name === 'caption_connection_alias');
    widget.value = 'Removed';
    await f.buttons.find(button => button.textContent === 'Refresh LLM Aliases').onclick();
    assert.equal(f.requests[0].path, '/dinki/llm/aliases');
    assert.equal(widget.value, 'Removed');
    assert.deepEqual([...widget.options.values], ['-- Select connection --', 'Home', 'Office', 'Removed']);
});

test('Captioning sends one photo at a time with alias and manually entered model', async () => {
    const f = fixture(async () => ({ caption: 'a person waving' }));
    captionOptions(f);
    const applied = [];
    f.node.widgets.find(w => w.name === 'image_gallery').dkstCaptionTargets = () => [
        { image: { filename: 'one.png' }, apply: text => { applied.push(text); return true; } },
        { image: { filename: 'two.png' }, apply: text => { applied.push(text); return false; } },
    ];
    await f.buttons.find(button => button.textContent === 'Auto Caption (Empty)').onclick();
    assert.equal(f.requests.length, 2);
    assert.equal(f.requests[0].path, '/dinki/qwen21/caption');
    assert.equal(f.requests[0].body.caption_connection_alias, 'Home');
    assert.equal(f.requests[0].body.caption_model, 'my-vision-model');
    assert.equal(f.requests[1].body.image.filename, 'two.png');
    assert.equal(f.requests[0].body.api_key, undefined);
    assert.equal(applied.length, 2);
    assert.match(f.widget.root.children.at(-1).textContent, /1 captions added, 1 edited\/removed/);
    assert.equal(f.buttons.find(button => button.textContent === 'Start Training').disabled, false);
    assert.equal(f.node.properties.dkst_qwen21_job_id, undefined);
});

test('Stopping captions finishes the current photo and blocks training until it returns', async () => {
    let resolve;
    const response = new Promise(done => { resolve = done; });
    const f = fixture(async () => response);
    captionOptions(f);
    const applied = [];
    f.node.widgets.find(w => w.name === 'image_gallery').dkstCaptionTargets = () => [1, 2].map(n => ({
        image: { filename: `${n}.png` }, apply: text => { applied.push(text); return true; }
    }));
    const running = f.buttons.find(button => button.textContent === 'Auto Caption (Empty)').onclick();
    assert.equal(f.buttons.find(button => button.textContent === 'Start Training').disabled, true);
    await f.buttons.find(button => button.textContent === 'Start Training').onclick();
    await f.buttons.find(button => button.textContent === 'Stop Captioning').onclick();
    resolve({ caption: 'a person waving' });
    await running;
    assert.equal(f.requests.length, 1);
    assert.equal(applied.length, 1);
    assert.match(f.widget.root.children.at(-1).textContent, /^Stopped/);
    assert.equal(f.buttons.find(button => button.textContent === 'Start Training').disabled, false);
});

test('Caption failures retain completed captions; removed nodes discard late responses', async () => {
    let count = 0;
    const f = fixture(async () => ++count === 1 ? { caption: 'first caption' } : { error: 'Vision model unavailable' });
    captionOptions(f);
    const applied = [];
    f.node.widgets.find(w => w.name === 'image_gallery').dkstCaptionTargets = () => [1, 2].map(n => ({
        image: { filename: `${n}.png` }, apply: text => { applied.push(text); return true; }
    }));
    await f.buttons.find(button => button.textContent === 'Auto Caption (Empty)').onclick();
    assert.deepEqual(applied, ['first caption']);
    assert.match(f.widget.root.children.at(-1).textContent, /1 captions added.*Vision model unavailable/);

    let finish;
    const late = new Promise(done => { finish = done; });
    const g = fixture(async () => late);
    captionOptions(g);
    let changed = false;
    g.node.widgets.find(w => w.name === 'image_gallery').dkstCaptionTargets = () => [
        { image: { filename: 'one.png' }, apply: () => { changed = true; return true; } }
    ];
    const running = g.buttons.find(button => button.textContent === 'Auto Caption (Empty)').onclick();
    g.node.onRemoved();
    finish({ caption: 'late text' });
    await running;
    assert.equal(changed, false);
});

test('H3 UI uses separate endpoints and job IDs with the common optional caption controls', async () => {
    const f = fixture(async () => ({ ...state }), 'h3');
    assert.equal(f.setting.id, 'DKST.Training.MinimaxH3.Models');
    assert.equal(f.requests.length, 0);
    await f.buttons.find(b => b.textContent === 'Start Training').onclick();
    assert.equal(f.requests[0].path, '/dinki/minimax_h3/training');
    assert.equal(f.node.properties.dkst_minimax_h3_job_id, state.id);
    assert.equal(f.node.properties.dkst_qwen21_job_id, undefined);
    assert.equal(f.widget.serialize, false);
    captionOptions(f);
    await f.buttons.find(b => b.textContent === 'Auto Caption (Empty)').onclick();
    assert.equal(f.requests.length, 1, 'caption requests stay blocked during training');
});

test('H3 configuration has a reference DiT and optional adapters, and generation checks both trainers', async () => {
    const f = fixture(async (path, data) => path === '/dinki/training' ? {active:{family:'qwen21',id:state.id}} :
        path.endsWith('/models') ? {choices:{}} : {}, 'h3');
    await assert.rejects(f.app.queuePrompt(0,1), /Pause or finish LoRA training/);
    assert.equal(f.requests[0].path, '/dinki/training');
    await f.buttons.find(b => b.textContent === 'Configure Models').onclick();
    const rows = f.body.firstChild.children.filter(c => c.tag === 'label');
    assert.equal(rows.length, 7);
    assert.match(rows[2].textContent, /ref2va/);
    for (const index of [2,5,6]) assert.equal(rows[index].firstChild.firstChild.textContent, 'None (optional)');
});
