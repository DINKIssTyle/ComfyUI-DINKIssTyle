const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.listeners = {}; this.scrollTop = 0; }
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    setAttribute(name, value) { this[name] = value; }
    click() { this.clickCount = (this.clickCount || 0) + 1; }
}
function fixture(handler, savedId = '') {
    let extension, tick, cleared = false, queued = 0;
    const requests = [];
    const context = {
        document: { createElement: tag => new Element(tag) },
        app: { registerExtension: e => extension = e, queuePrompt: () => queued++ },
        api: { apiURL: path => '/base' + path, async fetchApi(path, options) {
            const data = options ? JSON.parse(options.body) : undefined;
            requests.push({ path, data });
            const result = await handler(path, data);
            return { ok: !result.error, json: async () => result };
        } },
        setInterval: fn => { tick = fn; return 1; }, clearInterval: () => cleared = true,
        console,
    };
    const scroll = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_scroll.js'), 'utf8').replace(/^export /gm, '');
    const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_face_crop.js'), 'utf8').replace(/^import .*;\r?\n/gm, '');
    vm.runInNewContext(scroll + '\n' + source, context);
    const defaults = { image_gallery: '[{"filename":"photo.png","subfolder":"ref_images","type":"input"}]',
        output_folder: 'DKST_FaceCrops/person', filename_prefix: 'face', face_selection: 'All faces', crop_shape: 'BBox ratio',
        padding_percent: 30, limit_size: true, max_size: 1024, grid_split: '4x4', confidence: .8, result_job_id: savedId };
    const node = { comfyClass: 'DINKI_Face_Crop_Export', size: [400, 600], widgets: Object.entries(defaults).map(([name, value]) => ({ name, value })),
        setSize(size) { this.size = size; }, graph: { incrementVersion() {} }, setDirtyCanvas() {},
        addDOMWidget(name, type, root, options) { const widget = { name, type, root, options }; this.widgets.push(widget); return widget; } };
    extension.nodeCreated(node);
    const root = node.widgets.at(-1).root;
    return { node, root, buttons: root.children[0].children, status: root.children[1], viewport: root.children[2], requests,
        tick: () => tick(), get cleared() { return cleared; }, get queued() { return queued; } };
}
const id = 'a'.repeat(32);
const job = (status = 'completed', results = []) => ({ id, status, results, skipped: [], processed: 16, total: 16, output_dir: '/output/faces', error: '' });
const flush = () => new Promise(resolve => setImmediate(resolve));

test('Execute exports options, persists result id, never queues generation, and removes polling', async () => {
    const f = fixture(() => job());
    assert.equal(f.buttons[1].disabled, true);
    await f.buttons[0].listeners.click();
    assert.equal(f.requests[0].data.action, 'start');
    assert.equal(f.requests[0].data.grid_split, '4x4');
    assert.equal(f.node.widgets.find(w => w.name === 'result_job_id').value, id);
    assert.equal(f.node.widgets.at(-1).serialize, false);
    assert.equal(f.queued, 0);
    f.node.onRemoved(); assert.equal(f.cleared, true);
});

test('Restored completed job renders base-aware images and manifest without rerunning export', async () => {
    const crop = { filename: 'face_00001_01.png', width: 64, height: 100 };
    const f = fixture(() => job('completed', [crop]), id);
    await flush();
    assert.equal(f.requests.length, 1);
    assert.equal(f.requests[0].data, undefined);
    const image = f.viewport.children[0].children[0].children[0];
    assert.match(image.src, /^\/base\/dinki\/face_crop\/crop/);
    assert.equal(image.alt, crop.filename);
    assert.match(f.buttons[3].href, /manifest\?id=/);
    await f.tick(); assert.equal(f.requests.length, 1);
    f.node.onRemoved();
});

test('Polling appends crops without resetting scroll and cancel waits for worker completion', async () => {
    let status = 'processing', count = 1;
    const crop = n => ({ filename: `face_${n}.png`, width: 90, height: 120 });
    const f = fixture((path, data) => {
        if (data?.action === 'cancel') status = 'cancel_requested';
        return job(status, Array.from({ length: count }, (_, n) => crop(n)));
    }, id);
    await flush();
    f.viewport.scrollTop = 100;
    count = 2; await f.tick();
    assert.equal(f.viewport.children[0].children.length, 2);
    assert.equal(f.viewport.scrollTop, 100);
    assert.equal(f.buttons[0].disabled, true);
    await f.buttons[1].listeners.click();
    assert.equal(f.buttons[1].disabled, true);
    assert.match(f.status.textContent, /cancel_requested/);
    status = 'cancelled'; await f.tick();
    assert.equal(f.buttons[0].disabled, false);
    f.node.onRemoved();
});

test('Server rejection preserves prior export and displays actionable error', async () => {
    const f = fixture((path, data) => data ? { error: 'OpenCV is unavailable' } : job(), id);
    await flush(); await f.buttons[0].listeners.click();
    assert.equal(f.node.widgets.find(w => w.name === 'result_job_id').value, id);
    assert.equal(f.status.textContent, 'OpenCV is unavailable');
    assert.equal(f.buttons[0].disabled, false);
    f.node.onRemoved();
});

test('Old response cannot replace a different restored job', async () => {
    let release;
    const f = fixture(() => new Promise(resolve => release = resolve), id);
    f.node.widgets.find(w => w.name === 'result_job_id').value = 'b'.repeat(32);
    release(job('completed', [{ filename: 'stale.png' }]));
    await flush();
    assert.equal(f.viewport.children[0].children.length, 0);
    f.node.onRemoved();
});

test('Result wheel, arrows and slider work despite a canceled canvas event', async () => {
    const f = fixture(() => job());
    f.viewport.clientHeight = 260; f.viewport.scrollHeight = 1260;
    let stopped = false, prevented = false;
    f.viewport.listeners.wheel({ deltaY: 120, deltaMode: 0, defaultPrevented: true,
        preventDefault() { prevented = true; }, stopPropagation() { stopped = true; } });
    assert.equal(f.viewport.scrollTop, 120); assert.equal(stopped && prevented, true);
    const controls = f.root.children[3].children;
    controls[2].listeners.click(); assert.ok(f.viewport.scrollTop > 120);
    controls[1].value = '1000'; controls[1].listeners.input(); assert.equal(f.viewport.scrollTop, 1000);
    f.node.onRemoved();
});

test('ZIP download is enabled only for a completed export containing crops', async () => {
    for (const [status, count, enabled] of [['processing', 1, false], ['cancelled', 1, false], ['completed', 0, false], ['completed', 1, true]]) {
        const f = fixture(() => job(status, Array.from({ length: count }, () => ({ filename: 'face.png' }))), id);
        await flush(); assert.equal(f.buttons[4].disabled, !enabled); f.node.onRemoved();
    }
});

test('ZIP preparation prevents duplicate clicks and starts a native base-aware download', async () => {
    let release;
    const f = fixture((path, data) => path.endsWith('/zip') ? new Promise(resolve => release = resolve) : job('completed', [{ filename: 'face.png' }]), id);
    await flush();
    const pending = f.buttons[4].listeners.click();
    assert.equal(f.buttons[4].disabled, true);
    assert.equal(f.buttons[0].disabled, true);
    assert.match(f.buttons[4].textContent, /Preparing ZIP/);
    assert.match(f.status.textContent, /Preparing ZIP/);
    await f.buttons[4].listeners.click();
    assert.equal(f.requests.filter(r => r.path.endsWith('/zip')).length, 1);
    release({ id, filename: '인물_20261012.zip', url: '/dinki/face_crop/zip?id=' + id + '&token=abc', size: 1048576 });
    await pending;
    const link = f.buttons[5];
    assert.equal(link.clickCount, 1);
    assert.equal(link.download, '인물_20261012.zip');
    assert.match(link.href, /^\/base\/dinki\/face_crop\/zip/);
    assert.equal(link.style.display, 'inline');
    assert.match(f.status.textContent, /1.0 MB/);
    assert.equal(f.buttons[4].disabled, false);
    assert.equal(f.queued, 0);
    f.node.onRemoved();
});

test('ZIP server errors preserve the selected export and allow retry', async () => {
    const f = fixture(path => path.endsWith('/zip') ? { error: 'A crop file is missing' } : job('completed', [{ filename: 'face.png' }]), id);
    await flush(); await f.buttons[4].listeners.click();
    assert.match(f.status.textContent, /completed/);
    assert.match(f.status.textContent, /A crop file is missing/);
    assert.equal(f.buttons[4].disabled, false);
    assert.equal(f.buttons[5].clickCount, undefined);
    assert.equal(f.node.widgets.find(w => w.name === 'result_job_id').value, id);
    f.node.onRemoved();
});

test('ZIP completion cannot download after removal or a workflow selects another export', async () => {
    for (const remove of [true, false]) {
        let release;
        const f = fixture(path => path.endsWith('/zip') ? new Promise(resolve => release = resolve) : job('completed', [{ filename: 'face.png' }]), id);
        await flush(); const pending = f.buttons[4].listeners.click();
        if (remove) f.node.onRemoved();
        else f.node.widgets.find(w => w.name === 'result_job_id').value = 'b'.repeat(32);
        release({ id, filename: 'old.zip', url: '/zip', size: 10 }); await pending;
        assert.equal(f.buttons[5].clickCount, undefined);
        assert.equal(f.buttons[5].style.display, 'none');
        f.node.onRemoved();
    }
});
