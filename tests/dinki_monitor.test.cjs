const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function load() {
    let extension;
    const settings = new Map();
    const host = { isConnected: true, append(el) { el.parentElement = this; } };
    const document = {
        hidden: false, head: host, body: { append(el) { el.parentElement = this; } },
        getElementById() { return null; },
        querySelector() { return { parentElement: host }; },
        createElement() { return {
            dataset: {}, style: {}, children: [], setAttribute() {}, addEventListener() {},
            getBoundingClientRect() { return {left:24,top:64,width:500,height:32}; },
            setPointerCapture() {}, hasPointerCapture() { return true; }, releasePointerCapture() {},
            append(...items) { this.children.push(...items); },
            replaceChildren(...items) { this.children = items; },
            remove() { this.parentElement = null; },
        }; },
        addEventListener() {}, removeEventListener() {},
    };
    const timers = new Map();
    let timerId = 0;
    const context = {
        document, window: { innerWidth:1280, innerHeight:800, addEventListener() {}, removeEventListener() {} }, AbortController,
        setTimeout(fn, ms) { timers.set(++timerId, { fn, ms }); return timerId; },
        clearTimeout(id) { timers.delete(id); },
        app: { registerExtension(ext) { extension = ext; }, extensionManager: { setting: {
            get(key) { return settings.get(key); }, async set(key, value) { settings.set(key, value); }
        } } },
        api: { async fetchApi() { return { ok: true, json: async () => ({cpu_percent: 20, gpus: []}) }; } },
    };
    const source = fs.readFileSync(path.join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_monitor.js'), 'utf8');
    vm.runInNewContext(source.replace(/^import .*;\n/gm, '').replace(/^export /gm, '') + '\nglobalThis.Controller = SystemMonitor;', context);
    return { context, extension, settings, host, document, timers };
}

test('formats zero, unknown sensors, memory units and selected physical GPU', () => {
    const { context } = load();
    const data = { cpu_percent: 0, ram: {used_bytes: 2 ** 30, total_bytes: 8 * 2 ** 30}, gpus: [
        { index: 1, name: 'RTX', utilization: 80, temperature: 61, memory_used_mib: 2048, memory_total_mib: 8192 }
    ] };
    const values = context.monitorValues(data, 1);
    assert.equal(values.cpu, '0%');
    assert.equal(values.ram, '1.0/8.0 GiB');
    assert.equal(values.temperature, '61°C');
    assert.equal(values.vram, '2.0/8.0 GiB');
    assert.equal(context.monitorValues(data, 0).gpu, '—');
    assert.equal(context.monitorValues(null, 0).cpu, '—');
});

test('mounts in toolbar, polls once and schedules the next sample', async () => {
    const { context, host, timers, settings } = load();
    settings.set('DKST.Monitor.Placement', 'Toolbar');
    const monitor = new context.Controller();
    await monitor.poll(0);
    assert.equal(monitor.element.parentElement, host);
    assert.equal(monitor.cells.cpu.textContent, '20%');
    assert.deepEqual([...timers.values()].map(x => x.ms), [2000]);
    monitor.destroy();
    assert.equal(timers.size, 0);
});

test('disabled or hidden monitor does not request data', () => {
    const { context, settings, document } = load();
    let calls = 0;
    context.api.fetchApi = async () => { calls++; };
    const monitor = new context.Controller();
    settings.set('DKST.Monitor.Enabled', false);
    monitor.restart();
    assert.equal(calls, 0);
    settings.set('DKST.Monitor.Enabled', true);
    document.hidden = true;
    monitor.restart();
    assert.equal(calls, 0);
    monitor.destroy();
});

test('request failure clears stale numbers and schedules retry', async () => {
    const { context, timers } = load();
    const monitor = new context.Controller();
    monitor.render({cpu_percent: 75});
    context.api.fetchApi = async () => { throw new Error('offline'); };
    await monitor.poll(0);
    assert.equal(monitor.cells.cpu.textContent, '—');
    assert.equal(monitor.element.dataset.offline, 'true');
    assert.equal(timers.size, 1);
    monitor.destroy();
});

test('late result cannot restart polling after monitor is disabled', async () => {
    const { context, settings, timers } = load();
    let finish;
    context.api.fetchApi = () => new Promise(resolve => { finish = resolve; });
    const monitor = new context.Controller();
    const pending = monitor.poll(0);
    settings.set('DKST.Monitor.Enabled', false);
    monitor.restart();
    finish({ok: true, json: async () => ({cpu_percent: 99})});
    await pending;
    assert.equal(monitor.cells.cpu.textContent, '—');
    assert.equal(timers.size, 0);
    monitor.destroy();
});

test('saved per-metric visibility and order apply with deterministic ties', () => {
    const { context, settings } = load();
    ['CPU','GPU','Temperature','RAM','VRAM'].forEach((key, index) => settings.set('DKST.Monitor.Order' + key, index + 1));
    settings.set('DKST.Monitor.ShowTemperature', false);
    settings.set('DKST.Monitor.OrderVRAM', 1);
    settings.set('DKST.Monitor.OrderCPU', 5);
    const monitor = new context.Controller();
    assert.equal(monitor.layout(), true);
    assert.deepEqual(monitor.element.children.slice(1,-1), [monitor.items.vram, monitor.items.gpu, monitor.items.ram, monitor.items.cpu]);
    settings.set('DKST.Monitor.ShowTemperature', true);
    settings.set('DKST.Monitor.OrderCPU', 1);
    monitor.layout();
    assert.deepEqual(monitor.element.children.slice(1,-1), [monitor.items.cpu, monitor.items.vram, monitor.items.gpu, monitor.items.temperature, monitor.items.ram]);
    monitor.destroy();
});

test('hiding every metric stops polling, re-enabling one restores it', async () => {
    const { context, settings, document, timers } = load();
    let calls = 0;
    context.api.fetchApi = async () => { calls++; return {ok:true, json:async()=>({cpu_percent: 10})}; };
    const monitor = new context.Controller();
    for (const key of ['CPU', 'GPU', 'Temperature', 'RAM', 'VRAM']) settings.set('DKST.Monitor.Show' + key, false);
    monitor.restart();
    assert.equal(calls, 0);
    assert.equal(monitor.element.parentElement, null);
    assert.equal(timers.size, 0);
    settings.set('DKST.Monitor.ShowCPU', true);
    monitor.restart();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(calls, 1);
    assert.equal(monitor.element.parentElement, document.body);
    assert.deepEqual(monitor.element.children.slice(1,-1), [monitor.items.cpu]);
    monitor.destroy();
});

test('RAM and VRAM percentage options are independent and handle missing totals', () => {
    const { context } = load();
    const data = {ram:{used_bytes:2, total_bytes:8},gpus:[{index:0,memory_used_mib:2048,memory_total_mib:8192}]};
    assert.equal(context.monitorValues(data,0,true,false).ram, '25%');
    assert.equal(context.monitorValues(data,0,true,false).vram, '2.0/8.0 GiB');
    assert.equal(context.monitorValues(data,0,false,true).vram, '25%');
    assert.equal(context.monitorValues({ram:{used_bytes:0,total_bytes:0}},0,true,true).ram, '—');
});

test('floating placement restores position, clamps to viewport, and docks back', () => {
    const {context, settings, document, host} = load();
    const monitor = new context.Controller();
    settings.set('DKST.Monitor.Placement', 'Floating');
    settings.set('DKST.Monitor.Position', '{"x":9000,"y":-20}');
    monitor.mount();
    assert.equal(monitor.element.parentElement, document.body);
    assert.equal(monitor.element.style.left, '772px');
    assert.equal(monitor.element.style.top, '8px');
    settings.set('DKST.Monitor.Placement', 'Toolbar');
    monitor.mount();
    assert.equal(monitor.element.parentElement, host);
    assert.equal(monitor.element.style.left, '');
    monitor.destroy();
});

test('drag detaches monitor and saves floating position', async () => {
    const {context, settings, document} = load();
    const monitor = new context.Controller();
    monitor.mount();
    monitor.startDrag({button:0,pointerId:1,clientX:30,clientY:70,preventDefault(){}});
    monitor.moveDrag({pointerId:1,clientX:130,clientY:170});
    assert.equal(monitor.element.parentElement, document.body);
    await monitor.endDrag({pointerId:1});
    assert.equal(settings.get('DKST.Monitor.Placement'), 'Floating');
    assert.deepEqual(JSON.parse(settings.get('DKST.Monitor.Position')), {x:124,y:164});
    monitor.destroy();
});

test('drag-list setting replaces numeric controls and overrides legacy ordering', () => {
    const {context, settings, extension} = load();
    assert.equal(typeof extension.settings.find(s => s.id === 'DKST.Monitor.Layout').type, 'function');
    assert.ok(extension.settings.filter(s => s.id.startsWith('DKST.Monitor.Order')).every(s => s.type === 'hidden'));
    settings.set('DKST.Monitor.Layout', [
        {id:'VRAM',visible:true}, {id:'CPU',visible:false}, {id:'GPU',visible:true},
        {id:'Temperature',visible:true}, {id:'RAM',visible:true}
    ]);
    const monitor = new context.Controller();
    monitor.layout();
    assert.deepEqual(monitor.element.children.slice(1,-1), [monitor.items.vram,monitor.items.gpu,monitor.items.temperature,monitor.items.ram]);
    monitor.destroy();
});

test('saved list normalizes unknown and duplicate metrics without losing items', () => {
    const {context} = load();
    const result = context.metricLayout([{id:'VRAM',visible:false},{id:'bad'},{id:'VRAM'}]);
    assert.equal(result.length,5);
    assert.equal(result[0].id,'VRAM');
    assert.equal(result[0].visible,false);
    assert.equal(new Set(result.map(x=>x.id)).size,5);
});
