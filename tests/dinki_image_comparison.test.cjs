const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_image_comparison.js'), 'utf8');

function element(tagName = 'div') {
    return {
        tagName: tagName.toUpperCase(), style: {}, children: [], listeners: {},
        append(...children) { this.children.push(...children); },
        appendChild(child) { this.children.push(child); },
        setAttribute(name, value) { this[name] = value; },
        removeAttribute(name) { delete this[name]; },
        addEventListener(name, callback) { this.listeners[name] = callback; },
        remove() { this.removed = true; },
        contains() { return false; },
        click() { this.clicked = true; },
        setPointerCapture(id) { this.capture = id; },
        hasPointerCapture(id) { return this.capture === id; },
        releasePointerCapture() { this.capture = null; },
        getBoundingClientRect() { return { left: 20, width: 200 }; },
    };
}

function fixture() {
    let extension;
    let opened, fetched;
    const elements = [];
    const observers = [];
    const createElement = tag => {
        const value = element(tag);
        elements.push(value);
        return value;
    };
    const body = createElement('body');
    const app = { nodeOutputs: {}, registerExtension(value) { extension = value; } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: { apiURL: path => path },
        document: { createElement, body, addEventListener() {}, removeEventListener() {} },
        window: { innerWidth: 1000, innerHeight: 800, open: url => { opened = url; } },
        fetch: async url => { fetched = url; return { ok: true, blob: async() => ({}) }; },
        URL: { createObjectURL: () => 'blob:comparison', revokeObjectURL() {} },
        URLSearchParams, queueMicrotask, setTimeout() {},
        ResizeObserver: class {
            constructor(callback) { this.callback = callback; observers.push(this); }
            observe() {}
            disconnect() { this.disconnected = true; }
        },
    });
    class CompareNode {
        constructor(properties = {}) {
            this.id = 7;
            this.comfyClass = 'DINKI_Image_Comparison';
            this.properties = properties;
            this.widgets = [{ name: 'mode', value: 'Slide' }];
            this.size = [420, 340];
            this.onNodeCreated();
        }
        addDOMWidget(name, type, root) {
            this.root = root;
            return { options: {} };
        }
        setDirtyCanvas() {}
    }
    extension.beforeRegisterNodeDef(CompareNode, { name: 'DINKI_Image_Comparison' });
    return { app, extension, CompareNode, elements, observers,
        get opened() { return opened; }, get fetched() { return fetched; } };
}

const output = { dkst_comparison: [
    { filename: 'first.png', type: 'temp' },
    { filename: 'second.png', type: 'temp' },
    { filename: 'difference.png', type: 'temp' },
] };

function parts(node) {
    const [toolbar, viewport, minimap] = node.root.children;
    const panArea = viewport.children[0];
    const canvas = panArea.children[0];
    const [first, second, difference, divider] = canvas.children;
    const mapCanvas = minimap.children[0];
    const [mapFirst, mapSecond, mapDifference, mapWindow] = mapCanvas.children;
    return { toolbar, viewport, panArea, canvas, first, second, difference, divider,
        minimap, mapCanvas, mapFirst, mapSecond, mapDifference, mapWindow,
        button: label => toolbar.children.find(item => item.textContent === label) };
}

test('Slide follows pointer position and Difference displays the computed preview', () => {
    const { CompareNode } = fixture();
    const node = new CompareNode();
    node.onExecuted(output);
    const { first, second, difference, divider, viewport } = parts(node);
    assert.match(first.src, /first\.png/);
    assert.equal(first.style.objectFit, 'contain');
    assert.equal(first.style.objectPosition, 'center');
    assert.equal(second.style.objectFit, 'contain');
    assert.equal(difference.style.objectFit, 'contain');
    assert.equal(second.style.display, 'block');
    viewport.listeners.pointermove({ clientX: 70 });
    assert.equal(second.style.clipPath, 'inset(0 75% 0 0)');
    assert.equal(divider.style.left, '25%');
    node.onWidgetChanged('mode', 'Difference');
    assert.equal(difference.style.display, 'block');
    assert.equal(first.style.display, 'none');
    assert.equal(divider.style.display, 'none');
    node.widgets[0].callback('Slide');
    assert.equal(second.style.display, 'block');
});

test('comparison restores its own images after a tab switch', async () => {
    const { app, extension, CompareNode } = fixture();
    const first = new CompareNode();
    first.onExecuted(output);
    app.nodeOutputs[7] = { dkst_comparison: [
        { filename: 'other-1.png' }, { filename: 'other-2.png' }, { filename: 'other-diff.png' },
    ] };
    const restored = new CompareNode(structuredClone(first.properties));
    extension.loadedGraphNode(restored);
    assert.match(parts(restored).first.src, /first\.png/);
    restored.widgets[0].value = 'Difference';
    restored.onConfigure();
    await Promise.resolve();
    assert.equal(parts(restored).difference.style.display, 'block');
});

test('right-click menu changes mode and opens or saves the selected comparison images', async () => {
    const context = fixture();
    const node = new context.CompareNode();
    node.onExecuted(output);
    const event = { button: 2, clientX: 40, clientY: 40,
        preventDefault() { this.prevented = true; },
        stopPropagation() { this.stopped = true; },
        stopImmediatePropagation() { this.stopped = true; } };
    node.root.listeners.contextmenu(event);
    assert.equal(event.prevented, true);
    assert.equal(event.stopped, true);
    const buttons = context.elements.filter(item => item.tagName === 'BUTTON' && /^(Mode:|Open Image|Save Image)/.test(item.textContent));
    assert.deepEqual(buttons.map(item => item.textContent), [
        'Mode: Slide', 'Mode: Difference', 'Open Image 1', 'Save Image 1',
        'Open Image 2', 'Save Image 2',
    ]);
    await buttons[1].onclick();
    assert.equal(node.widgets[0].value, 'Difference');
    assert.equal(parts(node).difference.style.display, 'block');
    await buttons[2].onclick();
    assert.match(context.opened, /filename=first\.png/);
    await buttons[5].onclick();
    assert.match(context.fetched, /filename=second\.png/);
    assert.match(context.fetched, /type=temp/);
    const link = context.elements.find(item => item.tagName === 'A');
    assert.equal(link.download, 'second.png');
    assert.equal(link.clicked, true);
    const options = [];
    node.getExtraMenuOptions(null, options);
    assert.deepEqual(options.map(item => item.content), buttons.map(item => item.textContent));
});

test('zoom scales the same aligned images and Difference without changing inputs or node size', () => {
    const { CompareNode } = fixture();
    const node = new CompareNode();
    node.onExecuted({ ...output, resolution: ['1600 × 900'] });
    const p = parts(node);
    const sources = [p.first.src, p.second.src, p.difference.src];
    assert.deepEqual(p.toolbar.children.map(button => button.textContent), ['25%', '50%', '75%', '100%', '150%', '200%', '400%', 'Fit']);
    assert.equal(p.canvas.style.width, '100%');
    assert.equal(p.first.style.objectFit, 'contain');
    for (const [label, width, height] of [['25%', 400, 225], ['50%', 800, 450], ['75%', 1200, 675], ['100%', 1600, 900],
        ['150%', 2400, 1350], ['200%', 3200, 1800], ['400%', 6400, 3600]]) {
        p.button(label).onclick();
        assert.equal(p.canvas.style.width, `${width}px`);
        assert.equal(p.canvas.style.height, `${height}px`);
        assert.equal(p.viewport.style.overflow, 'auto');
        assert.equal(p.button(label)['aria-pressed'], 'true');
        assert.deepEqual([p.first.src, p.second.src, p.difference.src], sources);
        for (const image of [p.first, p.second, p.difference]) {
            assert.equal(image.style.width, '100%');
            assert.equal(image.style.height, '100%');
            assert.equal(image.style.objectPosition, 'center');
        }
        node.onWidgetChanged('mode', 'Difference');
        assert.equal(p.difference.style.display, 'block');
        assert.equal(p.canvas.style.width, `${width}px`);
        assert.deepEqual(node.size, [420, 340]);
    }
    p.button('Fit').onclick();
    assert.equal(p.canvas.style.width, '100%');
    assert.equal(p.canvas.style.height, '100%');
    assert.equal(p.viewport.style.overflow, 'hidden');
});

function layout(p, width = 400, height = 300, graphScale = 1) {
    Object.assign(p.viewport, { clientWidth: width, clientHeight: height,
        offsetWidth: width, offsetHeight: height, scrollLeft: 0, scrollTop: 0 });
    Object.defineProperties(p.canvas, {
        clientWidth: { get: () => p.canvas.style.width === '100%' ? p.viewport.clientWidth : parseFloat(p.canvas.style.width) },
        clientHeight: { get: () => p.canvas.style.height === '100%' ? p.viewport.clientHeight : parseFloat(p.canvas.style.height) },
    });
    p.viewport.getBoundingClientRect = () => ({ left: 100, top: 100,
        width: p.viewport.offsetWidth * graphScale, height: p.viewport.offsetHeight * graphScale });
    p.canvas.getBoundingClientRect = () => ({
        left: 100 + (Math.max(0, (p.viewport.clientWidth - p.canvas.clientWidth) / 2) - p.viewport.scrollLeft) * graphScale,
        top: 100 + (Math.max(0, (p.viewport.clientHeight - p.canvas.clientHeight) / 2) - p.viewport.scrollTop) * graphScale,
        width: p.canvas.clientWidth * graphScale, height: p.canvas.clientHeight * graphScale,
    });
    p.mapCanvas.getBoundingClientRect = () => ({ left: 500, top: 300,
        width: parseFloat(p.mapCanvas.style.width) * graphScale, height: parseFloat(p.mapCanvas.style.height) * graphScale });
}

function mapEvent(p, x, y, pointerId = 1) {
    const bounds = p.mapCanvas.getBoundingClientRect();
    return { button: 0, pointerId, clientX: bounds.left + x * bounds.width,
        clientY: bounds.top + y * bounds.height,
        preventDefault() { this.prevented = true; }, stopPropagation() { this.stopped = true; } };
}

test('minimap follows the visible region, mode, scroll and node resizing', () => {
    const context = fixture();
    const node = new context.CompareNode();
    const p = parts(node);
    layout(p);
    node.onExecuted({ ...output, resolution: ['1600 × 900'] });
    assert.equal(p.minimap.style.display, 'none');
    p.button('400%').onclick();
    assert.equal(p.minimap.style.display, 'block');
    assert.equal(p.mapFirst.src, p.first.src);
    assert.equal(p.mapSecond.src, p.second.src);
    assert.equal(p.mapDifference.src, p.difference.src);
    assert.equal(p.mapWindow.style.width, '6.25%');
    assert.ok(Math.abs(parseFloat(p.mapWindow.style.height) - 100 * 300 / 3600) < 1e-9);
    p.viewport.scrollLeft = 1200;
    p.viewport.scrollTop = 900;
    p.viewport.listeners.scroll();
    assert.equal(p.mapWindow.style.left, '18.75%');
    assert.equal(p.mapWindow.style.top, '25%');
    p.viewport.listeners.pointermove({ clientX: 200 });
    assert.equal(p.mapSecond.style.clipPath, p.second.style.clipPath);
    node.onWidgetChanged('mode', 'Difference');
    assert.equal(p.mapDifference.style.display, 'block');
    assert.equal(p.mapFirst.style.display, 'none');
    p.button('25%').onclick();
    Object.assign(p.viewport, { clientWidth: 800, clientHeight: 600, offsetWidth: 800, offsetHeight: 600 });
    context.observers[0].callback();
    assert.equal(p.minimap.style.display, 'none');
    Object.assign(p.viewport, { clientWidth: 200, clientHeight: 150, offsetWidth: 200, offsetHeight: 150 });
    context.observers[0].callback();
    assert.equal(p.minimap.style.display, 'block');
    p.button('Fit').onclick();
    assert.equal(p.minimap.style.display, 'none');
    node.onRemoved();
    assert.equal(context.observers[0].disconnected, true);
    assert.equal(p.mapFirst.src, undefined);
});

test('minimap click and captured drag pan the aligned canvas at graph zoom without moving the divider', () => {
    const { CompareNode } = fixture();
    const node = new CompareNode();
    const p = parts(node);
    layout(p, 400, 300, .75);
    node.onExecuted({ ...output, resolution: ['1600 × 900'] });
    p.button('200%').onclick();
    const dividerPosition = p.divider.style.left;
    const click = mapEvent(p, .8, .8);
    p.mapCanvas.listeners.pointerdown(click);
    assert.equal(click.prevented, true);
    assert.equal(click.stopped, true);
    assert.ok(Math.abs(p.viewport.scrollLeft - 2360) < 1e-9);
    assert.ok(Math.abs(p.viewport.scrollTop - 1290) < 1e-9);
    assert.equal(p.mapCanvas.capture, 1);
    p.mapCanvas.listeners.pointerup(click);
    assert.equal(p.mapCanvas.capture, null);
    // Grab off-center within the rectangle: no jump when the drag starts.
    const grab = mapEvent(p, .82, .81);
    p.mapCanvas.listeners.pointerdown(grab);
    assert.ok(Math.abs(p.viewport.scrollLeft - 2360) < 1e-9);
    assert.ok(Math.abs(p.viewport.scrollTop - 1290) < 1e-9);
    p.mapCanvas.listeners.pointermove(mapEvent(p, .62, .61));
    assert.ok(Math.abs(p.viewport.scrollLeft - 1720) < 1e-9);
    assert.ok(Math.abs(p.viewport.scrollTop - 930) < 1e-9);
    p.mapCanvas.listeners.pointermove(mapEvent(p, 2, 2));
    assert.equal(p.viewport.scrollLeft, 2800);
    assert.equal(p.viewport.scrollTop, 1500);
    assert.equal(p.divider.style.left, dividerPosition);
    p.mapCanvas.listeners.pointercancel(grab);
    p.mapCanvas.listeners.pointermove(mapEvent(p, 0, 0));
    assert.equal(p.viewport.scrollLeft, 2800);
});

test('minimap handles one-axis overflow, keyboard navigation and zoom around the inspected area', () => {
    const { CompareNode, extension } = fixture();
    const node = new CompareNode();
    const p = parts(node);
    layout(p);
    node.onExecuted({ ...output, resolution: ['1600 × 400'] });
    p.button('50%').onclick();
    assert.equal(p.minimap.style.display, 'block');
    assert.equal(p.mapWindow.style.height, '100%');
    const click = mapEvent(p, .9, .5);
    p.mapCanvas.listeners.pointerdown(click);
    p.mapCanvas.listeners.pointerup(click);
    assert.equal(p.viewport.scrollLeft, 400);
    assert.equal(p.viewport.scrollTop, 0);
    const key = { key: 'ArrowLeft', preventDefault() {}, stopPropagation() {} };
    p.mapCanvas.listeners.keydown(key);
    assert.equal(p.viewport.scrollLeft, 300);
    p.button('200%').onclick();
    assert.equal(p.viewport.scrollLeft, 1800); // center stays at 62.5% of the aligned image
    assert.equal(p.viewport.scrollTop, 250);
    const restored = new CompareNode(structuredClone(node.properties));
    extension.loadedGraphNode(restored);
    assert.equal(parts(restored).canvas.style.width, '3200px');
    assert.equal(parts(restored).button('200%')['aria-pressed'], 'true');
});

test('slider uses zoomed canvas screen coordinates after scrolling and graph zoom', () => {
    const { CompareNode } = fixture();
    const node = new CompareNode();
    node.onExecuted({ ...output, resolution: ['1600 × 900'] });
    const p = parts(node);
    p.button('50%').onclick();
    // 800 CSS pixels shown at 75% graph zoom, with the canvas scrolled left.
    p.canvas.getBoundingClientRect = () => ({ left: -100, width: 600 });
    p.viewport.listeners.pointermove({ clientX: 50 });
    assert.equal(p.second.style.clipPath, 'inset(0 75% 0 0)');
    assert.equal(p.divider.style.left, '25%');
    p.button('100%').onclick();
    assert.equal(p.divider.style.left, '25%');
    p.canvas.getBoundingClientRect = () => ({ left: -200, width: 1200 });
    p.viewport.listeners.pointermove({ clientX: 400 });
    assert.equal(p.divider.style.left, '50%');
    p.viewport.listeners.pointermove({ clientX: -400 });
    assert.equal(p.divider.style.left, '0%');
});

test('zoom and divider restore across reload and persist when input resolution changes', async () => {
    const { CompareNode, extension } = fixture();
    const node = new CompareNode();
    node.onExecuted({ ...output, resolution: ['1600 × 900'] });
    parts(node).button('75%').onclick();
    parts(node).viewport.listeners.pointermove({ clientX: 70 });
    const restored = new CompareNode(structuredClone(node.properties));
    extension.loadedGraphNode(restored);
    assert.equal(parts(restored).canvas.style.width, '1200px');
    assert.equal(parts(restored).divider.style.left, '25%');
    restored.onConfigure();
    await Promise.resolve();
    assert.equal(parts(restored).canvas.style.height, '675px');
    restored.onExecuted({ ...output, resolution: ['800 × 1200'] });
    assert.equal(parts(restored).canvas.style.width, '600px');
    assert.equal(parts(restored).canvas.style.height, '900px');
    assert.equal(restored.properties.dkstComparison.zoom, .75);
    assert.equal(parts(restored).divider.style.left, '25%');
});

test('legacy previews use decoded dimensions for zoom and controls keep canvas gestures separate', () => {
    const { CompareNode } = fixture();
    const node = new CompareNode();
    node.onExecuted(output);
    const p = parts(node);
    assert.equal(p.button('100%').disabled, true);
    p.first.naturalWidth = 1200;
    p.first.naturalHeight = 800;
    p.first.onload();
    p.button('25%').onclick();
    assert.equal(p.canvas.style.width, '300px');
    let stopped = 0;
    p.toolbar.listeners.pointerdown({ stopPropagation() { stopped++; } });
    p.viewport.listeners.wheel({ stopPropagation() { stopped++; } });
    assert.equal(stopped, 2);
    p.button('Fit').onclick();
    p.viewport.listeners.wheel({ stopPropagation() { stopped++; } });
    assert.equal(stopped, 2);
    node.onRemoved();
    assert.equal(p.first.src, undefined);
    assert.equal(p.first.onload, null);
});
