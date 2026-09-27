const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_appearance.js'), 'utf8');

function fixture(initialSettings = {}) {
    let extension;
    const settingsStore = { ...initialSettings };
    const canvas = {
        dirty: false,
        setDirty(v1, v2) { this.dirty = true; },
    };
    const app = {
        canvas,
        registerExtension(value) { extension = value; },
        ui: {
            settings: {
                getSettingValue(id, fallback) {
                    return settingsStore[id] !== undefined ? settingsStore[id] : fallback;
                },
                setSettingValue(id, value) {
                    settingsStore[id] = value;
                },
            },
        },
    };

    class MockLGraphCanvas {
        drawNode(node, ctx) {
            ctx.records.push({ type: 'origDrawNode', node });
        }
    }

    class MockLGraphNode {
        onDrawForeground(ctx) {
            ctx.records.push({ type: 'origOnDrawForeground', node: this });
        }
    }

    const context = {
        app,
        LGraphCanvas: MockLGraphCanvas,
        LGraphNode: MockLGraphNode,
        LiteGraph: { NODE_TITLE_HEIGHT: 30, NODE_COLLAPSED_WIDTH: 140 },
        console,
        Map,
        URL,
        encodeURIComponent,
        document: {
            elements: {},
            getElementById(id) { return this.elements[id] || null; },
            createElement(tag) { return { tag, id: '', textContent: '' }; },
            head: {
                appendChild(el) { context.document.elements[el.id] = el; },
            },
        },
    };

    const scriptCode = source
        .replace(/^import .*;\r?\n/gm, '')
        .replace(/export const ([a-zA-Z0-9_]+)\s*=/g, 'globalThis.$1 = var_$1 =')
        .replace(/export function ([a-zA-Z0-9_]+)/g, 'globalThis.$1 = function $1')
        .replace(/const PREFIX/g, 'var PREFIX')
        .replace(/const ([a-zA-Z0-9_]+)\s*=/g, 'var $1 =');

    vm.runInNewContext(scriptCode, context);
    return { app, extension, context, settingsStore };
}

test('DINKI.Appearance registers settings under DKST.Appearance with English options', () => {
    const { extension } = fixture();
    assert.equal(extension.name, 'DINKI.Appearance');
    assert.ok(Array.isArray(extension.settings));

    const iconSetting = extension.settings.find(s => s.id === 'DKST.Appearance.PinIcon');
    assert.ok(iconSetting, 'PinIcon setting exists');
    assert.equal(iconSetting.type, 'combo');
    assert.deepEqual(JSON.parse(JSON.stringify(iconSetting.options)), ['Default', 'Lock', 'Circle']);
    assert.equal(iconSetting.category, undefined);
    assert.deepEqual(iconSetting.id.split('.'), ['DKST', 'Appearance', 'PinIcon']);

    const colorSetting = extension.settings.find(s => s.id === 'DKST.Appearance.PinIconColor');
    assert.ok(colorSetting, 'PinIconColor setting exists');
    assert.equal(colorSetting.type, 'combo');
    assert.deepEqual(JSON.parse(JSON.stringify(colorSetting.options)), ['Red', 'Orange', 'Yellow', 'Blue', 'Green', 'Purple', 'White', 'Gray', 'Black']);
    assert.equal(colorSetting.category, undefined);
    assert.deepEqual(colorSetting.id.split('.'), ['DKST', 'Appearance', 'PinIconColor']);
});

test('createPinIconSvg generates cute rounded lock SVG with requested characteristics', () => {
    const { context } = fixture();
    const lockSvg = context.createPinIconSvg('Lock', 'Red');

    assert.ok(lockSvg.includes('<svg'), 'is valid svg');
    assert.ok(lockSvg.includes('rx="4" ry="4"'), 'has rounded body with rx/ry');
    assert.ok(lockSvg.includes('stroke-linecap="round"'), 'has rounded stroke cap');
    assert.ok(lockSvg.includes('#ef4444'), 'uses red fill color');
    assert.ok(lockSvg.includes('#ffffff'), 'uses contrast keyhole color');

    // Default pin svg
    const pinSvg = context.createPinIconSvg('Default', 'Blue');
    assert.ok(pinSvg.includes('#3b82f6'), 'uses blue fill color');

    // Circle svg
    const circleSvg = context.createPinIconSvg('Circle', 'Green');
    assert.ok(circleSvg.includes('#22c55e'), 'uses green fill color');

    // Korean aliases also work for backward compatibility
    const aliasSvg = context.createPinIconSvg('자물쇠', '초록');
    assert.ok(aliasSvg.includes('rx="4" ry="4"'));
    assert.ok(aliasSvg.includes('#22c55e'));
});

test('color normalization and palette supports all 9 requested colors', () => {
    const { context } = fixture();
    const colors = ['Red', 'Orange', 'Yellow', 'Blue', 'Green', 'Purple', 'White', 'Gray', 'Black'];

    for (const color of colors) {
        const normalized = context.normalizeColor(color);
        assert.equal(normalized, color);
        assert.ok(context.COLOR_PALETTE[normalized], `palette entry for ${color} exists`);
        assert.ok(context.COLOR_PALETTE[normalized].fill.startsWith('#'), `${color} has valid hex`);
    }
});

test('beforeRegisterNodeDef hooks onDrawForeground and draws pin icon on pinned nodes', () => {
    const { extension, context } = fixture({
        'DKST.Appearance.PinIcon': 'Lock',
        'DKST.Appearance.PinIconColor': 'Orange',
    });

    class DummyNodeType {
        onDrawForeground(ctx) {
            ctx.records.push({ type: 'original' });
        }
    }

    extension.beforeRegisterNodeDef(DummyNodeType);

    const mockCtx = {
        records: [],
        save() { this.records.push({ type: 'save' }); },
        restore() { this.records.push({ type: 'restore' }); },
        translate(x, y) { this.records.push({ type: 'translate', x, y }); },
        scale(sx, sy) { this.records.push({ type: 'scale', sx, sy }); },
        beginPath() { this.records.push({ type: 'beginPath' }); },
        arc() {},
        lineTo() {},
        moveTo() {},
        stroke() {},
        fill() { this.records.push({ type: 'fill' }); },
        closePath() {},
        fillRect(x, y, w, h) { this.records.push({ type: 'fillRect', x, y, w, h }); },
        roundRect(x, y, w, h, r) { this.records.push({ type: 'roundRect', x, y, w, h, r }); },
    };

    const nodeInstance = new DummyNodeType();
    nodeInstance.flags = { pinned: false };
    nodeInstance.size = [180, 100];

    nodeInstance.onDrawForeground(mockCtx);
    assert.equal(mockCtx.records.filter(r => r.type === 'save').length, 0, 'did not draw for unpinned');

    mockCtx.records = [];
    nodeInstance.flags.pinned = true;
    nodeInstance.onDrawForeground(mockCtx);
    assert.ok(mockCtx.records.some(r => r.type === 'fillRect' || r.type === 'roundRect'), 'erased default pin area');
    assert.ok(mockCtx.records.some(r => r.type === 'save'), 'drew custom pin icon');
});

test('setting change updates DOM style and triggers canvas setDirty', () => {
    const { app, extension, context } = fixture();
    app.canvas.dirty = false;

    const iconSetting = extension.settings.find(s => s.id === 'DKST.Appearance.PinIcon');
    iconSetting.onChange('Lock');
    assert.equal(app.canvas.dirty, true, 'canvas setDirty called on icon change');

    app.canvas.dirty = false;
    const colorSetting = extension.settings.find(s => s.id === 'DKST.Appearance.PinIconColor');
    colorSetting.onChange('Yellow');
    assert.equal(app.canvas.dirty, true, 'canvas setDirty called on color change');

    const styleEl = context.document.getElementById('dkst-pin-appearance-style');
    assert.ok(styleEl, 'DOM style element created');
    assert.ok(styleEl.textContent.includes('--dkst-pin-color: #eab308'), 'DOM style updated with yellow color');
});

test('SVG icon assets exist only in the single required location: ComfyUI-DINKIssTyle/icons', () => {
    const { existsSync } = require('node:fs');
    const iconsDir = join(__dirname, '../ComfyUI-DINKIssTyle/icons');
    const jsIconsDir = join(__dirname, '../ComfyUI-DINKIssTyle/js/icons');
    const resourceIconsDir = join(__dirname, '../resource/icons');

    for (const name of ['pin_lock.svg', 'pin_default.svg', 'pin_circle.svg']) {
        assert.ok(existsSync(join(iconsDir, name)), `ComfyUI-DINKIssTyle/icons/${name} exists`);
    }

    assert.equal(existsSync(jsIconsDir), false, 'ComfyUI-DINKIssTyle/js/icons must not exist');
    assert.equal(existsSync(resourceIconsDir), false, 'resource/icons outside node pack must not exist');
});
