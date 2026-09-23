const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8').replace(/^import .*;\r?\n/gm, '');

function fixture(overrides = {}) {
    const context = vm.createContext({ app: { registerExtension() {} }, api: {}, ...overrides });
    vm.runInContext(source, context);
    return context;
}

test('copy starts the clipboard write before the download resolves', async () => {
    let resolveFetch, written;
    const blob = { type: 'image/png' };
    const ctx = fixture({
        fetch: () => new Promise(resolve => { resolveFetch = resolve; }),
        ClipboardItem: class { constructor(data) { this.data = data; } },
        navigator: { clipboard: { write(items) { written = items; return items[0].data['image/png']; } } }
    });
    const result = ctx.copyImageToClipboard('/view?filename=test.png');
    assert.ok(written, 'write must occur within the user gesture');
    resolveFetch({ ok: true, blob: async() => blob });
    assert.equal(await result, blob);
});

test('non-PNG images convert at full resolution and release bitmap resources', async () => {
    let closed = false, drawn;
    const png = { type: 'image/png' };
    const bitmap = { width: 1600, height: 900, close() { closed = true; } };
    const canvas = { getContext: () => ({ drawImage(...args) { drawn = args; } }),
        toBlob(callback, type) { assert.equal(type, 'image/png'); callback(png); } };
    const ctx = fixture({
        fetch: async() => ({ ok: true, blob: async() => ({ type: 'image/webp' }) }),
        createImageBitmap: async() => bitmap,
        document: { createElement: () => canvas }
    });
    assert.equal(await ctx.clipboardImagePNG('/view'), png);
    assert.deepEqual([canvas.width, canvas.height], [1600, 900]);
    assert.deepEqual(drawn, [bitmap, 0, 0]);
    assert.ok(closed);
});

test('paste prefers PNG and uploads the first image item, skipping text', async () => {
    let uploaded, requested;
    const blob = { type: 'image/png' };
    const ctx = fixture({ navigator: { clipboard: { read: async() => [
        { types: ['text/plain'] },
        { types: ['image/jpeg', 'image/png'], getType: async(type) => { requested = type; return blob; } }
    ] } } });
    await ctx.pasteImageFromClipboard({ dkstUploadClipboardImage: async(value) => { uploaded = value; } });
    assert.equal(requested, 'image/png');
    assert.equal(uploaded, blob);
});

test('unsupported clipboard and image-free clipboard report actionable failures', async () => {
    const ctx = fixture();
    assert.throws(() => ctx.copyImageToClipboard('/view'), /HTTPS or localhost/);
    await assert.rejects(ctx.pasteImageFromClipboard({}), /Ctrl\+V \/ Cmd\+V/);
    ctx.navigator = { clipboard: { read: async() => [{ types: ['text/plain'] }] } };
    await assert.rejects(ctx.pasteImageFromClipboard({}), /No image found/);
});

test('permission denial is caught by the node menu and does not upload', async () => {
    let message;
    const ctx = fixture({
        navigator: { clipboard: { read: async() => { throw new Error('Permission denied'); } } },
        alert(value) { message = value; }
    });
    await ctx.clipboardMenuAction('Paste Image', () => ctx.pasteImageFromClipboard({
        dkstUploadClipboardImage() { assert.fail('must not upload'); }
    })).callback();
    assert.equal(message, 'Paste Image: Permission denied');
});
