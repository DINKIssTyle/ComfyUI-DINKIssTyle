const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join, resolve, dirname } = require('node:path');
const vm = require('node:vm');
const { JSDOM } = require('jsdom');

const root = resolve(__dirname, '../ComfyUI-DINKIssTyle/js');
const settle = () => new Promise(resolve => setImmediate(resolve));

async function startup(blockLibraries = false) {
    const dom = new JSDOM('<!doctype html><html><head></head><body></body></html>', { runScripts: 'outside-only' });
    const context = dom.getInternalVMContext();
    context.console = { error() {} };
    let extension;
    const app = { canvas: {}, registerExtension(value) { extension = value; } };
    const requests = [];
    const modules = new Map();
    const appModule = new vm.SyntheticModule(['app'], function() { this.setExport('app', app); }, { context });
    async function load(specifier, parent = join(root, 'entry.js')) {
        if (specifier === '/scripts/app.js') return appModule;
        const path = resolve(dirname(parent), specifier);
        if (modules.has(path)) return modules.get(path);
        requests.push(path);
        // Model a server that supplies JavaScript MIME only for .js files.
        if (!path.endsWith('.js')) throw new Error('Unsupported JavaScript MIME');
        if (blockLibraries && path.includes('/vendor/')) throw new Error('Dependency unavailable');
        const module = new vm.SourceTextModule(readFileSync(path, 'utf8'), {
            context, identifier: path,
            async importModuleDynamically(specifier, referencing) {
                const imported = await load(specifier, referencing.identifier);
                await imported.evaluate();
                return imported;
            },
        });
        modules.set(path, module);
        await module.link((specifier, referencing) => load(specifier, referencing.identifier));
        return module;
    }
    const entry = await load('./dinki_text_note.js');
    await entry.evaluate();
    class Note {
        constructor() {
            this.widgets = [{ name: 'text', value: '# 한글\n\n**note**', options: {} }];
            this.properties = {};
            this.size = [360, 280];
            this.graph = { incrementVersion() {} };
            this.onNodeCreated();
        }
        addDOMWidget(name, type, element) {
            this.root = element;
            dom.window.document.body.appendChild(element);
            return { options: {} };
        }
        setDirtyCanvas() {}
    }
    extension.beforeRegisterNodeDef(Note, { name: 'DINKI_Text_Note' });
    const node = new Note();
    return { node, dom, requests };
}

test('real extension module registers its toolbar before loading renderer dependencies', async () => {
    const { node, dom, requests } = await startup();
    const buttons = node.root.querySelectorAll('button');
    assert.deepEqual(Array.from(buttons, button => button.textContent), ['Lock', 'Copy']);
    assert.equal(node.widgets[0].hidden, true);
    assert.equal(requests.some(path => path.includes('/vendor/')), false);
    buttons[0].click();
    await settle();
    await settle();
    assert.equal(node.root.querySelector('.dkst-note-preview h1').textContent, '한글');
    assert.equal(node.root.querySelector('strong').textContent, 'note');
    assert.equal(requests.filter(path => path.includes('/vendor/')).length, 2);
    node.onRemoved();
    dom.window.close();
});

test('dependency failure leaves toolbar, editing and Lock functional with a readable source fallback', async () => {
    const { node, dom } = await startup(true);
    const [lock, copy] = node.root.querySelectorAll('button');
    lock.click();
    await settle();
    const output = node.root.querySelector('.dkst-note-preview');
    assert.equal(output.textContent, node.widgets[0].value);
    assert.match(output.title, /failed/);
    assert.equal(copy.disabled, false);
    assert.equal(node.root.querySelector('textarea').readOnly, true);
    lock.click();
    const textarea = node.root.querySelector('textarea');
    textarea.value = '# Changed';
    textarea.dispatchEvent(new dom.window.Event('input'));
    assert.equal(node.widgets[0].value, '# Changed');
    lock.click();
    await settle();
    assert.equal(output.textContent, '# Changed');
    node.onRemoved();
    dom.window.close();
});

test('Unlock during renderer loading cancels the preview and relocking renders correctly', async () => {
    const { node, dom } = await startup();
    const lock = node.root.querySelector('button');
    lock.click();
    lock.click();
    await settle();
    await settle();
    const preview = node.root.querySelector('.dkst-note-preview');
    assert.equal(preview.style.display, 'none');
    assert.equal(node.root.querySelector('textarea').readOnly, false);
    assert.equal(preview.querySelector('h1'), null);
    lock.click();
    await settle();
    assert.equal(preview.style.display, 'block');
    assert.equal(preview.querySelector('h1').textContent, '한글');
    assert.equal(lock.textContent, 'Unlock');
    node.onRemoved();
    dom.window.close();
});
