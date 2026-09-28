const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

test('pasted Image Load preview retains its temp source after tab reconstruction', async () => {
    let extension;
    let tempExists = true;
    const scheduled = [];
    const app = { registerExtension(value) {
        if (value.name === 'DINKI.ImageLoad') extension = value;
    } };
    function element(tag) {
        return {
            tag, style: {}, children: [],
            append(...children) { this.children.push(...children); },
            addEventListener() {}, removeAttribute(name) { delete this[name]; },
        };
    }
    class LoadedImage {
        set src(value) {
            this._src = value;
            this.naturalWidth = 640;
            this.naturalHeight = 480;
            this.onload?.();
        }
        get src() { return this._src; }
    }
    const api = {
        apiURL: url => url,
        async fetchApi(url, options) {
            if (options?.method === 'HEAD') {
                return { ok: tempExists, status: tempExists ? 200 : 404 };
            }
            return { ok: true, async json() {
                return url.endsWith('/categories') ? { categories: ['', 'folder'] }
                    : { files: ['ordinary.png'] };
            } };
        },
    };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api, document: { createElement: element }, Image: LoadedImage,
        URLSearchParams, requestAnimationFrame: fn => fn(),
        setTimeout: fn => { scheduled.push(fn); }, console,
    });
    class Node {
        constructor(properties) {
            this.properties = properties;
            this.widgets = [
                { name: 'category', value: 'folder' },
                { name: 'filename', value: 'DKST_Paste_saved.png' },
                { name: 'source_type', value: 'input' },
            ];
            this.onNodeCreated();
        }
        addDOMWidget(name, type, preview) { this.preview = preview; return {}; }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Node, { name: 'DINKI_Image_Load' });
    const node = new Node({ dkstImageLoad: {
        category: 'folder', filename: 'DKST_Paste_saved.png', source_type: 'temp',
    } });
    node.onConfigure();
    while (scheduled.length) scheduled.shift()();
    await new Promise(resolve => setImmediate(resolve));

    assert.equal(node.widgets[2].value, 'temp');
    assert.equal(node.widgets[2].hidden, true);
    assert.equal(node.widgets.find(widget => widget.name === 'image').hidden, true);
    assert.equal(node.widgets[1].value, 'DKST_Paste_saved.png');
    assert.deepEqual(Array.from(node.widgets[1].options.values),
        ['DKST_Paste_saved.png', 'ordinary.png']);
    assert.match(node.preview.children[0].src, /filename=DKST_Paste_saved.png/);
    assert.match(node.preview.children[0].src, /type=temp/);
    assert.equal(node.preview.style.display, 'flex');
    assert.equal(node.dkstImageResolution, '640 × 480');

    tempExists = false;
    const stale = new Node({ dkstImageLoad: {
        category: 'folder', filename: 'DKST_Paste_deleted.png', source_type: 'temp',
    } });
    stale.widgets[1].value = 'DKST_Paste_deleted.png';
    stale.onConfigure();
    while (scheduled.length) scheduled.shift()();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(stale.widgets[0].value, 'folder');
    assert.equal(stale.widgets[1].value, 'ordinary.png');
    assert.equal(stale.widgets[2].value, 'input');
    assert.deepEqual(Array.from(stale.widgets[1].options.values), ['ordinary.png']);
    assert.equal(stale.properties.dkstImageLoad, undefined);
});

test('both Image Load nodes repopulate the selected category after async load and tab return', async () => {
    for (const nodeClass of ['DINKI_Image_Load', 'DINKI_Image_Load_Crop']) {
        let extension, tick, releaseFirstCategories, releaseStaleCategory;
        const scheduled = [];
        const requests = [];
        let files = ['forest.png'];
        const graph = { _nodes: [] };
        const app = { canvas: { graph }, registerExtension(value) {
            if (value.name === 'DINKI.ImageLoad') extension = value;
        } };
        const element = () => ({ style: {}, children: [], append(...children) {
            this.children.push(...children);
        }, addEventListener() {}, removeAttribute() {} });
        const api = {
            apiURL: url => url,
            async fetchApi(url) {
                requests.push(url);
                if (url.endsWith('/categories')) {
                    if (!releaseFirstCategories) {
                        return new Promise(resolve => {
                            releaseFirstCategories = () => resolve({ ok: true,
                                json: async() => ({ categories: ['', 'folder'] }) });
                        });
                    }
                    return { ok: true, json: async() => ({ categories: ['', 'folder'] }) };
                }
                return { ok: true, json: async() => ({ files: url.includes('category=folder') ? files : ['root.png'] }) };
            },
        };
        vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
            app, api, document: { createElement: element },
            window: { addEventListener() {} },
            Image: class { set src(value) {
                this._src = value;
                this.naturalWidth = 640;
                this.naturalHeight = 480;
                this.onload?.();
            } get src() { return this._src; } },
            URLSearchParams, requestAnimationFrame: fn => fn(),
            setTimeout: fn => scheduled.push(fn),
            setInterval: fn => { tick = fn; return 1; }, console,
        });
        const combo = (name, value) => {
            const widget = { name, value, notifications: 0, _options: { values: [] } };
            if (name === 'filename') {
                widget.options = { values: ['root.png'] };
                widget._state = { options: { values: ['root.png'] } };
                return widget;
            }
            Object.defineProperty(widget, 'options', {
                get() { return this._options; },
                set(options) { this._options = options; this.notifications++; },
            });
            return widget;
        };
        class Node {
            constructor() {
                this.comfyClass = nodeClass;
                this.graph = graph;
                this.properties = {};
                this.widgets = [combo('category', ''), combo('filename', ''),
                    combo('source_type', 'input')];
                this.onNodeCreated();
            }
            addDOMWidget() { return {}; }
            addWidget(type, name, value, callback, options) {
                const widget = { type, name, value, callback, options };
                this.widgets.push(widget);
                return widget;
            }
            setDirtyCanvas() {}
        }
        await extension.beforeRegisterNodeDef(Node, { name: nodeClass });
        const node = new Node();
        graph._nodes.push(node);
        const drain = async() => {
            while (scheduled.length) scheduled.shift()();
            await new Promise(resolve => setImmediate(resolve));
        };

        await drain(); // Initial category request is still waiting.
        // ComfyUI may invoke the initial root combo callback while restoring
        // widget values. Its async cleanup must not install root options later.
        node.dkstDeleteTemporaryImage = () => new Promise(resolve => {
            releaseStaleCategory = resolve;
        });
        const staleCallback = node.widgets[0].callback('');
        node.widgets[0].value = 'folder';
        node.widgets[1].value = 'forest.png';
        const initialFilenameOptions = node.widgets[1].options;
        node.onConfigure({});
        extension.loadedGraphNode(node);
        await drain();
        assert.equal(node.widgets[0].value, 'folder');
        assert.equal(node.widgets[1].value, 'forest.png');
        assert.deepEqual(Array.from(node.widgets[1].options.values), ['forest.png']);
        assert.deepEqual(Array.from(node.widgets[1]._state.options.values), ['forest.png'],
            'Nodes 2.0 reads the registered widget state for its dropdown');
        assert.notEqual(node.widgets[1].options, initialFilenameOptions,
            `${nodeClass} must replace writable combo options for the Nodes 2.0 popup`);
        assert.ok(node.widgets[0].notifications > 0,
            `${nodeClass} must notify accessor-backed combo renderers`);
        releaseStaleCategory();
        await staleCallback;
        await drain();
        assert.deepEqual(Array.from(node.widgets[1].options.values), ['forest.png'],
            'a delayed root-category callback must not replace the folder menu');
        releaseFirstCategories();
        await drain();
        assert.equal(node.widgets[0].value, 'folder', 'late initial response must not reset category');

        extension.setup();
        await drain();
        app.canvas.graph = { _nodes: [] };
        tick();
        files = ['forest.png', 'lake.png'];
        app.canvas.graph = graph;
        tick();
        await drain();
        assert.deepEqual(Array.from(node.widgets[1].options.values), ['forest.png', 'lake.png']);
        assert.deepEqual(Array.from(node.widgets[1]._state.options.values), ['forest.png', 'lake.png']);
        assert.equal(node.widgets[1].value, 'forest.png');
        files = ['forest.png', 'lake.png', 'river.png'];
        extension.afterConfigureGraph();
        await drain();
        assert.deepEqual(Array.from(node.widgets[1].options.values),
            ['forest.png', 'lake.png', 'river.png']);
        assert.ok(requests.some(url => url.includes('category=folder')));
    }
});
