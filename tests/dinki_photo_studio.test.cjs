const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const source = readFileSync(join(__dirname, "../ComfyUI-DINKIssTyle/js/dinki_photo_studio.js"), "utf8");

function fixture() {
    let extension;
    const calls = [];
    const queueCalls = [];
    const messages = [];
    const listeners = new Map();
    const presets = { Portrait: { light_exposure: 1.3, lens_blur_apply: false } };
    const defaults = {
        light_exposure: 0, light_contrast: 0, light_highlights: 0,
        light_shadows: 0, light_whites: 0, light_blacks: 0,
        color_white_balance: "Off", effects_texture: 0,
        detail_sharpening: 0, optics_distortion: 0,
        lens_blur_apply: false, lens_blur_focus: 0.5,
        lens_blur_bokeh: 4, lens_blur_aperture_blades: 9,
        lens_blur_bokeh_boost: 0,
        lens_blur_depth_blur_radius: 5, lens_blur_depth_sigma: 2,
        grain_seed: 0, depth_near_is_white: true,
    };
    const app = {
        registerExtension(value) { extension = value; },
        ui: { dialog: { show(message) { messages.push(message); } } },
        async queuePrompt(...args) { queueCalls.push(args); return true; },
    };
    const api = {
        addEventListener(name, handler) {
            if (!listeners.has(name)) listeners.set(name, new Set());
            listeners.get(name).add(handler);
        },
        removeEventListener(name, handler) { listeners.get(name)?.delete(handler); },
        async fetchApi(path, options) {
        calls.push({ path, options });
        if (options?.method === "POST") {
            const body = JSON.parse(options.body);
            presets[body.name] = body.settings;
            return { ok: true, async json() { return { presets: { ...presets } }; } };
        }
        return { ok: true, async json() { return { presets: { ...presets }, defaults }; } };
        },
    };
    const window = { prompt() { return "My Look"; }, alert() {} };
    const depthPixels = Uint8ClampedArray.from([
        0, 0, 0, 255, 64, 64, 64, 255,
        128, 128, 128, 255, 255, 255, 255, 255,
    ]);
    class FakeImage {
        width = 2;
        height = 2;
        naturalWidth = 2;
        naturalHeight = 2;
        set src(value) { this.source = value; this.onload?.(); }
    }
    const document = { createElement() {
        return { getContext() { return {
            drawImage() {}, getImageData() { return { data: depthPixels }; },
        }; } };
    } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ""),
        { app, api, window, Image: FakeImage, document, queueMicrotask,
            NODE_CLASS: "DINKI_Photo_Studio", isSetting: name => /^(light|color|effects|detail|optics|lens_blur|depth_near_is_white|grain_seed)/.test(name),
            inputSignature: () => "signature", workflowImages: () => [], openPhotoStudio: async () => {},
            setNodeValue(node, name, value) { const w = node.widgets.find(w => w.name === name); if (w) w.value = value; },
            request: async () => ({ presets, defaults }),
        });
    const nodeType = function () {};
    extension.beforeRegisterNodeDef(nodeType, { name: "DINKI_Photo_Studio" });
    const node = {
        id: 42,
        pos: [100, 200],
        widgets: [
            { name: "active", value: false },
            { name: "preset", value: "Custom", type: "text", options: {} },
            { name: "light_exposure", value: 0.4 },
            { name: "light_contrast", value: 0 },
            { name: "light_highlights", value: 0 },
            { name: "light_shadows", value: 0 },
            { name: "light_whites", value: 0 },
            { name: "light_blacks", value: 0 },
            { name: "color_white_balance", value: "Off" },
            { name: "effects_texture", value: 0 },
            { name: "detail_sharpening", value: 0 },
            { name: "optics_distortion", value: 0 },
            { name: "lens_blur_apply", value: false },
            { name: "lens_blur_focus", value: 0.5 },
            { name: "lens_blur_bokeh", value: 4 },
            { name: "lens_blur_aperture_blades", value: 9 },
            { name: "lens_blur_bokeh_boost", value: 0 },
            { name: "lens_blur_depth_blur_radius", value: 5 },
            { name: "lens_blur_depth_sigma", value: 2 },
            { name: "depth_near_is_white", value: true },
            { name: "grain_seed", value: 0 },
        ],
        size: [320, 700],
        setDirtyCanvas() {},
        addWidget(type, name, value, callback) {
            const widget = { type, name, value, callback };
            this.widgets.push(widget);
            return widget;
        },
        addCustomWidget(widget) {
            this.widgets.push(widget);
            return widget;
        },
    };
    nodeType.prototype.onNodeCreated.call(node);
    const emit = (name, detail) => {
        for (const handler of listeners.get(name) ?? []) handler({ detail });
    };
    return { node, nodeType, extension, app, calls, queueCalls, messages, emit, defaults, presets, listeners };
}

test("compact node keeps hidden values serializable and presets do not replace restored settings", async () => {
    const { node } = fixture();
    await new Promise(resolve => setImmediate(resolve));
    const preset = node.widgets.find(widget => widget.name === "preset");
    const exposure = node.widgets.find(widget => widget.name === "light_exposure");
    assert.equal(preset.type, "combo");
    assert.deepEqual([...preset.options.values], ["Custom", "Default", "Portrait"]);
    assert.equal(node.widgets.filter(w => w.type === "button").length, 1);
    assert.equal(node.widgets.at(-1).name, "Open Photo Studio");
    assert.equal(node.widgets.at(-1).serialize, false);
    assert.equal(exposure.hidden, true);
    assert.notEqual(exposure.serialize, false);
    assert.equal(exposure.value, .4);
    preset.callback("Portrait"); assert.equal(exposure.value, 1.3);
    preset.callback("Default"); assert.equal(exposure.value, 0);
    assert.deepEqual([...node.size], [300, 280]);
});

test("old positional workflows with serialized Save buttons keep control values", () => {
    const { node, nodeType } = fixture();
    const oldWidgets = node.widgets.filter(widget => widget.serialize !== false &&
        !["lens_blur_aperture_blades", "lens_blur_depth_blur_radius",
            "lens_blur_depth_sigma", "auto_light_request"].includes(widget.name));
    const changes = {
        light_exposure: 1.25, optics_distortion: 30,
        lens_blur_apply: true, lens_blur_focus: 0.8,
        lens_blur_bokeh: 3.5, lens_blur_bokeh_boost: 15,
        grain_seed: 99, depth_near_is_white: false,
    };
    const values = oldWidgets.map(widget => changes[widget.name] ?? widget.value);
    values.splice(oldWidgets.findIndex(widget => widget.name === "preset") + 1, 0, null, null);
    const info = { widgets_values: values };
    nodeType.prototype.onConfigure.call(node, JSON.parse(JSON.stringify(info)));
    assert.equal(node.widgets.find(widget => widget.name === "light_exposure").value, 1.25);
    assert.equal(node.widgets.find(widget => widget.name === "optics_distortion").value, 30);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_apply").value, true);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_bokeh_boost").value, 15);
    assert.equal(node.widgets.find(widget => widget.name === "grain_seed").value, 99);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_aperture_blades").value, 9);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_depth_blur_radius").value, 5);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_depth_sigma").value, 2);
    assert.equal(node.widgets.find(widget => widget.name === "auto_light_request"), undefined);
});

test("recent positional workflows without Aperture Blades restore following values", () => {
    const { node, nodeType } = fixture();
    const oldWidgets = node.widgets.filter(widget => widget.serialize !== false &&
        !["lens_blur_aperture_blades", "lens_blur_depth_blur_radius",
            "lens_blur_depth_sigma", "auto_light_request"].includes(widget.name));
    const values = oldWidgets.map(widget => widget.name === "lens_blur_bokeh_boost" ? 25 : widget.value);
    values.push(false); // Existing hidden Auto request in previously saved workflows.
    nodeType.prototype.onConfigure.call(node, { widgets_values: values });
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_bokeh_boost").value, 25);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_aperture_blades").value, 9);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_depth_blur_radius").value, 5);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_depth_sigma").value, 2);
});

test("workflow reload restores named control values despite widget placeholders", () => {
    const { node, nodeType } = fixture();
    const get = name => node.widgets.find(widget => widget.name === name);
    get("light_exposure").value = 1.1;
    get("lens_blur_focus").value = 0.73;
    get("lens_blur_depth_blur_radius").value = 7;
    get("lens_blur_depth_sigma").value = 3.2;
    const info = { widgets_values: node.widgets.map(widget =>
        widget.serialize === false ? null : widget.value) };
    nodeType.prototype.onSerialize.call(node, info);
    assert.equal(info.properties.dkstPhotoStudioSettings.lens_blur_focus, 0.73);
    assert.ok(info.widgets_values.every(value => value !== null));
    get("light_exposure").value = 0;
    get("lens_blur_focus").value = 0.5;
    get("lens_blur_depth_blur_radius").value = 5;
    get("lens_blur_depth_sigma").value = 2;
    nodeType.prototype.onConfigure.call(node, JSON.parse(JSON.stringify(info)));
    assert.equal(get("light_exposure").value, 1.1);
    assert.equal(get("lens_blur_focus").value, 0.73);
    assert.equal(get("lens_blur_depth_blur_radius").value, 7);
    assert.equal(get("lens_blur_depth_sigma").value, 3.2);
});

test("previous positional workflows preserve values around the new depth controls", () => {
    const { node, nodeType } = fixture();
    const stored = node.widgets.filter(widget => widget.serialize !== false &&
        !["lens_blur_depth_blur_radius", "lens_blur_depth_sigma"].includes(widget.name));
    const changes = { lens_blur_bokeh_boost: 42, depth_near_is_white: false, grain_seed: 17 };
    const values = stored.map(widget => changes[widget.name] ?? widget.value);
    nodeType.prototype.onConfigure.call(node, { widgets_values: values });
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_bokeh_boost").value, 42);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_depth_blur_radius").value, 5);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_depth_sigma").value, 2);
    assert.equal(node.widgets.find(widget => widget.name === "depth_near_is_white").value, false);
    assert.equal(node.widgets.find(widget => widget.name === "grain_seed").value, 17);
});

test("positional workflows ignore serialized visual separators", () => {
    const { node, nodeType } = fixture();
    const stored = node.widgets.filter(widget => widget.serialize !== false);
    const values = stored.map(widget => ({
        light_exposure: -0.8, lens_blur_focus: 0.82,
        lens_blur_depth_sigma: 4.1, grain_seed: 33,
    })[widget.name] ?? widget.value);
    values.splice(2, 0, null, null);
    values.splice(values.length - 3, 0, null);
    nodeType.prototype.onConfigure.call(node, { widgets_values: values });
    assert.equal(node.widgets.find(widget => widget.name === "light_exposure").value, -0.8);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_focus").value, 0.82);
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_depth_sigma").value, 4.1);
    assert.equal(node.widgets.find(widget => widget.name === "grain_seed").value, 33);
});

test("execution source tokens are transient and cached output is restored", async () => {
    const { node, nodeType, app, extension, emit, listeners } = fixture();
    emit("executed", { node: "42", output: { photo_studio_source: ["source-token"] } });
    assert.equal(node.dkstPhotoStudioSource, "source-token");
    const info = {};
    nodeType.prototype.onSerialize.call(node, info);
    assert.equal(JSON.stringify(info).includes("source-token"), false);
    node.comfyClass = "DINKI_Photo_Studio";
    app.nodeOutputs = { 42: { photo_studio_source: ["cached-token"] } };
    extension.loadedGraphNode(node);
    await new Promise(resolve => queueMicrotask(resolve));
    assert.equal(node.dkstPhotoStudioSource, "cached-token");
    node.onRemoved();
    assert.equal(listeners.get("executed").size, 0);
});
