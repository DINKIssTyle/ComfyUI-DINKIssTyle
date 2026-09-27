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
        { app, api, window, Image: FakeImage, document, queueMicrotask });
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

test("loading presets keeps workflow widget values until selection", async () => {
    const { node } = fixture();
    await new Promise(resolve => setImmediate(resolve));
    const preset = node.widgets.find(widget => widget.name === "preset");
    const exposure = node.widgets.find(widget => widget.name === "light_exposure");
    assert.equal(preset.type, "combo");
    assert.deepEqual([...preset.options.values], ["Custom", "Default", "Portrait"]);
    assert.deepEqual(node.widgets.slice(0, 6).map(widget => widget.name),
        ["active", "preset", "Save", "Save As", "Auto", "Reset"]);
    assert.deepEqual(node.widgets.filter(widget => widget.type === "dkst_photo_section").map(widget => widget.name), [
        "__dkst_photo_section_light", "__dkst_photo_section_color",
        "__dkst_photo_section_effects", "__dkst_photo_section_detail",
        "__dkst_photo_section_optics", "__dkst_photo_section_lens_blur",
    ]);
    assert.equal(node.widgets.find(widget => widget.name === "light_exposure").label, "Exposure");
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_apply").label, "Apply");
    assert.equal(node.widgets.find(widget => widget.name === "lens_blur_aperture_blades").label,
        "Aperture Blades");
    assert.ok(node.widgets.filter(widget => widget.type === "dkst_photo_section").every(widget => widget.serialize === false));
    assert.equal(exposure.value, 0.4);
    preset.callback("Portrait");
    assert.equal(exposure.value, 1.3);
    preset.callback("Default");
    assert.equal(exposure.value, 0);
});

test("Save As persists current values and keeps node size", async () => {
    const { node, calls, defaults } = fixture();
    await new Promise(resolve => setImmediate(resolve));
    const button = node.widgets.find(widget => widget.name === "Save As");
    await button.callback();
    const request = calls.find(call => call.options?.method === "POST");
    assert.equal(request.path, "/dinki/photo-studio/presets");
    const payload = JSON.parse(request.options.body);
    assert.equal(payload.name, "My Look");
    assert.equal(payload.settings.light_exposure, 0.4);
    assert.equal(Object.keys(payload.settings).length, Object.keys(defaults).length);
    assert.equal(payload.settings.auto_light_request, undefined);
    assert.equal(payload.overwrite, false);
    assert.deepEqual(node.size, [320, 700]);
    assert.equal(node.widgets.find(widget => widget.name === "preset").value, "My Look");
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
    assert.equal(info.dkst_photo_settings.lens_blur_focus, 0.73);
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

test("Auto applies the last suggestion immediately without queueing", async () => {
    const { node, queueCalls, emit } = fixture();
    const get = name => node.widgets.find(widget => widget.name === name);
    emit("executed", { node: "999", output: { auto_light_suggestion: [{ light_exposure: 2 }] } });
    assert.equal(get("light_exposure").value, 0.4);
    const light = {
        light_exposure: 0.75, light_contrast: 8, light_highlights: -24,
        light_shadows: 16, light_whites: -9, light_blacks: 5,
    };
    emit("executed", { node: "42", output: { auto_light_suggestion: [light] } });
    assert.equal(get("light_exposure").value, 0.4);
    get("Auto").callback();
    for (const [name, value] of Object.entries(light)) assert.equal(get(name).value, value);
    assert.equal(get("color_white_balance").value, "Auto");
    assert.equal(get("preset").value, "Custom");
    assert.deepEqual(queueCalls, []);
});

test("Reset restores every section default while keeping Active unchanged", async () => {
    const { node, defaults } = fixture();
    await new Promise(resolve => setImmediate(resolve));
    const get = name => node.widgets.find(widget => widget.name === name);
    for (const name of Object.keys(defaults)) get(name).value = "changed";
    assert.equal(get("active").value, false);
    get("preset").value = "Portrait";
    await get("Reset").callback();
    for (const [name, value] of Object.entries(defaults)) assert.equal(get(name).value, value, name);
    assert.equal(get("preset").value, "Default");
    assert.equal(get("active").value, false);
});

test("Auto without a prior execution asks for an image run", () => {
    const { node, queueCalls, messages } = fixture();
    node.widgets.find(widget => widget.name === "Auto").callback();
    assert.equal(node.widgets.find(widget => widget.name === "light_exposure").value, 0.4);
    assert.deepEqual(queueCalls, []);
    assert.match(messages[0], /Run this node once with an image/);
});

test("depth preview below Apply sets Focus by click and drag", () => {
    const { node, emit } = fixture();
    const get = name => node.widgets.find(widget => widget.name === name);
    const preview = get("__dkst_depth_focus");
    assert.equal(node.widgets.indexOf(preview), node.widgets.indexOf(get("lens_blur_apply")) + 1);
    emit("executed", { node: "42", output: { depth_preview: ["data:image/png;base64,preview"] } });
    const ctx = Object.fromEntries([
        "save", "restore", "fillRect", "fillText", "drawImage", "beginPath", "arc",
        "stroke", "moveTo", "lineTo",
    ].map(name => [name, () => {}]));
    preview.draw(ctx, node, 320, 100, 210);
    const rect = preview.imageRect;
    assert.ok(rect);
    const at = (x, y) => ({ button: 0,
        canvasX: node.pos[0] + rect.x + rect.width * x,
        canvasY: node.pos[1] + rect.y + rect.height * y });
    const pointer = { eDown: at(0.75, 0.75) };
    assert.equal(preview.onPointerDown(pointer), true);
    assert.equal(get("lens_blur_focus").value, 1);
    pointer.onDrag(at(0.25, 0.75));
    assert.equal(get("lens_blur_focus").value, 0.5);
    get("depth_near_is_white").value = false;
    preview.draw(ctx, node, 320, 100, 210);
    preview.onPointerDown({ eDown: at(0.75, 0.25) });
    assert.equal(get("lens_blur_focus").value, 0.75);
    let markerDraws = 0;
    ctx.arc = () => { markerDraws++; };
    preview.draw(ctx, node, 320, 100, 210);
    assert.ok(markerDraws > 0);
    get("lens_blur_focus").callback(0.3);
    markerDraws = 0;
    preview.draw(ctx, node, 320, 100, 210);
    assert.equal(markerDraws, 0);
    emit("executed", { node: "42", output: { depth_preview: [null] } });
    assert.equal(preview.imageRect, null);
});

test("a new image run replaces the depth preview and Auto suggestion immediately", () => {
    const { node, nodeType } = fixture();
    const preview = node.widgets.find(widget => widget.name === "__dkst_depth_focus");
    const ctx = Object.fromEntries([
        "save", "restore", "fillRect", "fillText", "beginPath", "arc",
        "stroke", "moveTo", "lineTo",
    ].map(name => [name, () => {}]));
    const drawn = [];
    ctx.drawImage = image => drawn.push(image.source);
    nodeType.prototype.onExecuted.call(node, {
        depth_preview: ["data:image/png;base64,first"],
        auto_light_suggestion: [{ light_exposure: 0.25 }],
    });
    preview.draw(ctx, node, 320, 100, 210);
    nodeType.prototype.onExecuted.call(node, {
        depth_preview: ["data:image/png;base64,second"],
        auto_light_suggestion: [{ light_exposure: 1.25 }],
    });
    preview.draw(ctx, node, 320, 100, 210);
    assert.deepEqual(drawn, ["data:image/png;base64,first", "data:image/png;base64,second"]);
    node.widgets.find(widget => widget.name === "Auto").callback();
    assert.equal(node.widgets.find(widget => widget.name === "light_exposure").value, 1.25);
});

test("loading a workflow restores the latest cached depth preview", async () => {
    const { node, extension, app } = fixture();
    node.comfyClass = "DINKI_Photo_Studio";
    app.nodeOutputs = { 42: { depth_preview: ["data:image/png;base64,cached"] } };
    extension.loadedGraphNode(node);
    await new Promise(resolve => queueMicrotask(resolve));
    const preview = node.widgets.find(widget => widget.name === "__dkst_depth_focus");
    const drawn = [];
    const ctx = Object.fromEntries([
        "save", "restore", "fillRect", "fillText", "beginPath", "arc",
        "stroke", "moveTo", "lineTo",
    ].map(name => [name, () => {}]));
    ctx.drawImage = image => drawn.push(image.source);
    preview.draw(ctx, node, 320, 100, 210);
    assert.deepEqual(drawn, ["data:image/png;base64,cached"]);
});
