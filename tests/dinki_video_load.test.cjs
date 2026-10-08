const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");
const { JSDOM } = require("jsdom");

const tick = async () => { await new Promise(resolve => setImmediate(resolve)); await new Promise(resolve => setImmediate(resolve)); };
async function fixture(handler) {
    const dom = new JSDOM("<body></body>");
    const document = dom.window.document;
    const callbacks = [];
    const context2d = new Proxy({}, { get: () => () => {} });
    dom.window.HTMLCanvasElement.prototype.getContext = () => context2d;
    const media = dom.window.HTMLMediaElement.prototype;
    Object.defineProperty(media, "paused", { configurable: true, get() { return !this.playing; } });
    media.load = function () {};
    media.pause = function () { this.playing = false; this.dispatchEvent(new dom.window.Event("pause")); };
    media.play = function () { this.playing = true; this.dispatchEvent(new dom.window.Event("play")); return Promise.resolve(); };
    dom.window.HTMLElement.prototype.setPointerCapture = function () {};
    dom.window.HTMLElement.prototype.releasePointerCapture = function () {};
    const app = { registerExtension(extension) { callbacks.push(extension); }, nodeOutputs: {} };
    const metadata = filename => ({ width: 640, height: 480, duration: 10, fps: 30000 / 1001,
        source_key: filename, preview: { filename, subfolder: "", type: "input" } });
    const calls = [];
    const api = { addEventListener() {}, removeEventListener() {}, apiURL: url => url,
        async fetchApi(url, options) {
            calls.push({ url, options });
            const custom = await handler?.(url, options, metadata);
            if (custom) return custom;
            let data;
            if (url.endsWith("categories")) data = { categories: ["", "clips"] };
            else if (url.includes("/files?")) data = { files: ["first.mp4", "second.mp4"] };
            else if (url.includes("/metadata?")) data = metadata(new URL(url, "http://test").searchParams.get("filename"));
            else data = { preview: { filename: "proxy.mp4", type: "temp", subfolder: "" } };
            return { ok: true, json: async () => data };
        },
    };
    const context = vm.createContext({ app, api, document, URLSearchParams, FormData: dom.window.FormData,
        Image: dom.window.Image, queueMicrotask,
        requestAnimationFrame: () => 1, cancelAnimationFrame() {}, ResizeObserver: class { observe() {} disconnect() {} } });
    const modules = new Map();
    const appModule = new vm.SyntheticModule(["app"], function () { this.setExport("app", app); }, { context });
    const apiModule = new vm.SyntheticModule(["api"], function () { this.setExport("api", api); }, { context });
    function module(name) {
        if (!modules.has(name)) modules.set(name, new vm.SourceTextModule(
            readFileSync(join(__dirname, "../ComfyUI-DINKIssTyle/js", name), "utf8"), { context, identifier: name }));
        return modules.get(name);
    }
    const entry = module("dinki_video_load.js");
    await entry.link(specifier => {
        if (specifier.endsWith("scripts/app.js")) return appModule;
        if (specifier.endsWith("scripts/api.js")) return apiModule;
        return module(specifier.replace(/^\.\//, ""));
    });
    await entry.evaluate();
    const defaults = { category: "", filename: "first.mp4", aspect_ratio: "Original", custom_width: 1,
        custom_height: 1, crop_x: 0, crop_y: 0, crop_width: 1, crop_height: 1,
        resolution_multiple: 8, megapixels: 1, trim_in: 0, trim_out: 0, fps_mode: "Original", output_fps: 24 };
    class Node {
        constructor() {
            this.widgets = Object.entries(defaults).map(([name, value]) => ({ name, value, options: {} }));
            this.comfyClass = "DINKI_Video_Load_Crop"; this.size = [300, 300]; this.id = 1;
            this.graph = { incrementVersion() {} };
        }
        addDOMWidget(name, type, element, options) {
            document.body.append(element);
            const widget = { name, type, element, options };
            this.widgets.push(widget); return widget;
        }
        setDirtyCanvas() {}
        expandToFitContent() {}
    }
    for (const extension of callbacks) extension.beforeRegisterNodeDef?.(Node, { name: "DINKI_Video_Load_Crop" });
    const node = new Node();
    node.onNodeCreated();
    const get = name => node.widgets.find(w => w.name === name);
    const editor = node.dkstVideoEditor;
    Object.defineProperties(editor.video, { videoWidth: { value: 640 }, videoHeight: { value: 480 }, readyState: { value: 2 } });
    const change = (input, value) => { input.value = String(value); input.dispatchEvent(new dom.window.Event("change")); };
    const pointer = (element, type, x, target = element) => {
        const event = new dom.window.Event(type, { bubbles: true });
        Object.assign(event, { button: 0, clientX: x, pointerId: 1 });
        target.dispatchEvent(event);
    };
    editor.timeline.getBoundingClientRect = () => ({ left: 0, width: 100 });
    await tick();
    editor.video.onloadedmetadata?.();
    return { node, editor, get, dom, calls, change, pointer };
}

test("seconds input moves OUT relative to IN and clamps to source end", async () => {
    const { editor, get, change, node } = await fixture();
    change(editor.inputs[0], 3);
    change(editor.inputs[2], 2);
    assert.equal(get("trim_out").value, 5);
    change(editor.inputs[2], 100);
    assert.equal(get("trim_out").value, 10);
    assert.equal(editor.inputs[2].value, "7.000");
    node.onRemoved();
});

test("typing seconds updates OUT immediately before the field loses focus", async () => {
    const { editor, get, dom, node } = await fixture();
    editor.inputs[0].focus();
    editor.inputs[0].value = "1";
    editor.inputs[0].dispatchEvent(new dom.window.Event("input"));
    editor.inputs[2].focus();
    editor.inputs[2].value = "2";
    editor.inputs[2].dispatchEvent(new dom.window.Event("input"));
    assert.equal(get("trim_out").value, 3);
    assert.equal(editor.inputs[1].value, "3.000");
    node.onRemoved();
});

test("timeline supports IN/OUT drag, independent seeking and keyboard frame steps", async () => {
    const { editor, get, pointer, dom, node } = await fixture();
    pointer(editor.timeline, "pointerdown", 20, editor.handles[0]);
    pointer(editor.timeline, "pointerup", 20);
    assert.equal(get("trim_in").value, 2);
    pointer(editor.timeline, "pointerdown", 80, editor.handles[1]);
    pointer(editor.timeline, "pointerup", 80);
    assert.equal(get("trim_out").value, 8);
    pointer(editor.timeline, "pointerdown", 50);
    pointer(editor.timeline, "pointerup", 50);
    assert.equal(editor.video.currentTime, 5);
    editor.handles[0].dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "ArrowRight" }));
    assert.ok(Math.abs(get("trim_in").value - (2 + 1001 / 30000)) < .000001);
    node.onRemoved();
});

test("FPS reset and full-range reset restore original modes", async () => {
    const { node, get, editor, change } = await fixture();
    get("fps_mode").value = "Custom"; get("output_fps").value = 12;
    change(editor.inputs[0], 2); change(editor.inputs[1], 5);
    const buttons = [...node.dkstCropRoot.querySelectorAll("button")];
    buttons.find(button => button.textContent === "Original FPS").click();
    assert.equal(get("fps_mode").value, "Original");
    assert.equal(get("output_fps").value, 30000 / 1001);
    buttons.find(button => button.textContent === "Full range").click();
    assert.equal(get("trim_in").value, 0);
    assert.equal(get("trim_out").value, 0);
    assert.equal(editor.inputs[1].value, "10.000");
    node.onRemoved();
});

test("named workflow settings survive restore and preserve node dimensions", async () => {
    const { node, get, editor } = await fixture();
    node.size = [520, 810];
    node.onConfigure({ properties: { dkstVideoLoadSettings: {
        category: "clips", filename: "second.mp4", crop_x: .2, crop_y: .1, crop_width: .6,
        crop_height: .6, trim_in: 2, trim_out: 6, fps_mode: "Custom", output_fps: 15,
    } } });
    await tick(); editor.video.onloadedmetadata?.();
    assert.equal(get("crop_x").value, .2);
    assert.equal(get("crop_width").value, .6);
    assert.equal(get("trim_in").value, 2);
    assert.equal(get("trim_out").value, 6);
    assert.equal(get("fps_mode").value, "Custom");
    assert.deepEqual(node.size, [520, 810]);
    const saved = {};
    node.onSerialize(saved);
    assert.equal(saved.properties.dkstVideoLoadSettings.output_fps, 15);
    assert.equal(saved.properties.dkstVideoLoadSettings.filename, "second.mp4");
    node.onRemoved();
});

test("stale metadata errors cannot overwrite a newer file selection", async () => {
    let rejectFirst;
    const fixturePromise = fixture(url => {
        if (url.includes("/metadata?") && url.includes("first.mp4")) return new Promise((_, reject) => { rejectFirst = reject; });
    });
    const { node, get, editor } = await fixturePromise;
    get("filename").value = "second.mp4"; get("filename").callback();
    await tick(); editor.video.onloadedmetadata?.();
    rejectFirst(new Error("old failure")); await tick();
    assert.match(node.dkstCropRoot.textContent, /Source 29\.970 FPS/);
    assert.doesNotMatch(node.dkstCropRoot.textContent, /old failure/);
    node.onRemoved();
});

test("browser decoding failure requests a proxy once and removal clears media", async () => {
    const { node, editor, calls } = await fixture();
    await editor.video.onerror();
    assert.ok(editor.video.src.includes("proxy.mp4"));
    await editor.video.onerror();
    assert.equal(calls.filter(call => call.url.endsWith("/preview")).length, 1);
    node.onRemoved();
    assert.equal(editor.video.getAttribute("src"), null);
});

test("video upload uses the selected category and selects the server-renamed file", async () => {
    const { node, get, dom, calls, editor } = await fixture(url => {
        if (url === "/upload/image") return { ok: true, json: async () =>
            ({ name: "new (1).MP4", subfolder: "clips", type: "input" }) };
        if (url.includes("/files?")) return { ok: true, json: async () =>
            ({ files: ["first.mp4", "new (1).MP4"] }) };
    });
    get("category").value = "clips";
    get("trim_in").value = 3; get("trim_out").value = 6;
    get("crop_x").value = .2; get("crop_width").value = .5;
    const uploaded = await node.dkstUploadVideo(new dom.window.File(["video"], "new.MP4", { type: "video/mp4" }));
    assert.equal(uploaded, true);
    const upload = calls.find(call => call.url === "/upload/image");
    assert.equal(upload.options.method, "POST");
    assert.equal(upload.options.body.get("image").name, "new.MP4");
    assert.equal(upload.options.body.get("subfolder"), "clips");
    assert.equal(upload.options.body.get("type"), "input");
    assert.equal(upload.options.body.get("overwrite"), "false");
    assert.equal(get("filename").value, "new (1).MP4");
    assert.ok(get("filename").options.values.includes("new (1).MP4"));
    assert.equal(get("trim_in").value, 0);
    assert.equal(get("trim_out").value, 0);
    assert.equal(get("crop_x").value, 0);
    assert.equal(get("crop_width").value, 1);
    assert.ok(editor.video.src.includes("new+%281%29.MP4"));
    const saved = {}; node.onSerialize(saved);
    assert.equal(saved.properties.dkstVideoLoadSettings.filename, "new (1).MP4");
    node.onRemoved();
});

test("Upload Video picker accepts video extensions and is cleaned up with the node", async () => {
    const { node, dom, get } = await fixture(url => {
        if (url === "/upload/image") return { ok: true, json: async () =>
            ({ name: "picked.mov", subfolder: "", type: "input" }) };
    });
    const input = dom.window.document.querySelector('input[type="file"]');
    assert.equal(input.accept, ".mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.ts");
    let opened = 0;
    input.click = () => { opened++; };
    [...node.dkstCropRoot.querySelectorAll("button")].find(button => button.textContent === "Upload Video").click();
    assert.equal(opened, 1);
    Object.defineProperty(input, "files", { value: [new dom.window.File(["video"], "picked.mov")] });
    input.dispatchEvent(new dom.window.Event("change"));
    await tick();
    assert.equal(get("filename").value, "picked.mov");
    node.onRemoved();
    assert.equal(dom.window.document.querySelector('input[type="file"]'), null);
    assert.equal(node.dkstUploadVideo, undefined);
});

test("upload failure preserves the current file and editing settings", async () => {
    const { node, get, dom, editor } = await fixture(url => {
        if (url === "/upload/image") return { ok: false, status: 413, json: async () => ({ error: "Video is too large" }) };
    });
    get("trim_in").value = 2; get("crop_x").value = .2;
    const source = editor.video.src;
    assert.equal(await node.dkstUploadVideo(new dom.window.File(["video"], "new.mp4")), false);
    assert.equal(get("filename").value, "first.mp4");
    assert.equal(get("trim_in").value, 2);
    assert.equal(get("crop_x").value, .2);
    assert.equal(editor.video.src, source);
    assert.match(node.dkstCropRoot.textContent, /Video is too large/);
    assert.equal([...node.dkstCropRoot.querySelectorAll("button")].find(button => button.textContent === "Upload Video").disabled, false);
    node.onRemoved();
});

test("completed upload cannot replace a newer user selection", async () => {
    let finish;
    const { node, get, dom } = await fixture(url => {
        if (url === "/upload/image") return new Promise(resolve => { finish = resolve; });
    });
    const pending = node.dkstUploadVideo(new dom.window.File(["video"], "old.mp4"));
    get("filename").value = "second.mp4"; get("filename").callback();
    await tick();
    finish({ ok: true, json: async () => ({ name: "old.mp4", subfolder: "", type: "input" }) });
    assert.equal(await pending, false);
    assert.equal(get("filename").value, "second.mp4");
    node.onRemoved();
});

test("dropping a video into the preview or native node uploads exactly once", async () => {
    let uploaded = 0;
    const { node, dom, get } = await fixture(url => {
        if (url === "/upload/image") {
            uploaded++;
            return { ok: true, json: async () => ({ name: `dropped${uploaded}.mkv`, subfolder: "", type: "input" }) };
        }
    });
    const transfer = { types: ["Files"], files: [new dom.window.File(["video"], "clip.mkv")] };
    assert.equal(node.onDragOver({ dataTransfer: { types: ["Files"], files: [] } }), true);
    const event = new dom.window.Event("drop", { bubbles: true, cancelable: true });
    Object.defineProperty(event, "dataTransfer", { value: transfer });
    node.dkstCropRoot.dispatchEvent(event);
    await tick();
    assert.equal(event.defaultPrevented, true);
    assert.equal(uploaded, 1);
    assert.equal(get("filename").value, "dropped1.mkv");
    assert.equal(node.onDragDrop({ dataTransfer: transfer }), true);
    await tick();
    assert.equal(uploaded, 2);
    assert.equal(get("filename").value, "dropped2.mkv");
    assert.equal(await node.dkstUploadVideo(new dom.window.File(["other"], "note.txt")), false);
    assert.equal(uploaded, 2);
    node.onRemoved();
});

test("plain-text HTTP upload errors still display the server status", async () => {
    const { node, dom, get } = await fixture(url => {
        if (url === "/upload/image") return { ok: false, status: 413, json: async () => { throw new SyntaxError("not JSON"); } };
    });
    assert.equal(await node.dkstUploadVideo(new dom.window.File(["video"], "new.mp4")), false);
    assert.match(node.dkstCropRoot.textContent, /Request failed \(413\)/);
    assert.equal(get("filename").value, "first.mp4");
    node.onRemoved();
});
