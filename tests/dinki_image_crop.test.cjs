const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const source = readFileSync(join(__dirname, "../ComfyUI-DINKIssTyle/js/dinki_image_crop.js"), "utf8");

function fixture(className = "DINKI_Image_Crop", deferImages = false, executionHandler = null) {
    let extension;
    const listeners = new Map();
    const app = { registerExtension(value) { extension = value; }, nodeOutputs: {} };
    const api = {
        addEventListener(name, fn) { listeners.set(name, fn); },
        removeEventListener(name) { listeners.delete(name); },
    };
    const images = [];
    class FakeImage {
        constructor() { images.push(this); }
        width = 400;
        height = 300;
        set src(value) { this.uri = value; if (!deferImages) this.onload?.(); }
    }
    const drawImages = [];
    const drawTexts = [];
    const context = new Proxy({
        drawImage(image) { drawImages.push(image.uri); },
        fillText(value) { drawTexts.push(value); },
    }, { get(target, key) {
        if (!target[key]) target[key] = () => {};
        return target[key];
    } });
    const observers = [];
    class FakeResizeObserver {
        constructor(callback) { this.callback = callback; observers.push(this); }
        observe(element) { this.element = element; }
        disconnect() { this.element = null; }
    }
    const document = {
        createElement(tag) {
            return {
                tag, style: {}, children: [], value: "", listeners: {},
                clientWidth: 340, clientHeight: 300,
                append(...items) { this.children.push(...items); },
                addEventListener(name, callback) { this.listeners[name] = callback; },
                setAttribute() {},
                getContext() { return context; },
                getBoundingClientRect() { return { left: 0, top: 0,
                    width: this.clientWidth * (this.displayScale || 1),
                    height: this.clientHeight * (this.displayScale || 1) }; },
                setPointerCapture() {}, releasePointerCapture() {},
            };
        },
    };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ""),
        { app, api, Image: FakeImage, document, ResizeObserver: FakeResizeObserver,
            queueMicrotask });
    const Node = function () {};
    if (executionHandler) Node.prototype.onExecuted = executionHandler;
    Node.prototype.onConfigure = function (info) {
        if (Array.isArray(info?.size)) this.size = [...info.size];
    };
    extension.beforeRegisterNodeDef(Node, { name: className });
    const node = {
        id: 8, comfyClass: className, pos: [100, 200], size: [300, 250],
        widgets: [
            ...(className === "DINKI_Image_Load_Crop" ? [
                { name: "category", value: "" }, { name: "filename", value: "first.png" },
            ] : []),
            { name: "aspect_ratio", value: "Original" },
            { name: "custom_width", value: 1 },
            { name: "custom_height", value: 1 },
            { name: "crop_x", value: 0 },
            { name: "crop_y", value: 0 },
            { name: "crop_width", value: 1 },
            { name: "crop_height", value: 1 },
            ...(className === "DINKI_Image_Load_Crop" ? [
                { name: "resolution_multiple", value: 8, type: "number" },
                { name: "megapixels", value: 1, type: "number" },
                { name: "crop_mode", value: "Crop", type: "combo" },
            ] : []),
        ],
        graph: { incrementVersion() {} }, setDirtyCanvas() {}, expandToFitContent() {},
        addCustomWidget(widget) { this.widgets.push(widget); return widget; },
        addDOMWidget(name, type, element, options) {
            const widget = { name, type, element, options,
                computeLayoutSize() { return {
                    minHeight: options.getMinHeight(), maxHeight: options.getMaxHeight(), minWidth: 0,
                }; },
            };
            this.widgets.push(widget);
            return widget;
        },
    };
    Node.prototype.onNodeCreated.call(node);
    const get = name => node.widgets.find(widget => widget.name === name);
    const output = (width, height, rect, uri = "data:preview") => ({
        source_size: [[width, height]], crop_rect: [rect], source_preview: [uri],
    });
    return { app, extension, Node, node, get, output, listeners, observers, drawImages, drawTexts, images };
}

test("Load & Crop reloads identical execution output after a file refresh clears the canvas", () => {
    const { node, output, drawImages } = fixture("DINKI_Image_Load_Crop");
    const result = output(400, 300, [0, 0, 400, 300], "data:executed");
    node.dkstCropOutput(result);
    node.dkstCropSourcePreview(null);
    node.dkstCropSourcePreview({ width: 400, height: 300, uri: "file:refreshed" }, "input::first.png");
    node.dkstCropOutput(result);
    assert.equal(drawImages.at(-1), "data:executed");
    node.dkstCropSourcePreview(null);
    node.dkstCropOutput(result);
    assert.equal(drawImages.at(-1), "data:executed");
});

test("a pending crop output cannot overwrite a newly selected file", () => {
    const { node, output, images, drawImages } = fixture("DINKI_Image_Load_Crop", true);
    node.dkstCropOutput(output(400, 300, [0, 0, 400, 300], "data:old"));
    node.dkstCropSourcePreview({ width: 400, height: 300, uri: "file:new" }, "input::new.png");
    images[0].onload();
    assert.equal(drawImages.at(-1), "file:new");
});

test("crop retries the same output after a failed image load", () => {
    const { node, output, images, drawImages } = fixture("DINKI_Image_Crop", true);
    const result = output(400, 300, [0, 0, 400, 300], "data:retry");
    node.dkstCropOutput(result);
    images[0].onerror();
    node.dkstCropOutput(result);
    assert.equal(images.length, 2);
    images[1].onload();
    assert.equal(drawImages.at(-1), "data:retry");
});

test("crop keeps the newest result when image decoding finishes out of order", () => {
    const { node, output, images, drawImages } = fixture("DINKI_Image_Crop", true);
    node.dkstCropOutput(output(400, 300, [0, 0, 400, 300], "data:first"));
    node.dkstCropOutput(output(400, 300, [0, 0, 400, 300], "data:second"));
    images[1].onload();
    images[0].onload();
    assert.equal(drawImages.at(-1), "data:second");
});

test("crop updates even when an upstream execution handler fails", () => {
    const { node, Node, output, drawImages } = fixture("DINKI_Image_Load_Crop", false,
        () => { throw new Error("native handler failed"); });
    for (const uri of ["data:first", "data:second"]) {
        assert.throws(() => Node.prototype.onExecuted.call(node,
            output(400, 300, [0, 0, 400, 300], uri)), /native handler failed/);
        assert.equal(drawImages.at(-1), uri);
    }
});

test("preset ratio centers a maximum crop and custom fields form a ratio pair", () => {
    const { node, get, output } = fixture();
    node.dkstCropOutput(output(400, 300, [0, 0, 400, 300]));
    get("aspect_ratio").value = "1:1";
    get("aspect_ratio").callback("1:1");
    assert.equal(get("crop_width").value, 0.75);
    assert.equal(get("crop_x").value, 0.125);
    assert.equal(get("crop_height").value, 1);
    get("aspect_ratio").value = "Custom";
    get("aspect_ratio").callback("Custom");
    const row = get("__dkst_crop_preview").element.children[0];
    const inputs = row.children[1].children.filter(child => child.tag === "input");
    inputs[0].value = "16";
    inputs[0].listeners.change();
    inputs[1].value = "9";
    inputs[1].listeners.change();
    assert.equal(get("custom_width").value, 16);
    assert.equal(get("custom_height").value, 9);
    assert.equal(get("crop_width").value, 1);
    assert.equal(get("crop_height").value, 0.75);
});

test("dragging moves inside bounds and corner resize keeps selected aspect", () => {
    const { node, get, output } = fixture();
    node.dkstCropOutput(output(400, 300, [0, 0, 400, 300]));
    get("aspect_ratio").value = "1:1";
    get("aspect_ratio").callback("1:1");
    const preview = get("__dkst_crop_preview");
    const canvas = preview.element.children[1];
    preview.render();
    const r = preview.imageRect;
    const event = (x, y) => ({ button: 0, pointerId: 1, clientX: x, clientY: y,
        preventDefault() {}, stopPropagation() {} });
    const down = event(r.x + r.w / 2, r.y + r.h / 2);
    canvas.listeners.pointerdown(down);
    canvas.listeners.pointermove(event(down.clientX + r.w, down.clientY + r.h));
    canvas.listeners.pointerup(event(down.clientX + r.w, down.clientY + r.h));
    assert.equal(get("crop_x").value, 0.25);
    assert.equal(get("crop_y").value, 0);
    const corner = event(r.x + get("crop_x").value * r.w,
        r.y + get("crop_y").value * r.h);
    canvas.listeners.pointerdown(corner);
    canvas.listeners.pointermove(event(corner.clientX + 30, corner.clientY + 20));
    canvas.listeners.pointerup(event(corner.clientX + 30, corner.clientY + 20));
    const pixelW = get("crop_width").value * 400;
    const pixelH = get("crop_height").value * 300;
    assert.ok(Math.abs(pixelW - pixelH) < 0.001);
    assert.ok(get("crop_x").value >= 0 && get("crop_y").value >= 0);
});

test("node resize enlarges the canvas without adding a separate Custom widget", () => {
    const { node, get, output, observers } = fixture();
    node.dkstCropOutput(output(400, 300, [0, 0, 400, 300]));
    const preview = get("__dkst_crop_preview");
    assert.equal(get("__dkst_custom_ratio"), undefined);
    const canvas = preview.element.children[1];
    const before = preview.imageRect.h;
    canvas.clientWidth = 500;
    canvas.clientHeight = 500;
    observers[0].callback();
    assert.ok(preview.imageRect.h > before);
    for (const name of ["custom_width", "custom_height", "crop_x", "crop_y",
        "crop_width", "crop_height"]) {
        assert.equal(get(name).hidden, true);
    }
});

test("pointer hit testing follows the graph zoom scale", () => {
    const { node, get, output } = fixture();
    node.dkstCropOutput(output(400, 300, [50, 0, 300, 300]));
    const preview = get("__dkst_crop_preview");
    const canvas = preview.element.children[1];
    canvas.displayScale = 0.5;
    const r = preview.imageRect;
    const initialX = get("crop_x").value;
    const pointer = (x, y) => ({ button: 0, pointerId: 2,
        clientX: x * 0.5, clientY: y * 0.5,
        preventDefault() {}, stopPropagation() {} });
    const startX = r.x + r.w / 2;
    const startY = r.y + r.h / 2;
    canvas.listeners.pointerdown(pointer(startX, startY));
    canvas.listeners.pointermove(pointer(startX + 20, startY));
    canvas.listeners.pointerup(pointer(startX + 20, startY));
    assert.ok(get("crop_x").value > initialX);
});

test("a new input image on the next execution replaces the preview", () => {
    const { node, output, listeners, drawImages } = fixture();
    const first = output(400, 300, [0, 0, 400, 300], "data:image/jpeg;base64,first");
    const second = output(400, 300, [0, 0, 400, 300], "data:image/jpeg;base64,second");
    listeners.get("executed")({ detail: { node: node.id, output: first } });
    assert.equal(drawImages.at(-1), first.source_preview[0]);
    listeners.get("executed")({ detail: { node: node.id, output: second } });
    assert.equal(drawImages.at(-1), second.source_preview[0]);
});

test("Load & Crop previews a newly selected file before a workflow run", () => {
    const { node, get, drawImages } = fixture("DINKI_Image_Load_Crop");
    get("aspect_ratio").value = "1:1";
    const first = { width: 400, height: 300, uri: "file:first" };
    node.dkstCropSourcePreview(first, "input::first.png");
    assert.equal(drawImages.at(-1), first.uri);
    assert.equal(get("crop_x").value, 0.125);
    assert.equal(get("crop_width").value, 0.75);
    const second = { width: 300, height: 400, uri: "file:second" };
    node.dkstCropSourcePreview(second, "input::second.png");
    assert.equal(drawImages.at(-1), second.uri);
    assert.equal(get("crop_x").value, 0);
    assert.equal(get("crop_y").value, 0.125);
    assert.equal(get("crop_height").value, 0.75);
});

test("Load & Crop uses native numeric controls after its flexible crop preview", () => {
    const { node, get } = fixture("DINKI_Image_Load_Crop");
    const position = name => node.widgets.indexOf(get(name));
    assert.ok(position("aspect_ratio") < position("__dkst_crop_preview"));
    assert.equal(node.widgets.filter(widget => widget.element).length, 1);
    const preview = get("__dkst_crop_preview");
    const [custom, canvas] = preview.element.children;
    assert.equal(custom.style.display, "none");
    assert.equal(canvas.tag, "canvas");
    assert.equal(preview.element.children.length, 3);
    assert.equal(preview.element.children[2].style.display, "none");
    assert.ok(position("crop_mode") < position("aspect_ratio"));
    assert.equal(preview.element.style.height, "100%");
    assert.equal(preview.element.style.contain, "size layout paint");
    assert.equal(canvas.style.height, "0");
    assert.equal(preview.options.getMinHeight(), 240);
    assert.equal(preview.element.style.minHeight, "240px");
    assert.deepEqual(node.widgets.slice(-3).map(widget => widget.name),
        ["__dkst_crop_preview", "resolution_multiple", "megapixels"]);
    assert.equal(get("resolution_multiple").type, "number");
    assert.equal(get("megapixels").type, "number");
    assert.equal(preview.computeSize, undefined);
    const layout = preview.computeLayoutSize();
    assert.equal(layout.minHeight, 240);
    assert.equal(layout.maxHeight, 10000);
    assert.equal(node.size[0], 370);
    assert.equal(node.size[1], 488);
    assert.equal(get("resolution_multiple").hidden, undefined);
    assert.equal(get("megapixels").hidden, undefined);
});

test("Load & Crop shows the selected generation size before execution", () => {
    const { node, get, drawTexts } = fixture("DINKI_Image_Load_Crop");
    get("aspect_ratio").value = "1:1";
    node.dkstCropSourcePreview({ width: 400, height: 300, uri: "file:first" }, "input::first.png");
    const megapixels = get("megapixels");
    megapixels.value = 1.68;
    megapixels.callback(1.68);
    assert.equal(megapixels.value, 1.68);
    assert.ok(drawTexts.includes("Source crop 300 × 300 px"));
    assert.equal(drawTexts.at(-1), "Output 1328 × 1328 px");
    get("resolution_multiple").value = 12;
    get("resolution_multiple").callback(12);
    megapixels.value = 0.98;
    megapixels.callback(0.98);
    assert.equal(drawTexts.at(-1), "Output 1008 × 1008 px");
});

test("Custom controls appear only for Custom and restore without an input image", () => {
    const { node, Node, get } = fixture("DINKI_Image_Load_Crop");
    const preview = get("__dkst_crop_preview");
    const row = preview.element.children[0];
    get("aspect_ratio").value = "Custom";
    get("aspect_ratio").callback("Custom");
    assert.equal(row.style.display, "flex");
    assert.equal(preview.options.getMinHeight(), 276);
    assert.equal(preview.element.style.minHeight, "276px");
    Node.prototype.onConfigure.call(node, { properties: { dkstCropSettings: {
        aspect_ratio: "Custom", custom_width: 16, custom_height: 9,
    } } });
    const [width, , height] = row.children[1].children;
    assert.equal(width.value, "16");
    assert.equal(height.value, "9");
    get("aspect_ratio").value = "16:9";
    get("aspect_ratio").callback("16:9");
    assert.equal(row.style.display, "none");
    assert.equal(preview.options.getMinHeight(), 240);
    assert.equal(get("custom_width").value, 16);
});

test("Load & Crop removes an old unconnected source_type socket", () => {
    const { node, Node } = fixture("DINKI_Image_Load_Crop");
    node.inputs = [{ name: "source_type", link: null },
        { name: "aspect_ratio", link: 42 }];
    node.removeInput = index => node.inputs.splice(index, 1);
    Node.prototype.onConfigure.call(node, { widgets_values: [] });
    assert.deepEqual(node.inputs.map(input => input.name), ["aspect_ratio"]);
    node.inputs.push({ name: "source_type", link: 99 });
    Node.prototype.onConfigure.call(node, { widgets_values: [] });
    assert.equal(node.inputs.at(-1).link, 99);
});

test("Load & Crop restores positional crop values after file selectors", () => {
    const { node, Node, get } = fixture("DINKI_Image_Load_Crop");
    Node.prototype.onConfigure.call(node, { widgets_values: [
        "", "portrait.png", "4:5", 4, 5, 0.1, 0.2, 0.7, 0.8, "16", "2MP",
    ] });
    assert.equal(get("aspect_ratio").value, "4:5");
    assert.equal(get("crop_x").value, 0.1);
    assert.equal(get("crop_height").value, 0.8);
    assert.equal(get("resolution_multiple").value, 16);
    assert.equal(get("megapixels").value, 2);
});

test("Load & Crop preserves size settings when saving the reordered widgets", () => {
    const { node, Node, get } = fixture("DINKI_Image_Load_Crop");
    get("resolution_multiple").value = "32";
    get("megapixels").value = "4MP";
    get("crop_x").value = 0.1;
    get("crop_height").value = 0.8;
    const info = { widgets_values: ["", "portrait.png", "4:5", 4, 5,
        "32", "4MP", 0.1, 0.2, 0.7, 0.8] };
    Node.prototype.onSerialize.call(node, info);
    assert.equal(info.properties.dkstCropSettings.resolution_multiple, "32");
    assert.equal(info.properties.dkstCropSettings.megapixels, "4MP");
    get("resolution_multiple").value = "8";
    get("megapixels").value = "1MP";
    get("crop_x").value = 0;
    get("crop_height").value = 1;
    Node.prototype.onConfigure.call(node, info);
    assert.equal(get("crop_x").value, 0.1);
    assert.equal(get("crop_height").value, 0.8);
    assert.equal(get("resolution_multiple").value, 32);
    assert.equal(get("megapixels").value, 4);
});

test("Load & Crop restores the reordered positional values without named settings", () => {
    const { node, Node, get } = fixture("DINKI_Image_Load_Crop");
    Node.prototype.onConfigure.call(node, { widgets_values: [
        "", "portrait.png", "4:5", 4, 5, "32", "4MP", 0.1, 0.2, 0.7, 0.8,
    ] });
    assert.equal(get("crop_x").value, 0.1);
    assert.equal(get("crop_height").value, 0.8);
    assert.equal(get("resolution_multiple").value, 32);
    assert.equal(get("megapixels").value, 4);
});

test("Load & Crop restores numeric size controls from old and current layouts", () => {
    for (const stored of [
        ["", "portrait.png", "4:5", 4, 5, 12, 0.56, 0.1, 0.2, 0.7, 0.8],
        ["", "portrait.png", "4:5", 4, 5, 0.1, 0.2, 0.7, 0.8, 12, 0.56],
        ["", "portrait.png", "4:5", 4, 5, 0.1, 0.2, 0.7, 0.8, "input", null, 12, 0.56],
        ["input", "portrait.png", "4:5", 4, 5, 0.1, 0.2, 0.7, 0.8, 12, 0.56, "temp"],
    ]) {
        const { node, Node, get } = fixture("DINKI_Image_Load_Crop");
        Node.prototype.onConfigure.call(node, { widgets_values: stored });
        assert.equal(get("crop_x").value, 0.1);
        assert.equal(get("crop_height").value, 0.8);
        assert.equal(get("resolution_multiple").value, 12);
        assert.equal(get("megapixels").value, 0.56);
    }
});

test("Load & Crop restores named crop and fractional size values without an array", () => {
    const { node, Node, get } = fixture("DINKI_Image_Load_Crop");
    Node.prototype.onConfigure.call(node, { widgets_values_named: {
        crop_x: 0.125, crop_height: 0.75, resolution_multiple: 12, megapixels: 0.56,
    } });
    assert.equal(get("crop_x").value, 0.125);
    assert.equal(get("crop_height").value, 0.75);
    assert.equal(get("resolution_multiple").value, 12);
    assert.equal(get("megapixels").value, 0.56);
});

test("serialized named settings survive UI widgets and saved preview does not reset crop", async () => {
    const { node, Node, extension, app, get, output } = fixture();
    node.dkstCropOutput(output(400, 300, [50, 0, 300, 300]));
    const info = { widgets_values: ["1:1", null, 1, 1, null, 0.125, 0, 0.75, 1] };
    Node.prototype.onSerialize.call(node, info);
    assert.equal(info.widgets_values.includes(null), false);
    assert.equal(info.properties.dkstCropSettings.crop_x, 0.125);
    get("crop_x").value = 0;
    Node.prototype.onConfigure.call(node, info);
    assert.equal(get("crop_x").value, 0.125);
    get("crop_x").value = 0.2;
    app.nodeOutputs[node.id] = output(400, 300, [50, 0, 300, 300]);
    extension.loadedGraphNode(node);
    await Promise.resolve();
    assert.equal(get("crop_x").value, 0.2);
});

test("reloading or refreshing workflow preserves user resized node dimensions", () => {
    const { node, Node, extension } = fixture("DINKI_Image_Load_Crop");
    assert.equal(node.size[0], 370);
    assert.equal(node.size[1], 488);
    // ComfyUI saves the user's size in the standard node size field.
    node.size = [550, 750];
    const info = { size: [...node.size], widgets_values: [],
        properties: { dkstCropSize: [370, 472] } };
    Node.prototype.onSerialize.call(node, info);
    assert.deepEqual(info.size, [550, 750]);
    assert.equal(info.properties.dkstCropSize, undefined);

    // Reopen workflow on a new node instance
    const reopened = fixture("DINKI_Image_Load_Crop");
    assert.equal(reopened.node.size[0], 370);
    assert.equal(reopened.node.size[1], 488);

    // A stale size property from an older workflow must not override the native size.
    info.properties.dkstCropSize = [370, 472];
    reopened.Node.prototype.onConfigure.call(reopened.node, info);
    assert.equal(reopened.node.size[0], 550);
    assert.equal(reopened.node.size[1], 750);
    assert.equal(reopened.get("__dkst_crop_preview").computeLayoutSize().minHeight, 240);

    // Graph finished loading event
    extension.loadedGraphNode(reopened.node);
    assert.equal(reopened.node.size[0], 550);
    assert.equal(reopened.node.size[1], 750);
});

test("Expand fits a full portrait inside a wide canvas and previews the output size", () => {
    const { node, get, drawTexts } = fixture("DINKI_Image_Load_Crop");
    node.dkstCropSourcePreview({ width: 1800, height: 3358, uri: "file:portrait" }, "portrait");
    get("aspect_ratio").value = "16:9";
    get("aspect_ratio").callback("16:9");
    get("crop_mode").value = "Expand";
    get("crop_mode").callback("Expand");
    assert.ok(get("crop_x").value < 0);
    assert.ok(get("crop_y").value < 0);
    assert.ok(get("crop_width").value > 1);
    assert.ok(get("crop_height").value >= 1);
    assert.ok(drawTexts.includes("Canvas 5984 × 3366 px"));
    assert.equal(drawTexts.at(-1), "Output 1368 × 768 px");
    const preview = get("__dkst_crop_preview");
    const [custom, canvas, controls] = preview.element.children;
    assert.equal(custom.style.display, "none");
    assert.equal(controls.style.display, "flex");
    assert.equal(preview.options.getMinHeight(), 276);
    const r = preview.imageRect;
    assert.ok(r.x + get("crop_x").value * r.w >= 8);
    assert.ok(r.x + (get("crop_x").value + get("crop_width").value) * r.w <= canvas.clientWidth - 8);

    get("crop_x").value = -3;
    controls.children[0].listeners.click({ preventDefault() {}, stopPropagation() {} });
    assert.ok(Math.abs(get("crop_x").value - (-2092 / 1800)) < 0.000001);
    get("crop_mode").value = "Crop";
    get("crop_mode").callback("Crop");
    assert.ok(get("crop_x").value >= 0);
    assert.ok(get("crop_width").value <= 1);
    assert.equal(controls.style.display, "none");
});

test("Expand drag crosses image bounds, freezes the viewport, and refits on release", () => {
    const { node, get } = fixture("DINKI_Image_Load_Crop");
    get("crop_mode").value = "Expand";
    node.dkstCropSourcePreview({ width: 400, height: 300, uri: "file:image" }, "image");
    const preview = get("__dkst_crop_preview");
    const canvas = preview.element.children[1];
    canvas.displayScale = 0.5;
    const pointer = (x, y) => ({ button: 0, pointerId: 2, clientX: x * 0.5, clientY: y * 0.5,
        preventDefault() {}, stopPropagation() {} });
    const before = { ...preview.imageRect };
    const start = pointer(before.x, before.y);
    canvas.listeners.pointerdown(start);
    canvas.listeners.pointermove(pointer(before.x - before.w / 2, before.y - before.h / 2));
    assert.ok(get("crop_x").value < 0);
    assert.ok(get("crop_y").value < 0);
    assert.ok(get("crop_width").value > 1);
    assert.equal(preview.imageRect.w, before.w);
    assert.equal(preview.imageRect.x, before.x);
    canvas.listeners.pointerup(pointer(before.x - before.w / 2, before.y - before.h / 2));
    assert.ok(preview.imageRect.w < before.w);
    const w = get("crop_width").value * 400, h = get("crop_height").value * 300;
    assert.ok(Math.abs(w / h - 4 / 3) < 0.00001);

    const r = preview.imageRect;
    const cx = r.x + (get("crop_x").value + get("crop_width").value / 2) * r.w;
    const cy = r.y + (get("crop_y").value + get("crop_height").value / 2) * r.h;
    const oldX = get("crop_x").value;
    canvas.listeners.pointerdown(pointer(cx, cy));
    canvas.listeners.pointermove(pointer(cx - r.w, cy));
    assert.ok(get("crop_x").value < oldX);
    canvas.listeners.pointercancel();
    assert.ok(preview.imageRect.x !== r.x);
});

test("Expand restores negative geometry from named and positional saves", () => {
    for (const info of [
        { properties: { dkstCropSettings: { crop_mode: "Expand", aspect_ratio: "16:9",
            crop_x: -1, crop_y: -0.01, crop_width: 3, crop_height: 1.02,
            resolution_multiple: 32, megapixels: 1.22 } } },
        { widgets_values: ["", "portrait.png", "Expand", "16:9", 1, 1, -1, -0.01, 3, 1.02, "input", 32, 1.22] },
        { widgets_values: ["", "portrait.png", "16:9", 1, 1, -1, -0.01, 3, 1.02, 32, 1.22, "input", "Expand"] },
    ]) {
        const { node, Node, get, output } = fixture("DINKI_Image_Load_Crop");
        Node.prototype.onConfigure.call(node, info);
        assert.equal(get("crop_mode").value, "Expand");
        assert.equal(get("crop_x").value, -1);
        assert.equal(get("crop_width").value, 3);
        assert.equal(get("resolution_multiple").value, 32);
        assert.equal(get("megapixels").value, 1.22);
        node.dkstCropOutput(output(300, 400, [-150, -25, 600, 450]));
        assert.equal(get("crop_x").value, -0.5);
        assert.equal(get("crop_height").value, 1.125);
        const saved = {};
        Node.prototype.onSerialize.call(node, saved);
        assert.equal(saved.properties.dkstCropSettings.crop_mode, "Expand");
        assert.equal(saved.properties.dkstCropSettings.crop_x, -0.5);
    }
});

test("a new file in Expand refits the full source and legacy saves default to Crop", () => {
    const { node, Node, get } = fixture("DINKI_Image_Load_Crop");
    get("crop_mode").value = "Expand";
    get("aspect_ratio").value = "1:1";
    node.dkstCropSourcePreview({ width: 300, height: 400, uri: "file:portrait" }, "first");
    assert.ok(get("crop_width").value > 1);
    node.dkstCropSourcePreview({ width: 600, height: 300, uri: "file:landscape" }, "second");
    assert.equal(get("crop_x").value, 0);
    assert.equal(get("crop_y").value, -0.5);
    assert.equal(get("crop_height").value, 2);
    Node.prototype.onConfigure.call(node, { widgets_values: [
        "", "portrait.png", "4:5", 4, 5, 0.1, 0.2, 0.7, 0.8, 16, 2,
    ] });
    assert.equal(get("crop_mode").value, "Crop");
    assert.equal(get("crop_x").value, 0.1);
});

test("Expand x offsets resembling old resolution multiples restore as coordinates", () => {
    for (const cropX of [4, 8, 32, 128]) {
        const { node, Node, get } = fixture("DINKI_Image_Load_Crop");
        Node.prototype.onConfigure.call(node, { widgets_values: [
            "", "portrait.png", "Expand", "16:9", 1, 1, cropX, -0.2, 3, 1.5, 16, 0.56,
        ] });
        assert.equal(get("crop_x").value, cropX);
        assert.equal(get("crop_y").value, -0.2);
        assert.equal(get("crop_width").value, 3);
        assert.equal(get("resolution_multiple").value, 16);
        assert.equal(get("megapixels").value, 0.56);
    }
});
