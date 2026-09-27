const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const source = readFileSync(join(__dirname, "../ComfyUI-DINKIssTyle/js/dinki_image_crop.js"), "utf8");

function fixture(className = "DINKI_Image_Crop") {
    let extension;
    const listeners = new Map();
    const app = { registerExtension(value) { extension = value; }, nodeOutputs: {} };
    const api = {
        addEventListener(name, fn) { listeners.set(name, fn); },
        removeEventListener(name) { listeners.delete(name); },
    };
    class FakeImage {
        width = 400;
        height = 300;
        set src(value) { this.uri = value; this.onload?.(); }
    }
    const drawImages = [];
    const context = new Proxy({ drawImage(image) { drawImages.push(image.uri); } }, { get(target, key) {
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
                { name: "resolution_multiple", value: "8" },
                { name: "megapixels", value: "1MP" },
            ] : []),
        ],
        graph: { incrementVersion() {} }, setDirtyCanvas() {}, expandToFitContent() {},
        addCustomWidget(widget) { this.widgets.push(widget); return widget; },
        addDOMWidget(name, type, element, options) {
            const widget = { name, type, element, options };
            this.widgets.push(widget);
            return widget;
        },
    };
    Node.prototype.onNodeCreated.call(node);
    const get = name => node.widgets.find(widget => widget.name === name);
    const output = (width, height, rect, uri = "data:preview") => ({
        source_size: [[width, height]], crop_rect: [rect], source_preview: [uri],
    });
    return { app, extension, Node, node, get, output, listeners, observers, drawImages };
}

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

test("Load & Crop keeps resolution controls below the canvas in one flexible widget", () => {
    const { node, get } = fixture("DINKI_Image_Load_Crop");
    const position = name => node.widgets.indexOf(get(name));
    assert.ok(position("aspect_ratio") < position("__dkst_crop_preview"));
    assert.equal(node.widgets.filter(widget => widget.element).length, 1);
    const preview = get("__dkst_crop_preview");
    const [custom, canvas, footer] = preview.element.children;
    const [resolutionRow, megapixelsRow] = footer.children;
    assert.equal(custom.style.display, "none");
    assert.equal(canvas.tag, "canvas");
    assert.equal(resolutionRow.children[0].textContent, "Multiple");
    assert.equal(megapixelsRow.children[0].textContent, "Megapixels");
    assert.equal(preview.element.style.height, "100%");
    assert.equal(preview.element.style.contain, "size layout paint");
    assert.equal(canvas.style.height, "0");
    assert.equal(preview.options.getMinHeight(), 308);
    assert.equal(preview.element.style.minHeight, "308px");
    assert.equal(node.widgets.filter(widget => !widget.hidden).at(-1).name,
        "__dkst_crop_preview");
    assert.equal(get("resolution_multiple").hidden, true);
    assert.equal(get("megapixels").hidden, true);
    megapixelsRow.children[1].value = "3MP";
    megapixelsRow.children[1].listeners.change();
    assert.equal(get("megapixels").value, "3MP");
    get("resolution_multiple").value = "16";
    preview.render();
    assert.equal(resolutionRow.children[1].value, "16");
});

test("Custom controls appear only for Custom and restore without an input image", () => {
    const { node, Node, get } = fixture("DINKI_Image_Load_Crop");
    const preview = get("__dkst_crop_preview");
    const row = preview.element.children[0];
    get("aspect_ratio").value = "Custom";
    get("aspect_ratio").callback("Custom");
    assert.equal(row.style.display, "flex");
    assert.equal(preview.options.getMinHeight(), 344);
    assert.equal(preview.element.style.minHeight, "344px");
    Node.prototype.onConfigure.call(node, { properties: { dkstCropSettings: {
        aspect_ratio: "Custom", custom_width: 16, custom_height: 9,
    } } });
    const [width, , height] = row.children[1].children;
    assert.equal(width.value, "16");
    assert.equal(height.value, "9");
    get("aspect_ratio").value = "16:9";
    get("aspect_ratio").callback("16:9");
    assert.equal(row.style.display, "none");
    assert.equal(preview.options.getMinHeight(), 308);
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
    assert.equal(get("resolution_multiple").value, "16");
    assert.equal(get("megapixels").value, "2MP");
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
    assert.equal(get("resolution_multiple").value, "32");
    assert.equal(get("megapixels").value, "4MP");
});

test("Load & Crop restores the reordered positional values without named settings", () => {
    const { node, Node, get } = fixture("DINKI_Image_Load_Crop");
    Node.prototype.onConfigure.call(node, { widgets_values: [
        "", "portrait.png", "4:5", 4, 5, "32", "4MP", 0.1, 0.2, 0.7, 0.8,
    ] });
    assert.equal(get("crop_x").value, 0.1);
    assert.equal(get("crop_height").value, 0.8);
    assert.equal(get("resolution_multiple").value, "32");
    assert.equal(get("megapixels").value, "4MP");
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
