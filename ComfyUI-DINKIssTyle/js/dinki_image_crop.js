import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASSES = new Set(["DINKI_Image_Crop", "DINKI_Image_Load_Crop"]);
const STORED = ["aspect_ratio", "custom_width", "custom_height", "crop_x", "crop_y", "crop_width", "crop_height"];
const SIZE_CONTROLS = ["resolution_multiple", "megapixels"];
const megapixelsNumber = value => Number(String(value).replace(/MP$/, ""));
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function targetSize(width, height, node) {
    const megapixels = megapixelsNumber(node.widgets.find(widget => widget.name === "megapixels")?.value);
    const multiple = Number(node.widgets.find(widget => widget.name === "resolution_multiple")?.value);
    if (!width || !height || !Number.isFinite(megapixels) || megapixels <= 0 ||
        !Number.isInteger(multiple) || multiple < 4 || multiple > 128 || multiple % 4) return null;
    const area = megapixels * 1024 * 1024;
    const ratio = width / height;
    const roundToMultiple = value => {
        const scaled = value / multiple;
        const floor = Math.floor(scaled);
        const fraction = scaled - floor;
        const rounded = fraction < 0.5 ? floor : fraction > 0.5 ? floor + 1 : floor + floor % 2;
        return Math.max(multiple, rounded * multiple);
    };
    const targetHeight = Math.sqrt(area / ratio);
    return [roundToMultiple(targetHeight * ratio), roundToMultiple(targetHeight)];
}

function removeUnusedSourceTypeSocket(node) {
    const index = node.inputs?.findIndex(input =>
        input.name === "source_type" && input.link == null) ?? -1;
    if (index >= 0) node.removeInput?.(index);
}

function gcd(a, b) {
    while (b) [a, b] = [b, a % b];
    return a;
}

function ratioParts(node, width, height) {
    const value = node.widgets.find(widget => widget.name === "aspect_ratio")?.value;
    if (value === "Original") {
        const divisor = gcd(width, height);
        return [width / divisor, height / divisor];
    }
    let a, b;
    if (value === "Custom") {
        a = Number(node.widgets.find(widget => widget.name === "custom_width")?.value);
        b = Number(node.widgets.find(widget => widget.name === "custom_height")?.value);
    } else {
        [a, b] = String(value).split(":").map(Number);
    }
    if (!Number.isInteger(a) || !Number.isInteger(b) || a < 1 || b < 1) return [width, height];
    const divisor = gcd(a, b);
    return [a / divisor, b / divisor];
}

function fittedSize(width, height, a, b, maxWidth, maxHeight) {
    const multiple = Math.min(Math.floor(width / a), Math.floor(height / b),
        Math.round(Math.min(maxWidth / a, maxHeight / b)));
    if (multiple >= 1) return [a * multiple, b * multiple];
    // Tiny crops cannot represent every ratio with exact integer dimensions.
    let w = clamp(Math.round(maxWidth), 1, width);
    let h = Math.max(1, Math.round(w * b / a));
    if (h > maxHeight) {
        h = clamp(Math.round(maxHeight), 1, height);
        w = Math.max(1, Math.round(h * a / b));
    }
    return [clamp(w, 1, width), clamp(h, 1, height)];
}

function values(node) {
    return Object.fromEntries(STORED.map(name => [name, node.widgets.find(widget => widget.name === name)]));
}

function rectFromWidgets(widgets) {
    return {
        x: Number(widgets.crop_x.value), y: Number(widgets.crop_y.value),
        w: Number(widgets.crop_width.value), h: Number(widgets.crop_height.value),
    };
}

function setWidget(node, widget, value) {
    if (!widget || widget.value === value) return;
    const old = widget.value;
    widget.value = value;
    widget.callback?.(value, app.canvas, node);
    node.onWidgetChanged?.(widget.name, value, old, widget);
}

function setRect(node, widgets, rect) {
    const safe = {
        x: clamp(rect.x, 0, 1), y: clamp(rect.y, 0, 1),
        w: clamp(rect.w, 0.000001, 1), h: clamp(rect.h, 0.000001, 1),
    };
    safe.x = Math.min(safe.x, 1 - safe.w);
    safe.y = Math.min(safe.y, 1 - safe.h);
    for (const [name, value] of Object.entries({
        crop_x: safe.x, crop_y: safe.y, crop_width: safe.w, crop_height: safe.h,
    })) setWidget(node, widgets[name], Math.round(value * 1000000) / 1000000);
    node.graph?.incrementVersion?.();
    node.setDirtyCanvas?.(true, true);
    node.dkstCropRender?.();
}

function centeredRect(node, width, height) {
    const [a, b] = ratioParts(node, width, height);
    const [w, h] = fittedSize(width, height, a, b, width, height);
    return { x: (width - w) / (2 * width), y: (height - h) / (2 * height),
        w: w / width, h: h / height };
}

function customRatioControls(node, state, widgets) {
    const row = document.createElement("div");
    row.style.cssText = "display:none;align-items:center;gap:8px;flex:0 0 28px;min-height:28px;box-sizing:border-box;width:100%";
    const label = document.createElement("span");
    label.textContent = "Custom ratio";
    label.style.cssText = "flex:0 0 84px;font-size:12px";
    const fields = document.createElement("div");
    fields.style.cssText = "display:flex;align-items:center;gap:6px;flex:1;min-width:0";
    const sources = [widgets.custom_width, widgets.custom_height];
    const inputs = sources.map((source, index) => {
        const input = document.createElement("input");
        input.type = "number";
        input.min = "1";
        input.max = "10000";
        input.step = "1";
        input.value = String(source.value);
        input.setAttribute("aria-label", index === 0 ? "Custom width ratio" : "Custom height ratio");
        input.style.cssText = "flex:1;width:0;min-width:0;height:28px;box-sizing:border-box;background:var(--comfy-input-bg,#32343a);color:var(--input-text,#eee);border:1px solid var(--border-color,#555);border-radius:5px;padding:0 8px;font:inherit";
        input.addEventListener("change", () => {
            const number = clamp(Math.round(Number(input.value) || 1), 1, 10000);
            input.value = String(number);
            setWidget(node, source, number);
            if (widgets.aspect_ratio.value === "Custom" && state.width) {
                setRect(node, widgets, centeredRect(node, state.width, state.height));
            }
        });
        return input;
    });
    const colon = document.createElement("span");
    colon.textContent = ":";
    fields.append(inputs[0], colon, inputs[1]);
    row.append(label, fields);
    return { row, sync() {
        const visible = widgets.aspect_ratio.value === "Custom";
        row.style.display = visible ? "flex" : "none";
        inputs.forEach((input, index) => { input.value = String(sources[index].value); });
    } };
}

function cropPreviewWidget(node, state, widgets) {
    const root = document.createElement("div");
    Object.assign(root.style, { width: "100%", height: "100%",
        flex: "1 1 0", display: "flex", flexDirection: "column",
        gap: "8px", color: "var(--fg-color,#bbb)", font: "12px sans-serif",
        overflow: "hidden", boxSizing: "border-box", contain: "size layout paint" });
    // Vue nodes stretch every DOM widget's grid row independently. Keep the
    // fixed controls and the flexible canvas in one widget to avoid empty rows.
    const custom = customRatioControls(node, state, widgets);
    const canvas = document.createElement("canvas");
    Object.assign(canvas.style, { width: "100%", height: "0", minHeight: "0",
        flex: "1 1 0", display: "block", borderRadius: "6px",
        touchAction: "none", cursor: "default" });
    root.append(custom.row, canvas);
    const controlsHeight = () => custom.row.style.display === "none" ? 0 : 36;
    const sync = () => {
        custom.sync();
        // WidgetDOM in Vue nodes does not apply the legacy height callbacks.
        // A CSS minimum keeps the preview usable there as well as on canvas nodes.
        root.style.minHeight = `${240 + controlsHeight()}px`;
    };
    sync();
    node.dkstCropSync = sync;
    let imageRect = null;
    let active = null;
    const widget = node.addDOMWidget("__dkst_crop_preview", "dkst-crop-preview", root, {
        hideOnZoom: false, getMinHeight: () => 240 + controlsHeight(),
        getMaxHeight: () => 10000, getHeight: () => 300 + controlsHeight(),
    });
    widget.serialize = false;
    widget.options ??= {};
    widget.options.serialize = false;

    function render() {
        sync();
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        if (!width || !height) return;
        const dpr = globalThis.devicePixelRatio || 1;
        const pixelWidth = Math.round(width * dpr);
        const pixelHeight = Math.round(height * dpr);
        if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
            canvas.width = pixelWidth;
            canvas.height = pixelHeight;
        }
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#15171a";
        ctx.fillRect(0, 0, width, height);
        if (!state.image || !state.width || !state.height) {
            imageRect = null;
            widget.imageRect = null;
            ctx.fillStyle = "#a0a4ac";
            ctx.font = "12px sans-serif";
            ctx.textAlign = "center";
            const message = node.comfyClass === "DINKI_Image_Load_Crop" ?
                "Select an image to preview and crop" : "Run the node to preview the input image";
            ctx.fillText(message, width / 2, height / 2);
            return;
        }
        const inset = 8;
        const availableWidth = Math.max(1, width - inset * 2);
        const loadCrop = node.comfyClass === "DINKI_Image_Load_Crop";
        const availableHeight = Math.max(1, height - (loadCrop ? 32 : 18) - inset * 2);
        const scale = Math.min(availableWidth / state.image.width, availableHeight / state.image.height);
        const iw = state.image.width * scale;
        const ih = state.image.height * scale;
        const ix = (width - iw) / 2;
        const iy = inset + (availableHeight - ih) / 2;
        imageRect = { x: ix, y: iy, w: iw, h: ih };
        widget.imageRect = imageRect;
        ctx.drawImage(state.image, ix, iy, iw, ih);
        const rect = rectFromWidgets(widgets);
        const rx = ix + rect.x * iw, ry = iy + rect.y * ih;
        const rw = rect.w * iw, rh = rect.h * ih;
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        ctx.fillRect(ix, iy, iw, Math.max(0, ry - iy));
        ctx.fillRect(ix, ry + rh, iw, Math.max(0, iy + ih - ry - rh));
        ctx.fillRect(ix, ry, Math.max(0, rx - ix), rh);
        ctx.fillRect(rx + rw, ry, Math.max(0, ix + iw - rx - rw), rh);
        ctx.strokeStyle = "#f5f7fa";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(rx + 0.75, ry + 0.75, Math.max(1, rw - 1.5), Math.max(1, rh - 1.5));
        ctx.fillStyle = "#f5f7fa";
        for (const [hx, hy] of [[rx, ry], [rx + rw, ry], [rx, ry + rh], [rx + rw, ry + rh]]) {
            ctx.fillRect(hx - 4, hy - 4, 8, 8);
        }
        const [a, b] = ratioParts(node, state.width, state.height);
        const [outW, outH] = fittedSize(state.width, state.height, a, b,
            Math.max(1, Math.round(rect.w * state.width)),
            Math.max(1, Math.round(rect.h * state.height)));
        ctx.fillStyle = "#bfc3cb";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "right";
        const sizeLabel = loadCrop ?
            `Source crop ${outW} × ${outH} px` : `${outW} × ${outH} px`;
        ctx.fillText(sizeLabel, width - 4, height - (loadCrop ? 18 : 4));
        if (loadCrop) {
            const output = targetSize(outW, outH, node);
            if (output) ctx.fillText(`Output ${output[0]} × ${output[1]} px`, width - 4, height - 4);
        }
    }

    function point(event) {
        const bounds = canvas.getBoundingClientRect();
        if (!bounds.width || !bounds.height) return null;
        return { x: (event.clientX - bounds.left) * canvas.clientWidth / bounds.width,
            y: (event.clientY - bounds.top) * canvas.clientHeight / bounds.height };
    }

    function hit(position) {
        if (!position || !imageRect) return null;
        const rect = rectFromWidgets(widgets);
        const corners = [
            [rect.x, rect.y, "tl"], [rect.x + rect.w, rect.y, "tr"],
            [rect.x, rect.y + rect.h, "bl"], [rect.x + rect.w, rect.y + rect.h, "br"],
        ];
        const corner = corners.find(([x, y]) => Math.hypot(
            position.x - imageRect.x - x * imageRect.w,
            position.y - imageRect.y - y * imageRect.h) <= 11);
        if (corner) return corner[2];
        return position.x >= imageRect.x + rect.x * imageRect.w &&
            position.x <= imageRect.x + (rect.x + rect.w) * imageRect.w &&
            position.y >= imageRect.y + rect.y * imageRect.h &&
            position.y <= imageRect.y + (rect.y + rect.h) * imageRect.h ? "move" : null;
    }

    function drag(event) {
        if (!active || !imageRect) return;
        const position = point(event);
        if (!position) return;
        const x = (position.x - imageRect.x) / imageRect.w;
        const y = (position.y - imageRect.y) / imageRect.h;
        const { mode, current, downX, downY, ratio } = active;
        if (mode === "move") {
            setRect(node, widgets, { ...current,
                x: clamp(current.x + x - downX, 0, 1 - current.w),
                y: clamp(current.y + y - downY, 0, 1 - current.h) });
            return;
        }
        const anchorX = mode.includes("l") ? current.x + current.w : current.x;
        const anchorY = mode.includes("t") ? current.y + current.h : current.y;
        const signX = mode.includes("l") ? -1 : 1;
        const signY = mode.includes("t") ? -1 : 1;
        const desiredW = Math.max(0, signX * (x - anchorX));
        const desiredH = Math.max(0, signY * (y - anchorY));
        const maxW = signX < 0 ? anchorX : 1 - anchorX;
        const maxH = signY < 0 ? anchorY : 1 - anchorY;
        const maxAllowedW = Math.min(maxW, maxH * ratio);
        const newW = clamp((desiredW + desiredH * ratio) / 2,
            Math.min(maxAllowedW, Math.max(1 / state.width, ratio / state.height)), maxAllowedW);
        const newH = newW / ratio;
        setRect(node, widgets, { x: signX < 0 ? anchorX - newW : anchorX,
            y: signY < 0 ? anchorY - newH : anchorY, w: newW, h: newH });
    }

    canvas.addEventListener("pointerdown", event => {
        if (event.button !== 0) return;
        const position = point(event);
        const mode = hit(position);
        if (!mode) return;
        const current = rectFromWidgets(widgets);
        const [a, b] = ratioParts(node, state.width, state.height);
        active = { mode, current,
            downX: (position.x - imageRect.x) / imageRect.w,
            downY: (position.y - imageRect.y) / imageRect.h,
            ratio: (a / b) * (state.height / state.width) };
        canvas.setPointerCapture?.(event.pointerId);
        event.preventDefault();
        event.stopPropagation();
    });
    canvas.addEventListener("pointermove", event => {
        if (active) {
            drag(event);
            event.preventDefault();
            event.stopPropagation();
        } else {
            const mode = hit(point(event));
            canvas.style.cursor = mode === "move" ? "move" :
                mode === "tl" || mode === "br" ? "nwse-resize" :
                    mode ? "nesw-resize" : "default";
        }
    });
    canvas.addEventListener("pointerup", event => {
        if (!active) return;
        drag(event);
        active = null;
        canvas.releasePointerCapture?.(event.pointerId);
        event.preventDefault();
        event.stopPropagation();
    });
    canvas.addEventListener("pointercancel", () => { active = null; });
    const observer = typeof ResizeObserver === "function" ? new ResizeObserver(render) : null;
    observer?.observe(root);
    widget.render = render;
    widget.destroy = () => observer?.disconnect();
    queueMicrotask(render);
    return widget;
}

app.registerExtension({
    name: "DINKIssTyle.ImageCrop",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_CLASSES.has(nodeData.name)) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalCreated?.apply(this, args);
            const node = this;
            const widgets = values(node);
            if (STORED.some(name => !widgets[name])) return result;
            const state = { image: null, width: 0, height: 0, uri: null,
                generation: 0, lastOutputRect: null };
            const preview = cropPreviewWidget(node, state, widgets);
            node.dkstCropRender = () => preview.render();
            node.widgets.splice(node.widgets.indexOf(preview), 1);
            node.widgets.push(preview);
            if (nodeData.name === "DINKI_Image_Load_Crop") {
                for (const name of SIZE_CONTROLS) {
                    const control = node.widgets.find(widget => widget.name === name);
                    if (!control) continue;
                    node.widgets.splice(node.widgets.indexOf(control), 1);
                    node.widgets.push(control);
                    const originalCallback = control.callback;
                    control.callback = function(value, ...callbackArgs) {
                        originalCallback?.call(this, value, ...callbackArgs);
                        preview.render();
                    };
                }
            }
            const hiddenControls = ["custom_width", "custom_height", "crop_x", "crop_y",
                "crop_width", "crop_height"];
            if (nodeData.name === "DINKI_Image_Load_Crop") {
                hiddenControls.push("source_type");
            }
            for (const name of hiddenControls) {
                const control = node.widgets.find(widget => widget.name === name);
                if (!control) continue;
                control.hidden = true;
                control.options ??= {};
                control.options.hidden = true;
                control.computeSize = () => [0, -4];
                control.draw = () => {};
            }

            if (nodeData.name === "DINKI_Image_Load_Crop") removeUnusedSourceTypeSocket(node);
            const originalRatioCallback = widgets.aspect_ratio.callback;
            widgets.aspect_ratio.callback = function (value, ...callbackArgs) {
                originalRatioCallback?.call(this, value, ...callbackArgs);
                node.dkstCropSync?.();
                if (state.width) setRect(node, widgets, centeredRect(node, state.width, state.height));
                else preview.render();
                node.expandToFitContent?.();
                node.setDirtyCanvas?.(true, true);
            };
            if (nodeData.name === "DINKI_Image_Load_Crop") {
                node.dkstCropSourcePreview = (image, sourceKey = null) => {
                    if (!image) {
                        state.image = null;
                        state.width = 0;
                        state.height = 0;
                        preview.render();
                        return;
                    }
                    const width = image.naturalWidth || image.width;
                    const height = image.naturalHeight || image.height;
                    if (!width || !height) return;
                    const changed = state.sourceKey != null && state.sourceKey !== sourceKey;
                    const first = !state.width;
                    state.sourceKey = sourceKey;
                    state.image = image;
                    state.width = width;
                    state.height = height;
                    const rect = rectFromWidgets(widgets);
                    if (changed || (first && rect.x === 0 && rect.y === 0 &&
                            rect.w === 1 && rect.h === 1)) {
                        setRect(node, widgets, centeredRect(node, width, height));
                    }
                    preview.render();
                };
            }
            node.dkstCropOutput = (output, restoreOnly = false) => {
                if (!output || typeof output !== "object") return;
                const size = output.source_size?.[0];
                const uri = output.source_preview?.[0];
                if (!Array.isArray(size) || !Number.isFinite(size[0]) ||
                    !Number.isFinite(size[1]) || typeof uri !== "string") return;
                const crop = output.crop_rect?.[0];
                const rectKey = Array.isArray(crop) ? crop.join(",") : "";
                const newOutput = uri !== state.uri || rectKey !== state.lastOutputRect;
                state.lastOutputRect = rectKey;
                state.width = size[0]; state.height = size[1];
                if (newOutput && !restoreOnly && Array.isArray(crop) && crop.length === 4) {
                    setRect(node, widgets, {
                        x: crop[0] / state.width, y: crop[1] / state.height,
                        w: crop[2] / state.width, h: crop[3] / state.height,
                    });
                }
                if (uri === state.uri) return;
                state.uri = uri;
                state.image = null;
                preview.render();
                const generation = ++state.generation;
                const image = new Image();
                image.onload = () => {
                    if (generation !== state.generation) return;
                    state.image = image;
                    preview.render();
                    node.setDirtyCanvas?.(true, true);
                };
                image.onerror = () => {
                    if (generation === state.generation) {
                        state.image = null;
                        preview.render();
                    }
                };
                image.src = uri;
            };
            node.dkstCropRestore = () => {
                node.dkstCropSync?.();
                preview.render();
                node.expandToFitContent?.();
                node.setDirtyCanvas?.(true, true);
            };
            const executedHandler = ({ detail }) => {
                if (String(detail?.node) === String(node.id) ||
                    String(detail?.display_node) === String(node.id)) {
                    node.dkstCropOutput(detail.output);
                }
            };
            api.addEventListener("executed", executedHandler);
            const originalRemoved = node.onRemoved;
            node.onRemoved = function (...removedArgs) {
                state.generation++;
                preview.destroy();
                api.removeEventListener("executed", executedHandler);
                delete node.dkstCropOutput;
                delete node.dkstCropRender;
                delete node.dkstCropSync;
                delete node.dkstCropSourcePreview;
                return originalRemoved?.apply(this, removedArgs);
            };
            // The graph restores its own serialized size after node creation.
            // Only establish a useful size for a newly added node here.
            node.size[0] = Math.max(node.size[0], 370);
            node.size[1] = Math.max(node.size[1],
                (nodeData.name === "DINKI_Image_Load_Crop" ? 164 : 64) +
                preview.options.getHeight());
            return result;
        };
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = originalConfigure?.apply(this, arguments);
            const widgets = values(this);
            const named = info?.properties?.dkstCropSettings;
            const namedValues = info?.widgets_values_named;
            const loadCrop = nodeData.name === "DINKI_Image_Load_Crop";
            // Old layouts place size controls before crop coordinates; current
            // layouts place them last, sometimes after the hidden source_type.
            const list = (Array.isArray(info?.widgets_values) ? info.widgets_values : []).filter(value => value != null);
            if (loadCrop) {
                const sourceIndex = [9, 11].find(index => list[index] === "input" || list[index] === "temp");
                if (sourceIndex !== undefined) list.splice(sourceIndex, 1);
            }
            const offset = loadCrop ? 2 : 0;
            const earlyMultiple = Number(list[5]);
            const sizeFirst = loadCrop && Number.isInteger(earlyMultiple) &&
                earlyMultiple >= 4 && earlyMultiple <= 128 && earlyMultiple % 4 === 0;
            const positional = Object.fromEntries(STORED.map((name, index) =>
                [name, list[offset + index + (sizeFirst && index >= 3 ? 2 : 0)]]));
            for (const name of STORED) {
                const candidate = named?.[name] ?? namedValues?.[name] ?? positional[name];
                if (widgets[name] && candidate !== undefined) widgets[name].value = candidate;
            }
            if (nodeData.name === "DINKI_Image_Load_Crop") {
                removeUnusedSourceTypeSocket(this);
                const sizeIndex = sizeFirst ? 5 : 9;
                const sizeValues = {
                    resolution_multiple: named?.resolution_multiple ??
                        namedValues?.resolution_multiple ?? list[sizeIndex],
                    megapixels: named?.megapixels ??
                        namedValues?.megapixels ?? list[sizeIndex + 1],
                };
                for (const name of SIZE_CONTROLS) {
                    const widget = this.widgets?.find(item => item.name === name);
                    if (!widget) continue;
                    const candidate = sizeValues[name] ?? widget.value;
                    const numeric = name === "megapixels" ? megapixelsNumber(candidate) : Number(candidate);
                    const valid = name === "megapixels"
                        ? Number.isFinite(numeric) && numeric >= 0.1 && numeric <= 64
                        : Number.isInteger(numeric) && numeric >= 4 && numeric <= 128 && numeric % 4 === 0;
                    if (valid) widget.value = numeric;
                }
            }
            this.dkstCropRestore?.();
            return result;
        };
        const originalSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            const result = originalSerialize?.apply(this, arguments);
            if (info) {
                if (Array.isArray(info.widgets_values)) {
                    info.widgets_values = info.widgets_values.filter(value => value != null);
                }
                info.properties ??= {};
                const names = nodeData.name === "DINKI_Image_Load_Crop" ?
                    [...STORED, ...SIZE_CONTROLS] : STORED;
                info.properties.dkstCropSettings = Object.fromEntries(names.map(name =>
                    [name, this.widgets.find(widget => widget.name === name)?.value]));
                delete info.properties.dkstCropSize;
            }
            return result;
        };
        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = originalExecuted?.apply(this, arguments);
            this.dkstCropOutput?.(output);
            return result;
        };
    },
    loadedGraphNode(node) {
        if (!NODE_CLASSES.has(node.comfyClass)) return;
        queueMicrotask(() => node.dkstCropOutput?.(app.nodeOutputs?.[node.id], true));
    },
});
