import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "DINKI_Photo_Studio";
const ENDPOINT = "/dinki/photo-studio/presets";
const SECTIONS = [
    ["light_exposure", "Light"],
    ["color_white_balance", "Color"],
    ["effects_texture", "Effects"],
    ["detail_sharpening", "Detail"],
    ["optics_distortion", "Optics"],
    ["lens_blur_apply", "Lens Blur"],
];
const EXTRA_LABELS = {
    depth_near_is_white: "Depth: white is near",
    grain_seed: "Grain seed",
    lens_blur_apply: "Apply",
    lens_blur_focus: "Focus",
    lens_blur_amount: "Blur Amount",
    lens_blur_bokeh: "Bokeh (f-number)",
    lens_blur_aperture_blades: "Aperture Blades",
    lens_blur_bokeh_boost: "Bokeh Boost",
    lens_blur_depth_blur_radius: "Depth Blur Radius",
    lens_blur_depth_sigma: "Depth Sigma",
};
const LEGACY_DEFAULTS = {
    lens_blur_aperture_blades: 9,
    lens_blur_depth_blur_radius: 5,
    lens_blur_depth_sigma: 2.0,
};

function friendlyName(name) {
    if (EXTRA_LABELS[name]) return EXTRA_LABELS[name];
    return name.split("_").slice(1).join(" ").replace(/\b\w/g, letter => letter.toUpperCase());
}

function sectionWidget(title) {
    return {
        type: "dkst_photo_section",
        name: `__dkst_photo_section_${title.toLowerCase().replace(/\s+/g, "_")}`,
        value: null,
        serialize: false,
        options: { serialize: false },
        computeSize(width) { return [width, 26]; },
        draw(ctx, node, width, y, height) {
            ctx.save();
            ctx.font = "600 12px sans-serif";
            ctx.fillStyle = "#aeb4be";
            ctx.textAlign = "left";
            const label = `• ${title}`;
            ctx.fillText(label, 12, y + height - 7);
            const start = 20 + ctx.measureText(label).width;
            if (start < width - 12) {
                ctx.strokeStyle = "#555b63";
                ctx.beginPath();
                ctx.moveTo(start, y + height - 11);
                ctx.lineTo(width - 12, y + height - 11);
                ctx.stroke();
            }
            ctx.restore();
        },
    };
}

function depthFocusWidget(node, focus, nearIsWhite, preset, setValue) {
    let image = null;
    let pixels = null;
    let point = null;
    const widget = {
        type: "dkst_depth_focus",
        name: "__dkst_depth_focus",
        value: null,
        serialize: false,
        options: { serialize: false },
        imageRect: null,
        computeSize(width) { return [width, 210]; },
        clearPoint() {
            point = null;
            node.setDirtyCanvas?.(true, true);
        },
        setPreview(nextImage, nextPixels) {
            image = nextImage;
            pixels = nextPixels;
            point = null;
            this.imageRect = null;
            node.setDirtyCanvas?.(true, true);
        },
        draw(ctx, graphNode, width, y, height) {
            const left = 8;
            const top = y + 4;
            const boxWidth = width - 16;
            const boxHeight = height - 8;
            ctx.save();
            ctx.fillStyle = "#15171a";
            ctx.fillRect(left, top, boxWidth, boxHeight);
            if (!image || !pixels) {
                this.imageRect = null;
                ctx.fillStyle = "#959ba5";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Run node to preview depth map", left + boxWidth / 2,
                    top + boxHeight / 2);
                ctx.restore();
                return;
            }
            const scale = Math.min(boxWidth / image.width, boxHeight / image.height);
            const drawnWidth = image.width * scale;
            const drawnHeight = image.height * scale;
            const x = left + (boxWidth - drawnWidth) / 2;
            const py = top + (boxHeight - drawnHeight) / 2;
            this.imageRect = { x, y: py, width: drawnWidth, height: drawnHeight };
            ctx.drawImage(image, x, py, drawnWidth, drawnHeight);
            if (point) {
                const px = x + point.x * drawnWidth;
                const fy = py + point.y * drawnHeight;
                ctx.beginPath();
                ctx.arc(px, fy, 8, 0, Math.PI * 2);
                ctx.lineWidth = 3;
                ctx.strokeStyle = "#101114";
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(px, fy, 7, 0, Math.PI * 2);
                ctx.lineWidth = 2;
                ctx.strokeStyle = "#fff";
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(px - 12, fy);
                ctx.lineTo(px + 12, fy);
                ctx.moveTo(px, fy - 12);
                ctx.lineTo(px, fy + 12);
                ctx.lineWidth = 1;
                ctx.stroke();
            }
            ctx.restore();
        },
        focusAt(canvasX, canvasY) {
            if (!pixels || !image || !this.imageRect || !focus) return false;
            const rect = this.imageRect;
            const x = Math.max(0, Math.min(1, (canvasX - node.pos[0] - rect.x) / rect.width));
            const y = Math.max(0, Math.min(1, (canvasY - node.pos[1] - rect.y) / rect.height));
            const ix = Math.min(image.width - 1, Math.floor(x * image.width));
            const iy = Math.min(image.height - 1, Math.floor(y * image.height));
            const raw = pixels[(iy * image.width + ix) * 4] / 255;
            const value = nearIsWhite?.value === false ? 1 - raw : raw;
            setValue(focus, Math.round(value * 100) / 100);
            point = { x, y };
            if (preset.value !== "Custom") setValue(preset, "Custom");
            node.graph?.incrementVersion?.();
            node.setDirtyCanvas?.(true, true);
            return true;
        },
        onPointerDown(pointer) {
            const event = pointer.eDown;
            const rect = this.imageRect;
            if (!rect || !pixels || !event || event.button !== 0) return false;
            const x = event.canvasX - node.pos[0];
            const y = event.canvasY - node.pos[1];
            if (x < rect.x || x > rect.x + rect.width ||
                y < rect.y || y > rect.y + rect.height) return false;
            this.focusAt(event.canvasX, event.canvasY);
            pointer.onDrag = move => this.focusAt(move.canvasX, move.canvasY);
            pointer.onDragEnd = up => this.focusAt(up.canvasX, up.canvasY);
            pointer.onClick = up => this.focusAt(up.canvasX, up.canvasY);
            return true;
        },
        mouse(event, pos) {
            if (!pixels || !this.imageRect) return false;
            const pressed = event.type === "mousedown" || event.type === "pointerdown";
            const moving = (event.type === "mousemove" || event.type === "pointermove") &&
                event.buttons === 1;
            if ((pressed && event.button === 0) || moving) {
                return this.focusAt(node.pos[0] + pos[0], node.pos[1] + pos[1]);
            }
            return false;
        },
    };
    return widget;
}

function report(error) {
    const message = error?.message ?? String(error);
    if (app.ui?.dialog?.show) app.ui.dialog.show(message);
    else window.alert(message);
}

async function fetchPresets() {
    const response = await api.fetchApi(ENDPOINT);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load presets");
    return data;
}

app.registerExtension({
    name: "DINKIssTyle.PhotoStudio",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = originalConfigure?.apply(this, arguments);
            this.expandToFitContent?.();
            const serializable = this.widgets?.filter(widget =>
                widget.name === "active" || widget.name === "preset" ||
                /^(light|color|effects|detail|optics|lens_blur|depth_near_is_white|grain_seed)/.test(widget.name)) ?? [];
            const named = info?.properties?.dkstPhotoStudioSettings ??
                info?.dkst_photo_settings ?? info?.widgets_values_named;
            if (named && typeof named === "object" && !Array.isArray(named)) {
                for (const widget of serializable) {
                    if (Object.hasOwn(named, widget.name)) widget.value = named[widget.name];
                }
                return result;
            }
            if (!Array.isArray(info?.widgets_values)) return result;
            // Earlier workflows may contain null slots for buttons or section
            // headers and a trailing hidden Auto flag. No setting uses null.
            const values = info.widgets_values.filter(value => value !== null);
            if (typeof values.at(-1) === "boolean") values.pop();
            const missing = serializable.length - values.length;
            if (missing < 0 || missing > 3) return result;
            const omitted = new Set();
            if (missing === 1 || missing === 3) omitted.add("lens_blur_aperture_blades");
            if (missing === 2 || missing === 3) {
                omitted.add("lens_blur_depth_blur_radius");
                omitted.add("lens_blur_depth_sigma");
            }
            const stored = serializable.filter(widget => !omitted.has(widget.name));
            if (stored.length !== values.length) return result;
            stored.forEach((widget, index) => { widget.value = values[index]; });
            for (const name of omitted) {
                const widget = serializable.find(candidate => candidate.name === name);
                if (widget) widget.value = LEGACY_DEFAULTS[name];
            }
            return result;
        };
        const originalSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            const result = originalSerialize?.apply(this, arguments);
            if (info) {
                // ComfyUI versions that leave null holes for serialize:false
                // widgets read those same arrays back as compacted values.
                if (Array.isArray(info.widgets_values)) {
                    info.widgets_values = info.widgets_values.filter(value => value != null);
                }
                info.properties ??= {};
                info.properties.dkstPhotoStudioSettings = Object.fromEntries((this.widgets ?? [])
                    .filter(widget => widget.name === "active" || widget.name === "preset" ||
                        /^(light|color|effects|detail|optics|lens_blur|depth_near_is_white|grain_seed)/.test(widget.name))
                    .map(widget => [widget.name, widget.value]));
            }
            return result;
        };
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            const node = this;
            const widgets = node.widgets ?? [];
            const preset = widgets.find(widget => widget.name === "preset");
            if (!preset) return result;
            preset.label = "Preset";
            const active = widgets.find(widget => widget.name === "active");
            if (active) active.label = "Active";
            const settingsWidgets = widgets.filter(widget =>
                /^(light|color|effects|detail|optics|lens_blur|depth_near_is_white|grain_seed)/.test(widget.name));
            for (const widget of settingsWidgets) widget.label = friendlyName(widget.name);

            let presets = {};
            let defaults = {};
            let autoSuggestion = null;
            function setValue(widget, value) {
                if (!widget) return;
                const oldValue = widget.value;
                widget.value = value;
                widget.callback?.(value, app.canvas, node);
                node.onWidgetChanged?.(widget.name, value, oldValue, widget);
            }
            function applyValues(values) {
                for (const widget of settingsWidgets) {
                    if (Object.hasOwn(values, widget.name)) setValue(widget, values[widget.name]);
                }
                node.graph?.incrementVersion?.();
                node.setDirtyCanvas?.(true, true);
            }
            function options() {
                preset.type = "combo";
                preset.options ??= {};
                preset.options.values = ["Custom", "Default", ...Object.keys(presets).sort()];
                node.setDirtyCanvas?.(true, true);
            }
            const originalCallback = preset.callback;
            preset.callback = function (name, ...args) {
                originalCallback?.call(this, name, ...args);
                const values = name === "Default" ? defaults : presets[name];
                if (!values) return;
                applyValues(values);
            };
            options();

            async function refresh() {
                const data = await fetchPresets();
                presets = data.presets;
                defaults = data.defaults;
                options();
                // A saved workflow already carries its own settings. Loading
                // the preset list must never replace those persisted values.
            }
            async function save(name, overwrite) {
                if (!name) return;
                const settings = Object.fromEntries(settingsWidgets.map(widget => [widget.name, widget.value]));
                const response = await api.fetchApi(ENDPOINT, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name, settings, overwrite }),
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || "Could not save preset");
                presets = data.presets;
                preset.value = name;
                options();
            }
            const saveButton = node.addWidget("button", "Save", null, async () => {
                try {
                    if (preset.value === "Custom" || preset.value === "Default") {
                        const name = window.prompt("New preset name");
                        if (name !== null) await save(name.trim(), false);
                    } else {
                        await save(preset.value, true);
                    }
                } catch (error) { report(error); }
            }, { serialize: false });
            const saveAsButton = node.addWidget("button", "Save As", null, async () => {
                try {
                    const name = window.prompt("New preset name", preset.value === "Custom" ? "" : `${preset.value} Copy`);
                    if (name !== null) await save(name.trim(), false);
                } catch (error) { report(error); }
            }, { serialize: false });
            const autoButton = node.addWidget("button", "Auto", null, () => {
                if (!autoSuggestion) {
                    report(new Error("Run this node once with an image before using Auto."));
                    return;
                }
                applyValues({ ...autoSuggestion, color_white_balance: "Auto" });
                setValue(preset, "Custom");
            }, { serialize: false });
            const resetButton = node.addWidget("button", "Reset", null, async () => {
                try {
                    if (!Object.keys(defaults).length) await refresh();
                    applyValues(defaults);
                    setValue(preset, "Default");
                } catch (error) { report(error); }
            }, { serialize: false });
            const buttons = [saveButton, saveAsButton, autoButton, resetButton];
            for (const button of buttons) button.serialize = false;
            // Keep the preset actions beside their selector. The buttons are
            // created in this same position before workflow configuration.
            for (const button of buttons) node.widgets.splice(node.widgets.indexOf(button), 1);
            node.widgets.splice(node.widgets.indexOf(preset) + 1, 0, ...buttons);
            for (const [firstName, title] of SECTIONS) {
                const first = settingsWidgets.find(widget => widget.name === firstName);
                if (!first) continue;
                const marker = node.addCustomWidget(sectionWidget(title));
                node.widgets.splice(node.widgets.indexOf(marker), 1);
                node.widgets.splice(node.widgets.indexOf(first), 0, marker);
            }
            const focus = widgets.find(widget => widget.name === "lens_blur_focus");
            const nearIsWhite = widgets.find(widget => widget.name === "depth_near_is_white");
            const depthWidget = node.addCustomWidget(depthFocusWidget(
                node, focus, nearIsWhite, preset, setValue));
            node.widgets.splice(node.widgets.indexOf(depthWidget), 1);
            node.widgets.splice(node.widgets.indexOf(focus), 0, depthWidget);
            for (const setting of [focus, nearIsWhite]) {
                if (!setting) continue;
                const callback = setting.callback;
                setting.callback = function (...args) {
                    callback?.apply(this, args);
                    depthWidget.clearPoint();
                };
            }
            let depthGeneration = 0;
            let depthUri = null;
            function loadDepthPreview(uri) {
                if (uri === depthUri) return;
                depthUri = uri;
                const generation = ++depthGeneration;
                depthWidget.setPreview(null, null);
                if (!uri) return;
                const image = new Image();
                image.onload = () => {
                    if (generation !== depthGeneration) return;
                    const canvas = document.createElement("canvas");
                    canvas.width = image.naturalWidth || image.width;
                    canvas.height = image.naturalHeight || image.height;
                    const context = canvas.getContext("2d", { willReadFrequently: true });
                    if (!context) return;
                    context.drawImage(image, 0, 0);
                    depthWidget.setPreview(image,
                        context.getImageData(0, 0, canvas.width, canvas.height).data);
                };
                image.onerror = () => {
                    if (generation === depthGeneration) depthWidget.setPreview(null, null);
                };
                image.src = uri;
            }
            function applyExecutionOutput(output) {
                if (!output || typeof output !== "object") return;
                const preview = output.depth_preview;
                if (Array.isArray(preview)) loadDepthPreview(preview[0]);
                const values = output.auto_light_suggestion?.[0];
                if (values && typeof values === "object") autoSuggestion = values;
            }
            node.dkstPhotoStudioOutput = applyExecutionOutput;
            const executedHandler = ({ detail }) => {
                if (String(detail?.node) === String(node.id)) applyExecutionOutput(detail.output);
            };
            api.addEventListener("executed", executedHandler);
            const originalRemoved = node.onRemoved;
            node.onRemoved = function (...args) {
                depthGeneration++;
                api.removeEventListener("executed", executedHandler);
                delete node.dkstPhotoStudioOutput;
                return originalRemoved?.apply(this, args);
            };
            node.expandToFitContent?.();
            refresh().catch(report);
            return result;
        };
        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = originalExecuted?.apply(this, arguments);
            this.dkstPhotoStudioOutput?.(output);
            return result;
        };
    },
    loadedGraphNode(node) {
        if (node.comfyClass !== NODE_CLASS) return;
        queueMicrotask(() => node.dkstPhotoStudioOutput?.(app.nodeOutputs?.[node.id]));
    },
});
