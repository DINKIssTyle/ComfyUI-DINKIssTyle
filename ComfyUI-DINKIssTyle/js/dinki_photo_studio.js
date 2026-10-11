import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { NODE_CLASS, isSetting, settingsOf, setNodeValue, openPhotoStudio, request, workflowImages, inputSignature } from "./dinki_photo_studio_editor.js";

const LEGACY_DEFAULTS = { lens_blur_aperture_blades: 9, lens_blur_depth_blur_radius: 5, lens_blur_depth_sigma: 2.0 };
function report(error) {
    const message = error?.message ?? String(error);
    if (app.ui?.dialog?.show) app.ui.dialog.show(message);
    else window.alert(message);
}

app.registerExtension({
    name: "DINKIssTyle.PhotoStudio",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Image-node context menu opens an editor using the selected preview.
        // Apply creates a separate Photo Studio branch; existing links stay put.
        const originalMenu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const result = originalMenu?.apply(this, arguments);
            const candidates = workflowImages(this.graph);
            const images = app.nodeOutputs?.[this.id]?.images ?? this.imgs ?? [];
            const selected = this.imageIndex ?? this.overIndex ?? 0;
            const index = Number.isInteger(selected) && selected >= 0 && selected < images.length ? selected : 0;
            const candidate = candidates.find(c => c.previewNodeId === this.id && c.index === index);
            if (nodeData.name === NODE_CLASS) options.push({ content: "Open Photo Studio", callback: () => openPhotoStudio(this).catch(report) });
            else if (candidate) options.push({ content: "Open in Photo Studio", callback: () => openPhotoStudio(null, candidate).catch(report) });
            return result;
        };
        if (nodeData.name !== NODE_CLASS) return;
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = originalConfigure?.apply(this, arguments);
            this.dkstPhotoStudioCompact?.();
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
            const preset = node.widgets?.find(w => w.name === "preset");
            if (!preset) return result;
            const active = node.widgets.find(w => w.name === "active");
            if (active) active.label = "Active";
            preset.label = "Preset";
            for (const control of node.widgets.filter(w => isSetting(w.name))) {
                control.hidden = true;
                // Retain widgets and their serializeValue hooks for API prompts
                // and promoted inputs, but remove their graph-canvas rows.
                if (!String(control.type).startsWith("converted")) control.type = "hidden";
                control.computeSize = () => [0, -4];
            }
            let presets = {}, defaults = {};
            const options = () => {
                preset.type = "combo";
                preset.options ??= {};
                preset.options.values = ["Custom", "Default", ...Object.keys(presets).sort()];
                node.setDirtyCanvas?.(true, true);
            };
            node.dkstPhotoStudioPresetData = values => { presets = values; options(); };
            const originalPreset = preset.callback;
            preset.callback = function (name, ...args) {
                originalPreset?.call(this, name, ...args);
                const values = name === "Default" ? defaults : presets[name];
                if (!values) return;
                for (const [key, value] of Object.entries(values)) {
                    if (!node.inputs?.some(i => (i.name === key || i.widget?.name === key) && i.link != null)) setNodeValue(node, key, value);
                }
                node.graph?.incrementVersion?.(); node.setDirtyCanvas?.(true, true);
            };
            options();
            const open = node.addWidget("button", "Open Photo Studio", null, () => openPhotoStudio(node).catch(report), { serialize: false });
            open.serialize = false;
            if (node.addDOMWidget) {
                const box = document.createElement("div");
                Object.assign(box.style, { width: "100%", height: "100%", minHeight: "100px", display: "flex", alignItems: "center", justifyContent: "center", background: "#171b20", overflow: "hidden", borderRadius: "6px" });
                const image = document.createElement("img");
                image.alt = "Photo Studio preview";
                Object.assign(image.style, { width: "100%", height: "100%", objectFit: "contain", display: "none" });
                const hint = document.createElement("span");
                hint.textContent = "Open editor to choose an image";
                Object.assign(hint.style, { color: "#98a8b9", font: "12px system-ui", padding: "10px", textAlign: "center" });
                box.append(image, hint);
                image.addEventListener("load", () => { image.style.display = "block"; hint.style.display = "none"; });
                const preview = node.addDOMWidget("__dkst_photo_preview", "dkst_photo_preview", box, {
                    serialize: false, hideOnZoom: false, getMinHeight: () => 110, getMaxHeight: () => 160, getHeight: () => 120,
                });
                preview.serialize = false;
                node.dkstPhotoStudioThumbnail = image;
            }
            const compute = node.computeSize;
            if (compute) node.computeSize = function () {
                const size = compute.apply(this, arguments);
                return [Math.max(280, size[0]), Math.max(280, (this.inputs?.length || 2) * 22 + 190)];
            };
            node.dkstPhotoStudioCompact = () => {
                const size = node.computeSize?.() || [300, 280];
                if (node.setSize) node.setSize([Math.max(300, size[0]), size[1]]);
                else node.size = [Math.max(300, size[0]), size[1]];
            };
            const output = data => {
                if (data?.photo_studio_preview_error) delete node.dkstPhotoStudioSource;
                if (Array.isArray(data?.photo_studio_source)) { node.dkstPhotoStudioSource = data.photo_studio_source[0]; node.dkstPhotoStudioInputSignature = inputSignature(node); }
                if (data?.photo_studio_preview?.[0] && node.dkstPhotoStudioThumbnail) node.dkstPhotoStudioThumbnail.src = data.photo_studio_preview[0];
            };
            node.dkstPhotoStudioOutput = output;
            const handler = ({ detail }) => { if (String(detail?.node) === String(node.id)) output(detail.output); };
            api.addEventListener("executed", handler);
            const removed = node.onRemoved;
            node.onRemoved = function (...args) {
                api.removeEventListener("executed", handler);
                delete node.dkstPhotoStudioSource;
                return removed?.apply(this, args);
            };
            node.dkstPhotoStudioCompact();
            request("presets").then(data => { presets = data.presets; defaults = data.defaults; options(); }).catch(report);
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
        queueMicrotask(() => { node.dkstPhotoStudioCompact?.(); node.dkstPhotoStudioOutput?.(app.nodeOutputs?.[node.id]); });
    },
});
