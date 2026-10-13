import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ROOT = "/dinki/photo-studio";
export const NODE_CLASS = "DINKI_Photo_Studio";
export const isSetting = name => /^(light_|color_|effects_|detail_|optics_|lens_blur_|depth_near_is_white$|grain_seed$)/.test(name);
export const settingsOf = node => Object.fromEntries((node?.widgets ?? []).filter(w => isSetting(w.name)).map(w => [w.name, w.value]));
const widget = (node, name) => node?.widgets?.find(w => w.name === name);
let currentEditor = null;
let closeImageMenu;

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = filename;
    document.body.appendChild(link);
    try { link.click(); } finally { link.remove(); setTimeout(() => URL.revokeObjectURL(url), 60000); }
}

export function photoImageActions(getUrl, getFilename = () => "DKST_Photo_Studio.png") {
    return [
        { content: "Open Image", disabled: !getUrl(), callback: async () => {
            const url = getUrl();
            if (!url) return;
            if (!url.startsWith("data:")) return window.open(url, "_blank", "noopener,noreferrer");
            const popup = window.open("", "_blank");
            if (popup) popup.opener = null;
            try {
                const objectUrl = URL.createObjectURL(await (await fetch(url)).blob());
                if (popup) popup.location.href = objectUrl;
                setTimeout(() => URL.revokeObjectURL(objectUrl), 60000);
            } catch (error) { popup?.close(); throw error; }
        } },
        { content: "Save Image", disabled: !getUrl(), callback: async () => {
            const url = getUrl();
            if (!url) return;
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Image download failed (${response.status})`);
            downloadBlob(await response.blob(), getFilename());
        } },
    ];
}

export function showPhotoImageMenu(event, actions) {
    event.preventDefault(); event.stopPropagation(); event.stopImmediatePropagation?.();
    closeImageMenu?.();
    const menu = document.createElement("div");
    menu.className = "dkst-photo-context-menu";
    Object.assign(menu.style, { position: "fixed", zIndex: "100001", background: "#252b33", color: "#eee",
        border: "1px solid #63748a", borderRadius: "6px", padding: "5px",
        left: `${Math.max(0, Math.min(event.clientX, window.innerWidth - 200))}px`,
        top: `${Math.max(0, Math.min(event.clientY, window.innerHeight - 150))}px` });
    const dismiss = e => { if (!menu.contains(e.target)) close(); };
    const escape = e => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
    const close = () => {
        menu.remove(); document.removeEventListener("pointerdown", dismiss, true);
        document.removeEventListener("keydown", escape, true);
        if (closeImageMenu === close) closeImageMenu = null;
    };
    closeImageMenu = close;
    for (const action of actions) {
        const item = document.createElement("button");
        item.textContent = action.content; item.type = "button"; item.disabled = !!action.disabled;
        Object.assign(item.style, { display: "block", width: "100%", background: "transparent", color: "inherit",
            border: "0", textAlign: "left", padding: "9px 12px", cursor: "pointer" });
        item.onclick = async () => { close(); try { await action.callback(); } catch (error) { window.alert(error.message); } };
        menu.appendChild(item);
    }
    document.body.appendChild(menu);
    document.addEventListener("pointerdown", dismiss, true); document.addEventListener("keydown", escape, true);
}

export function setNodeValue(node, name, value, invokeCallback = true) {
    const w = widget(node, name);
    if (!w) return;
    const previous = w.value;
    w.value = value;
    if (invokeCallback) w.callback?.(value, app.canvas, node);
    node.onWidgetChanged?.(name, value, previous, w);
}

export function descriptorOf(value) {
    if (value?.filename) return { filename: value.filename, subfolder: value.subfolder || "", type: value.type || "input" };
    const src = typeof value === "string" ? value : value?.src;
    if (!src) return null;
    try {
        const url = new URL(src, window.location.href);
        if (!url.pathname.endsWith("/view")) return null;
        const filename = url.searchParams.get("filename");
        return filename ? { filename, subfolder: url.searchParams.get("subfolder") || "", type: url.searchParams.get("type") || "output" } : null;
    } catch { return null; }
}

export function imageUrl(descriptor) {
    return api.apiURL(`/view?${new URLSearchParams(descriptor)}`);
}

function inputDescriptor(value) {
    if (typeof value !== "string" || !value) return null;
    const annotated = value.match(/ \[(input|output|temp)\]$/);
    const name = annotated ? value.slice(0, annotated.index) : value;
    const slash = name.lastIndexOf("/");
    return { filename: name.slice(slash + 1), subfolder: slash < 0 ? "" : name.slice(0, slash), type: annotated?.[1] || "input" };
}

export function linkedImage(node, name = "image") {
    const input = node?.inputs?.find(i => i.name === name);
    const link = input?.link != null ? node.graph?.links?.[input.link] : null;
    const source = link && node.graph?.getNodeById(link.origin_id);
    return source ? { node: source, slot: link.origin_slot } : null;
}

export function previewImages(node) {
    if (node.comfyClass === NODE_CLASS || node.type === NODE_CLASS) {
        const image = node.dkstPhotoStudioImage ?? node.properties?.dkstPhotoStudioPreview;
        return image ? [image] : app.nodeOutputs?.[node.id]?.photo_studio_image ?? [];
    }
    if (node.comfyClass === "DINKI_Preview_Image" || node.type === "DINKI_Preview_Image") {
        // The DOM Viewer restores its own saved preview before the output cache.
        // Prefer the descriptors actually displayed, including a restored batch.
        if (node.dkstPreviewImages != null) return node.dkstPreviewImages;
        const stored = node.properties?.dkstPreview?.dkst_images;
        if (stored?.length) return stored;
        return app.nodeOutputs?.[node.id]?.dkst_images ?? app.nodeOutputs?.[node.id]?.images ?? [];
    }
    return app.nodeOutputs?.[node.id]?.images ?? node.imgs ?? [];
}

export function selectedImageCandidate(node) {
    const images = previewImages(node);
    const selected = node.dkstPreviewImageIndex ?? node.properties?.dkstPreview?.selectedIndex ??
        node.imageIndex ?? node.overIndex ?? 0;
    const index = Number.isInteger(selected) && selected >= 0 && selected < images.length ? selected : 0;
    return workflowImages(node.graph).find(c => c.previewNodeId === node.id && c.index === index) || null;
}

// Only resolve actual IMAGE ports. Never substitute a sampler's ancestor input
// for the sampler output, or bypass an unexecuted crop/composite.
export function workflowImages(graph = app.graph) {
    const candidates = [];
    for (const node of graph?._nodes ?? []) {
        const outputs = (node.outputs ?? []).map((o, slot) => ({ o, slot })).filter(({ o }) => o.type === "IMAGE");
        if (node.comfyClass === "LoadImage" || node.type === "LoadImage" || node.comfyClass === "DINKI_Photo_Studio_Source" || node.type === "DINKI_Photo_Studio_Source") {
            const descriptor = inputDescriptor(widget(node, "image")?.value);
            if (descriptor && outputs.length) candidates.push({ descriptor, nodeId: node.id, slot: outputs[0].slot, label: node.title || "Load Image", previewNodeId: node.id, index: 0, direct: true, loader: (node.comfyClass === "LoadImage" || node.type === "LoadImage") ? "LoadImage" : "PhotoStudioSource", interpretation: widget(node, "interpretation")?.value });
            continue;
        }
        const images = previewImages(node);
        images.forEach((image, index) => {
            const descriptor = descriptorOf(image);
            if (!descriptor) return;
            let origin = outputs.length === 1 ? { node, slot: outputs[0].slot } : null;
            if (!outputs.length) origin = linkedImage(node, "images") || linkedImage(node, "image");
            candidates.push({ descriptor, nodeId: origin?.node.id, slot: origin?.slot,
                label: `${node.title || node.comfyClass || node.type} · ${index + 1}`, previewNodeId: node.id, index,
                // A file snapshot is used when a port would produce an entire
                // batch or the UI does not identify which IMAGE output it is.
                direct: !!origin && images.length === 1 });
        });
    }
    return candidates;
}

export function inputSignature(node) {
    const visited = new Set();
    const collect = source => {
        if (!source || visited.has(source.id)) return null;
        visited.add(source.id);
        return [source.id, source.type, (source.widgets || []).filter(w => w.serialize !== false).map(w => [w.name, w.value]),
            (source.inputs || []).map(input => {
                const link = source.graph?.links?.[input.link];
                return link ? [input.name, link.origin_slot, collect(source.graph.getNodeById(link.origin_id))] : null;
            })];
    };
    return JSON.stringify(["image", "depth_image"].map(name => {
        const linked = linkedImage(node, name);
        return linked ? [linked.slot, collect(linked.node)] : null;
    }));
}

export function candidateFor(node, name = "image") {
    const linked = linkedImage(node, name);
    if (!linked) return null;
    return workflowImages(node.graph).find(c => c.nodeId === linked.node.id && c.slot === linked.slot && c.direct) || null;
}

export async function request(path, payload) {
    const response = await api.fetchApi(`${ROOT}/${path}`, payload === undefined ? undefined : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    let data;
    try { data = await response.json(); } catch { throw new Error(`Photo Studio request failed (${response.status})`); }
    if (!response.ok) { const error = new Error(data.error || `Photo Studio request failed (${response.status})`); error.status = response.status; throw error; }
    return data;
}

function element(tag, className, text, parent) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text != null) el.textContent = text;
    parent?.appendChild(el);
    return el;
}
function button(text, parent, action) {
    const el = element("button", "", text, parent);
    el.type = "button";
    el.addEventListener("click", action);
    return el;
}
function labelFor(name) {
    const special = { depth_near_is_white: "Depth: white is near", grain_seed: "Grain seed", lens_blur_bokeh: "Bokeh (f-number)", lens_blur_aperture_blades: "Aperture blades", lens_blur_depth_blur_radius: "Depth blur radius", lens_blur_depth_sigma: "Depth sigma" };
    return special[name] || name.replace(/^(light|color|effects|detail|optics|lens_blur)_/, "").replaceAll("_", " ").replace(/^./, s => s.toUpperCase());
}
function ensureStyles() {
    if (document.getElementById("dkst-photo-styles")) return;
    const link = element("link", null, null, document.head);
    link.id = "dkst-photo-styles";
    link.rel = "stylesheet";
    link.href = new URL("./dinki_photo_studio.css", import.meta.url).href;
}

function createNode(graph, type, position) {
    const node = globalThis.LiteGraph?.createNode(type);
    if (!node) throw new Error(`Node type ${type} is unavailable`);
    node.pos = position;
    graph.add(node);
    node.dkstPhotoStudioCompact?.();
    queueMicrotask(() => { if (node.graph === graph) node.dkstPhotoStudioCompact?.(); });
    return node;
}

export function connectCandidate(graph, candidate, target, inputName, created = []) {
    if (!candidate) return;
    let source = candidate.direct ? graph.getNodeById(candidate.nodeId) : null;
    if (candidate.direct && !source) throw new Error("The selected image node was removed. Choose an image again.");
    let slot = candidate.slot;
    if (!source) {
        source = createNode(graph, "DINKI_Photo_Studio_Source", [target.pos[0] - 330, target.pos[1] + (inputName === "depth_image" ? 260 : 0)]);
        created.push(source);
        const d = candidate.descriptor;
        setNodeValue(source, "image", `${d.subfolder ? `${d.subfolder}/` : ""}${d.filename}${d.type === "input" ? "" : ` [${d.type}]`}`);
        setNodeValue(source, "interpretation", inputName === "depth_image" ? "Depth map" : "Image");
        slot = source.outputs?.findIndex(o => o.type === "IMAGE");
    }
    const input = target.inputs?.findIndex(i => i.name === inputName);
    if (input == null || input < 0 || slot == null || slot < 0 || !source.outputs?.[slot] || source.outputs[slot].type !== "IMAGE") throw new Error(`Cannot connect ${inputName}`);
    if (!source.connect(slot, target, input)) throw new Error(`Could not connect ${inputName}`);
}

export async function openPhotoStudio(node = null, sourceCandidate = null) {
    if (currentEditor) { currentEditor.root.focus(); return currentEditor; }
    ensureStyles();
    const editor = new PhotoStudioEditor(node, sourceCandidate);
    currentEditor = editor;
    try { await editor.start(); } catch (error) { editor.message(error.message, true); }
    return editor;
}

export class PhotoStudioEditor {
    constructor(node, candidate) {
        this.node = node;
        this.graph = node?.graph || app.graph;
        this.imageCandidate = candidate || candidateFor(node);
        this.depthCandidate = candidateFor(node, "depth_image");
        this.depthChanged = false;
        this.sourceChanged = !!candidate;
        this.capture = candidate || (node?.dkstPhotoStudioInputSignature && node.dkstPhotoStudioInputSignature !== inputSignature(node)) ? null : node?.dkstPhotoStudioSource;
        this.depthCapture = this.capture;
        this.appGraph = app.graph;
        this.session = null;
        this.closed = false;
        this.sequence = 0;
        this.sourceSequence = 0;
        this.history = [];
        this.redoHistory = [];
        this.controls = new Map();
        this.zoomValue = "fit";
        this.comparisonPosition = 50;
        this.previousFocus = document.activeElement;
        this.root = element("div", "dkst-photo-overlay", null, document.body);
        this.root.tabIndex = -1;
        this.root.setAttribute("role", "dialog");
        this.root.setAttribute("aria-modal", "true");
        this.root.setAttribute("aria-label", "DKST Photo Studio");
        this.root.innerHTML = `<section class="dkst-photo-editor">
            <header><div class="dkst-photo-title">DKST <strong>Photo Studio</strong><small>Non-destructive adjustments</small></div><div class="dkst-photo-actions"></div></header>
            <div class="dkst-photo-body"><main><div class="dkst-photo-image-tools"></div><div class="dkst-photo-view-frame"><div class="dkst-photo-viewport"><div class="dkst-photo-pan-area"><div class="dkst-photo-pair"><img class="dkst-photo-before" alt="Original image"><img class="dkst-photo-after" alt="Adjusted image"><img class="dkst-photo-depth" alt="Focus depth map"><div class="dkst-photo-divider"></div><span class="dkst-photo-before-label">Before</span><span class="dkst-photo-after-label">After</span></div></div><div class="dkst-photo-empty">Choose an image from your workflow or import a file.</div></div><div class="dkst-photo-minimap" hidden><div class="dkst-photo-map" tabindex="0" role="group" aria-label="Photo Studio minimap. Click or drag to navigate; arrow keys to pan."><img alt="Image overview"><div class="dkst-photo-map-window"></div></div></div></div><div class="dkst-photo-view-tools"></div></main><aside><div class="dkst-photo-histogram"><canvas width="256" height="66" aria-label="RGB histogram"></canvas></div><div class="dkst-photo-depth-tools"></div><div class="dkst-photo-panels"></div></aside></div>
            <footer><div class="dkst-photo-status" role="status" aria-live="polite"></div><div class="dkst-photo-footer-actions"></div></footer>
        </section>`;
        const query = selector => this.root.querySelector(selector);
        this.status = query(".dkst-photo-status");
        this.before = query(".dkst-photo-before");
        this.after = query(".dkst-photo-after");
        this.depth = query(".dkst-photo-depth");
        this.pair = query(".dkst-photo-pair");
        this.viewport = query(".dkst-photo-viewport");
        this.panArea = query(".dkst-photo-pan-area");
        this.divider = query(".dkst-photo-divider");
        this.minimap = query(".dkst-photo-minimap");
        this.map = query(".dkst-photo-map");
        this.mapImage = this.map.querySelector("img");
        this.mapWindow = query(".dkst-photo-map-window");
        this.empty = query(".dkst-photo-empty");
        this.histogram = query("canvas");
        this.focusMarker = element("span", "dkst-photo-focus-marker", null, query(".dkst-photo-viewport"));
        this.focusMarker.hidden = true;
        this.panels = query(".dkst-photo-panels");
        const actions = query(".dkst-photo-actions");
        this.preset = element("select", "", null, actions);
        this.preset.setAttribute("aria-label", "Preset");
        this.preset.addEventListener("change", () => this.change({ ...this.defaults, ...(this.preset.value === "Default" ? {} : this.presets[this.preset.value]) }, this.preset.value));
        button("Save preset", actions, () => this.savePreset());
        button("Save as…", actions, () => this.savePreset(true));
        button("Auto", actions, () => { if (this.auto) this.change({ ...this.draft, ...this.auto }); else this.message("Choose an image first.", true); }).title = "Adjust tone gently while preserving white balance and highlight detail.";
        button("Reset", actions, () => this.change({ ...this.defaults }, "Default"));
        this.undoButton = button("Undo", actions, () => this.undo());
        this.redoButton = button("Redo", actions, () => this.redo());
        const imageTools = query(".dkst-photo-image-tools");
        button("Choose image…", imageTools, () => this.choose("image"));
        button("Import image…", imageTools, () => this.importFile("image"));
        this.refreshInput = button("Refresh input", imageTools, () => {
            if (this.node) { this.capture = this.node.dkstPhotoStudioInputSignature === inputSignature(this.node) ? this.node.dkstPhotoStudioSource : null; this.depthCapture = this.capture; this.imageCandidate = candidateFor(this.node); }
            this.loadSource();
        });
        this.imageLabel = element("span", "dkst-photo-source-label", "", imageTools);
        this.exportTarget = element("select", "", null, imageTools);
        this.exportTarget.setAttribute("aria-label", "Image to open or save");
        this.exportTarget.innerHTML = '<option value="adjusted">Adjusted image</option><option value="original">Original image</option>';
        this.openImageButton = button("Open Image", imageTools, () => this.exportImage(false));
        this.saveImageButton = button("Save Image", imageTools, () => this.exportImage(true));
        this.openImageButton.disabled = this.saveImageButton.disabled = true;
        this.saveImageButton.title = "Save a PNG at the original image resolution";
        const depthTools = query(".dkst-photo-depth-tools");
        element("strong", "", "Depth map", depthTools);
        const depthButtons = element("div", "", null, depthTools);
        button("Choose…", depthButtons, () => this.choose("depth"));
        button("Import…", depthButtons, () => this.importFile("depth"));
        button("Clear", depthButtons, () => this.select(null, "depth"));
        this.depthLabel = element("small", "", "Optional · required for Lens Blur", depthTools);
        const viewTools = query(".dkst-photo-view-tools");
        this.compare = button("Before / After", viewTools, () => {
            this.pair.classList.toggle("compare");
            this.compare.classList.toggle("selected", this.pair.classList.contains("compare"));
            this.updateComparison(); this.updateFocusMarker();
        });
        this.comparisonMode = element("select", "", null, viewTools);
        this.comparisonMode.setAttribute("aria-label", "Comparison mode");
        this.comparisonMode.innerHTML = '<option value="Slide">Slide</option><option value="Difference">Difference</option>';
        this.comparisonMode.addEventListener("change", () => {
            this.pair.classList.add("compare"); this.compare.classList.add("selected"); this.updateComparison();
        });
        this.showDepth = button("Show depth", viewTools, () => { this.pair.classList.toggle("show-depth"); this.updateComparison(); this.updateMinimap(); this.updateFocusMarker(); });
        this.focusButton = button("Pick focus", viewTools, () => { this.pickFocus = !this.pickFocus; this.focusButton.classList.toggle("selected", this.pickFocus); this.pair.classList.toggle("pick-focus", this.pickFocus); });
        this.quality = element("select", "", null, viewTools);
        this.quality.setAttribute("aria-label", "Preview quality");
        this.quality.innerHTML = '<option value="fast">Fast preview</option><option value="original">Original resolution processing</option>';
        this.quality.addEventListener("change", () => this.schedule());
        this.zoom = element("select", "", null, viewTools);
        this.zoom.setAttribute("aria-label", "Preview zoom");
        this.zoom.innerHTML = '<option value="fit">Fit</option><option value="0.25">25%</option><option value="0.5">50%</option><option value="1">100%</option><option value="1.5">150%</option><option value="2">200%</option><option value="3">300%</option><option value="4">400%</option>';
        this.zoom.addEventListener("change", () => this.setZoom(this.zoom.value));
        this.pair.addEventListener("click", e => this.focusAt(e));
        this.pair.addEventListener("pointermove", event => {
            if (this.pickFocus || !this.pair.classList.contains("compare") || this.comparisonMode.value !== "Slide") return;
            const bounds = this.pair.getBoundingClientRect();
            if (bounds.width) this.comparisonPosition = Math.max(0, Math.min(100, (event.clientX - bounds.left) / bounds.width * 100));
            this.updateComparison();
        });
        this.viewport.addEventListener("scroll", () => { this.updateMinimap(); this.updateFocusMarker(); });
        this.installMinimap();
        const footer = query(".dkst-photo-footer-actions");
        button("Cancel", footer, () => this.close());
        this.applyButton = button("Apply", footer, () => this.apply());
        this.applyButton.className = "primary";
        this.applyButton.disabled = true;
        this.keyHandler = e => {
            if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); const picker = this.root.querySelector(".dkst-photo-picker"); if (picker) picker.remove(); else this.close(); }
            if (e.key === "Tab") {
                const targets = [...(this.root.querySelector(".dkst-photo-picker") || this.root).querySelectorAll('button:not(:disabled), select:not(:disabled), input:not(:disabled), [tabindex="0"]')].filter(el => !el.closest("details:not([open])") && !el.hidden);
                const first = targets[0], last = targets.at(-1);
                if (first && e.shiftKey && (document.activeElement === first || document.activeElement === this.root)) { e.preventDefault(); last.focus(); }
                else if (last && !e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
            }
            // Stop LiteGraph keyboard shortcuts while editing values.
            e.stopPropagation();
        };
        this.root.addEventListener("keydown", this.keyHandler);
        this.root.addEventListener("wheel", e => e.stopPropagation(), { passive: true });
        this.root.focus();
        this.resized = () => this.setZoom(this.zoomValue);
        window.addEventListener("resize", this.resized);
        this.resizeObserver = typeof ResizeObserver === "function" ? new ResizeObserver(this.resized) : null;
        this.resizeObserver?.observe(this.viewport);
        for (const image of [this.before, this.after, this.depth]) image.addEventListener("load", () => { this.updateMinimap(); this.updateFocusMarker(); });
        this.watchGraph = setInterval(() => {
            if (app.graph !== this.appGraph || (this.node && this.graph.getNodeById(this.node.id) !== this.node)) this.close();
        }, 1000);
        this.executed = ({ detail }) => {
            if (this.node && String(detail?.node) === String(this.node.id)) this.message("New input is available. Refresh input to preview it with your current draft settings.");
        };
        api.addEventListener("executed", this.executed);
    }

    message(text, error = false) {
        if (this.closed) return;
        this.status.textContent = text;
        this.status.classList.toggle("error", error);
    }

    async exportImage(save) {
        if (!this.session || this.closed || this.exporting) return;
        const popup = save ? null : window.open("", "_blank");
        if (popup) popup.opener = null;
        this.exporting = true;
        this.openImageButton.disabled = this.saveImageButton.disabled = true;
        const session = this.session;
        this.message("Preparing image at original resolution…");
        try {
            const response = await api.fetchApi(`${ROOT}/export`, { method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session, settings: { ...this.draft }, original: this.exportTarget.value === "original" }),
            });
            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                throw new Error(data.error || `Image export failed (${response.status})`);
            }
            const blob = await response.blob();
            if (this.closed || session !== this.session) { popup?.close(); return; }
            if (save) downloadBlob(blob, "DKST_Photo_Studio.png");
            else {
                if (!popup) throw new Error("Allow pop-ups to open the image, or use Save Image.");
                const url = URL.createObjectURL(blob);
                popup.location.href = url;
                setTimeout(() => URL.revokeObjectURL(url), 60000);
            }
            this.message("Original-resolution image ready.");
        } catch (error) { popup?.close(); this.message(error.message, true); }
        finally {
            this.exporting = false;
            if (!this.closed) this.openImageButton.disabled = this.saveImageButton.disabled = !this.session;
        }
    }

    visibleArea() {
        const bounds = this.pair.getBoundingClientRect(), view = this.viewport.getBoundingClientRect();
        if (!bounds.width || !bounds.height || !view.width || !view.height) return null;
        const clamp = value => Math.max(0, Math.min(1, value));
        return { left: clamp((view.left - bounds.left) / bounds.width), top: clamp((view.top - bounds.top) / bounds.height),
            right: clamp((view.left + this.viewport.clientWidth - bounds.left) / bounds.width),
            bottom: clamp((view.top + this.viewport.clientHeight - bounds.top) / bounds.height) };
    }

    setZoom(value) {
        const area = this.visibleArea();
        this.zoomValue = value === "fit" ? "fit" : Math.max(.25, Math.min(4, Number(value) || 1));
        const { clientWidth: w, clientHeight: h } = this.viewport;
        if (!w || !h || !this.width || !this.height) return;
        const scale = this.zoomValue === "fit" ? Math.min(w / this.width, h / this.height) : this.zoomValue;
        this.pair.style.width = `${Math.max(1, this.width * scale)}px`;
        this.pair.style.height = `${Math.max(1, this.height * scale)}px`;
        this.panArea.style.width = this.zoomValue === "fit" ? "100%" : "max-content";
        this.panArea.style.height = this.zoomValue === "fit" ? "100%" : "max-content";
        this.viewport.style.overflow = this.zoomValue === "fit" ? "hidden" : "auto";
        this.navigate(area ? (area.left + area.right) / 2 : .5, area ? (area.top + area.bottom) / 2 : .5);
        this.updateFocusMarker();
    }

    navigate(x, y) {
        const w = this.pair.clientWidth, h = this.pair.clientHeight;
        this.viewport.scrollLeft = Math.max(0, Math.min(w - this.viewport.clientWidth, x * w - this.viewport.clientWidth / 2));
        this.viewport.scrollTop = Math.max(0, Math.min(h - this.viewport.clientHeight, y * h - this.viewport.clientHeight / 2));
        this.updateMinimap(); this.updateFocusMarker();
    }

    updateComparison() {
        const compare = this.pair.classList.contains("compare") && !this.pair.classList.contains("show-depth");
        const difference = compare && this.comparisonMode.value === "Difference";
        this.pair.classList.toggle("difference", difference);
        this.after.style.clipPath = compare && !difference ? `inset(0 ${100 - this.comparisonPosition}% 0 0)` : "none";
        this.divider.style.left = `${this.comparisonPosition}%`;
        this.divider.hidden = !compare || difference;
        this.compare.setAttribute("aria-pressed", String(this.pair.classList.contains("compare")));
        this.updateMinimap();
    }

    updateMinimap() {
        const w = this.pair.clientWidth, h = this.pair.clientHeight;
        const active = this.session && this.zoomValue !== "fit" && w && h &&
            (w > this.viewport.clientWidth || h > this.viewport.clientHeight);
        this.minimap.hidden = !active;
        if (!active) return;
        const scale = Math.min(Math.min(150, this.viewport.clientWidth * .3) / w, Math.min(110, this.viewport.clientHeight * .3) / h);
        this.map.style.width = `${w * scale}px`; this.map.style.height = `${h * scale}px`;
        const source = this.pair.classList.contains("show-depth") ? this.depth : this.after;
        if (source.getAttribute("src") && this.mapImage.getAttribute("src") !== source.getAttribute("src")) this.mapImage.src = source.src;
        const area = this.visibleArea();
        if (area) Object.assign(this.mapWindow.style, { left: `${area.left * 100}%`, top: `${area.top * 100}%`,
            width: `${(area.right - area.left) * 100}%`, height: `${(area.bottom - area.top) * 100}%` });
    }

    installMinimap() {
        const point = event => {
            const rect = this.map.getBoundingClientRect();
            return rect.width && rect.height ? { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height } : null;
        };
        this.map.addEventListener("pointerdown", event => {
            if (event.button !== 0 || this.minimap.hidden) return;
            const p = point(event), area = this.visibleArea();
            if (!p || !area) return;
            event.preventDefault(); event.stopPropagation();
            const inside = p.x >= area.left && p.x <= area.right && p.y >= area.top && p.y <= area.bottom;
            this.mapDrag = { id: event.pointerId, x: inside ? p.x - (area.left + area.right) / 2 : 0, y: inside ? p.y - (area.top + area.bottom) / 2 : 0 };
            this.map.setPointerCapture(event.pointerId);
            this.navigate(p.x - this.mapDrag.x, p.y - this.mapDrag.y);
        });
        this.map.addEventListener("pointermove", event => {
            if (this.mapDrag?.id !== event.pointerId) return;
            const p = point(event); if (p) this.navigate(p.x - this.mapDrag.x, p.y - this.mapDrag.y);
        });
        const end = event => {
            if (this.mapDrag?.id !== event.pointerId) return;
            if (this.map.hasPointerCapture(event.pointerId)) this.map.releasePointerCapture(event.pointerId);
            this.mapDrag = null;
        };
        this.map.addEventListener("pointerup", end); this.map.addEventListener("pointercancel", end);
        this.map.addEventListener("lostpointercapture", () => { this.mapDrag = null; });
        this.map.addEventListener("keydown", event => {
            const direction = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
            if (!direction) return;
            event.preventDefault(); event.stopPropagation();
            const area = this.visibleArea();
            if (area) this.navigate((area.left + area.right) / 2 + direction[0] * (area.right - area.left) * .25,
                (area.top + area.bottom) / 2 + direction[1] * (area.bottom - area.top) * .25);
        });
    }

    async start() {
        const data = await request("presets");
        if (this.closed) return;
        this.defaults = data.defaults;
        this.presets = data.presets;
        this.schema = data.controls;
        this.draft = { ...this.defaults, ...settingsOf(this.node) };
        this.updatePresets(widget(this.node, "preset")?.value || "Custom");
        this.buildControls();
        this.sync();
        if (this.capture || this.imageCandidate) await this.loadSource();
        else this.message("Select an existing workflow image, import a file, or run this node once to capture its actual input.");
    }

    updatePresets(selected = "Custom") {
        this.preset.replaceChildren();
        for (const name of ["Custom", "Default", ...Object.keys(this.presets).sort()]) {
            const option = element("option", "", name, this.preset);
            option.value = name;
        }
        this.preset.value = selected in this.presets || selected === "Default" ? selected : "Custom";
    }

    isConnected(name) {
        return this.node?.inputs?.some(i => (i.widget?.name === name || i.name === name) && i.link != null);
    }

    buildControls() {
        const groups = [["light_", "Light"], ["color_", "Color"], ["effects_", "Effects"], ["detail_", "Detail"], ["optics_", "Optics"], ["lens_blur_", "Lens Blur"]];
        for (const [prefix, title] of groups) {
            const panel = element("details", "", null, this.panels);
            panel.open = title === "Light" || title === "Color";
            element("summary", "", title, panel);
            const names = Object.keys(this.defaults).filter(name => name.startsWith(prefix) || (title === "Effects" && name === "grain_seed") || (title === "Lens Blur" && name === "depth_near_is_white"));
            for (const name of names) {
                const spec = this.schema[name];
                const row = element("label", "dkst-photo-control", null, panel);
                element("span", "", labelFor(name), row);
                row.title = spec.tooltip || "";
                const connected = this.isConnected(name);
                const input = element(spec.kind === "choice" ? "select" : "input", "", null, row);
                input.setAttribute("aria-label", labelFor(name));
                if (spec.kind === "choice") {
                    for (const value of spec.values) { const option = element("option", "", value, input); option.value = value; }
                } else if (spec.kind === "bool") input.type = "checkbox";
                else {
                    input.type = "number";
                    input.min = spec.min; input.max = spec.max; input.step = spec.step;
                    const slider = element("input", "dkst-photo-slider", null, row);
                    slider.type = "range"; slider.min = spec.min; slider.max = spec.max; slider.step = spec.step;
                    slider.setAttribute("aria-label", `${labelFor(name)} slider`);
                    slider.disabled = connected;
                    slider.addEventListener("pointerdown", () => this.beginGesture());
                    slider.addEventListener("keydown", () => this.beginGesture());
                    slider.addEventListener("input", () => this.controlChange(name, Number(slider.value), false));
                    slider.addEventListener("change", () => { this.gesture = false; });
                    this.controls.set(name, { input, slider });
                }
                input.disabled = connected;
                if (connected) { row.classList.add("connected"); row.title = "Controlled by a connected workflow input. Disconnect it to edit here; preview uses the stored widget value."; }
                let numberGesture = false;
                const readNumber = () => {
                    if (!input.value || !input.validity.valid) return null;
                    const value = Number(input.value);
                    return !Number.isFinite(value) || value < spec.min || value > spec.max || (spec.kind === "int" && !Number.isInteger(value)) ? null : value;
                };
                if (input.type === "number") {
                    input.addEventListener("input", () => {
                        const value = readNumber();
                        if (value === null) return;
                        this.controlChange(name, value, !numberGesture, true);
                        numberGesture = true;
                    });
                    input.addEventListener("blur", () => { numberGesture = false; this.sync(); });
                }
                input.addEventListener("change", () => {
                    if (input.type === "number") return;
                    const value = spec.kind === "bool" ? input.checked : spec.kind === "choice" ? input.value : Number(input.value);
                    if (spec.kind !== "choice" && spec.kind !== "bool" && (!Number.isFinite(value) || value < spec.min || value > spec.max || (spec.kind === "int" && !Number.isInteger(value)))) { this.sync(); return; }
                    this.controlChange(name, value, true);
                });
                if (!this.controls.has(name)) this.controls.set(name, { input });
            }
        }
    }

    remember() {
        this.history.push({ settings: { ...this.draft }, preset: this.preset.value });
        if (this.history.length > 60) this.history.shift();
        this.redoHistory = [];
    }
    beginGesture() { if (!this.gesture) { this.remember(); this.gesture = true; } }
    controlChange(name, value, remember, preserveInput = false) {
        if (remember) this.remember();
        if (name.startsWith("lens_blur_") || name === "depth_near_is_white" || name === "optics_distortion") { this.focusPoint = null; this.focusMarker.hidden = true; }
        this.draft[name] = value;
        if (name === "color_temperature" || name === "color_tint") this.draft.color_white_balance = "Custom";
        this.preset.value = "Custom";
        this.sync(preserveInput ? name : null); this.schedule();
    }
    change(settings, preset = "Custom") {
        this.remember();
        this.focusPoint = null; this.focusMarker.hidden = true;
        for (const name of Object.keys(settings)) if (!this.isConnected(name)) this.draft[name] = settings[name];
        this.preset.value = preset;
        this.sync(); this.schedule();
    }
    undo() {
        const previous = this.history.pop();
        if (!previous) return;
        this.redoHistory.push({ settings: { ...this.draft }, preset: this.preset.value });
        this.focusPoint = null; this.focusMarker.hidden = true;
        this.draft = previous.settings; this.preset.value = previous.preset;
        this.gesture = false; this.sync(); this.schedule();
    }
    redo() {
        const next = this.redoHistory.pop();
        if (!next) return;
        this.history.push({ settings: { ...this.draft }, preset: this.preset.value });
        this.focusPoint = null; this.focusMarker.hidden = true;
        this.draft = next.settings; this.preset.value = next.preset;
        this.gesture = false; this.sync(); this.schedule();
    }
    sync(preserveInput = null) {
        for (const [name, { input, slider }] of this.controls) {
            if (name !== preserveInput) { if (input.type === "checkbox") input.checked = !!this.draft[name]; else input.value = this.draft[name]; }
            if (slider) slider.value = this.draft[name];
        }
        this.undoButton.disabled = !this.history.length;
        this.redoButton.disabled = !this.redoHistory.length;
    }

    async savePreset(asNew = false) {
        try {
            let name = this.preset.value;
            const overwrite = !asNew && name !== "Custom" && name !== "Default";
            if (!overwrite) name = window.prompt("Preset name", asNew && name !== "Custom" ? `${name} Copy` : "");
            if (name == null) return;
            const saved = { ...this.draft };
            const data = await request("presets", { name: name.trim(), settings: saved, overwrite });
            if (this.closed) return;
            this.presets = data.presets; this.updatePresets(JSON.stringify(saved) === JSON.stringify(this.draft) ? name.trim() : "Custom");
            this.message("Preset saved.");
        } catch (error) { this.message(error.message, true); }
    }

    async choose(kind) {
        this.root.querySelector(".dkst-photo-picker")?.remove();
        const picker = element("div", "dkst-photo-picker", null, this.root.querySelector("section"));
        const header = element("header", "", null, picker);
        element("strong", "", kind === "depth" ? "Choose depth map · same scene and crop" : "Choose workflow image", header);
        button("Close", header, () => picker.remove());
        const grid = element("div", "dkst-photo-image-grid", null, picker);
        const candidates = workflowImages(this.graph);
        for (const candidate of candidates) {
            const card = button("", grid, () => { picker.remove(); this.select(candidate, kind); });
            const image = element("img", "", null, card);
            image.alt = candidate.label; image.src = imageUrl(candidate.descriptor);
            element("span", "", candidate.label, card);
            const dimensions = element("small", "", candidate.descriptor.filename, card);
            image.addEventListener("load", () => { dimensions.textContent = `${image.naturalWidth} × ${image.naturalHeight} · ${candidate.direct ? "IMAGE connection" : "file snapshot"}`; });
        }
        if (!candidates.length) element("p", "", "No available images yet. Load an image, run an image-producing node, or import a file.", grid);
        header.querySelector("button").focus();
    }

    importFile(kind) {
        const input = element("input");
        input.type = "file"; input.accept = "image/*";
        input.addEventListener("change", async () => {
            if (!input.files?.[0]) return;
            this.message("Uploading image…");
            try {
                const form = new FormData(); form.append("image", input.files[0]); form.append("overwrite", "false");
                const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || "Image upload failed");
                if (!this.closed) await this.select({ descriptor: { filename: data.name, subfolder: data.subfolder || "", type: data.type || "input" }, label: input.files[0].name, direct: false }, kind);
            } catch (error) { this.message(error.message, true); }
        });
        input.click();
    }

    async select(candidate, kind) {
        if (kind === "depth") { this.depthCandidate = candidate; this.depthChanged = true; }
        else { this.imageCandidate = candidate; this.capture = null; this.sourceChanged = true; }
        await this.loadSource();
    }

    async loadSource() {
        const generation = ++this.sourceSequence;
        this.sequence++;
        this.applyButton.disabled = true;
        this.openImageButton.disabled = this.saveImageButton.disabled = true;
        this.showDepth.disabled = this.focusButton.disabled = true;
        const old = this.session; this.session = null;
        if (old) request("release", { session: old }).catch(() => {});
        if (!this.capture && !this.imageCandidate) { this.message("Choose the main image first."); return; }
        this.focusPoint = null; this.focusMarker.hidden = true;
        this.message("Loading original image…");
        try {
            const data = await request("session", {
                ...(this.capture ? { source_session: this.capture } : { image: this.imageCandidate.descriptor, image_loader: this.imageCandidate.loader, image_interpretation: this.imageCandidate.interpretation }),
                use_source_depth: !this.depthChanged,
                ...(!this.depthChanged && !this.capture && this.depthCapture ? { depth_source_session: this.depthCapture } : {}),
                ...((this.depthChanged || !this.capture) && this.depthCandidate ? { depth: this.depthCandidate.descriptor, depth_loader: this.depthCandidate.loader, depth_interpretation: this.depthCandidate.interpretation } : {}),
            });
            if (this.closed || generation !== this.sourceSequence) { request("release", { session: data.session }).catch(() => {}); return; }
            this.session = data.session; this.auto = data.auto; this.hasDepth = data.has_depth;
            this.width = data.width; this.height = data.height;
            this.before.src = data.original; this.after.src = data.original;
            this.empty.hidden = true;
            this.pair.classList.remove("show-depth");
            this.applyButton.disabled = false;
            this.openImageButton.disabled = this.saveImageButton.disabled = !!this.exporting;
            this.showDepth.disabled = this.focusButton.disabled = !data.has_depth;
            this.imageLabel.textContent = `${this.capture ? "Captured node input" : this.imageCandidate?.label || "Image file"} · ${data.width} × ${data.height} · first frame`;
            this.depthLabel.textContent = data.has_depth ? this.depthCandidate?.label || "Connected depth input" : "No depth map · Lens Blur requires one";
            this.viewport.scrollLeft = this.viewport.scrollTop = 0;
            this.setZoom(this.zoomValue); this.navigate(.5, .5); this.updateComparison();
            this.schedule(0);
        } catch (error) {
            if (generation !== this.sourceSequence || this.closed) return;
            if (error.status === 410 && this.capture && this.imageCandidate) {
                this.capture = null; this.depthCapture = null;
                await this.loadSource();
            } else this.message(error.message, true);
        }
    }

    schedule(delay = 180) {
        this.sequence++;
        clearTimeout(this.timer);
        this.timer = setTimeout(() => this.render(), delay);
    }
    async render() {
        if (this.closed || !this.session) return;
        if (this.rendering) { this.pending = true; return; }
        this.rendering = true; this.pending = false;
        const sequence = this.sequence;
        const session = this.session;
        this.message("Updating preview…");
        try {
            const data = await request("preview", { session, settings: { ...this.draft }, quality: this.quality.value });
            if (this.closed || sequence !== this.sequence || session !== this.session) return;
            this.after.src = data.preview;
            this.updateMinimap();
            this.renderedSequence = sequence;
            if (data.depth_preview) this.depth.src = data.depth_preview;
            this.drawHistogram(data.histogram);
            const external = Object.keys(this.draft).some(name => this.isConnected(name));
            this.message(`${widget(this.node, "active")?.value === false ? "Node is inactive; enable Active for workflow output. " : ""}${data.proxy ? "Proxy preview · pixel effects may differ. " : "Original resolution processing. "}${external ? "Connected controls use stored values in this preview. " : ""}Apply saves settings; run the workflow to update downstream images.`);
        } catch (error) { if (sequence === this.sequence) this.message(error.message, true); }
        finally { this.rendering = false; if (this.pending && !this.closed) this.render(); }
    }
    drawHistogram(values) {
        let context;
        try { context = this.histogram.getContext("2d"); } catch { return; }
        if (!context || !values) return;
        context.clearRect(0, 0, 256, 66);
        const peak = Math.max(1, ...values.flat());
        values.forEach((channel, c) => {
            context.beginPath(); context.moveTo(0, 66);
            channel.forEach((value, i) => context.lineTo(i * 256 / channel.length, 66 - value / peak * 62));
            context.lineTo(256, 66); context.closePath();
            context.fillStyle = ["#e86a6a88", "#62d48b88", "#629bea88"][c]; context.fill();
        });
    }
    async focusAt(event) {
        if (!this.pickFocus || !this.hasDepth || !this.session || this.isConnected("lens_blur_focus")) return;
        if (this.renderedSequence !== undefined && this.renderedSequence !== this.sequence) { this.message("Wait for the current preview before picking focus."); return; }
        const image = this.pair.classList.contains("show-depth") ? this.depth : this.after;
        if (event.target !== image || !this.width) return;
        const rect = image.getBoundingClientRect();
        const scale = Math.min(rect.width / this.width, rect.height / this.height);
        const w = this.width * scale, h = this.height * scale;
        const x = (event.clientX - rect.left - (rect.width - w) / 2) / w;
        const y = (event.clientY - rect.top - (rect.height - h) / 2) / h;
        if (x < 0 || x > 1 || y < 0 || y > 1) return;
        const generation = this.sequence, session = this.session;
        try {
            const data = await request("focus", { session, settings: { ...this.draft }, point: [x, y] });
            if (!this.closed && generation === this.sequence && session === this.session) {
                this.controlChange("lens_blur_focus", data.focus, true);
                this.focusPoint = { x, y }; this.updateFocusMarker();
            }
        } catch (error) { this.message(error.message, true); }
    }

    updateFocusMarker() {
        if (!this.focusPoint || !this.width) { this.focusMarker.hidden = true; return; }
        const image = this.pair.classList.contains("show-depth") ? this.depth : this.after;
        const viewport = this.root.querySelector(".dkst-photo-viewport");
        const rect = image.getBoundingClientRect(), parent = viewport.getBoundingClientRect();
        const scale = Math.min(rect.width / this.width, rect.height / this.height);
        const w = this.width * scale, h = this.height * scale;
        this.focusMarker.style.left = `${rect.left - parent.left + viewport.scrollLeft + (rect.width - w) / 2 + this.focusPoint.x * w}px`;
        this.focusMarker.style.top = `${rect.top - parent.top + viewport.scrollTop + (rect.height - h) / 2 + this.focusPoint.y * h}px`;
        this.focusMarker.hidden = false;
    }

    async apply() {
        if (!this.session || this.closed) return;
        if (this.draft.lens_blur_apply && !this.hasDepth) { this.message("Select a depth map before applying Lens Blur.", true); return; }
        const graph = this.graph;
        if (!graph || (this.node && graph.getNodeById(this.node.id) !== this.node)) { this.message("The target node is no longer in this workflow.", true); return; }
        this.applyButton.disabled = true;

        try {
            for (const name of ["imageCandidate", "depthCandidate"]) {
                const candidate = this[name];
                if (candidate && !candidate.direct && candidate.descriptor.type !== "input") {
                    const descriptor = await request("materialize", { image: candidate.descriptor });
                    if (this.closed) return;
                    this[name] = { ...candidate, descriptor };
                }
            }
        } catch (error) { this.message(error.message, true); this.applyButton.disabled = false; return; }
        if (this.closed || app.graph !== this.appGraph || (this.node && graph.getNodeById(this.node.id) !== this.node)) { this.close(); return; }
        const created = [];
        const links = (this.node?.inputs ?? []).map(input => input.link != null ? graph.links?.[input.link] : null);
        const previous = settingsOf(this.node);
        const previousPreset = widget(this.node, "preset")?.value;
        let node = this.node;
        graph.beforeChange?.();
        try {
            if (!node) { node = createNode(graph, NODE_CLASS, [app.canvas?.graph_mouse?.[0] || 100, app.canvas?.graph_mouse?.[1] || 100]); created.push(node); }
            if (this.sourceChanged || !this.node) connectCandidate(graph, this.imageCandidate, node, "image", created);
            if (this.depthChanged) {
                if (this.depthCandidate) connectCandidate(graph, this.depthCandidate, node, "depth_image", created);
                else { const index = node.inputs?.findIndex(i => i.name === "depth_image"); if (index >= 0) node.disconnectInput(index); }
            } else if (!this.node && this.depthCandidate) connectCandidate(graph, this.depthCandidate, node, "depth_image", created);
            for (const [name, value] of Object.entries(this.draft)) if (!this.isConnected(name)) setNodeValue(node, name, value);
            node.dkstPhotoStudioPresetData?.(this.presets);
            setNodeValue(node, "preset", this.preset.value, false);
            node.graph?.incrementVersion?.(); node.setDirtyCanvas?.(true, true);
            node.dkstPhotoStudioSetPreview?.(this.after.src);
            node.dkstPhotoStudioCompact?.();
            this.close();
        } catch (error) {
            for (const added of created.reverse()) graph.remove(added);
            if (this.node) {
                for (const name of ["image", "depth_image"]) {
                    const index = this.node.inputs.findIndex(i => i.name === name);
                    if (index < 0) continue;
                    this.node.disconnectInput(index);
                    const link = links[index];
                    if (link) graph.getNodeById(link.origin_id)?.connect(link.origin_slot, this.node, index);
                }
                for (const [name, value] of Object.entries(previous)) setNodeValue(this.node, name, value);
                setNodeValue(this.node, "preset", previousPreset, false);
            }
            this.applyButton.disabled = false;
            this.message(error.message, true);
        }
        finally { graph.afterChange?.(); }
    }

    close() {
        if (this.closed) return;
        this.closed = true; this.sequence++; this.sourceSequence++;
        clearTimeout(this.timer);
        clearInterval(this.watchGraph);
        window.removeEventListener("resize", this.resized);
        this.resizeObserver?.disconnect();
        closeImageMenu?.();
        api.removeEventListener("executed", this.executed);
        if (this.session) request("release", { session: this.session }).catch(() => {});
        this.root.remove();
        this.previousFocus?.focus?.();
        if (currentEditor === this) currentEditor = null;
    }
}
