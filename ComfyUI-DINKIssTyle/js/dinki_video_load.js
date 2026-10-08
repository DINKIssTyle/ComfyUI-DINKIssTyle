import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import "./dinki_image_crop.js";

const CLASS = "DINKI_Video_Load_Crop";
const SETTINGS = ["category", "filename", "aspect_ratio", "custom_width", "custom_height",
    "crop_x", "crop_y", "crop_width", "crop_height", "resolution_multiple", "megapixels",
    "trim_in", "trim_out", "fps_mode", "output_fps"];
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const rounded = value => Math.round(value * 1000000) / 1000000;
const VIDEO_ACCEPT = ".mp4,.mov,.webm,.mkv,.avi,.m4v,.mpg,.mpeg,.ts";
const isVideoFile = file => VIDEO_ACCEPT.split(",").some(extension =>
    file?.name?.toLowerCase().endsWith(extension));

function mountEditor(node) {
    const widgets = Object.fromEntries(SETTINGS.map(name => [name, node.widgets.find(w => w.name === name)]));
    if (Object.values(widgets).some(w => !w) || !node.dkstCropRoot) return;
    const state = { duration: 0, fps: 24, generation: 0, removed: false, animation: 0, source: null,
        uploading: false };
    const root = node.dkstCropRoot;
    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;flex-direction:column;gap:8px;flex:0 0 auto;padding:0 4px 4px;box-sizing:border-box";
    const video = document.createElement("video");
    video.style.display = "none";
    video.preload = "auto";
    video.playsInline = true;
    video.muted = true;
    // Editing previews are silent; audio is retained in the VIDEO output.
    root.append(video, controls);
    node.dkstVideoControlsHeight = 206;
    node.hideOutputImages = true;
    node.onDrawBackground = function () {};

    const row = () => {
        const element = document.createElement("div");
        element.style.cssText = "display:flex;align-items:center;gap:6px;min-width:0;flex-wrap:wrap";
        controls.append(element);
        return element;
    };
    const button = (parent, text, callback) => {
        const element = document.createElement("button");
        element.type = "button";
        element.textContent = text;
        element.style.cssText = "background:var(--comfy-input-bg,#32343a);color:var(--input-text,#eee);border:1px solid var(--border-color,#555);border-radius:5px;padding:4px 8px;cursor:pointer";
        element.addEventListener("click", callback);
        parent.append(element);
        return element;
    };
    const information = document.createElement("div");
    information.style.cssText = "font-size:11px;line-height:16px;overflow-wrap:anywhere;white-space:pre-line;min-height:32px";
    controls.append(information);
    const fileActions = row();
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = VIDEO_ACCEPT;
    fileInput.style.display = "none";
    document.body.append(fileInput);
    const uploadButton = button(fileActions, "Upload Video", () => fileInput.click());
    uploadButton.title = "Upload a video into the selected category folder";
    button(fileActions, "Refresh files", () => refresh(true).catch(showError));
    fileInput.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        fileInput.value = "";
        if (file) uploadVideo(file);
    });
    const toolbar = row();
    const play = button(toolbar, "▶ Play", () => {
        if (!state.duration) return;
        const [start, end] = windowValues();
        if (video.currentTime < start || video.currentTime >= end) video.currentTime = start;
        if (video.paused) video.play().catch(error => showError(error));
        else video.pause();
    });
    button(toolbar, "Full range", () => setWindow(0, state.duration, true));
    button(toolbar, "Original FPS", () => {
        set("fps_mode", "Original");
        set("output_fps", state.fps);
        sync();
    });
    const positionLabel = document.createElement("span");
    positionLabel.style.cssText = "margin-left:auto;font-variant-numeric:tabular-nums";
    toolbar.append(positionLabel);

    const timeline = document.createElement("div");
    timeline.style.cssText = "position:relative;height:30px;flex:0 0 30px;background:#25272c;border-radius:5px;touch-action:none;margin:0 8px;cursor:crosshair";
    timeline.setAttribute("aria-label", "Video trim timeline");
    controls.append(timeline);
    const selection = document.createElement("div");
    selection.style.cssText = "position:absolute;top:6px;bottom:6px;background:#527fba88;border:1px solid #85b8fa;pointer-events:none";
    timeline.append(selection);
    const handles = ["IN", "OUT"].map(label => {
        const element = document.createElement("button");
        element.type = "button";
        element.textContent = label === "IN" ? "[" : "]";
        element.title = `${label} trim handle — drag or use arrow keys`;
        element.setAttribute("aria-label", `${label} trim handle`);
        element.style.cssText = "position:absolute;top:0;bottom:0;width:16px;transform:translateX(-50%);border:0;border-radius:4px;background:#85b8fa;color:#142337;cursor:ew-resize;touch-action:none;font-weight:bold;padding:0;z-index:2";
        timeline.append(element);
        return element;
    });
    const playhead = document.createElement("div");
    playhead.style.cssText = "position:absolute;top:0;bottom:0;width:2px;background:#fff;pointer-events:none;z-index:3";
    timeline.append(playhead);
    const fields = row();
    const inputs = ["IN", "OUT", "Length"].map(label => {
        const wrapper = document.createElement("label");
        wrapper.style.cssText = "display:flex;gap:4px;align-items:center;flex:1;min-width:80px";
        const text = document.createElement("span");
        text.textContent = label;
        const input = document.createElement("input");
        input.type = "number"; input.min = "0"; input.step = "0.001";
        input.setAttribute("aria-label", `${label} seconds`);
        input.style.cssText = "width:0;flex:1;min-width:35px;background:var(--comfy-input-bg,#32343a);color:var(--input-text,#eee);border:1px solid var(--border-color,#555);border-radius:4px;padding:4px";
        wrapper.append(text, input);
        fields.append(wrapper);
        return input;
    });

    function set(name, value) {
        const widget = widgets[name];
        const old = widget.value;
        widget.value = value;
        if (old !== value) {
            node.onWidgetChanged?.(name, value, old, widget);
            node.graph?.incrementVersion?.();
            node.setDirtyCanvas?.(true, true);
        }
    }
    function windowValues() {
        const start = Number(widgets.trim_in.value) || 0;
        const end = Number(widgets.trim_out.value) || state.duration;
        return [start, end];
    }
    function showError(error) { information.textContent = error.message || String(error); }
    function sync() {
        const [start, end] = windowValues();
        const length = Math.max(0, end - start);
        inputs.forEach((input, i) => {
            if (document.activeElement !== input) input.value = [start, end, length][i].toFixed(3);
            input.disabled = !state.duration;
            input.max = String(state.duration);
        });
        const percent = seconds => `${clamp(state.duration ? seconds / state.duration : 0, 0, 1) * 100}%`;
        handles[0].style.left = selection.style.left = percent(start);
        handles[1].style.left = percent(end);
        selection.style.width = percent(length);
        playhead.style.left = percent(video.currentTime || 0);
        handles.forEach(handle => { handle.disabled = !state.duration; });
        play.disabled = !state.duration;
        play.textContent = video.paused ? "▶ Play" : "Ⅱ Pause";
        positionLabel.textContent = `${(video.currentTime || 0).toFixed(3)} / ${state.duration.toFixed(3)}s`;
        widgets.output_fps.disabled = widgets.fps_mode.value === "Original";
        if (state.source) {
            const fps = widgets.fps_mode.value === "Original" ? state.fps : Number(widgets.output_fps.value);
            const frames = Math.max(1, Math.ceil(length * fps - 1e-8));
            const mp = Number(widgets.megapixels.value);
            const gib = frames * mp * 1024 * 1024 * 3 * 4 / 1024 ** 3;
            information.textContent = `${state.source.width} × ${state.source.height} · ${state.duration.toFixed(3)}s · Source ${state.fps.toFixed(3)} FPS · Silent preview\nOutput ${fps.toFixed(3)} FPS · ~${frames} frames · ~${gib.toFixed(2)} GiB frame batch`;
        }
        node.dkstCropRender?.();
    }
    function setWindow(start, end, full = false) {
        if (!state.duration) return;
        const minimum = Math.min(state.duration, 1 / state.fps);
        start = clamp(start, 0, Math.max(0, state.duration - minimum));
        end = clamp(end, start + minimum, state.duration);
        set("trim_in", rounded(start));
        set("trim_out", full ? 0 : rounded(end));
        const position = clamp(video.currentTime || 0, start, end - Math.min(minimum, (end - start) / 2));
        video.currentTime = position;
        sync();
    }
    inputs.forEach((input, i) => {
        const update = commit => {
            if (input.value === "") return;
            const value = Number(input.value);
            if (!Number.isFinite(value)) return;
            const [start, end] = windowValues();
            if (i === 0) setWindow(Math.min(value, end - 1 / state.fps), end);
            else setWindow(start, i === 1 ? value : start + value);
            if (commit) input.value = (i === 0 ? Number(widgets.trim_in.value) : i === 1 ? windowValues()[1] :
                windowValues()[1] - windowValues()[0]).toFixed(3);
        };
        input.addEventListener("input", () => update(false));
        input.addEventListener("change", () => update(true));
    });
    let dragging = null;
    const timelineTime = event => {
        const bounds = timeline.getBoundingClientRect();
        return clamp((event.clientX - bounds.left) / bounds.width, 0, 1) * state.duration;
    };
    const drag = event => {
        if (!state.duration) return;
        const time = timelineTime(event);
        const [start, end] = windowValues();
        if (dragging === "in") setWindow(Math.min(time, end - 1 / state.fps), end);
        else if (dragging === "out") setWindow(start, time);
        else { video.currentTime = time; sync(); }
    };
    timeline.addEventListener("pointerdown", event => {
        if (event.button !== 0 || !state.duration) return;
        dragging = event.target === handles[0] ? "in" : event.target === handles[1] ? "out" : "seek";
        video.pause();
        timeline.setPointerCapture(event.pointerId);
        drag(event); event.preventDefault(); event.stopPropagation();
    });
    timeline.addEventListener("pointermove", event => { if (dragging) { drag(event); event.stopPropagation(); } });
    timeline.addEventListener("pointerup", event => {
        if (dragging) drag(event);
        dragging = null; timeline.releasePointerCapture?.(event.pointerId);
        event.stopPropagation();
    });
    timeline.addEventListener("pointercancel", () => { dragging = null; });
    handles.forEach((handle, i) => handle.addEventListener("keydown", event => {
        if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
        const [start, end] = windowValues();
        const delta = (event.key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 1 : 1 / state.fps);
        if (i === 0) setWindow(Math.min(start + delta, end - 1 / state.fps), end);
        else setWindow(start, end + delta);
        event.preventDefault(); event.stopPropagation();
    }));
    function animate() {
        if (state.removed) return;
        const [start, end] = windowValues();
        if (video.currentTime >= end || video.currentTime < start) video.currentTime = start;
        sync();
        if (!video.paused) state.animation = requestAnimationFrame(animate);
    }
    video.addEventListener("play", () => { cancelAnimationFrame(state.animation); animate(); });
    video.addEventListener("pause", () => { cancelAnimationFrame(state.animation); sync(); });
    video.addEventListener("seeked", sync);
    video.addEventListener("ended", () => {
        video.currentTime = windowValues()[0];
        video.play().catch(showError);
    });

    for (const name of ["trim_in", "trim_out"]) {
        widgets[name].hidden = true;
        widgets[name].options ??= {}; widgets[name].options.hidden = true;
        widgets[name].computeSize = () => [0, -4]; widgets[name].draw = () => {};
    }
    for (const name of ["fps_mode", "output_fps", "megapixels", "resolution_multiple"]) {
        const previous = widgets[name].callback;
        widgets[name].callback = function (...args) { previous?.apply(this, args); sync(); };
    }
    function setValues(widget, values) {
        let owner = widget;
        while (owner && !Object.getOwnPropertyDescriptor(owner, "options")) owner = Object.getPrototypeOf(owner);
        const descriptor = owner && Object.getOwnPropertyDescriptor(owner, "options");
        const list = values.length ? values : [""];
        const options = { ...(widget.options ?? {}), values: list };
        if (!descriptor || descriptor.writable || descriptor.set) widget.options = options;
        else widget.options.values = list;
        if (widget._state?.options) widget._state.options = { ...widget._state.options, values: list };
    }
    async function request(url, options) {
        const response = await api.fetchApi(url, options);
        let data;
        try {
            data = await response.json();
        } catch (error) {
            if (!response.ok) throw new Error(`Request failed (${response.status})`);
            throw error;
        }
        if (!response.ok || data.error) throw new Error(data.error || `Request failed (${response.status})`);
        return data;
    }
    function mediaURL(descriptor, key) {
        return api.apiURL(`/view?${new URLSearchParams({ ...descriptor, format: "video", v: key })}`);
    }
    async function uploadVideo(file) {
        if (state.removed || state.uploading) return false;
        if (!isVideoFile(file)) {
            showError(new Error("Choose an MP4, MOV, WebM, MKV, AVI, M4V, MPG, MPEG or TS video."));
            return false;
        }
        const generation = state.generation;
        const category = widgets.category.value || "";
        state.uploading = true;
        uploadButton.disabled = true;
        uploadButton.textContent = "Uploading…";
        video.pause();
        information.textContent = `Uploading ${file.name}…`;
        try {
            const body = new FormData();
            body.append("image", file, file.name);
            body.append("type", "input");
            body.append("subfolder", category);
            body.append("overwrite", "false");
            const data = await request("/upload/image", { method: "POST", body });
            if (generation !== state.generation || state.removed) return false;
            if (typeof data.name !== "string" || !data.name || data.type !== "input") {
                throw new Error("The server returned an invalid video upload result.");
            }
            set("category", data.subfolder ?? category);
            set("filename", data.name);
            resetSourceSettings();
            await refresh(true);
            return true;
        } catch (error) {
            if (generation === state.generation && !state.removed) showError(error);
            return false;
        } finally {
            state.uploading = false;
            uploadButton.disabled = false;
            uploadButton.textContent = "Upload Video";
        }
    }
    const droppedVideo = event => Array.from(event.dataTransfer?.files || []).find(isVideoFile);
    const previousDragOver = node.onDragOver;
    const previousDragDrop = node.onDragDrop;
    node.onDragOver = function (event) {
        return !!droppedVideo(event) || Array.from(event.dataTransfer?.types || []).includes("Files") ||
            previousDragOver?.call(this, event) || false;
    };
    node.onDragDrop = function (event) {
        const file = droppedVideo(event);
        if (!file) return previousDragDrop?.call(this, event) || false;
        uploadVideo(file);
        return true;
    };
    root.addEventListener("dragover", event => {
        // Browsers may expose only MIME types until the file is dropped.
        if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    });
    root.addEventListener("drop", event => {
        const file = droppedVideo(event);
        if (!file) return;
        event.preventDefault();
        event.stopPropagation();
        uploadVideo(file);
    });
    async function load(generation, category, filename) {
        if (!filename) { information.textContent = "Select a video file."; return; }
        const data = await request(`/dinki/video-load/metadata?${new URLSearchParams({ category, filename })}`);
        if (generation !== state.generation || state.removed) return;
        state.source = data;
        state.duration = data.duration; state.fps = data.fps;
        video.dkstSourceWidth = data.width; video.dkstSourceHeight = data.height;
        if (widgets.fps_mode.value === "Original") set("output_fps", data.fps);
        const [start, end] = windowValues();
        setWindow(start, end, Number(widgets.trim_out.value) === 0);
        let proxy = false;
        video.onloadedmetadata = () => {
            if (generation !== state.generation || state.removed) return;
            video.currentTime = windowValues()[0];
            node.dkstCropSourcePreview?.(video, `${category}/${filename}`);
            sync();
        };
        video.onloadeddata = () => { if (generation === state.generation) sync(); };
        video.onerror = async () => {
            if (generation !== state.generation || state.removed) return;
            if (proxy) return showError(new Error("Video preview could not be decoded."));
            proxy = true; information.textContent = "Preparing browser preview…";
            try {
                const result = await request("/dinki/video-load/preview", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ category, filename }),
                });
                if (generation !== state.generation || state.removed) return;
                video.src = mediaURL(result.preview, data.source_key); video.load();
            } catch (error) { if (generation === state.generation) showError(error); }
        };
        video.src = mediaURL(data.preview, data.source_key); video.load(); sync();
    }
    async function refresh(preserve = false, filesOnly = false) {
        const generation = ++state.generation;
        video.pause(); video.onloadedmetadata = video.onloadeddata = video.onerror = null;
        video.removeAttribute("src"); video.load();
        state.duration = 0; state.source = null;
        node.dkstCropSourcePreview?.(null);
        sync(); information.textContent = "Loading video information…";
        try {
            const category = widgets.category.value || "";
            const filename = widgets.filename.value || "";
            if (!filesOnly) {
                const data = await request("/dinki/video-load/categories");
                if (generation !== state.generation || state.removed) return;
                setValues(widgets.category, data.categories);
            }
            const data = await request(`/dinki/video-load/files?${new URLSearchParams({ category })}`);
            if (generation !== state.generation || state.removed) return;
            setValues(widgets.filename, data.files);
            set("filename", preserve && filename ? filename : data.files[0] || "");
            await load(generation, category, widgets.filename.value);
        } catch (error) {
            if (generation === state.generation && !state.removed) showError(error);
        }
    }
    function resetSourceSettings() {
        set("trim_in", 0); set("trim_out", 0);
        for (const [name, value] of Object.entries({ crop_x: 0, crop_y: 0, crop_width: 1, crop_height: 1 })) set(name, value);
    }
    widgets.category.callback = () => {
        resetSourceSettings();
        refresh(false, true).catch(showError);
    };
    widgets.filename.callback = () => {
        resetSourceSettings();
        refresh(true, true).catch(showError);
    };
    node.dkstUploadVideo = uploadVideo;
    node.dkstVideoLoadRestore = () => refresh(true).catch(showError);
    node.dkstVideoEditor = { video, inputs, timeline, handles, refresh, sync };
    const previousRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        state.removed = true; state.generation++;
        cancelAnimationFrame(state.animation);
        fileInput.remove();
        delete node.dkstUploadVideo;
        video.onloadedmetadata = video.onloadeddata = video.onerror = null;
        video.pause(); video.removeAttribute("src"); video.load();
        return previousRemoved?.apply(this, args);
    };
    node.size[0] = Math.max(node.size[0], 430);
    node.size[1] = Math.max(node.size[1], 696);
    node.dkstCropRestore?.();
    queueMicrotask(() => { if (!state.removed) node.dkstVideoLoadRestore(); });
}

app.registerExtension({
    name: "DINKIssTyle.VideoLoad",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== CLASS) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            mountEditor(this);
            return result;
        };
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = configure?.apply(this, arguments);
            const saved = info?.properties?.dkstVideoLoadSettings;
            if (saved) for (const name of SETTINGS) {
                const widget = this.widgets?.find(w => w.name === name);
                if (widget && saved[name] !== undefined) widget.value = saved[name];
            }
            queueMicrotask(() => this.dkstVideoLoadRestore?.());
            return result;
        };
        const serialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            const result = serialize?.apply(this, arguments);
            info.properties ??= {};
            info.properties.dkstVideoLoadSettings = Object.fromEntries(SETTINGS.map(name =>
                [name, this.widgets?.find(w => w.name === name)?.value]));
            return result;
        };
    },
});
