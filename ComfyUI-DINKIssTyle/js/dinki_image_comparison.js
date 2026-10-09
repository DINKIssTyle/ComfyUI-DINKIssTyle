import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

function comparisonFileUrl(descriptor) {
    return api.apiURL(`/view?${new URLSearchParams(descriptor)}`);
}

async function saveComparisonFile(descriptor) {
    const response = await fetch(comparisonFileUrl(descriptor));
    if (!response.ok) throw new Error(`Image download failed (${response.status})`);
    const objectUrl = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = descriptor.filename;
    document.body.appendChild(link);
    try {
        link.click();
    } finally {
        link.remove();
        setTimeout(() => URL.revokeObjectURL(objectUrl), 60000);
    }
}

function showComparisonMenu(event, actions) {
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation?.();
    const menu = document.createElement("div");
    Object.assign(menu.style, {
        position: "fixed", zIndex: "100000", background: "#252525", color: "white",
        padding: "5px", border: "1px solid #555", borderRadius: "6px",
        left: `${Math.max(0, Math.min(event.clientX, window.innerWidth - 200))}px`,
        top: `${Math.max(0, Math.min(event.clientY, window.innerHeight - 260))}px`,
    });
    const close = () => {
        menu.remove();
        document.removeEventListener("pointerdown", dismiss, true);
        document.removeEventListener("keydown", escape, true);
    };
    const dismiss = e => { if (!menu.contains(e.target)) close(); };
    const escape = e => { if (e.key === "Escape") close(); };
    for (const action of actions) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = action.content;
        button.disabled = !!action.disabled;
        Object.assign(button.style, {
            display: "block", width: "100%", padding: "9px 12px", textAlign: "left",
            background: "transparent", color: "inherit", border: "0", cursor: "pointer",
        });
        button.onclick = async() => {
            close();
            if (button.disabled) return;
            try { await action.callback(); } catch (error) { alert(error.message); }
        };
        menu.appendChild(button);
    }
    document.body.appendChild(menu);
    document.addEventListener("pointerdown", dismiss, true);
    document.addEventListener("keydown", escape, true);
}

app.registerExtension({
    name: "DINKI.ImageComparison",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DINKI_Image_Comparison") return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments);
            const root = document.createElement("div");
            Object.assign(root.style, {
                position: "relative",
                width: "100%", height: "100%", minHeight: "0",
                display: "flex", flexDirection: "column", contain: "size layout paint",
                overflow: "hidden", background: "#181818", borderRadius: "6px",
                userSelect: "none",
            });
            const toolbar = document.createElement("div");
            Object.assign(toolbar.style, {
                display: "flex", flexWrap: "wrap", alignItems: "center", gap: "4px",
                padding: "4px 6px", flex: "0 0 auto",
            });
            const viewport = document.createElement("div");
            Object.assign(viewport.style, {
                flex: "1 1 0", minWidth: "0", minHeight: "0", overflow: "hidden",
            });
            const panArea = document.createElement("div");
            Object.assign(panArea.style, {
                display: "grid", placeItems: "center", minWidth: "100%", minHeight: "100%",
            });
            const canvas = document.createElement("div");
            canvas.style.position = "relative";
            const makeImage = () => {
                const image = document.createElement("img");
                Object.assign(image.style, {
                    position: "absolute", inset: "0", width: "100%", height: "100%",
                    objectFit: "contain", objectPosition: "center", pointerEvents: "none",
                });
                image.draggable = false;
                return image;
            };
            const first = makeImage();
            const second = makeImage();
            const difference = makeImage();
            const divider = document.createElement("div");
            Object.assign(divider.style, {
                position: "absolute", top: "0", bottom: "0", left: "50%",
                width: "2px", background: "#fff", boxShadow: "0 0 3px #000",
                pointerEvents: "none",
            });
            const hint = document.createElement("div");
            hint.textContent = "Run the workflow to compare images";
            Object.assign(hint.style, {
                position: "absolute", inset: "0", display: "grid", placeItems: "center",
                color: "#bbb", font: "12px sans-serif", pointerEvents: "none",
            });
            canvas.append(first, second, difference, divider, hint);
            panArea.appendChild(canvas);
            viewport.appendChild(panArea);
            const minimap = document.createElement("div");
            Object.assign(minimap.style, {
                position: "absolute", right: "10px", bottom: "10px", zIndex: "2",
                padding: "5px", background: "rgba(20,20,20,.9)", border: "1px solid #777",
                borderRadius: "5px", boxShadow: "0 2px 8px #0008", display: "none",
            });
            const mapCanvas = document.createElement("div");
            mapCanvas.tabIndex = 0;
            mapCanvas.setAttribute("role", "group");
            mapCanvas.setAttribute("aria-label", "Comparison minimap. Click or drag to navigate; arrow keys to pan.");
            mapCanvas.title = "Click to jump · Drag to pan · Arrow keys to pan";
            Object.assign(mapCanvas.style, {
                position: "relative", overflow: "hidden", background: "#000",
                cursor: "crosshair", touchAction: "none",
            });
            const mapFirst = makeImage();
            const mapSecond = makeImage();
            const mapDifference = makeImage();
            const mapWindow = document.createElement("div");
            Object.assign(mapWindow.style, {
                position: "absolute", boxSizing: "border-box", border: "2px solid #fff",
                background: "#ffffff20", boxShadow: "0 0 2px #000", pointerEvents: "none",
            });
            mapCanvas.append(mapFirst, mapSecond, mapDifference, mapWindow);
            minimap.appendChild(mapCanvas);
            root.append(toolbar, viewport, minimap);
            const widget = this.addDOMWidget("dkst_image_comparison", "DKST_IMAGE_COMPARISON", root, {
                hideOnZoom: false, getMinHeight: () => 180,
                getMaxHeight: () => 700, getHeight: () => 300,
            });
            widget.serialize = false;
            widget.options.serialize = false;

            let descriptors = [];
            let position = 50;
            let naturalWidth = 0;
            let naturalHeight = 0;
            const validZoom = value => ["fit", .25, .5, .75, 1, 1.5, 2, 4].includes(value);
            let zoom = validZoom(this.properties?.dkstComparison?.zoom) ? this.properties.dkstComparison.zoom : "fit";
            let selectedMode = this.widgets?.find(item => item.name === "mode")?.value || "Slide";
            const imageUrl = comparisonFileUrl;
            const zoomButtons = [];
            const clamp = (value, max = 1) => Math.max(0, Math.min(max, value));
            let mapDrag = null;
            const visibleArea = () => {
                const bounds = canvas.getBoundingClientRect();
                const view = viewport.getBoundingClientRect();
                if (!bounds.width || !bounds.height || !viewport.offsetWidth || !viewport.offsetHeight) return null;
                const scaleX = view.width / viewport.offsetWidth;
                const scaleY = view.height / viewport.offsetHeight;
                return {
                    left: clamp((view.left - bounds.left) / bounds.width),
                    top: clamp((view.top - bounds.top) / bounds.height),
                    right: clamp((view.left + viewport.clientWidth * scaleX - bounds.left) / bounds.width),
                    bottom: clamp((view.top + viewport.clientHeight * scaleY - bounds.top) / bounds.height),
                };
            };
            const updateMinimap = () => {
                const active = descriptors.length === 3 && zoom !== "fit" && naturalWidth && naturalHeight
                    && viewport.clientWidth > 0 && viewport.clientHeight > 0
                    && (canvas.clientWidth > viewport.clientWidth || canvas.clientHeight > viewport.clientHeight);
                minimap.style.display = active ? "block" : "none";
                if (!active) return;
                const scale = Math.min(Math.min(132, viewport.clientWidth * .33) / naturalWidth,
                    Math.min(100, viewport.clientHeight * .33) / naturalHeight);
                mapCanvas.style.width = `${naturalWidth * scale}px`;
                mapCanvas.style.height = `${naturalHeight * scale}px`;
                const area = visibleArea();
                if (!area) return;
                Object.assign(mapWindow.style, {
                    left: `${area.left * 100}%`, top: `${area.top * 100}%`,
                    width: `${(area.right - area.left) * 100}%`,
                    height: `${(area.bottom - area.top) * 100}%`,
                });
            };
            const navigate = (x, y) => {
                viewport.scrollLeft = clamp(x * canvas.clientWidth - viewport.clientWidth / 2,
                    Math.max(0, canvas.clientWidth - viewport.clientWidth));
                viewport.scrollTop = clamp(y * canvas.clientHeight - viewport.clientHeight / 2,
                    Math.max(0, canvas.clientHeight - viewport.clientHeight));
                updateMinimap();
            };
            const mapPoint = event => {
                // Screen bounds account for ComfyUI's graph zoom as well.
                const bounds = mapCanvas.getBoundingClientRect();
                if (!bounds.width || !bounds.height) return null;
                return { x: (event.clientX - bounds.left) / bounds.width,
                    y: (event.clientY - bounds.top) / bounds.height };
            };
            mapCanvas.addEventListener("pointerdown", event => {
                if (event.button !== 0 || minimap.style.display === "none") return;
                const point = mapPoint(event);
                const area = visibleArea();
                if (!point || !area) return;
                event.preventDefault();
                event.stopPropagation();
                const inside = point.x >= area.left && point.x <= area.right
                    && point.y >= area.top && point.y <= area.bottom;
                mapDrag = { id: event.pointerId,
                    x: inside ? point.x - (area.left + area.right) / 2 : 0,
                    y: inside ? point.y - (area.top + area.bottom) / 2 : 0 };
                mapCanvas.setPointerCapture(event.pointerId);
                navigate(point.x - mapDrag.x, point.y - mapDrag.y);
            });
            mapCanvas.addEventListener("pointermove", event => {
                event.stopPropagation();
                if (!mapDrag || event.pointerId !== mapDrag.id) return;
                const point = mapPoint(event);
                if (point) navigate(point.x - mapDrag.x, point.y - mapDrag.y);
            });
            const endMapDrag = event => {
                event.stopPropagation();
                if (!mapDrag || event.pointerId !== mapDrag.id) return;
                if (mapCanvas.hasPointerCapture(event.pointerId)) mapCanvas.releasePointerCapture(event.pointerId);
                mapDrag = null;
            };
            mapCanvas.addEventListener("pointerup", endMapDrag);
            mapCanvas.addEventListener("pointercancel", endMapDrag);
            mapCanvas.addEventListener("lostpointercapture", () => { mapDrag = null; });
            mapCanvas.addEventListener("keydown", event => {
                const direction = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
                if (!direction) return;
                event.preventDefault();
                event.stopPropagation();
                const area = visibleArea();
                if (area) navigate((area.left + area.right) / 2 + direction[0] * (area.right - area.left) * .25,
                    (area.top + area.bottom) / 2 + direction[1] * (area.bottom - area.top) * .25);
            });
            for (const eventName of ["mousedown", "click", "wheel"]) {
                minimap.addEventListener(eventName, event => event.stopPropagation());
            }
            viewport.addEventListener("scroll", updateMinimap);
            const resizeObserver = typeof ResizeObserver === "function" ? new ResizeObserver(updateMinimap) : null;
            resizeObserver?.observe(viewport);
            resizeObserver?.observe(canvas);
            const persistView = () => {
                if (this.properties?.dkstComparison) {
                    Object.assign(this.properties.dkstComparison, { zoom, position });
                }
            };
            const applyZoom = () => {
                const fit = zoom === "fit" || !naturalWidth || !naturalHeight;
                viewport.style.overflow = fit ? "hidden" : "auto";
                panArea.style.width = fit ? "100%" : "max-content";
                panArea.style.height = fit ? "100%" : "max-content";
                canvas.style.width = fit ? "100%" : `${naturalWidth * zoom}px`;
                canvas.style.height = fit ? "100%" : `${naturalHeight * zoom}px`;
                for (const [button, value] of zoomButtons) {
                    button.disabled = value === zoom || (value !== "fit" && (!naturalWidth || !naturalHeight));
                    button.setAttribute("aria-pressed", String(value === zoom));
                }
                updateMinimap();
            };
            for (const [label, value] of [["25%", .25], ["50%", .5], ["75%", .75], ["100%", 1],
                ["150%", 1.5], ["200%", 2], ["400%", 4], ["Fit", "fit"]]) {
                const button = document.createElement("button");
                button.type = "button";
                button.textContent = label;
                button.title = value === "fit" ? "Fit the aligned comparison canvas to the preview" : `${label} of the aligned comparison canvas`;
                Object.assign(button.style, {
                    height: "24px", padding: "0 8px", border: "1px solid #555",
                    borderRadius: "4px", background: "#2b2b2b", color: "#eee", cursor: "pointer",
                });
                button.onclick = () => {
                    const area = visibleArea();
                    zoom = value;
                    persistView();
                    applyZoom();
                    if (area && value !== "fit") navigate((area.left + area.right) / 2, (area.top + area.bottom) / 2);
                    this.setDirtyCanvas?.(true, true);
                };
                zoomButtons.push([button, value]);
                toolbar.appendChild(button);
            }
            for (const eventName of ["pointerdown", "mousedown", "click"]) {
                toolbar.addEventListener(eventName, event => event.stopPropagation());
            }
            viewport.addEventListener("wheel", event => {
                if (zoom !== "fit") event.stopPropagation();
            });
            first.onload = () => {
                if (!naturalWidth || !naturalHeight) {
                    naturalWidth = first.naturalWidth;
                    naturalHeight = first.naturalHeight;
                    applyZoom();
                }
            };
            const show = () => {
                const ready = descriptors.length === 3;
                const slide = selectedMode !== "Difference";
                first.style.display = ready && slide ? "block" : "none";
                second.style.display = ready && slide ? "block" : "none";
                difference.style.display = ready && !slide ? "block" : "none";
                divider.style.display = ready && slide ? "block" : "none";
                hint.style.display = ready ? "none" : "grid";
                second.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
                mapFirst.style.display = first.style.display;
                mapSecond.style.display = second.style.display;
                mapDifference.style.display = difference.style.display;
                mapSecond.style.clipPath = second.style.clipPath;
                divider.style.left = `${position}%`;
                canvas.style.cursor = slide ? "ew-resize" : "default";
            };
            viewport.addEventListener("pointermove", event => {
                if (selectedMode === "Difference" || descriptors.length !== 3) return;
                // All three prepared previews share this canvas. Its actual
                // screen bounds include graph zoom, centering and scroll offset.
                const bounds = canvas.getBoundingClientRect();
                if (!bounds.width) return;
                position = Math.max(0, Math.min(100, 100 * (event.clientX - bounds.left) / bounds.width));
                persistView();
                show();
            });
            const menuActions = () => {
                const changeMode = mode => {
                    const modeWidget = this.widgets?.find(item => item.name === "mode");
                    if (modeWidget) {
                        const previous = modeWidget.value;
                        modeWidget.value = mode;
                        modeWidget.callback?.(mode);
                        this.onWidgetChanged?.("mode", mode, previous, modeWidget);
                    }
                    this.dkstSetComparisonMode?.(mode);
                    this.setDirtyCanvas?.(true, true);
                };
                const fileAction = (content, index, save) => ({
                    content, disabled: !descriptors[index],
                    callback: () => save
                        ? saveComparisonFile(descriptors[index])
                        : window.open(comparisonFileUrl(descriptors[index]), "_blank", "noopener,noreferrer"),
                });
                return [
                    { content: "Mode: Slide", callback: () => changeMode("Slide") },
                    { content: "Mode: Difference", callback: () => changeMode("Difference") },
                    fileAction("Open Image 1", 0, false),
                    fileAction("Save Image 1", 0, true),
                    fileAction("Open Image 2", 1, false),
                    fileAction("Save Image 2", 1, true),
                ];
            };
            for (const eventName of ["pointerdown", "mousedown"]) {
                root.addEventListener(eventName, event => {
                    if (event.button === 2) event.stopPropagation();
                });
            }
            root.addEventListener("contextmenu", event => showComparisonMenu(event, menuActions()));
            const extraMenu = this.getExtraMenuOptions;
            this.getExtraMenuOptions = function(canvas, options) {
                const result = extraMenu?.apply(this, arguments);
                options.push(...menuActions());
                return result;
            };
            this.dkstSetComparisonMode = value => {
                selectedMode = value === "Difference" ? "Difference" : "Slide";
                show();
            };
            this.dkstSetComparisonImages = (message, persist = true) => {
                descriptors = message?.dkst_comparison || [];
                const size = /^(\d+)\s*[×x]\s*(\d+)$/.exec(message?.resolution?.[0] || "");
                naturalWidth = size ? Number(size[1]) : 0;
                naturalHeight = size ? Number(size[2]) : 0;
                if (!persist) {
                    if (validZoom(message?.zoom)) zoom = message.zoom;
                    if (Number.isFinite(message?.position)) position = Math.max(0, Math.min(100, message.position));
                }
                if (persist) {
                    this.properties ??= {};
                    this.properties.dkstComparison = {
                        dkst_comparison: descriptors,
                        resolution: message?.resolution || [],
                        zoom, position,
                    };
                }
                for (const [image, item] of [[first, descriptors[0]], [second, descriptors[1]],
                    [difference, descriptors[2]], [mapFirst, descriptors[0]],
                    [mapSecond, descriptors[1]], [mapDifference, descriptors[2]]]) {
                    if (item) image.src = imageUrl(item);
                    else image.removeAttribute("src");
                }
                applyZoom();
                show();
                this.setDirtyCanvas?.(true, true);
            };
            this.dkstRestoreComparison = () => {
                const stored = this.properties?.dkstComparison;
                const output = app.nodeOutputs?.[this.id];
                this.dkstSetComparisonMode(this.widgets?.find(item => item.name === "mode")?.value);
                if (stored?.dkst_comparison?.length === 3) this.dkstSetComparisonImages(stored, false);
                else if (output?.dkst_comparison?.length === 3) this.dkstSetComparisonImages(output, false);
            };
            const changed = this.onWidgetChanged;
            this.onWidgetChanged = function(name, value) {
                const changedResult = changed?.apply(this, arguments);
                if (name === "mode") this.dkstSetComparisonMode?.(value);
                return changedResult;
            };
            const node = this;
            const modeWidget = this.widgets?.find(item => item.name === "mode");
            if (modeWidget) {
                const callback = modeWidget.callback;
                modeWidget.callback = function(value) {
                    const callbackResult = callback?.apply(this, arguments);
                    node.dkstSetComparisonMode?.(value);
                    return callbackResult;
                };
            }
            const configured = this.onConfigure;
            this.onConfigure = function() {
                const configuredResult = configured?.apply(this, arguments);
                queueMicrotask(() => this.dkstRestoreComparison?.());
                return configuredResult;
            };
            const removed = this.onRemoved;
            this.onRemoved = function() {
                resizeObserver?.disconnect();
                mapDrag = null;
                first.onload = null;
                for (const image of [first, second, difference, mapFirst, mapSecond, mapDifference]) image.removeAttribute("src");
                return removed?.apply(this, arguments);
            };
            applyZoom();
            show();
            return result;
        };

        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            const result = executed?.apply(this, arguments);
            this.dkstSetComparisonImages?.(message);
            return result;
        };
    },
    loadedGraphNode(node) {
        if (node.comfyClass === "DINKI_Image_Comparison") node.dkstRestoreComparison?.();
    },
});
