import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "DINKI.ImageComparison",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DINKI_Image_Comparison") return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments);
            const root = document.createElement("div");
            Object.assign(root.style, {
                position: "relative", width: "100%", height: "100%", minHeight: "180px",
                overflow: "hidden", background: "#181818", borderRadius: "6px",
                cursor: "ew-resize", userSelect: "none", touchAction: "none",
            });
            const makeImage = () => {
                const image = document.createElement("img");
                Object.assign(image.style, {
                    position: "absolute", inset: "0", width: "100%", height: "100%",
                    objectFit: "contain", pointerEvents: "none",
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
            root.append(first, second, difference, divider, hint);
            const widget = this.addDOMWidget("dkst_image_comparison", "DKST_IMAGE_COMPARISON", root, {
                hideOnZoom: false, getMinHeight: () => 180,
                getMaxHeight: () => 700, getHeight: () => 300,
            });
            widget.serialize = false;
            widget.options.serialize = false;

            let descriptors = [];
            let position = 50;
            let selectedMode = this.widgets?.find(item => item.name === "mode")?.value || "Slide";
            const imageUrl = item => api.apiURL(`/view?${new URLSearchParams(item)}`);
            const show = () => {
                const ready = descriptors.length === 3;
                const slide = selectedMode !== "Difference";
                first.style.display = ready && slide ? "block" : "none";
                second.style.display = ready && slide ? "block" : "none";
                difference.style.display = ready && !slide ? "block" : "none";
                divider.style.display = ready && slide ? "block" : "none";
                hint.style.display = ready ? "none" : "grid";
                second.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
                divider.style.left = `${position}%`;
                root.style.cursor = slide ? "ew-resize" : "default";
            };
            root.addEventListener("pointermove", event => {
                if (selectedMode === "Difference" || descriptors.length !== 3) return;
                const bounds = root.getBoundingClientRect();
                if (!bounds.width) return;
                position = Math.max(0, Math.min(100, 100 * (event.clientX - bounds.left) / bounds.width));
                show();
            });
            this.dkstSetComparisonMode = value => {
                selectedMode = value === "Difference" ? "Difference" : "Slide";
                show();
            };
            this.dkstSetComparisonImages = (message, persist = true) => {
                descriptors = message?.dkst_comparison || [];
                if (persist) {
                    this.properties ??= {};
                    this.properties.dkstComparison = {
                        dkst_comparison: descriptors,
                        resolution: message?.resolution || [],
                    };
                }
                for (const [image, item] of [[first, descriptors[0]], [second, descriptors[1]],
                    [difference, descriptors[2]]]) {
                    if (item) image.src = imageUrl(item);
                    else image.removeAttribute("src");
                }
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
