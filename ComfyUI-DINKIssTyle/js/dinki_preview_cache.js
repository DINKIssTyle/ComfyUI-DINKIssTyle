import { app } from "/scripts/app.js";

app.registerExtension({
    name: "DINKI.PreviewImage.Cache",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DINKI_Preview_Image_Cache") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const widget = this.widgets?.find(widget => widget.name === "cache_file");
            if (widget) {
                // Keep the managed reference serialized but out of both node UIs.
                widget.type = "converted-widget";
                widget.hidden = true;
                widget.computeSize = () => [0, -4];
            }
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            try {
                return executed?.apply(this, arguments);
            } finally {
                const files = message?.dkst_cache_file;
                const file = Array.isArray(files) ? [...files].reverse().find(value =>
                    typeof value === "string" && /^[0-9a-f]{32}\.npy$/.test(value)) : undefined;
                const widget = this.widgets?.find(widget => widget.name === "cache_file");
                if (widget && file !== undefined && widget.value !== file) {
                    const previous = widget.value;
                    widget.value = file;
                    widget.callback?.call(widget, file);
                    this.onWidgetChanged?.("cache_file", file, previous);
                    this.graph?.change?.();
                    if (this.graph && this.graph !== app.rootGraph) app.rootGraph?.change?.();
                    this.setDirtyCanvas?.(true, true);
                }
            }
        };
    },
});
