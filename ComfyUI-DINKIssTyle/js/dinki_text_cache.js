import { app } from "/scripts/app.js";

app.registerExtension({
    name: "DINKI.TextMultilineCache",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DINKI_Text_Multiline_Cache") return;
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            const values = message?.dkst_cached_text;
            const text = Array.isArray(values)
                ? [...values].reverse().find(value => typeof value === "string" && value !== "") : undefined;
            const widget = this.widgets?.find(widget => widget.name === "text");
            if (!widget || text === undefined || widget.value === text) return result;
            widget.value = text;
            widget.callback?.call(widget, text);
            this.onWidgetChanged?.("text", text);
            this.graph?.change?.();
            // A nested graph's cached text is part of the root workflow too.
            if (this.graph && this.graph !== app.rootGraph) app.rootGraph?.change?.();
            this.setDirtyCanvas?.(true, true);
            return result;
        };
    },
});
