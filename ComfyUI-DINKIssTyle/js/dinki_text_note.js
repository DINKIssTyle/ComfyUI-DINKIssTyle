import { app } from "/scripts/app.js";

function copyWithSelection(text) {
    const field = document.createElement("textarea");
    field.value = text;
    Object.assign(field.style, {
        position: "fixed", left: "-10000px", top: "0", opacity: "0",
    });
    const focused = document.activeElement;
    document.body.appendChild(field);
    try {
        field.focus();
        field.select();
        if (!document.execCommand?.("copy")) {
            throw new Error("The browser blocked copying. Select the note and use Ctrl+C or Cmd+C.");
        }
    } finally {
        field.remove();
        focused?.focus?.();
    }
}

async function copyNoteText(text) {
    // Async Clipboard is restricted to secure contexts. The selection command
    // can copy during a user click on ordinary HTTP when the browser permits it.
    if (globalThis.isSecureContext && globalThis.navigator?.clipboard?.writeText) {
        try {
            await navigator.clipboard.writeText(text);
            return;
        } catch {
            copyWithSelection(text);
            return;
        }
    }
    copyWithSelection(text);
}

app.registerExtension({
    name: "DINKI.TextNote",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DINKI_Text_Note") return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments);
            const textWidget = this.widgets?.find(widget => widget.name === "text");
            if (!textWidget) return result;
            textWidget.hidden = true;
            if (textWidget.options) textWidget.options.hidden = true;

            const root = document.createElement("div");
            const toolbar = document.createElement("div");
            const lockButton = document.createElement("button");
            const copyButton = document.createElement("button");
            const textarea = document.createElement("textarea");
            Object.assign(root.style, {
                width: "100%", height: "100%", minHeight: "0", display: "flex",
                flexDirection: "column", overflow: "hidden", boxSizing: "border-box",
                background: "#202020", borderRadius: "6px",
            });
            Object.assign(toolbar.style, {
                display: "flex", flex: "0 0 auto", gap: "6px", padding: "6px",
                borderBottom: "1px solid #444",
            });
            for (const button of [lockButton, copyButton]) {
                button.type = "button";
                Object.assign(button.style, {
                    border: "1px solid #555", borderRadius: "4px", padding: "4px 10px",
                    background: "#333", color: "#eee", cursor: "pointer",
                });
            }
            copyButton.textContent = "Copy";
            textarea.value = String(textWidget.value ?? "");
            textarea.placeholder = "Write a note…";
            textarea.setAttribute("aria-label", "Note text");
            Object.assign(textarea.style, {
                flex: "1 1 0", minHeight: "0", width: "100%", boxSizing: "border-box",
                padding: "10px", border: "0", outline: "none", resize: "none",
                background: "transparent", color: "#eee", font: "14px/1.5 sans-serif",
            });
            toolbar.append(lockButton, copyButton);
            root.append(toolbar, textarea);
            const widget = this.addDOMWidget("dkst_text_note", "DKST_TEXT_NOTE", root, {
                hideOnZoom: false,
                getMinHeight: () => 180,
                getMaxHeight: () => 10000,
                getHeight: () => 240,
            });
            widget.serialize = false;
            widget.options ??= {};
            widget.options.serialize = false;

            const sync = () => {
                const text = String(textWidget.value ?? "");
                if (textarea.value !== text) textarea.value = text;
                const locked = this.properties?.dkstNoteLocked === true;
                textarea.readOnly = locked;
                lockButton.textContent = locked ? "Unlock" : "Lock";
                lockButton.setAttribute("aria-pressed", String(locked));
                lockButton.style.background = locked ? "#526d94" : "#333";
                lockButton.title = locked ? "Unlock note" : "Lock note";
            };
            this.dkstSyncTextNote = sync;
            sync();

            textarea.addEventListener("input", () => {
                if (textarea.readOnly) {
                    textarea.value = String(textWidget.value ?? "");
                    return;
                }
                const value = textarea.value;
                const previous = textWidget.value;
                if (value === previous) return;
                textWidget.value = value;
                textWidget.options?.setValue?.(value);
                textWidget.callback?.call(textWidget, value, app.canvas, this);
                this.onWidgetChanged?.("text", value, previous, textWidget);
                this.graph?.incrementVersion?.();
                this.setDirtyCanvas?.(true, true);
            });
            lockButton.addEventListener("click", () => {
                this.properties ??= {};
                this.properties.dkstNoteLocked = !this.properties.dkstNoteLocked;
                sync();
                this.graph?.incrementVersion?.();
                this.setDirtyCanvas?.(true, true);
            });
            copyButton.addEventListener("click", () => {
                copyNoteText(String(textWidget.value ?? "")).catch(error => {
                    alert(`Copy: ${error.message}`);
                });
            });

            const configured = this.onConfigure;
            this.onConfigure = function() {
                const configuredResult = configured?.apply(this, arguments);
                queueMicrotask(() => this.dkstSyncTextNote?.());
                return configuredResult;
            };
            this.setSize?.([360, 280]);
            return result;
        };
    },
    loadedGraphNode(node) {
        if (node.comfyClass === "DINKI_Text_Note") node.dkstSyncTextNote?.();
    },
});
