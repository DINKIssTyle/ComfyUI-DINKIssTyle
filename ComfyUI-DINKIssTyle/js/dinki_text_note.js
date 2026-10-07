import { app } from "/scripts/app.js";
import { renderNoteMarkdown, installNoteMarkdownStyles } from "./dinki_note_markdown.js";

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

function validNodeSize(size) {
    return size?.length === 2 && Number.isFinite(size[0]) && size[0] > 0 &&
        Number.isFinite(size[1]) && size[1] > 0;
}

function currentNodeSize(node) {
    // Nodes 2.0 can hold the resized geometry in its layout store.
    try { void node.renderingSize; } catch {}
    return validNodeSize(node.size) ? [node.size[0], node.size[1]] : null;
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
            installNoteMarkdownStyles();
            textWidget.hidden = true;
            if (textWidget.options) textWidget.options.hidden = true;

            const root = document.createElement("div");
            const toolbar = document.createElement("div");
            const lockButton = document.createElement("button");
            const copyButton = document.createElement("button");
            const textarea = document.createElement("textarea");
            const preview = document.createElement("div");
            Object.assign(root.style, {
                width: "100%", height: "100%", minHeight: "0", display: "flex",
                flexDirection: "column", overflow: "hidden", boxSizing: "border-box",
                background: "#202020", borderRadius: "6px",
            });
            Object.assign(toolbar.style, {
                display: "flex", flex: "0 0 auto", gap: "6px", padding: "6px",
                flexWrap: "wrap",
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
            copyButton.setAttribute("aria-live", "polite");
            textarea.value = String(textWidget.value ?? "");
            textarea.placeholder = "Write a note…";
            textarea.setAttribute("aria-label", "Note text");
            Object.assign(textarea.style, {
                flex: "1 1 0", minHeight: "0", width: "100%", boxSizing: "border-box",
                padding: "10px", border: "0", outline: "none", resize: "none",
                background: "transparent", color: "#eee", font: "14px/1.5 sans-serif",
            });
            preview.className = "dkst-note-preview";
            preview.tabIndex = 0;
            preview.setAttribute("role", "region");
            preview.setAttribute("aria-label", "Note Markdown preview");
            Object.assign(preview.style, {
                flex: "1 1 0", minHeight: "0", width: "100%", boxSizing: "border-box",
                padding: "10px", color: "#eee", font: "14px/1.5 sans-serif",
            });
            // Preserve native selection, scrolling and clipboard shortcuts inside
            // the note without passing those gestures to the graph canvas.
            for (const name of ["pointerdown", "mousedown", "dblclick", "wheel", "keydown"]) {
                root.addEventListener(name, event => event.stopPropagation());
            }
            toolbar.append(lockButton, copyButton);
            root.append(toolbar, textarea, preview);
            const widget = this.addDOMWidget("dkst_text_note", "DKST_TEXT_NOTE", root, {
                hideOnZoom: false,
                getMinHeight: () => 180,
                getMaxHeight: () => 10000,
                getHeight: () => 240,
            });
            widget.serialize = false;
            widget.options ??= {};
            widget.options.serialize = false;

            let renderedText;
            let renderingText;
            let renderGeneration = 0;
            const sync = () => {
                const text = String(textWidget.value ?? "");
                if (textarea.value !== text) textarea.value = text;
                // Lock is the single persisted state: locked notes render
                // Markdown, unlocked notes show their editable source.
                const locked = this.properties?.dkstNoteLocked === true;
                const previewing = locked;
                if (this.properties) delete this.properties.dkstNoteViewMode;
                textarea.style.display = previewing ? "none" : "block";
                preview.style.display = previewing ? "block" : "none";
                if (!previewing && renderingText !== undefined) {
                    renderGeneration++;
                    renderingText = undefined;
                }
                if (previewing && renderedText !== text && renderingText !== text) {
                    const generation = ++renderGeneration;
                    renderingText = text;
                    renderedText = undefined;
                    const isCurrent = () => generation === renderGeneration;
                    preview.textContent = "Loading preview…";
                    Promise.resolve().then(() => renderNoteMarkdown(preview, text, isCurrent)).then(success => {
                        if (!isCurrent()) return;
                        renderingText = undefined;
                        if (success !== false) {
                            renderedText = text;
                            preview.removeAttribute("title");
                        }
                    }).catch(error => {
                        if (!isCurrent()) return;
                        renderingText = undefined;
                        // Keep the source readable if rendering fails; retry on
                        // the next sync instead of caching a failed render.
                        preview.textContent = text;
                        preview.style.whiteSpace = "pre-wrap";
                        preview.title = "Preview failed to load. Unlock and Lock to retry.";
                        console.error("DKST note Markdown preview failed", error);
                    });
                }
                textarea.readOnly = locked;
                lockButton.textContent = locked ? "Unlock" : "Lock";
                lockButton.setAttribute("aria-pressed", String(locked));
                lockButton.style.background = locked ? "#526d94" : "#333";
                lockButton.title = locked ? "Unlock and edit Markdown source" : "Lock and preview Markdown";
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
                sync();
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
            let copyFeedbackTimer;
            copyButton.addEventListener("click", async () => {
                clearTimeout(copyFeedbackTimer);
                copyButton.disabled = true;
                copyButton.textContent = "Copying…";
                try {
                    await copyNoteText(String(textWidget.value ?? ""));
                    copyButton.textContent = "Copied!";
                    copyButton.style.background = "#326547";
                } catch (error) {
                    copyButton.textContent = "Copy failed";
                    copyButton.style.background = "#8a3d3d";
                    alert(`Copy: ${error.message}`);
                } finally {
                    copyButton.disabled = false;
                    copyFeedbackTimer = setTimeout(() => {
                        copyButton.textContent = "Copy";
                        copyButton.style.background = "#333";
                    }, 1800);
                }
            });

            const configured = this.onConfigure;
            this.onConfigure = function(info) {
                const configuredResult = configured?.apply(this, arguments);
                const savedSize = validNodeSize(info?.size) ? [info.size[0], info.size[1]] : null;
                if (savedSize) this.setSize?.(savedSize);
                queueMicrotask(() => {
                    if (savedSize) {
                        const current = currentNodeSize(this);
                        if (current?.[0] !== savedSize[0] || current?.[1] !== savedSize[1]) {
                            this.setSize?.(savedSize);
                        }
                    }
                    this.dkstSyncTextNote?.();
                });
                return configuredResult;
            };
            const removed = this.onRemoved;
            this.onRemoved = function() {
                renderGeneration++;
                clearTimeout(copyFeedbackTimer);
                return removed?.apply(this, arguments);
            };
            const serialized = this.onSerialize;
            this.onSerialize = function(info) {
                const serializedResult = serialized?.apply(this, arguments);
                const size = currentNodeSize(this);
                if (size) info.size = size;
                return serializedResult;
            };
            // Node configuration restores its saved size after creation. A
            // setSize() here can overwrite that value in the frontend layout.
            if (this.size) {
                this.size[0] = Math.max(this.size[0], 360);
                this.size[1] = Math.max(this.size[1], 280);
            }
            return result;
        };
    },
    loadedGraphNode(node) {
        if (node.comfyClass === "DINKI_Text_Note") node.dkstSyncTextNote?.();
    },
});
