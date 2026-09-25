// ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/js/dinki_nodes.js

import { app, ComfyApp } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Convert the old orientation dropdown when loading existing workflows.
app.registerExtension({
    name: "DINKI.PhotoSpecifications.Orientation",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DINKI_photo_specifications") return;
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = onConfigure?.apply(this, arguments);
            const widget = getWidget(this, "orientation");
            if (widget?.value === "Portrait" || widget?.value === "Landscape") {
                widget.value = widget.value === "Landscape";
            }
            return result;
        };
    },
});

// 공통 헬퍼
function getWidget(node, name) {
  return node.widgets?.find(w => w.name === name);
}
function ensureLater(fn) {
  requestAnimationFrame(() => setTimeout(fn, 0));
}

async function clipboardImagePNG(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Image download failed (${response.status})`);
    const blob = await response.blob();
    if (blob.type === "image/png") return blob;
    const bitmap = await createImageBitmap(blob);
    try {
        const canvas = document.createElement("canvas");
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        canvas.getContext("2d").drawImage(bitmap, 0, 0);
        return await new Promise((resolve, reject) => canvas.toBlob(
            png => png ? resolve(png) : reject(new Error("Unable to convert image to PNG.")), "image/png",
        ));
    } finally {
        bitmap.close();
    }
}

function copyImageToClipboard(url) {
    if (!globalThis.navigator?.clipboard?.write || typeof ClipboardItem === "undefined") {
        throw new Error("Copy Image requires clipboard support and HTTPS or localhost.");
    }
    // Start write during the click gesture; Safari also requires this before
    // the asynchronous image download/conversion completes.
    return navigator.clipboard.write([new ClipboardItem({ "image/png": clipboardImagePNG(url) })]);
}

async function pasteImageFromClipboard(node) {
    if (!globalThis.navigator?.clipboard?.read) {
        throw new Error("Paste Image requires HTTPS or localhost and clipboard support. Select this node and press Ctrl+V / Cmd+V instead.");
    }
    const items = await navigator.clipboard.read();
    for (const item of items) {
        const type = item.types.find(type => type === "image/png") || item.types.find(type => type.startsWith("image/"));
        if (type) {
            await node.dkstUploadClipboardImage(await item.getType(type));
            return;
        }
    }
    throw new Error("No image found in the clipboard. Copy an image first.");
}

function clipboardMenuAction(content, action) {
    return { content, callback: async() => {
        try { await action(); } catch (error) { alert(`${content}: ${error.message}`); }
    } };
}

function previewImageActions(descriptor) {
    const url = api.apiURL(`/view?${new URLSearchParams(descriptor)}`);
    return [
        { content: "Open Image", callback: () => window.open(url, "_blank", "noopener,noreferrer") },
        { content: "Copy Image", callback: () => copyImageToClipboard(url) },
        { content: "Save Image", callback: async() => {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`Image download failed (${response.status})`);
            const objectUrl = URL.createObjectURL(await response.blob());
            const link = document.createElement("a");
            link.href = objectUrl;
            link.download = descriptor.filename;
            document.body.appendChild(link);
            link.click();
            link.remove();
            setTimeout(() => URL.revokeObjectURL(objectUrl), 60000);
        } },
    ];
}

function showPreviewImageMenu(event, descriptor) {
    event.preventDefault();
    event.stopImmediatePropagation();
    const menu = document.createElement("div");
    Object.assign(menu.style, {
        position: "fixed", zIndex: "100000", background: "#252525", color: "white",
        padding: "5px", border: "1px solid #555", borderRadius: "6px",
        left: `${Math.max(0, Math.min(event.clientX, window.innerWidth - 190))}px`,
        top: `${Math.max(0, Math.min(event.clientY, window.innerHeight - 100))}px`,
    });
    const close = () => {
        menu.remove();
        document.removeEventListener("pointerdown", dismiss, true);
        document.removeEventListener("keydown", escape, true);
    };
    const dismiss = e => { if (!menu.contains(e.target)) close(); };
    const escape = e => { if (e.key === "Escape") close(); };
    for (const action of previewImageActions(descriptor)) {
        const button = document.createElement("button");
        button.textContent = action.content;
        Object.assign(button.style, { display: "block", width: "100%", padding: "9px 12px", textAlign: "left", background: "transparent", color: "inherit", border: "0", cursor: "pointer" });
        button.onclick = async() => {
            close();
            try { await action.callback(); } catch (error) { alert(error.message); }
        };
        menu.appendChild(button);
    }
    document.body.appendChild(menu);
    document.addEventListener("pointerdown", dismiss, true);
    document.addEventListener("keydown", escape, true);
}

// ============================================================
// 1. DINKI Prompt Selector Logic
// ============================================================
app.registerExtension({
    name: "DINKI.PromptSelector.Logic",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DINKI_PromptSelector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const originalWidget = this.widgets.find(w => w.name === "title");

                const comboWidget = this.addWidget(
                    "combo",
                    "title",
                    "",
                    (value) => {
                        originalWidget.value = value;
                    },
                    { values: [] }
                );
                comboWidget.serialize = false;
                originalWidget.hidden = true;
                
                const refreshButton = this.addWidget(
                    "button",
                    "🔄 Refresh Prompts",
                    null,
                    () => refreshPromptList(true)
                );

                const refreshPromptList = async (force) => {
                    try {
                        if (force || !comboWidget.options.values || comboWidget.options.values.length === 0) {
                            const response = await api.fetchApi('/get-csv-prompts');
                            const titles = await response.json();
                            
                            comboWidget.options.values = titles;
                            
                            if (!titles.includes(comboWidget.value) && titles.length > 0) {
                                comboWidget.value = titles[0];
                            } else if (titles.length === 0) {
                                comboWidget.value = "";
                            }
                        }
                    } catch (error) {
                        console.error("❌ Error refreshing DINKI prompt list:", error);
                    } finally {
                        if (comboWidget.callback) {
                            comboWidget.callback(comboWidget.value);
                        }
                    }
                };

                refreshPromptList(false);

                this.widgets.splice(this.widgets.indexOf(originalWidget), 1);
                this.widgets.splice(0, 0, comboWidget);
            };
        }
    },
});

// ============================================================
// 2. DINKI Prompt Selector Live Attach v2
// ============================================================
const livePromptContexts = new Map();
const livePromptObjectIds = new WeakMap();
let livePromptNextId = 0;
let livePromptMonitor;
function syncLivePromptSubgraphs(baseline = false) {
    if (app.configuringGraph) return;
    const seen = new Set();
    const visiting = new Set();
    const identity = node => {
        if (!livePromptObjectIds.has(node)) livePromptObjectIds.set(node, ++livePromptNextId);
        return livePromptObjectIds.get(node);
    };
    const visit = (graph, bindings = new Map(), path = []) => {
        if (!graph || visiting.has(graph)) return;
        visiting.add(graph);
        for (const node of graph.nodes ?? graph._nodes ?? []) {
            const nodeBindings = bindings.get(node) ?? {};
            if (node.__dinki_live_sync) {
                const key = [...path, identity(node)].join("/");
                seen.add(key);
                let context = livePromptContexts.get(key);
                const initial = !context;
                if (!context) {
                    context = { live: true, node };
                    livePromptContexts.set(key, context);
                }
                context.bindings = nodeBindings;
                context.read = name => nodeBindings[name]?.[0]?.widget.value ?? getWidget(node, name)?.value;
                node.__dinki_live_sync(context, baseline || initial);
            }
            if (node.subgraph) {
                const childBindings = new Map();
                const inherited = Object.fromEntries(Object.entries(nodeBindings).map(([name, chain]) => [name, chain[0]?.widget.value]));
                nodeModePromotedValues(node, inherited, (widget, target, name, sourceName) => {
                    if (!widget) return;
                    if (!childBindings.has(target)) childBindings.set(target, {});
                    childBindings.get(target)[name] = [...(nodeBindings[sourceName] ?? []), { node, widget }];
                });
                visit(node.subgraph, childBindings, [...path, identity(node)]);
            }
        }
        visiting.delete(graph);
    };
    visit(app.rootGraph ?? app.graph);
    for (const [key, context] of livePromptContexts) {
        if (!seen.has(key)) {
            context.live = false;
            livePromptContexts.delete(key);
        }
    }
}

app.registerExtension({
  name: "DINKI.PromptSelectorLive.Attach.v2",
  setup() {
    if (livePromptMonitor === undefined) {
      livePromptMonitor = setInterval(() => syncLivePromptSubgraphs(), 100);
    }
  },
  afterConfigureGraph() {
    // Loading a saved append-mode workflow must not append its preset again.
    syncLivePromptSubgraphs(true);
  },
  async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
    if (nodeData?.name !== "DINKI_PromptSelectorLive") return;

    if (nodeType.prototype.__dinki_live_patched) return;
    nodeType.prototype.__dinki_live_patched = true;

    async function attach(node) {
      if (node.__dinki_live_attached || node.__dinki_live_attach_pending) return;
      node.__dinki_live_attach_pending = true;

      const attachWhenReady = (attempt = 0) => {
        const titleW = getWidget(node, "title");
        const textW  = getWidget(node, "text");
        if (!titleW || !textW) {
          // Vue/Nodes 2.0 may create widgets after the node lifecycle hook.
          if (attempt < 120) requestAnimationFrame(() => attachWhenReady(attempt + 1));
          else node.__dinki_live_attach_pending = false;
          return;
        }
        node.__dinki_live_attached = true;
        node.__dinki_live_attach_pending = false;

        const setTextValue = (value, context) => {
          const oldValue = textW.value;

          // `value` is enough for classic widgets. Nodes 2.0 multiline
          // widgets additionally keep their value in a Vue/DOM value store.
          textW.value = value;
          textW.options?.setValue?.(value);

          const inputEl = textW.inputEl || textW.element;
          if (inputEl && "value" in inputEl && inputEl.value !== value) {
            inputEl.value = value;
            inputEl.dispatchEvent?.(new Event("input", { bubbles: true }));
            inputEl.dispatchEvent?.(new Event("change", { bubbles: true }));
          }

          try {
            textW.callback?.call(textW, value, app.canvas, node);
          } catch (e) {
            console.warn("DINKI Live text callback error:", e);
          }
          node.onWidgetChanged?.("text", value, oldValue, textW);
          node.graph?.incrementVersion?.();
          // A promoted text input can own a separate value store. Update every
          // host along the same instance path so the visible text and queued
          // prompt agree, including renamed/nested inputs such as text_1.
          for (const { node: host, widget } of context?.bindings.text ?? []) {
            if (widget.value === value) continue;
            widget.value = value;
            widget.options?.setValue?.(value);
            host.graph?.incrementVersion?.();
            host.setDirtyCanvas?.(true, true);
          }
        };

        if (!node.__dinki_live_clear_added) {
          node.addWidget("button", "Clear", null, () => {
            const tW = getWidget(node, "text");
            if (tW) {
              tW.value = "";
              node.setDirtyCanvas(true);
            }
          });
          node.__dinki_live_clear_added = true;
        }

        if (!node.__dinki_live_refresh_added) {
          node.addWidget("button", "🔄 Refresh Prompts", null, async () => {
            try {
              const res = await api.fetchApi("/get-csv-prompts");
              if (!res.ok) throw new Error(`HTTP ${res.status}`);
              const titles = await res.json();

              if (!titleW.options) titleW.options = {};
              titleW.options.values = Array.isArray(titles) ? titles : [];

              if (!titleW.options.values.includes(titleW.value)) {
                titleW.value = titleW.options.values.length ? titleW.options.values[0] : "";
              }

              if (titleW.callback) titleW.callback(titleW.value);
              node.setDirtyCanvas(true);
            } catch (e) {
              console.error("DINKI Live refresh error:", e);
            }
          });
          node.__dinki_live_refresh_added = true;
        }

        if (!node.__dinki_live_cb_wrapped) {
          const origCb = titleW.callback;
          let observedTitle = titleW.value;

          const loadSelectedPrompt = async (value, context) => {
            titleW.value = value;
            observedTitle = value;
            if (context) context.observedTitle = value;
            const read = name => context ? context.read(name) : getWidget(node, name)?.value;
            const state = context ?? node;
            const sepVal = read("separator") ?? "\n";
            const sig = JSON.stringify([value, read("mode") || "append", sepVal, read("text")]);
            if (state.__dinki_last_apply_sig === sig) return;
            state.__dinki_last_apply_sig = sig;
            const requestId = (state.__dinki_live_request_id || 0) + 1;
            state.__dinki_live_request_id = requestId;

            try {
              const res = await api.fetchApi("/dinki/prompts");
              if (!res.ok) throw new Error(`HTTP ${res.status}`);
              const map = await res.json();
              // Ignore a slow response if another preset was selected while
              // this request was in flight.
              if (requestId !== state.__dinki_live_request_id || read("title") !== value || context?.live === false || app.configuringGraph) return;
              const selectedTitle = String(value ?? titleW.value ?? "").trim();
              let picked = (map && selectedTitle) ? (map[selectedTitle] || "") : "";
              if (!picked && map && typeof map === "object") {
                const matchedTitle = Object.keys(map).find(
                  key => key.trim().toLocaleLowerCase() === selectedTitle.toLocaleLowerCase()
                );
                if (matchedTitle) picked = map[matchedTitle] || "";
              }
              const mode = read("mode") || "append";
              let sep = sepVal;
              if (sep === "\\n") sep = "\n";
              if (sep === "\\n\\n") sep = "\n\n";
              if (!picked) {
                if (selectedTitle && selectedTitle !== "-- None --") {
                  console.warn("[DINKI Live] Selected title was not found in /dinki/prompts:", selectedTitle);
                }
                return;
              }

              if (mode === "replace") {
                setTextValue(picked, context);
              } else if (mode === "append") {
                const currentText = read("text") || "";
                let nextValue;
                if (!currentText) nextValue = picked;
                else nextValue = (sep && !currentText.endsWith(sep))
                  ? currentText + sep + picked
                  : currentText + picked;
                setTextValue(nextValue, context);
              }
              node.setDirtyCanvas(true, true);
              console.info("[DINKI Live] Prompt applied:", selectedTitle, `(${picked.length} chars)`);
            } catch (e) {
              console.error("DINKI Live fetch/prompts error:", e);
            } finally {
              if (requestId === state.__dinki_live_request_id) {
                setTimeout(() => { state.__dinki_last_apply_sig = null; }, 0);
              }
            }
          };

          node.__dinki_live_sync = (context, baseline) => {
            const title = context.read("title");
            if (baseline) {
              context.observedTitle = title;
              context.__dinki_live_request_id = (context.__dinki_live_request_id || 0) + 1;
            } else if (title !== context.observedTitle) {
              loadSelectedPrompt(title, context);
            }
          };

          const loadFromWidget = value => {
            const contexts = [...livePromptContexts.values()].filter(context =>
              context.node === node && (!context.bindings.title?.length || context.read("title") === value));
            return contexts.length
              ? Promise.all(contexts.map(context => loadSelectedPrompt(value, context)))
              : loadSelectedPrompt(value);
          };

          titleW.callback = function (value, ...args) {
            // Some ComfyUI widget implementations expect their callback's
            // `this` value to be the widget.  A callback error must not stop
            // the live prompt lookup.
            try {
              origCb?.call(titleW, value, ...args);
            } catch (e) {
              console.warn("DINKI Live original title callback error:", e);
            }
            return loadFromWidget(value);
          };

          // Nodes 2.0 reports widget edits through the node notification and
          // may not invoke the classic widget callback at all.
          const origWidgetChanged = node.onWidgetChanged;
          node.onWidgetChanged = function (name, value) {
            const result = origWidgetChanged?.apply(this, arguments);
            if (name === "title") loadFromWidget(value);
            return result;
          };

          // Final compatibility path: a few Nodes 2.0 renderers update the
          // store-backed combo without calling either legacy hook.
          const origDrawForeground = node.onDrawForeground;
          node.onDrawForeground = function () {
            const result = origDrawForeground?.apply(this, arguments);
            if (titleW.value !== observedTitle) {
              observedTitle = titleW.value;
              loadFromWidget(observedTitle);
            }
            return result;
          };

          node.__dinki_live_cb_wrapped = true;
          syncLivePromptSubgraphs();
        }
      };

      ensureLater(() => attachWhenReady());
    }

    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origCreated?.apply(this, arguments);
      attach(this);
      return r;
    };

    const origAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function () {
      const r = origAdded?.apply(this, arguments);
      attach(this);
      return r;
    };
  },
});


// ============================================================
// 3. DINKI Prompt Selector Auto Reset
// ============================================================
function resetTitleWidget(node) {
  const w = node?.widgets?.find(w => w.name === "title");
  if (!w) return;
  const noneIdx = (w.options || []).indexOf("-- None --");
  if (noneIdx >= 0) {
    w.value = "-- None --";
  } else {
    w.value = (w.options && w.options[0]) || w.value;
  }
  if (w.callback) try { w.callback(w.value); } catch (e) {}
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "DINKI.PromptSelector.AutoReset",
  async setup() {
    api.addEventListener("executedNode", ({ detail }) => {
      const { node } = detail || {};
      if (!node) return;
      const targetNames = ["DINKI_PromptSelector", "DINKI_PromptSelectorLive"];
      if (!targetNames.includes(node?.comfyClass)) return;
      resetTitleWidget(node);
    });
  }
});


// ============================================================
// 4. DINKI Node Switch Logic
// ============================================================
function applyNodeSwitch(node, changedName, changedValue, values = {}) {
    // app.graph is the currently displayed graph, which may be a different subgraph.
    const graph = node.graph;
    if (!graph || app.configuringGraph) return;

    const idWidget = getWidget(node, "node_ids");
    const toggleWidget = getWidget(node, "active");
    if (!idWidget || !toggleWidget) return;

    // Some widget renderers notify before committing widget.value.
    const idsText = changedName === "node_ids" ? changedValue : (values.node_ids ?? idWidget.value);
    const isActive = changedName === "active" ? changedValue : (values.active ?? toggleWidget.value);
    const ids = new Set(String(idsText ?? "").split(",").map(id => id.trim()).filter(Boolean));
    let changed = false;

    for (const target of graph.nodes ?? graph._nodes ?? []) {
        if (target === node || !ids.has(String(target.id))) continue;
        const mode = isActive ? (target.mode === 4 ? 0 : target.mode) : 4;
        if (target.mode !== mode) {
            target.mode = mode;
            changed = true;
        }
    }
    if (changed) {
        graph.change?.();
        graph.setDirtyCanvas?.(true, true);
    }
}

function setNodeChangeLabels(node, widget, on, off) {
    if (!widget) return;
    if (widget.options?.on === on && widget.options?.off === off) return;
    // Some Vue renderers retain the original options object. Update that object
    // first; newer widgets also need their options setter to notify the store.
    const options = widget.options ?? {};
    Object.assign(options, { on, off });
    let owner = widget;
    while (owner && !Object.getOwnPropertyDescriptor(owner, "options")) owner = Object.getPrototypeOf(owner);
    const descriptor = owner && Object.getOwnPropertyDescriptor(owner, "options");
    if (descriptor?.set) {
        widget.options = { ...options };
    } else if (!widget.options && (!descriptor || descriptor.writable)) {
        widget.options = options;
    }
    node.graph?.incrementVersion?.();
    node.setDirtyCanvas?.(true, true);
}

function syncNodeChangeLabels(node, changedName, changedValue, values = {}) {
    const value = name => name === changedName ? changedValue : (values[name] ?? getWidget(node, name)?.value);
    setNodeChangeLabels(node, getWidget(node, "active"),
        String(value("group_1_label") ?? "").trim() || "Group 1",
        String(value("group_2_label") ?? "").trim() || "Group 2");
}

function applyNodeChange(node, changedName, changedValue, values = {}) {
    const value = name => name === changedName ? changedValue : (values[name] ?? getWidget(node, name)?.value);
    syncNodeChangeLabels(node, changedName, changedValue, values);
    if (changedName === "group_1_label" || changedName === "group_2_label") return;
    const graph = node.graph;
    if (!graph || app.configuringGraph) return;
    const names = ["node_ids_1", "node_ids_2", "active"];
    if (names.some(name => !getWidget(node, name))) return;
    const parseIds = text => new Set(String(text ?? "").split(",").map(id => id.trim()).filter(Boolean));
    const first = parseIds(value("node_ids_1"));
    const second = parseIds(value("node_ids_2"));
    const enabled = value("active") ? first : second;
    const disabled = value("active") ? second : first;
    const disabledMode = value("disable_mode") === "Mute" ? 2 : 4;
    let changed = false;
    for (const target of graph.nodes ?? graph._nodes ?? []) {
        if (target === node) continue;
        const id = String(target.id);
        // Shared IDs stay enabled whichever group is selected.
        const mode = enabled.has(id) ? 0 : disabled.has(id) ? disabledMode : target.mode;
        if (target.mode !== mode) {
            target.mode = mode;
            changed = true;
        }
    }
    if (changed) {
        graph.change?.();
        graph.setDirtyCanvas?.(true, true);
    }
}

// Read promoted input values through their real links rather than display labels
// (which users can rename). New frontends keep these values only on the host.
function nodeModePromotedValues(host, inherited, onSource) {
    const graph = host.subgraph;
    const result = new Map();
    // Earlier frontends expose proxy widgets instead of linked graph inputs.
    const widgets = host.widgets ?? [];
    const proxies = host.properties?.proxyWidgets ?? [];
    for (const [index, widget] of widgets.entries()) {
        const source = widget._overlay;
        const [id, name] = source?.isProxyWidget
            ? [source.nodeId, source.widgetName] : (proxies[index] ?? []);
        if (id == null || String(id) === "-1" || !name) continue;
        const target = (graph.nodes ?? graph._nodes ?? []).find(node => String(node.id) === String(id));
        const value = inherited[widget.name] ?? widget.value;
        if (target && value !== undefined) {
            if (!result.has(target)) result.set(target, {});
            result.get(target)[name] = value;
            onSource?.(widget, target, name, widget.name);
        }
    }
    for (const input of host.inputs ?? []) {
        if (input.link != null && inherited[input.name] === undefined) continue;
        const widget = host.getWidgetFromSlot?.(input) ?? input._widget ?? getWidget(host, input.widget?.name ?? input.name);
        const value = inherited[input.name] ?? widget?.value;
        if (value === undefined) continue;
        const slot = graph.inputNode?.slots?.find(slot => slot.name === input.name);
        for (const id of slot?.linkIds ?? []) {
            const link = graph.getLink?.(id) ?? graph.links?.[id];
            if (!link) continue;
            const resolved = link.resolve?.(graph);
            const target = resolved?.inputNode ?? graph.getNodeById?.(link.target_id);
            const targetInput = resolved?.input ?? target?.inputs?.[link.target_slot];
            if (!target || !targetInput) continue;
            const targetWidget = target.getWidgetFromSlot?.(targetInput);
            const name = target.subgraph ? targetInput.name : (targetWidget?.name ?? targetInput.widget?.name);
            if (!name) continue;
            if (!result.has(target)) result.set(target, {});
            result.get(target)[name] = value;
            onSource?.(widget, target, name, input.name);
        }
    }
    return result;
}

// Both controls need the same classic/Nodes 2.0 and workflow lifecycle hooks.
function registerNodeModeControl(extensionName, nodeClass, widgetNames, apply) {
    const previous = new WeakMap();
    const watchedHosts = new WeakSet();
    const watchedWidgets = new WeakSet();
    let monitor;
    let pendingSync = false;
    const scheduleSync = () => {
        if (pendingSync) return;
        pendingSync = true;
        queueMicrotask(() => {
            pendingSync = false;
            syncSubgraphs();
        });
    };
    const watchHost = host => {
        if (!watchedHosts.has(host)) {
            watchedHosts.add(host);
            const original = host.onWidgetChanged;
            host.onWidgetChanged = function() {
                const result = original?.apply(this, arguments);
                scheduleSync();
                return result;
            };
        }
        for (const widget of host.widgets ?? []) {
            if (watchedWidgets.has(widget)) continue;
            watchedWidgets.add(widget);
            const original = widget.callback;
            widget.callback = function() {
                const result = original?.apply(this, arguments);
                scheduleSync();
                return result;
            };
        }
    };
    const syncSubgraphs = () => {
        if (app.configuringGraph) return;
        const visiting = new Set();
        const visit = (graph, overrides = new Map(), nested = false) => {
            if (!graph || visiting.has(graph)) return;
            visiting.add(graph);
            for (const node of graph.nodes ?? graph._nodes ?? []) {
                const values = overrides.get(node) ?? {};
                // Nodes 2.0 can write text fields without a classic callback,
                // including controls on the root graph. Labels are UI-only:
                // refreshing them must not toggle any target node modes.
                if (nodeClass === "DINKI_Node_Change" && node.comfyClass === nodeClass) {
                    syncNodeChangeLabels(node, undefined, undefined, values);
                }
                if (nested && node.comfyClass === nodeClass) {
                    const effective = Object.fromEntries(widgetNames.map(name => [name, values[name] ?? getWidget(node, name)?.value]));
                    const old = previous.get(node);
                    // A tab switch or workflow load can restore target modes without
                    // changing the promoted selector value. Reassert Node Change's
                    // effective selection; apply() only dirties the graph if needed.
                    if (nodeClass === "DINKI_Node_Change" || !old || old.graph !== graph || widgetNames.some(name => !Object.is(old.values[name], effective[name]))) {
                        apply(node, undefined, undefined, effective);
                        previous.set(node, { graph, values: effective });
                    }
                }
                if (node.subgraph) {
                    watchHost(node);
                    visit(node.subgraph, nodeModePromotedValues(node, values), true);
                    // Copy labels after children have updated, including nested promotions.
                    if (nodeClass === "DINKI_Node_Change") {
                        nodeModePromotedValues(node, values, (widget, target, name) => {
                            if (!(target.subgraph || (target.comfyClass === nodeClass && name === "active"))) return;
                            const source = getWidget(target, name);
                            if (source?.options?.on != null && source.options.off != null) {
                                setNodeChangeLabels(node, widget, source.options.on, source.options.off);
                            }
                        });
                    }
                }
            }
            visiting.delete(graph);
        };
        visit(app.rootGraph ?? app.graph);
    };
app.registerExtension({
    name: extensionName,
    setup() {
        // Promoted widgets can update only a Vue value store, without invoking
        // either the inner callback or a value setter. Observe effective values
        // without replacing framework accessors; redraw only when they change.
        if (monitor === undefined) {
            monitor = setInterval(syncSubgraphs, 100);
            app.canvas?.canvas?.addEventListener("subgraph-opened", syncSubgraphs);
            syncSubgraphs();
        }
    },
    nodeCreated(node) {
        if (node.subgraph) {
            watchHost(node);
            scheduleSync();
        }
        if (node.comfyClass !== nodeClass) return;

        // Nodes 2.0 and programmatic widget updates use the node notification.
        const onWidgetChanged = node.onWidgetChanged;
        node.onWidgetChanged = function (name, value) {
            const result = onWidgetChanged?.apply(this, arguments);
            if (widgetNames.includes(name)) {
                apply(this, name, value);
            }
            return result;
        };

        // Keep classic canvas widgets working and preserve other extensions' callbacks.
        for (const name of widgetNames) {
            const widget = getWidget(node, name);
            if (!widget) continue;
            const callback = widget.callback;
            widget.callback = function (value) {
                const result = callback?.apply(this, arguments);
                apply(node, name, value);
                return result;
            };
        }

        // Defer until insertion/configuration has completed, without a fixed timer.
        for (const hook of ["onAdded", "onConfigure"]) {
            const original = node[hook];
            node[hook] = function () {
                const result = original?.apply(this, arguments);
                queueMicrotask(() => apply(this));
                return result;
            };
        }
    },
    afterConfigureGraph() {
        const visited = new Set();
        const syncGraph = (graph) => {
            if (!graph || visited.has(graph)) return;
            visited.add(graph);
            for (const node of graph.nodes ?? graph._nodes ?? []) {
                if (node.comfyClass === nodeClass) apply(node);
                if (node.subgraph) syncGraph(node.subgraph);
            }
        };
        syncGraph(app.rootGraph ?? app.graph);
        syncSubgraphs();
    }
});
}

registerNodeModeControl("DINKI.NodeSwitch", "DINKI_Node_Switch", ["node_ids", "active"], applyNodeSwitch);
registerNodeModeControl("DINKI.NodeChange", "DINKI_Node_Change",
    ["node_ids_1", "node_ids_2", "active", "disable_mode", "group_1_label", "group_2_label"], applyNodeChange);


// ============================================================
// 5. DINKI Color LUT Logic (Upload & Preview)
// ============================================================

// 5-1. Basic LUT Node Upload
app.registerExtension({
    name: "DINKIssTyle.ColorLUT.Upload",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DINKI_Color_Lut") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;

                this.addWidget("button", "Upload .cube", "Upload", () => {
                    const fileInput = document.createElement("input");
                    Object.assign(fileInput, {
                        type: "file", accept: ".cube", style: "display: none",
                        onchange: async () => {
                            if (fileInput.files.length > 0) await uploadFile(fileInput.files[0]);
                        },
                    });
                    document.body.appendChild(fileInput);
                    fileInput.click();
                    document.body.removeChild(fileInput);
                });

                async function uploadFile(file) {
                    try {
                        const body = new FormData();
                        body.append("image", file);
                        body.append("subfolder", "luts");
                        body.append("type", "input");
                        body.append("overwrite", "true");

                        const resp = await api.fetchApi("/upload/image", { method: "POST", body });

                        if (resp.status === 200) {
                            const data = await resp.json();
                            const filename = data.name;
                            const lutWidget = node.widgets.find((w) => w.name === "lut_name");
                            if (lutWidget) {
                                if (!lutWidget.options.values.includes(filename)) {
                                    lutWidget.options.values.push(filename);
                                }
                                lutWidget.value = filename;
                                app.graph.setDirtyCanvas(true);
                            }
                            alert(`Uploaded: ${filename}`);
                        } else {
                            alert("Upload failed: " + resp.statusText);
                        }
                    } catch (error) {
                        alert("Error uploading file: " + error);
                    }
                }
                return r;
            };
        }
    },
});

// 5-2. Preview LUT Node Logic
app.registerExtension({
    name: "DINKIssTyle.ColorLUT.PreviewInteractive",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        if (nodeData.name === "DINKI_Color_Lut_Preview") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;

                this.previewImage = new Image();
                this.previewUrl = null;

                this.previewImage.onload = () => { app.graph.setDirtyCanvas(true); };

                const lutWidget = this.widgets.find((w) => w.name === "lut_name");
                const strengthWidget = this.widgets.find((w) => w.name === "strength");

                const requestPreview = async () => {
                    const lutName = lutWidget.value;
                    const strength = strengthWidget.value;

                    try {
                        const resp = await api.fetchApi("/dinki/preview_lut", {
                            method: "POST",
                            body: JSON.stringify({ lut_name: lutName, strength: strength }),
                        });

                        if (resp.status === 200) {
                            const blob = await resp.blob();
                            if (node.previewUrl) URL.revokeObjectURL(node.previewUrl);
                            const url = URL.createObjectURL(blob);
                            node.previewUrl = url; 
                            node.previewImage.src = url;
                        }
                    } catch (e) {
                        console.error("DINKI LUT Preview Error:", e);
                    }
                };

                if (lutWidget) lutWidget.callback = requestPreview;
                if (strengthWidget) strengthWidget.callback = requestPreview;

                api.addEventListener("executed", ({ detail }) => {
                    if (detail?.node == node.id) requestPreview();
                });

                this.addWidget("button", "Upload .cube", "Upload", () => {
                    const fileInput = document.createElement("input");
                    Object.assign(fileInput, {
                        type: "file", accept: ".cube", style: "display: none",
                        onchange: async () => {
                            if (fileInput.files.length > 0) await uploadFile(fileInput.files[0]);
                        },
                    });
                    document.body.appendChild(fileInput);
                    fileInput.click();
                    document.body.removeChild(fileInput);
                });

                async function uploadFile(file) {
                    try {
                        const body = new FormData();
                        body.append("image", file);
                        body.append("subfolder", "luts");
                        body.append("type", "input");
                        body.append("overwrite", "true");
                        const resp = await api.fetchApi("/upload/image", { method: "POST", body });

                        if (resp.status === 200) {
                            const data = await resp.json();
                            const filename = data.name;
                            const lutWidget = node.widgets.find((w) => w.name === "lut_name");
                            if (lutWidget) {
                                if (!lutWidget.options.values.includes(filename)) lutWidget.options.values.push(filename);
                                lutWidget.value = filename;
                                requestPreview(); 
                            }
                            alert(`Uploaded: ${filename}`);
                        } else { alert("Upload failed: " + resp.statusText); }
                    } catch (error) { alert("Error: " + error); }
                }
                return r;
            };

            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function (_, options) {
                if (getExtraMenuOptions) getExtraMenuOptions.apply(this, arguments);
                if (this.previewUrl) {
                    options.push(
                        {
                            content: "Open Preview Image",
                            callback: () => { window.open(this.previewUrl, "_blank"); },
                        },
                        {
                            content: "Save Preview Image",
                            callback: () => {
                                const lutName = this.widgets.find((w) => w.name === "lut_name")?.value || "lut";
                                const cleanName = lutName.replace(".cube", "");
                                const a = document.createElement("a");
                                a.href = this.previewUrl;
                                a.setAttribute("download", `preview_${cleanName}.png`);
                                document.body.appendChild(a);
                                a.click();
                                document.body.removeChild(a);
                            },
                        }
                    );
                }
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) onDrawForeground.apply(this, arguments);
                if (this.previewImage && this.previewImage.src) {
                    const w = this.size[0]; const h = this.size[1];
                    const headerHeight = 50; const drawH = h - headerHeight - 10;
                    if (drawH > 0) {
                        const imgW = this.previewImage.width; const imgH = this.previewImage.height;
                        const ratio = Math.min(w / imgW, drawH / imgH);
                        const finalW = imgW * ratio; const finalH = imgH * ratio;
                        const x = (w - finalW) / 2; const y = headerHeight + (drawH - finalH) / 2;
                        ctx.save();
                        ctx.drawImage(this.previewImage, x, y + 10, finalW, finalH);
                        ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
                        ctx.strokeRect(x, y + 10, finalW, finalH);
                        ctx.restore();
                    }
                }
            };
        }
    },
});

// ============================================================
// 6. DINKI Adobe XMP Logic (Upload & Preview)
// ============================================================

// 6-1. [추가] Basic XMP Node Upload (이 부분이 빠져 있었음)
app.registerExtension({
    name: "DINKIssTyle.AdobeXMP.Upload",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DINKI_adobe_xmp") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;

                this.addWidget("button", "Upload .xmp", "Upload", () => {
                    const fileInput = document.createElement("input");
                    Object.assign(fileInput, {
                        type: "file", accept: ".xmp", style: "display: none",
                        onchange: async () => {
                            if (fileInput.files.length > 0) await uploadFile(fileInput.files[0]);
                        },
                    });
                    document.body.appendChild(fileInput);
                    fileInput.click();
                    document.body.removeChild(fileInput);
                });

                async function uploadFile(file) {
                    try {
                        const body = new FormData();
                        body.append("image", file);
                        body.append("subfolder", "adobe_xmp");
                        body.append("type", "input");
                        body.append("overwrite", "true");
                        const resp = await api.fetchApi("/upload/image", { method: "POST", body });

                        if (resp.status === 200) {
                            const data = await resp.json();
                            const filename = data.name;
                            const xmpWidget = node.widgets.find((w) => w.name === "xmp_file");
                            if (xmpWidget) {
                                if (!xmpWidget.options.values.includes(filename)) xmpWidget.options.values.push(filename);
                                xmpWidget.value = filename;
                                app.graph.setDirtyCanvas(true);
                            }
                            alert(`Uploaded: ${filename}`);
                        } else { alert("Upload failed: " + resp.statusText); }
                    } catch (error) { alert("Error: " + error); }
                }
                return r;
            };
        }
    },
});

// 6-2. Preview XMP Node Logic
app.registerExtension({
    name: "DINKIssTyle.AdobeXMP.PreviewInteractive",
    // [수정] 오타 수정: beforeRegisterDef -> beforeRegisterNodeDef
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        if (nodeData.name === "DINKI_Adobe_XMP_Preview") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this;

                this.previewImage = new Image();
                this.previewUrl = null;

                this.previewImage.onload = () => { app.graph.setDirtyCanvas(true); };

                const xmpWidget = this.widgets.find((w) => w.name === "xmp_file");
                const strengthWidget = this.widgets.find((w) => w.name === "strength");

                const requestPreview = async () => {
                    const xmpFile = xmpWidget.value;
                    const strength = strengthWidget.value;

                    try {
                        const resp = await api.fetchApi("/dinki/preview_xmp", {
                            method: "POST",
                            body: JSON.stringify({ xmp_file: xmpFile, strength: strength }),
                        });

                        if (resp.status === 200) {
                            const blob = await resp.blob();
                            if (node.previewUrl) URL.revokeObjectURL(node.previewUrl);
                            const url = URL.createObjectURL(blob);
                            node.previewUrl = url; 
                            node.previewImage.src = url;
                        }
                    } catch (e) {
                        console.error("DINKI XMP Preview Error:", e);
                    }
                };

                if (xmpWidget) xmpWidget.callback = requestPreview;
                if (strengthWidget) strengthWidget.callback = requestPreview;

                api.addEventListener("executed", ({ detail }) => {
                    if (detail?.node == node.id) requestPreview();
                });

                this.addWidget("button", "Upload .xmp", "Upload", () => {
                    const fileInput = document.createElement("input");
                    Object.assign(fileInput, {
                        type: "file", accept: ".xmp", style: "display: none",
                        onchange: async () => {
                            if (fileInput.files.length > 0) await uploadFile(fileInput.files[0]);
                        },
                    });
                    document.body.appendChild(fileInput);
                    fileInput.click();
                    document.body.removeChild(fileInput);
                });

                async function uploadFile(file) {
                    try {
                        const body = new FormData();
                        body.append("image", file);
                        body.append("subfolder", "adobe_xmp");
                        body.append("type", "input");
                        body.append("overwrite", "true");
                        const resp = await api.fetchApi("/upload/image", { method: "POST", body });

                        if (resp.status === 200) {
                            const data = await resp.json();
                            const filename = data.name;
                            const xmpWidget = node.widgets.find((w) => w.name === "xmp_file");
                            if (xmpWidget) {
                                if (!xmpWidget.options.values.includes(filename)) xmpWidget.options.values.push(filename);
                                xmpWidget.value = filename;
                                requestPreview(); 
                            }
                            alert(`Uploaded: ${filename}`);
                        } else { alert("Upload failed: " + resp.statusText); }
                    } catch (error) { alert("Error: " + error); }
                }
                return r;
            };

            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function (_, options) {
                if (getExtraMenuOptions) getExtraMenuOptions.apply(this, arguments);
                if (this.previewUrl) {
                    options.push(
                        {
                            content: "Open Preview Image",
                            callback: () => { window.open(this.previewUrl, "_blank"); },
                        },
                        {
                            content: "Save Preview Image",
                            callback: () => {
                                const xmpName = this.widgets.find((w) => w.name === "xmp_file")?.value || "preset";
                                const cleanName = xmpName.replace(".xmp", "");
                                const a = document.createElement("a");
                                a.href = this.previewUrl;
                                a.setAttribute("download", `preview_${cleanName}.png`);
                                document.body.appendChild(a);
                                a.click();
                                document.body.removeChild(a);
                            },
                        }
                    );
                }
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) onDrawForeground.apply(this, arguments);
                if (this.previewImage && this.previewImage.src) {
                    const w = this.size[0]; const h = this.size[1];
                    const headerHeight = 50; const drawH = h - headerHeight - 10;
                    if (drawH > 0) {
                        const imgW = this.previewImage.width; const imgH = this.previewImage.height;
                        const ratio = Math.min(w / imgW, drawH / imgH);
                        const finalW = imgW * ratio; const finalH = imgH * ratio;
                        const x = (w - finalW) / 2; const y = headerHeight + (drawH - finalH) / 2;
                        ctx.save();
                        ctx.drawImage(this.previewImage, x, y + 10, finalW, finalH);
                        ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
                        ctx.strokeRect(x, y + 10, finalW, finalH);
                        ctx.restore();
                    }
                }
            };
        }
    },
});

// ============================================================
// 7. DINKI Video Player Logic (Fixed for Temp/Output)
// ============================================================
app.registerExtension({
    name: "DINKI.VideoPlayer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DINKI_Video_Player") {
            
            // 1. 노드 실행 시 (파일 수신)
            nodeType.prototype.onExecuted = function(message) {
                // Python에서 보낸 데이터 확인
                // 기존: return {"ui": {"video": ["filename.mp4"]}} -> 문자열
                // 변경: return {"ui": {"video": [{"filename":..., "type":..., "subfolder":...}]}} -> 객체
                
                const videoData = message.video[0];
                let filename, type, subfolder;

                if (typeof videoData === 'string') {
                    // 구버전 호환성 (문자열인 경우)
                    filename = videoData;
                    type = 'output';
                    subfolder = '';
                } else {
                    // 신버전 (객체인 경우)
                    filename = videoData.filename;
                    type = videoData.type || 'output';
                    subfolder = videoData.subfolder || '';
                }
                
                // 확장자 추출 및 소문자 변환
                const ext = filename.split('.').pop().toLowerCase();
                
                // 기존 위젯 제거 (새 영상 재생을 위해)
                if (this.videoWidget) {
                    this.videoWidget.element.remove();
                    this.videoWidget = null;
                }

                // [중요] URL 생성 시 type과 subfolder를 동적으로 반영하도록 수정됨
                const queryParams = new URLSearchParams({
                    filename: filename,
                    type: type,
                    subfolder: subfolder,
                    format: 'video',
                    t: Date.now()
                });
                const fileUrl = api.apiURL(`/view?${queryParams.toString()}`);

                // 컨테이너 생성
                const div = document.createElement("div");
                Object.assign(div.style, {
                    position: "absolute",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    pointerEvents: "auto",
                    zIndex: "10",
                    backgroundColor: "#000",
                    overflow: "hidden"
                });

                let contentElement;

                // 포맷에 따른 태그 생성
                if (['mp4', 'webm', 'mov'].includes(ext)) {
                    contentElement = document.createElement("video");
                    Object.assign(contentElement, {
                        controls: true,
                        autoplay: true,
                        loop: true,
                        muted: true, // 자동 재생 정책 준수
                    });
                } else {
                    // 이미지 포맷 (gif, webp 등)
                    contentElement = document.createElement("img");
                    Object.assign(contentElement.style, {
                        objectFit: "contain",
                    });
                }

                // 소스 연결 및 스타일 설정
                contentElement.src = fileUrl;
                contentElement.style.width = "100%";
                contentElement.style.height = "100%";
                contentElement.style.maxWidth = "100%";
                contentElement.style.maxHeight = "100%";

                div.appendChild(contentElement);
                document.body.appendChild(div);

                this.videoWidget = {
                    element: div,
                    content: contentElement,
                };

                // 노드 크기 최소값 보정
                const currentSize = this.getSize();
                if (currentSize[0] < 300) this.setSize([300, 300]); 

                app.graph.setDirtyCanvas(true);
            };

            // 2. 위치 동기화 (기존 로직 유지)
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) onDrawForeground.apply(this, arguments);

                if (!this.videoWidget) return;

                const div = this.videoWidget.element;
                
                if (this.flags.collapsed) {
                    div.style.display = "none";
                    return;
                }

                const scale = app.canvas.ds.scale;
                const offset = app.canvas.ds.offset;

                const realX = (this.pos[0] + offset[0]) * scale;
                const realY = (this.pos[1] + offset[1]) * scale;
                
                const titleHeight = LiteGraph.NODE_TITLE_HEIGHT || 30;
                const realWidth = this.size[0] * scale;
                const realHeight = (this.size[1] - titleHeight) * scale;

                // 화면 밖 체크
                if (realX + realWidth < 0 || realY + realHeight < 0 || 
                    realX > window.innerWidth || realY > window.innerHeight) {
                    div.style.display = "none";
                    return;
                }

                div.style.display = "flex";
                div.style.left = `${realX}px`;
                div.style.top = `${realY + (titleHeight * scale)}px`;
                div.style.width = `${realWidth}px`;
                div.style.height = `${realHeight}px`;
            };

            // 3. 삭제 처리 (기존 로직 유지)
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                if (onRemoved) onRemoved.apply(this, arguments);
                if (this.videoWidget) {
                    this.videoWidget.element.remove();
                    this.videoWidget = null;
                }
            };
        }
    }
});



// ============================================================
// 8. DINKI String Switch RT (Fixed & Final)
// ============================================================
app.registerExtension({
    name: "DINKI.StringSwitchRT",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DINKI_String_Switch_RT") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const node = this;

                // 1. 위젯 찾기
                const comboIndex = node.widgets.findIndex(w => w.name === "select_string");
                // [변경] 여러 개의 string_ 대신 하나의 input_text 위젯을 찾습니다.
                const textInputWidget = node.widgets.find(w => w.name === "input_text");

                if (comboIndex === -1 || !textInputWidget) {
                    console.warn("DINKI Warning: Necessary widgets not found.");
                    return;
                }

                // 2. 구버전 UI 호환을 위한 위젯 교체 (이전 답변과 동일 로직)
                const originalWidget = node.widgets[comboIndex];
                const originalValue = originalWidget.value;
                let comboWidget;

                if (originalWidget.type !== "combo") {
                    node.widgets.splice(comboIndex, 1);
                    comboWidget = node.addWidget("combo", "select_string", originalValue, originalWidget.callback, { values: [] });
                    node.widgets.pop();
                    node.widgets.splice(comboIndex, 0, comboWidget);
                } else {
                    comboWidget = originalWidget;
                }

                // Keep slash-separated strings as literal values, including model IDs.
                comboWidget.options.getOptionLabel = value => value ?? "";
                const originalComboCallback = comboWidget.callback;
                comboWidget.callback = function (value, ...args) {
                    const values = comboWidget.options.values;
                    if (typeof value === "string" && !values.includes(value)) {
                        const matches = values.filter(line => line.endsWith("/" + value));
                        if (matches.length === 1) value = matches[0];
                    }
                    comboWidget.value = value;
                    return originalComboCallback?.call(this, value, ...args);
                };

                // [핵심 변경] 3. 줄 바꿈 기준으로 드랍다운 목록 업데이트
                const updateCombo = () => {
                    // 텍스트 박스의 값을 줄바꿈(\n)으로 자릅니다.
                    // trim()을 사용하여 양쪽 공백을 제거하고, 빈 줄은 필터링(제외)합니다.
                    const rawText = textInputWidget.value || "";
                    const lines = rawText.split("\n")
                                         .map(line => line.trim())
                                         .filter(line => line.length > 0);

                    comboWidget.options.values = lines;

                    // 현재 선택된 값이 목록에 없으면(예: 텍스트를 지웠을 때) 첫 번째 값 선택
                    if (!lines.includes(comboWidget.value) && lines.length > 0) {
                        comboWidget.value = lines[0];
                    }
                };

                // 4. 멀티라인 텍스트 위젯에 리스너 연결
                const originalCallback = textInputWidget.callback;
                textInputWidget.callback = function (value) {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    
                    // 타이핑 할 때마다 드랍다운 목록 갱신
                    updateCombo();
                    
                    // 캔버스 갱신
                    app.graph.setDirtyCanvas(true, true);
                };

                // 5. 초기화
                requestAnimationFrame(() => {
                    updateCombo();
                    // 저장된 값이 유효하다면 복구
                    if (comboWidget.options.values.includes(originalValue)) {
                        comboWidget.value = originalValue;
                    }
                });
            };
        }
    },
});



// ============================================================
// 9. DINKI Note
// ============================================================
app.registerExtension({
    name: "DINKI.Note.Display",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // "DINKI Note" 노드에만 적용
        if (nodeData.name === "DINKI_Note") {
            
            // 노드가 생성될 때 기본 크기를 좀 더 크게 설정
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                this.setSize([300, 300]); // 기본 크기 (가로, 세로)
            };

            // 화면 그리기 함수 오버라이딩
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) onDrawForeground.apply(this, arguments);

                // 1. 위젯 값 가져오기
                const directionWidget = this.widgets.find(w => w.name === "direction");
                const textWidget = this.widgets.find(w => w.name === "text");

                const directionValue = directionWidget ? directionWidget.value : "";
                const textValue = textWidget ? textWidget.value : "";

                // 위젯 영역 아래부터 그리기를 시작하기 위해 높이 계산 (대략적인 위젯 높이 제외)
                // 위젯들이 가려지지 않도록 margin을 줍니다.
                const startY = 100; 

                ctx.save(); // 그리기 상태 저장

                // --- 2. 이모지 그리기 (아주 크게) ---
                ctx.font = "80px Arial"; // 이모지 크기 설정
                ctx.fillStyle = "white";
                ctx.textAlign = "center";
                ctx.textBaseline = "top";
                
                // 노드 가로 중앙에 이모지 배치
                ctx.fillText(directionValue, this.size[0] / 2, startY);

                // --- 3. 텍스트 그리기 (크게) ---
                const fontSize = 24; // ★ 여기서 텍스트 폰트 크기 조절 ★
                ctx.font = "bold " + fontSize + "px Arial"; 
                ctx.fillStyle = "#ddd"; // 글자색 (밝은 회색)
                
                // 이모지 아래로 위치 잡기
                let textY = startY + 90; 
                const lineHeight = fontSize * 1.4;
                const maxWidth = this.size[0] - 20; // 좌우 여백 10px씩

                // 텍스트 줄바꿈 처리 (Word Wrap)
                const words = textValue.split('\n'); // 엔터키 기준 먼저 분리
                
                for (let i = 0; i < words.length; i++) {
                    const line = words[i];

                    let tempLine = "";
                    const chars = line.split("");
                    
                    for(let n = 0; n < chars.length; n++) {
                        let testLine = tempLine + chars[n];
                        let metrics = ctx.measureText(testLine);
                        let testWidth = metrics.width;
                        
                        if (testWidth > maxWidth && n > 0) {
                            ctx.fillText(tempLine, this.size[0] / 2, textY);
                            tempLine = chars[n];
                            textY += lineHeight;
                        } else {
                            tempLine = testLine;
                        }
                    }
                    ctx.fillText(tempLine, this.size[0] / 2, textY);
                    textY += lineHeight;
                }

                ctx.restore(); // 그리기 상태 복구
            };
        }
    },
});


// ============================================================
// 10. DINKI Sampler Preset
// ============================================================
app.registerExtension({
    name: "DINKI.SamplerPreset",
    async nodeCreated(node, app) {
        if (node.comfyClass !== "DINKI_Sampler_Preset_JS") return;

        const modelWidget = node.widgets.find((w) => w.name === "model");
        const presetWidget = node.widgets.find((w) => w.name === "preset");

        if (!modelWidget || !presetWidget) return;

        // API 데이터 가져오기
        const response = await api.fetchApi("/dinki/sampler_presets");
        if (response.status !== 200) {
            presetWidget.options.values = ["API Error"];
            return;
        }
        const presetData = await response.json();

        // === [핵심 수정] 프리셋 목록 업데이트 함수 ===
        // targetValue: 이 값이 목록에 있다면 그 값을 선택하고(불러오기 복구), 없다면 첫 번째 값 선택
        const updatePresets = (selectedModel, targetValue = null) => {
            const presets = presetData[selectedModel];
            
            if (presets && presets.length > 0) {
                // 1. 목록 갱신
                const newOptions = presets.map(p => p.display);
                presetWidget.options.values = newOptions;

                // 2. 값 설정 로직 (저장된 값 유지 vs 초기화)
                if (targetValue && newOptions.includes(targetValue)) {
                    // 저장된 값(targetValue)이 현재 목록에 유효하게 존재하면 유지
                    presetWidget.value = targetValue;
                } else {
                    // 유효하지 않거나 새로운 모델 선택 시 첫 번째 값으로 초기화
                    presetWidget.value = newOptions[0];
                }
            } else {
                presetWidget.options.values = ["No Presets Found"];
                presetWidget.value = "No Presets Found";
            }

            node.setDirtyCanvas(true, true); 
        };

        // 모델 변경 콜백 (사용자가 직접 변경 시)
        const originalCallback = modelWidget.callback;
        modelWidget.callback = function (value) {
            // 사용자가 모델을 바꿀 때는 기존 프리셋이 의미가 없으므로 
            // 두 번째 인자를 null로 주어 첫 번째 값으로 리셋시킴
            updatePresets(value, null);
            
            if (originalCallback) {
                originalCallback.call(this, value);
            }
        };

        // === [핵심 수정] 초기 실행 로직 ===
        // 노드가 생성될 때(워크플로우 로딩 시)
        if (modelWidget.value) {
            // 현재 저장되어 있는 프리셋 값(presetWidget.value)을 
            // updatePresets 함수에 전달하여 유지 시도
            updatePresets(modelWidget.value, presetWidget.value);
        }
    }
});


// ============================================================
// 11. DINKI Node Check
// ============================================================
app.registerExtension({
    name: "Dinki.NodeCheck",
    async setup() {
        const canvas = app.canvas;
        if (!canvas || canvas.__dinki_node_check_attached) return;
        canvas.__dinki_node_check_attached = true;

        const updateSelection = () => {
            const graph = canvas.graph || app.graph;
            if (!graph) return;
            // selectedItems preserves selection order and also contains groups
            // and reroutes. Only actual nodes in the displayed graph count.
            const nodes = graph.nodes || graph._nodes || [];
            const nodeSet = new Set(nodes);
            const selected = Array.from(canvas.selectedItems ?? Object.values(canvas.selected_nodes || {}))
                .filter(item => nodeSet.has(item));
            const selectedNodeId = selected.length ? String(selected[selected.length - 1].id) : "None";

            for (const node of nodes) {
                if (node.comfyClass !== "DINKI_Node_Check" && node.type !== "DINKI_Node_Check") continue;
                const widget = getWidget(node, "selected_node_id");
                if (!widget || widget.value === selectedNodeId) continue;
                const oldValue = widget.value;
                widget.value = selectedNodeId;
                widget.options?.setValue?.(selectedNodeId);
                widget.callback?.call(widget, selectedNodeId, canvas, node);
                node.onWidgetChanged?.(widget.name, selectedNodeId, oldValue, widget);
                node.setDirtyCanvas?.(true, true);
            }
        };

        // Vue nodes call select/deselect directly, bypassing onSelectionChange.
        // Batch deselectAll + select into one update after selection settles.
        let pending = false;
        const scheduleUpdate = () => {
            if (pending) return;
            pending = true;
            queueMicrotask(() => {
                pending = false;
                updateSelection();
            });
        };
        for (const name of ["select", "deselect", "deselectAll", "onSelectionChange"]) {
            const original = canvas[name];
            if (name !== "onSelectionChange" && typeof original !== "function") continue;
            canvas[name] = function(...args) {
                const result = original?.apply(this, args);
                scheduleUpdate();
                return result;
            };
        }
    },
    
    nodeCreated(node, app) {
        if (node.comfyClass === "DINKI_Node_Check") {

            const size = node.computeSize();
            node.setSize(size);

            const widget = getWidget(node, "selected_node_id");
            if (widget) {
                setTimeout(() => {
                    if (widget.inputEl) {
                        widget.inputEl.readOnly = true;
                        widget.inputEl.style.opacity = 0.6;
                    }
                }, 100);
            }
        }
    }
});



// ============================================================
// 12. DINKI Anchor
// ============================================================
app.registerExtension({
    name: "Dinki.Anchor",
    setup() {
        // 전역 키다운 이벤트 리스너 추가
        window.addEventListener("keydown", (e) => {
            // 1. 텍스트 입력 중일 때는 단축키 무시
            const activeTag = document.activeElement.tagName.toUpperCase();
            if (activeTag === "INPUT" || activeTag === "TEXTAREA") {
                return;
            }

            const graph = app.graph;
            if (!graph) return;

            // 2. 모든 DINKI_Anchor 노드 찾기
            const anchorNodes = graph.findNodesByType("DINKI_Anchor");
            if (!anchorNodes || anchorNodes.length === 0) return;

            // 3. 눌린 키와 매칭되는 노드 찾기
            anchorNodes.forEach(node => {
                const shortcutWidget = node.widgets[0]; // shortcut_key
                const zoomWidget = node.widgets[1];     // zoom_levels

                if (shortcutWidget && shortcutWidget.value === e.key) {
                    // 단축키 매칭됨 -> 이동 실행
                    handleAnchorMove(node, zoomWidget.value);
                }
            });
        });
    }
});


// ============================================================
// 13. DKST Image (Viewer) resolution overlay
// ============================================================
app.registerExtension({
    name: "DINKI.PreviewImage.Resolution",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DINKI_Preview_Image") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments);
            const container = document.createElement("div");
            Object.assign(container.style, {
                width: "100%", height: "100%", minHeight: "0", display: "flex",
                flexDirection: "column", overflow: "hidden", background: "#181818",
                contain: "size layout paint", borderRadius: "6px",
            });
            const image = document.createElement("img");
            Object.assign(image.style, {
                width: "100%", height: "0", minHeight: "0", flex: "1 1 0",
                objectFit: "contain", display: "none", pointerEvents: "auto",
            });
            const resolution = document.createElement("div");
            Object.assign(resolution.style, {
                flex: "0 0 28px", textAlign: "center", lineHeight: "28px", color: "#eee",
            });
            const selector = document.createElement("select");
            selector.style.display = "none";
            container.append(image, resolution, selector);
            const widget = this.addDOMWidget("dkst_preview_image", "DKST_IMAGE_PREVIEW", container, {
                hideOnZoom: false, getMinHeight: () => 120,
                getMaxHeight: () => 320, getHeight: () => 240,
            });
            widget.serialize = false;
            widget.options.serialize = false;
            let descriptors = [];
            const selected = () => descriptors[Number(selector.value) || 0];
            const extraMenu = this.getExtraMenuOptions;
            this.getExtraMenuOptions = function(canvas, options) {
                const result = extraMenu?.apply(this, arguments);
                if (selected()) options.push(clipboardMenuAction("Copy Image", () =>
                    copyImageToClipboard(api.apiURL(`/view?${new URLSearchParams(selected())}`))));
                return result;
            };
            const display = () => {
                const item = selected();
                if (!item) {
                    image.removeAttribute("src");
                    image.style.display = "none";
                    resolution.textContent = "";
                    return;
                }
                image.src = api.apiURL(`/view?${new URLSearchParams({ ...item, t: String(Date.now()) })}`);
                image.style.display = "block";
            };
            selector.onchange = () => {
                if (this.properties?.dkstPreview) {
                    this.properties.dkstPreview.selectedIndex = Number(selector.value) || 0;
                }
                display();
            };
            image.onload = () => {
                resolution.textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
            };
            for (const eventName of ["pointerdown", "mousedown"]) {
                image.addEventListener(eventName, event => {
                    if (event.button === 2) event.stopPropagation();
                });
            }
            image.addEventListener("contextmenu", event => {
                if (selected()) showPreviewImageMenu(event, selected());
            });
            this.dkstUpdatePreview = (message, persist = true) => {
                descriptors = message?.dkst_images || message?.images || [];
                if (persist) {
                    this.properties ??= {};
                    this.properties.dkstPreview = {
                        dkst_images: descriptors,
                        resolution: message?.resolution || [],
                        selectedIndex: 0,
                    };
                }
                selector.replaceChildren();
                descriptors.forEach((item, index) => {
                    const option = document.createElement("option");
                    option.value = String(index);
                    option.textContent = `${index + 1} / ${descriptors.length}`;
                    selector.appendChild(option);
                });
                selector.value = String(Math.min(message?.selectedIndex || 0, Math.max(0, descriptors.length - 1)));
                selector.style.display = descriptors.length > 1 ? "block" : "none";
                resolution.textContent = message?.resolution?.[0] || "";
                display();
                this.setDirtyCanvas(true, true);
            };
            this.dkstRestorePreview = () => {
                const stored = this.properties?.dkstPreview;
                const output = app.nodeOutputs?.[this.id];
                if (stored?.dkst_images?.length) {
                    this.dkstUpdatePreview(stored, false);
                } else if (output?.dkst_images?.length || output?.images?.length) {
                    this.dkstUpdatePreview(output, false);
                }
            };
            const configured = this.onConfigure;
            this.onConfigure = function() {
                const configuredResult = configured?.apply(this, arguments);
                queueMicrotask(() => this.dkstRestorePreview?.());
                return configuredResult;
            };
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            executed?.apply(this, arguments);
            this.dkstUpdatePreview?.(message);
        };
    },
    loadedGraphNode(node) {
        if (node.comfyClass === "DINKI_Preview_Image") {
            node.dkstRestorePreview?.();
        }
    },
});


// ============================================================
// 14. DKST Image (Load)
// ============================================================
app.registerExtension({
    name: "DINKI.ImageLoad",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DINKI_Image_Load") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            const categoryWidget = getWidget(node, "category");
            const filenameWidget = getWidget(node, "filename");
            const sourceTypeWidget = getWidget(node, "source_type");
            if (!categoryWidget || !filenameWidget || !sourceTypeWidget) return result;

            sourceTypeWidget.type = "converted-widget";
            sourceTypeWidget.computeSize = () => [0, -4];

            const previewContainer = document.createElement("div");
            Object.assign(previewContainer.style, {
                width: "100%",
                height: "100%",
                minHeight: "0",
                display: "none",
                flexDirection: "column",
                alignItems: "stretch",
                justifyContent: "center",
                overflow: "hidden",
                background: "#181818",
                borderRadius: "6px",
                boxSizing: "border-box",
                pointerEvents: "none",
                contain: "size layout paint",
            });

            const previewElement = document.createElement("img");
            Object.assign(previewElement.style, {
                width: "100%",
                height: "0",
                minHeight: "0",
                maxHeight: "100%",
                flex: "1 1 0",
                objectFit: "contain",
                display: "block",
                pointerEvents: "auto",
            });

            const resolutionElement = document.createElement("div");
            Object.assign(resolutionElement.style, {
                flex: "0 0 28px",
                lineHeight: "28px",
                textAlign: "center",
                color: "#f0f0f0",
                font: "13px sans-serif",
                background: "#181818",
            });
            previewContainer.append(previewElement, resolutionElement);

            if (typeof node.addDOMWidget === "function") {
                const previewWidget = node.addDOMWidget(
                    "dkst_image_preview",
                    "DKST_IMAGE_PREVIEW",
                    previewContainer,
                    {
                        hideOnZoom: false,
                        getMinHeight: () => 120,
                        getMaxHeight: () => 320,
                        getHeight: () => 240,
                    },
                );
                previewWidget.serialize = false;
            }

            const setValues = (widget, values) => {
                widget.options ??= {};
                widget.options.values = values.length ? values : [""];
            };

            let previewGeneration = 0;
            let refreshGeneration = 0;
            const rememberSelection = () => {
                node.properties ??= {};
                if (sourceTypeWidget.value === "temp" && filenameWidget.value?.startsWith("DKST_Paste_")) {
                    node.properties.dkstImageLoad = {
                        filename: filenameWidget.value,
                        category: categoryWidget.value || "",
                        source_type: "temp",
                    };
                } else {
                    delete node.properties.dkstImageLoad;
                }
            };

            const showPreview = (filename = filenameWidget.value) => {
                const generation = ++previewGeneration;
                rememberSelection();
                if (!filename) {
                    node.dkstLoadedImage = null;
                    node.dkstImageResolution = "";
                    previewElement.removeAttribute("src");
                    resolutionElement.textContent = "";
                    previewContainer.style.display = "none";
                    node.setDirtyCanvas(true, true);
                    return;
                }

                const params = new URLSearchParams({
                    filename,
                    subfolder: sourceTypeWidget.value === "temp" ? "" : (categoryWidget.value || ""),
                    type: sourceTypeWidget.value || "input",
                    t: String(Date.now()),
                });
                const image = new Image();
                image.onload = () => {
                    if (generation !== previewGeneration) return;
                    node.dkstLoadedImage = image;
                    node.dkstImageResolution = `${image.naturalWidth} × ${image.naturalHeight}`;
                    previewElement.src = image.src;
                    resolutionElement.textContent = node.dkstImageResolution;
                    previewContainer.style.display = "flex";
                    node.setDirtyCanvas(true, true);
                };
                image.onerror = () => {
                    if (generation !== previewGeneration) return;
                    node.dkstLoadedImage = null;
                    node.dkstImageResolution = "Unable to preview image";
                    previewElement.removeAttribute("src");
                    resolutionElement.textContent = node.dkstImageResolution;
                    previewContainer.style.display = "flex";
                    node.setDirtyCanvas(true, true);
                };
                image.src = api.apiURL(`/view?${params.toString()}`);
            };

            const refreshFiles = async(category, preferredFilename = null, sourceType = "input",
                generation = ++refreshGeneration) => {
                const params = new URLSearchParams({ category: category || "" });
                const response = await api.fetchApi(`/dinki/image-load/files?${params.toString()}`);
                if (!response.ok) throw new Error(`Unable to load image list (${response.status})`);
                const data = await response.json();
                if (generation !== refreshGeneration) return;
                const temporary = sourceType === "temp" && preferredFilename?.startsWith("DKST_Paste_");
                const files = temporary
                    ? [preferredFilename, ...(data.files || []).filter(name => name !== preferredFilename)]
                    : (data.files || []);
                setValues(filenameWidget, files);
                sourceTypeWidget.value = temporary ? "temp" : "input";
                filenameWidget.value = files.includes(preferredFilename) ? preferredFilename : (files[0] || "");
                node.dkstTemporaryFilename = temporary ? preferredFilename : null;
                showPreview(filenameWidget.value);
                node.setDirtyCanvas(true, true);
            };

            const refreshCategories = async(preferredCategory = null, preferredFilename = null,
                sourceType = "input") => {
                const generation = ++refreshGeneration;
                const response = await api.fetchApi("/dinki/image-load/categories");
                if (!response.ok) throw new Error(`Unable to load image categories (${response.status})`);
                const data = await response.json();
                if (generation !== refreshGeneration) return;
                const categories = data.categories || [""];
                setValues(categoryWidget, categories);
                categoryWidget.value = categories.includes(preferredCategory)
                    ? preferredCategory
                    : "";
                await refreshFiles(categoryWidget.value, preferredFilename, sourceType, generation);
            };

            const originalCategoryCallback = categoryWidget.callback;
            categoryWidget.callback = async function(value) {
                originalCategoryCallback?.apply(this, arguments);
                await node.dkstDeleteTemporaryImage?.();
                refreshFiles(value).catch(console.error);
            };

            const originalFilenameCallback = filenameWidget.callback;
            filenameWidget.callback = async function(value) {
                originalFilenameCallback?.apply(this, arguments);
                if (sourceTypeWidget.value === "temp" && value !== node.dkstTemporaryFilename) {
                    await node.dkstDeleteTemporaryImage();
                    await refreshFiles(categoryWidget.value, value);
                    return;
                }
                showPreview(value);
            };

            node.dkstRefreshImageLoader = refreshCategories;
            node.dkstRestoreImageLoader = () => {
                const saved = node.properties?.dkstImageLoad;
                const temporary = saved?.source_type === "temp";
                return refreshCategories(
                    temporary ? saved.category : categoryWidget.value,
                    temporary ? saved.filename : filenameWidget.value,
                    temporary ? "temp" : sourceTypeWidget.value,
                );
            };
            // The native mask editor reads/writes a widget named `image`.
            // Keep this bridge out of the visible controls and prompt inputs.
            const editorWidget = node.addWidget("combo", "image", "", () => {}, { values: [] });
            editorWidget.type = "converted-widget";
            editorWidget.computeSize = () => [0, -4];
            editorWidget.serialize = false;
            editorWidget.options.serialize = false;
            let editorValue = "";
            Object.defineProperty(editorWidget, "value", {
                configurable: true,
                get: () => editorValue,
                set: (value) => {
                    if (typeof value !== "string" || value === editorValue) return;
                    editorValue = value;
                    const match = value.match(/^(.*?)(?: \[(input|temp|output)\])?$/);
                    const path = match[1].replaceAll("\\", "/");
                    const slash = path.lastIndexOf("/");
                    const descriptor = {
                        filename: path.slice(slash + 1),
                        subfolder: slash < 0 ? "" : path.slice(0, slash),
                        type: match[2] || "input",
                    };
                    const applyEditedImage = async() => {
                        if (descriptor.type === "input") {
                            await node.dkstDeleteTemporaryImage?.();
                            await refreshCategories(descriptor.subfolder, descriptor.filename);
                        } else {
                            const response = await api.fetchApi(`/view?${new URLSearchParams(descriptor)}`);
                            if (!response.ok) throw new Error("Unable to load edited image");
                            await node.dkstUploadClipboardImage(await response.blob());
                        }
                    };
                    applyEditedImage().catch((error) => {
                        console.error(error);
                        alert(`Unable to apply edited image: ${error.message}`);
                    });
                },
            });

            let fileInput;
            const openImageFilePicker = () => {
                if (!fileInput) {
                    fileInput = document.createElement("input");
                    fileInput.type = "file";
                    fileInput.accept = "image/*";
                    fileInput.style.display = "none";
                    fileInput.onchange = async() => {
                        const file = fileInput.files?.[0];
                        fileInput.value = ""; // Allow choosing the same file again.
                        if (!file) return;
                        try {
                            await node.dkstUploadDroppedImage(file);
                        } catch (error) {
                            alert(`Unable to upload image: ${error.message}`);
                        }
                    };
                    document.body.appendChild(fileInput);
                }
                // Keep this synchronous with the menu click so the browser can
                // open the OS file picker without losing user activation.
                fileInput.click();
            };

            let closeImageMenu = () => {};
            for (const eventName of ["pointerdown", "mousedown"]) {
                previewElement.addEventListener(eventName, event => {
                    if (event.button === 2) event.stopPropagation();
                });
            }
            previewElement.addEventListener("contextmenu", (event) => {
                if (!node.dkstLoadedImage) return;
                event.preventDefault();
                event.stopPropagation();
                closeImageMenu();
                const menu = document.createElement("div");
                Object.assign(menu.style, {
                    position: "fixed", zIndex: "100000", background: "#252525",
                    color: "white", padding: "5px", border: "1px solid #555",
                    borderRadius: "6px", minWidth: "180px",
                    left: `${Math.max(0, Math.min(event.clientX, window.innerWidth - 200))}px`,
                    top: `${Math.max(0, Math.min(event.clientY, window.innerHeight - 180))}px`,
                });
                menu.setAttribute("role", "menu");
                const dismiss = (e) => { if (!menu.contains(e.target)) closeImageMenu(); };
                const escape = (e) => { if (e.key === "Escape") closeImageMenu(); };
                closeImageMenu = () => {
                    menu.remove();
                    document.removeEventListener("pointerdown", dismiss, true);
                    document.removeEventListener("keydown", escape, true);
                };
                const addAction = (label, action) => {
                    const button = document.createElement("button");
                    button.textContent = label;
                    button.setAttribute("role", "menuitem");
                    Object.assign(button.style, {
                        display: "block", width: "100%", padding: "9px 12px",
                        textAlign: "left", color: "inherit", background: "transparent",
                        border: "0", cursor: "pointer",
                    });
                    button.onclick = async() => {
                        closeImageMenu();
                        try { await action(); } catch (error) { alert(error.message); }
                    };
                    menu.appendChild(button);
                };
                addAction("Upload Image", openImageFilePicker);
                addAction("Paste Image", () => pasteImageFromClipboard(node));
                addAction("Open Image", () => window.open(node.dkstLoadedImage.src, "_blank", "noopener,noreferrer"));
                addAction("Open Mask Editor", () => {
                    if (typeof ComfyApp.open_maskeditor !== "function") {
                        throw new Error("ComfyUI mask editor is unavailable.");
                    }
                    const descriptor = {
                        filename: filenameWidget.value,
                        subfolder: sourceTypeWidget.value === "temp" ? "" : (categoryWidget.value || ""),
                        type: sourceTypeWidget.value || "input",
                    };
                    editorValue = `${descriptor.subfolder ? descriptor.subfolder + "/" : ""}${descriptor.filename} [${descriptor.type}]`;
                    node.imgs = [node.dkstLoadedImage];
                    node.imageIndex = 0;
                    ComfyApp.copyToClipspace(node);
                    ComfyApp.clipspace.images = [descriptor];
                    ComfyApp.clipspace_return_node = node;
                    ComfyApp.open_maskeditor();
                });
                document.body.appendChild(menu);
                document.addEventListener("pointerdown", dismiss, true);
                document.addEventListener("keydown", escape, true);
                menu.querySelector("button")?.focus();
            });
            const onRemoved = node.onRemoved;
            node.onRemoved = function() {
                closeImageMenu();
                fileInput?.remove();
                return onRemoved?.apply(this, arguments);
            };

            const extensionForBlob = (blob) => ({
                "image/jpeg": "jpg",
                "image/webp": "webp",
                "image/gif": "gif",
                "image/avif": "avif",
            })[blob.type] || "png";

            const uploadImage = async(file, type, uploadName) => {
                const upload = new File([file], uploadName, { type: file.type || "image/png" });
                const body = new FormData();
                body.append("image", upload);
                body.append("subfolder", type === "input" ? (categoryWidget.value || "") : "");
                body.append("type", type);
                body.append("overwrite", "false");

                const response = await api.fetchApi("/upload/image", { method: "POST", body });
                if (!response.ok) throw new Error(`Image upload failed (${response.status})`);
                return response.json();
            };

            node.dkstDeleteTemporaryImage = async() => {
                if (sourceTypeWidget.value !== "temp" || !filenameWidget.value) return;
                const filename = node.dkstTemporaryFilename || filenameWidget.value;
                node.dkstTemporaryFilename = null;
                sourceTypeWidget.value = "input";
                const response = await api.fetchApi("/dinki/image-load/delete-temp", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ filename }),
                });
                if (!response.ok) {
                    console.warn(`Unable to delete temporary pasted image (${response.status})`);
                }
            };

            node.dkstUploadClipboardImage = async(blob) => {
                ++refreshGeneration;
                const extension = extensionForBlob(blob);
                const data = await uploadImage(
                    blob,
                    "temp",
                    `DKST_Paste_${Date.now()}.${extension}`,
                );
                await node.dkstDeleteTemporaryImage();
                sourceTypeWidget.value = "temp";
                node.dkstTemporaryFilename = data.name;
                const params = new URLSearchParams({ category: categoryWidget.value || "" });
                const response = await api.fetchApi(`/dinki/image-load/files?${params}`);
                if (!response.ok) throw new Error(`Unable to load image list (${response.status})`);
                const listing = await response.json();
                setValues(filenameWidget, [data.name, ...(listing.files || []).filter(name => name !== data.name)]);
                filenameWidget.value = data.name;
                showPreview(data.name);
                node.setDirtyCanvas(true, true);
            };

            // Also available from the node menu before any image is loaded.
            const extraMenu = node.getExtraMenuOptions;
            node.getExtraMenuOptions = function(canvas, options) {
                const result = extraMenu?.apply(this, arguments);
                options.push({ content: "Upload Image", callback: openImageFilePicker });
                options.push(clipboardMenuAction("Paste Image", () => pasteImageFromClipboard(node)));
                return result;
            };

            node.dkstUploadDroppedImage = async(file) => {
                const data = await uploadImage(file, "input", file.name);
                await node.dkstDeleteTemporaryImage();
                sourceTypeWidget.value = "input";
                await refreshCategories(data.subfolder || "", data.name);
            };

            node.onDragOver = (event) => Array.from(event.dataTransfer?.files || []).some(
                (file) => file.type.startsWith("image/"),
            );
            node.onDragDrop = async(event) => {
                const image = Array.from(event.dataTransfer?.files || []).find(
                    (file) => file.type.startsWith("image/"),
                );
                if (!image) return false;
                try {
                    await node.dkstUploadDroppedImage(image);
                } catch (error) {
                    console.error(error);
                    alert(`Unable to drop image: ${error.message}`);
                }
                return true;
            };

            ensureLater(() => {
                node.dkstRestoreImageLoader?.().catch(console.error);
            });
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = onConfigure?.apply(this, arguments);
            ensureLater(() => {
                this.dkstRestoreImageLoader?.().catch(console.error);
            });
            return result;
        };

        const onSelected = nodeType.prototype.onSelected;
        nodeType.prototype.onSelected = function() {
            const result = onSelected?.apply(this, arguments);
            const sourceType = getWidget(this, "source_type")?.value || "input";
            if (sourceType === "temp" || this.properties?.dkstImageLoad?.source_type === "temp") return result;
            const category = getWidget(this, "category")?.value || "";
            const filename = getWidget(this, "filename")?.value || "";
            this.dkstRefreshImageLoader?.(category, filename).catch(console.error);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            onExecuted?.apply(this, arguments);
            const resolution = message?.resolution?.[0];
            if (resolution) this.dkstImageResolution = resolution;
        };
    },

    setup() {
        window.addEventListener("paste", async(event) => {
            const activeElement = document.activeElement;
            if (activeElement?.matches?.("input, textarea, [contenteditable='true']")) return;

            const selected = Object.values(app.canvas?.selected_nodes || {}).find(
                (node) => node.comfyClass === "DINKI_Image_Load",
            );
            if (!selected?.dkstUploadClipboardImage) return;

            const imageItem = Array.from(event.clipboardData?.items || []).find(
                (item) => item.type.startsWith("image/"),
            );
            if (!imageItem) return;

            const image = imageItem.getAsFile();
            if (!image) return;
            event.preventDefault();
            event.stopImmediatePropagation();
            try {
                await selected.dkstUploadClipboardImage(image);
            } catch (error) {
                console.error(error);
                alert(`Unable to paste image: ${error.message}`);
            }
        }, true);
    },
});

/**
 * 화면 이동 및 줌 로직 처리 함수 (수정됨)
 */
function handleAnchorMove(node, zoomString) {
    const canvas = app.canvas;

    // 1. 줌 레벨 파싱
    let zooms = zoomString.split(',')
        .map(s => parseFloat(s.trim()))
        .filter(n => !isNaN(n))
        .map(n => n / 100); // %를 배율로 변환

    if (zooms.length === 0) zooms = [1.0];

    // 2. 현재 줌 인덱스 순환
    if (typeof node._dinki_zoom_index === "undefined") {
        node._dinki_zoom_index = 0;
    } else {
        node._dinki_zoom_index = (node._dinki_zoom_index + 1) % zooms.length;
    }

    const targetScale = zooms[node._dinki_zoom_index];

    // 3. [수정됨] 위치 이동 (좌상단 기준)
    // ds.offset은 '확대 비율'과 무관한 절대 좌표값이어야 합니다.
    // 노드의 위치(pos)를 음수(-)로 주면 해당 위치가 캔버스의 (0,0)이 됩니다.
    const targetX = -node.pos[0];
    const targetY = -node.pos[1];

    // 4. 적용
    canvas.ds.scale = targetScale;
    canvas.ds.offset = [targetX, targetY];

    // 5. 화면 갱신 강제
    canvas.setDirty(true, true);
}




// ============================================================
// 13. DINKI AutoFocus
// ============================================================
let isAnimating = false;
let targetState = null; // { x, y, scale }

app.registerExtension({
    name: "Dinki.AutoFocus",
    setup() {
        const canvas = app.canvas;
        if (!canvas || canvas.__dinki_auto_focus_attached) return;
        canvas.__dinki_auto_focus_attached = true;
        const displayedNodes = () => {
            const graph = canvas.graph || app.graph;
            return { graph, nodes: graph?.nodes || graph?._nodes || [] };
        };

        window.addEventListener("keydown", (e) => {
            const active = document.activeElement;
            if (active?.matches?.("input, textarea, [contenteditable='true']") ||
                active?.isContentEditable) return;
            const { nodes } = displayedNodes();
            for (const node of nodes) {
                if (node.comfyClass !== "DINKI_Auto_Focus" && node.type !== "DINKI_Auto_Focus") continue;
                const enabled = getWidget(node, "enable");
                const shortcut = getWidget(node, "shortcut_key")?.value;
                if (!enabled || !shortcut || String(shortcut).toLowerCase() !== e.key?.toLowerCase()) continue;
                const previous = enabled.value;
                const next = !previous;
                enabled.value = next;
                enabled.options?.setValue?.(next);
                enabled.callback?.call(enabled, next, canvas, node);
                node.onWidgetChanged?.("enable", next, previous, enabled);
                node.setDirtyCanvas?.(true, true);
            }
        });

        let pending = false;
        let preFocusView = null;
        const focusSelection = () => {
            const { graph, nodes } = displayedNodes();
            if (!graph) return;
            if (preFocusView && preFocusView.graph !== graph) preFocusView = null;
            const nodeSet = new Set(nodes);
            const selected = Array.from(canvas.selectedItems ?? Object.values(canvas.selected_nodes || {}))
                .filter(node => nodeSet.has(node));
            if (!selected.length) {
                const previous = preFocusView;
                preFocusView = null;
                if (previous && getWidget(previous.control, "restore_on_deselect")?.value === true) {
                    const smoothness = Number(getWidget(previous.control, "smoothness")?.value);
                    startViewportMove(canvas, previous.x, previous.y, previous.scale,
                        Number.isFinite(smoothness) && smoothness > 0 ? Math.min(1, smoothness) : 0.2);
                }
                return;
            }
            const activeFocusNode = nodes.find(node =>
                (node.comfyClass === "DINKI_Auto_Focus" || node.type === "DINKI_Auto_Focus") &&
                getWidget(node, "enable")?.value === true);
            if (!activeFocusNode) return;

            const targetNode = selected[selected.length - 1];
            if (!targetNode || targetNode === activeFocusNode) return;

            const zoomLevel = Number(getWidget(activeFocusNode, "zoom_level")?.value);
            const smoothness = Number(getWidget(activeFocusNode, "smoothness")?.value);
            const fit = getWidget(activeFocusNode, "fit")?.value === true;
            if ((!fit && (!Number.isFinite(zoomLevel) || zoomLevel <= 0)) ||
                !Number.isFinite(smoothness) || smoothness <= 0) return;
            const originalView = preFocusView ?? {
                graph,
                x: canvas.ds.offset[0],
                y: canvas.ds.offset[1],
                scale: canvas.ds.scale,
                control: activeFocusNode,
            };
            if (startSmoothMove(canvas, targetNode, zoomLevel, Math.min(1, smoothness), fit)) {
                originalView.control = activeFocusNode;
                preFocusView = originalView;
            }
        };
        const scheduleFocus = () => {
            if (pending) return;
            pending = true;
            queueMicrotask(() => {
                pending = false;
                focusSelection();
            });
        };

        // Nodes 2.0 selects through these methods without calling the classic
        // onSelectionChange callback. Observe both paths after selection settles.
        for (const name of ["select", "deselect", "deselectAll", "onSelectionChange"]) {
            const original = canvas[name];
            if (name !== "onSelectionChange" && typeof original !== "function") continue;
            canvas[name] = function(...args) {
                const result = original?.apply(this, args);
                scheduleFocus();
                return result;
            };
        }
    }
});

/**
 * 부드러운 이동을 위한 애니메이션 함수
 */
function startSmoothMove(canvas, node, targetZoom, smoothness, fit = false) {
    const nodeWidth = Number(node.size?.[0]);
    const nodeHeight = Number(node.size?.[1]);
    const rect = canvas.canvas?.getBoundingClientRect();
    if (!rect?.width || !rect?.height || !Number.isFinite(nodeWidth) || !Number.isFinite(nodeHeight) ||
        nodeWidth <= 0 || nodeHeight <= 0) return false;
    const visibleWidth = rect.width;
    const visibleHeight = rect.height;
    if (fit) {
        const configuredMax = Number(canvas.ds?.max_scale);
        const maxScale = Number.isFinite(configuredMax) && configuredMax > 0 ? configuredMax : 3;
        targetZoom = Math.min(visibleWidth * 0.9 / nodeWidth,
            visibleHeight * 0.9 / nodeHeight, maxScale);
    }
    if (!Number.isFinite(targetZoom) || targetZoom <= 0) return false;
    const nodeCenterX = node.pos[0] + nodeWidth / 2;
    const nodeCenterY = node.pos[1] + nodeHeight / 2;

    // Center the node at the chosen scale, leaving a 5% margin in Fit mode.
    const targetOffsetX = (visibleWidth / 2) / targetZoom - nodeCenterX;
    const targetOffsetY = (visibleHeight / 2) / targetZoom - nodeCenterY;

    startViewportMove(canvas, targetOffsetX, targetOffsetY, targetZoom, smoothness);
    return true;
}

function startViewportMove(canvas, x, y, scale, smoothness) {
    targetState = {
        x, y, scale, smoothness,
        graph: canvas.graph || app.graph,
    };

    // An in-flight move adopts the new destination on its next frame.
    if (!isAnimating) {
        isAnimating = true;
        requestAnimationFrame(() => animateLoop(canvas));
    }
}

function animateLoop(canvas) {
    if (!targetState || targetState.graph !== (canvas.graph || app.graph)) {
        isAnimating = false;
        targetState = null;
        return;
    }

    // 현재 상태
    const currentX = canvas.ds.offset[0];
    const currentY = canvas.ds.offset[1];
    const currentScale = canvas.ds.scale;

    // Lerp (선형 보간) 공식: A + (B - A) * t
    // t가 작을수록 느리고 부드럽게(Elastic), 클수록 빠르게
    const t = targetState.smoothness; 
    
    // 다음 프레임 값 계산
    const nextX = currentX + (targetState.x - currentX) * t;
    const nextY = currentY + (targetState.y - currentY) * t;
    const nextScale = currentScale + (targetState.scale - currentScale) * t;

    // 적용
    canvas.ds.offset = [nextX, nextY];
    canvas.ds.scale = nextScale;
    canvas.setDirty(true, true);

    // 종료 조건: 목표에 충분히 가까워졌으면 멈춤 (jitter 방지)
    const dist = Math.abs(targetState.x - currentX) + Math.abs(targetState.y - currentY) + Math.abs(targetState.scale - currentScale);
    
    if (dist < 0.5) {
        // 정확한 목표값으로 딱 맞추고 종료
        canvas.ds.offset = [targetState.x, targetState.y];
        canvas.ds.scale = targetState.scale;
        canvas.setDirty(true, true);
        
        isAnimating = false;
        targetState = null; // 타겟 해제
    } else {
        // 계속 반복
        requestAnimationFrame(() => animateLoop(canvas));
    }
}
