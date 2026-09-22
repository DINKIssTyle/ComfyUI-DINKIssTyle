// ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/js/dinki_nodes.js

import { app, ComfyApp } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// 공통 헬퍼
function getWidget(node, name) {
  return node.widgets?.find(w => w.name === name);
}
function ensureLater(fn) {
  requestAnimationFrame(() => setTimeout(fn, 0));
}

function previewImageActions(descriptor) {
    const url = api.apiURL(`/view?${new URLSearchParams(descriptor)}`);
    return [
        { content: "Open Image", callback: () => window.open(url, "_blank", "noopener,noreferrer") },
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
app.registerExtension({
  name: "DINKI.PromptSelectorLive.Attach.v2",
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
        const modeW  = getWidget(node, "mode");
        const sepW   = getWidget(node, "separator");
        if (!titleW || !textW) {
          // Vue/Nodes 2.0 may create widgets after the node lifecycle hook.
          if (attempt < 120) requestAnimationFrame(() => attachWhenReady(attempt + 1));
          else node.__dinki_live_attach_pending = false;
          return;
        }
        node.__dinki_live_attached = true;
        node.__dinki_live_attach_pending = false;

        const setTextValue = (value) => {
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

          const loadSelectedPrompt = async (value) => {
            titleW.value = value;
            observedTitle = value;
            const sepVal = sepW?.value ?? "\n";
            const sig = JSON.stringify([value, modeW?.value || "append", sepVal, textW.value]);
            if (node.__dinki_last_apply_sig === sig) return;
            node.__dinki_last_apply_sig = sig;
            const requestId = (node.__dinki_live_request_id || 0) + 1;
            node.__dinki_live_request_id = requestId;

            try {
              const res = await api.fetchApi("/dinki/prompts");
              if (!res.ok) throw new Error(`HTTP ${res.status}`);
              const map = await res.json();
              // Ignore a slow response if another preset was selected while
              // this request was in flight.
              if (requestId !== node.__dinki_live_request_id || titleW.value !== value) return;
              const selectedTitle = String(value ?? titleW.value ?? "").trim();
              let picked = (map && selectedTitle) ? (map[selectedTitle] || "") : "";
              if (!picked && map && typeof map === "object") {
                const matchedTitle = Object.keys(map).find(
                  key => key.trim().toLocaleLowerCase() === selectedTitle.toLocaleLowerCase()
                );
                if (matchedTitle) picked = map[matchedTitle] || "";
              }
              const mode = modeW?.value || "append";
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
                setTextValue(picked);
              } else if (mode === "append") {
                let nextValue;
                if (!textW.value) nextValue = picked;
                else nextValue = (sep && !textW.value.endsWith(sep))
                  ? textW.value + sep + picked
                  : textW.value + picked;
                setTextValue(nextValue);
              }
              node.setDirtyCanvas(true, true);
              console.info("[DINKI Live] Prompt applied:", selectedTitle, `(${picked.length} chars)`);
            } catch (e) {
              console.error("DINKI Live fetch/prompts error:", e);
            } finally {
              setTimeout(() => { node.__dinki_last_apply_sig = null; }, 0);
            }
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
            return loadSelectedPrompt(value);
          };

          // Nodes 2.0 reports widget edits through the node notification and
          // may not invoke the classic widget callback at all.
          const origWidgetChanged = node.onWidgetChanged;
          node.onWidgetChanged = function (name, value) {
            const result = origWidgetChanged?.apply(this, arguments);
            if (name === "title") loadSelectedPrompt(value);
            return result;
          };

          // Final compatibility path: a few Nodes 2.0 renderers update the
          // store-backed combo without calling either legacy hook.
          const origDrawForeground = node.onDrawForeground;
          node.onDrawForeground = function () {
            const result = origDrawForeground?.apply(this, arguments);
            if (titleW.value !== observedTitle) {
              observedTitle = titleW.value;
              loadSelectedPrompt(observedTitle);
            }
            return result;
          };

          node.__dinki_live_cb_wrapped = true;
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
function applyNodeSwitch(node, changedName, changedValue) {
    // app.graph is the currently displayed graph, which may be a different subgraph.
    const graph = node.graph;
    if (!graph || app.configuringGraph) return;

    const idWidget = getWidget(node, "node_ids");
    const toggleWidget = getWidget(node, "active");
    if (!idWidget || !toggleWidget) return;

    // Some widget renderers notify before committing widget.value.
    const idsText = changedName === "node_ids" ? changedValue : idWidget.value;
    const isActive = changedName === "active" ? changedValue : toggleWidget.value;
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

app.registerExtension({
    name: "DINKI.NodeSwitch",
    nodeCreated(node) {
        if (node.comfyClass !== "DINKI_Node_Switch") return;

        // Nodes 2.0 and programmatic widget updates use the node notification.
        const onWidgetChanged = node.onWidgetChanged;
        node.onWidgetChanged = function (name, value) {
            const result = onWidgetChanged?.apply(this, arguments);
            if (name === "node_ids" || name === "active") {
                applyNodeSwitch(this, name, value);
            }
            return result;
        };

        // Keep classic canvas widgets working and preserve other extensions' callbacks.
        for (const name of ["node_ids", "active"]) {
            const widget = getWidget(node, name);
            if (!widget) continue;
            const callback = widget.callback;
            widget.callback = function (value) {
                const result = callback?.apply(this, arguments);
                applyNodeSwitch(node, name, value);
                return result;
            };
        }

        // Defer until insertion/configuration has completed, without a fixed timer.
        for (const hook of ["onAdded", "onConfigure"]) {
            const original = node[hook];
            node[hook] = function () {
                const result = original?.apply(this, arguments);
                queueMicrotask(() => applyNodeSwitch(this));
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
                if (node.comfyClass === "DINKI_Node_Switch") applyNodeSwitch(node);
                if (node.subgraph) syncGraph(node.subgraph);
            }
        };
        syncGraph(app.rootGraph ?? app.graph);
    }
});


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
        // 그래프가 로드된 후 실행
        const originalOnSelectionChange = LGraphCanvas.prototype.processNodeSelected;
        const canvas = app.canvas;
        const originalSelectionChange = canvas.onSelectionChange;
        
        canvas.onSelectionChange = function(nodes) {
            // 원래 기능 실행
            if (originalSelectionChange) {
                originalSelectionChange.apply(this, arguments);
            }

            // 1. 현재 선택된 노드 찾기
            let selectedNodeId = "None";
            const selected = Object.values(canvas.selected_nodes || {});
            
            if (selected.length > 0) {
                const targetNode = selected[selected.length - 1];
                selectedNodeId = String(targetNode.id);
                // 디버깅용: 콘솔에 선택된 ID 출력 (F12 눌러서 확인 가능)
                console.log("DINKI Check: Selected ID =", selectedNodeId);
            }

            // 2. 화면에 있는 모든 'DINKI_Node_Check' 노드 찾기
            const graph = app.graph;
            if (!graph) return;

            // [수정됨] findNodesByClass -> findNodesByType
            // ComfyUI에서 노드 타입 문자열("DINKI_Node_Check")로 찾을 때는 ByType을 써야 합니다.
            const checkNodes = graph.findNodesByType("DINKI_Node_Check");
            
            // 3. 찾은 노드들의 위젯 값 업데이트
            if (checkNodes && checkNodes.length > 0) {
                checkNodes.forEach(node => {
                    if (node.widgets && node.widgets[0]) {
                        // 값이 다를 때만 업데이트
                        if (node.widgets[0].value !== selectedNodeId) {
                            node.widgets[0].value = selectedNodeId;
                            node.setDirtyCanvas(true, true); 
                        }
                    }
                });
            }
        };
    },
    
    nodeCreated(node, app) {
        if (node.comfyClass === "DINKI_Node_Check") {

            const size = node.computeSize();
            node.setSize(size);

            if (node.widgets && node.widgets[0]) {
                setTimeout(() => {
                    if (node.widgets[0].inputEl) {
                        node.widgets[0].inputEl.readOnly = true;
                        node.widgets[0].inputEl.style.opacity = 0.6;
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
// 13. DKST Preview (Image) resolution overlay
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
            selector.onchange = display;
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
            this.dkstUpdatePreview = message => {
                descriptors = message?.dkst_images || message?.images || [];
                selector.replaceChildren();
                descriptors.forEach((item, index) => {
                    const option = document.createElement("option");
                    option.value = String(index);
                    option.textContent = `${index + 1} / ${descriptors.length}`;
                    selector.appendChild(option);
                });
                selector.value = "0";
                selector.style.display = descriptors.length > 1 ? "block" : "none";
                resolution.textContent = message?.resolution?.[0] || "";
                display();
                this.setDirtyCanvas(true, true);
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
            const output = app.nodeOutputs?.[node.id];
            if (output) node.dkstUpdatePreview?.(output);
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

            const showPreview = (filename = filenameWidget.value) => {
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
                    node.dkstLoadedImage = image;
                    node.dkstImageResolution = `${image.naturalWidth} × ${image.naturalHeight}`;
                    previewElement.src = image.src;
                    resolutionElement.textContent = node.dkstImageResolution;
                    previewContainer.style.display = "flex";
                    node.setDirtyCanvas(true, true);
                };
                image.onerror = () => {
                    node.dkstLoadedImage = null;
                    node.dkstImageResolution = "Unable to preview image";
                    previewElement.removeAttribute("src");
                    resolutionElement.textContent = node.dkstImageResolution;
                    previewContainer.style.display = "flex";
                    node.setDirtyCanvas(true, true);
                };
                image.src = api.apiURL(`/view?${params.toString()}`);
            };

            const refreshFiles = async(category, preferredFilename = null) => {
                sourceTypeWidget.value = "input";
                const params = new URLSearchParams({ category: category || "" });
                const response = await api.fetchApi(`/dinki/image-load/files?${params.toString()}`);
                if (!response.ok) throw new Error(`Unable to load image list (${response.status})`);
                const data = await response.json();
                setValues(filenameWidget, data.files || []);
                filenameWidget.value = data.files?.includes(preferredFilename)
                    ? preferredFilename
                    : (data.files?.[0] || "");
                showPreview(filenameWidget.value);
                node.setDirtyCanvas(true, true);
            };

            const refreshCategories = async(preferredCategory = null, preferredFilename = null) => {
                const response = await api.fetchApi("/dinki/image-load/categories");
                if (!response.ok) throw new Error(`Unable to load image categories (${response.status})`);
                const data = await response.json();
                const categories = data.categories || [""];
                setValues(categoryWidget, categories);
                categoryWidget.value = categories.includes(preferredCategory)
                    ? preferredCategory
                    : "";
                await refreshFiles(categoryWidget.value, preferredFilename);
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

            let closeImageMenu = () => {};
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
                    top: `${Math.max(0, Math.min(event.clientY, window.innerHeight - 100))}px`,
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
                    button.onclick = () => {
                        closeImageMenu();
                        try { action(); } catch (error) { alert(error.message); }
                    };
                    menu.appendChild(button);
                };
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
                refreshCategories(categoryWidget.value, filenameWidget.value).catch(console.error);
            });
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = onConfigure?.apply(this, arguments);
            ensureLater(() => {
                const category = getWidget(this, "category")?.value || "";
                const filename = getWidget(this, "filename")?.value || "";
                const sourceType = getWidget(this, "source_type");
                if (sourceType) sourceType.value = "input";
                this.dkstRefreshImageLoader?.(category, filename).catch(console.error);
            });
            return result;
        };

        const onSelected = nodeType.prototype.onSelected;
        nodeType.prototype.onSelected = function() {
            const result = onSelected?.apply(this, arguments);
            const sourceType = getWidget(this, "source_type")?.value || "input";
            if (sourceType === "temp") return result;
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
        // --------------------------------------------------------
        // 1. 단축키 리스너 (ON/OFF 토글)
        // --------------------------------------------------------
        window.addEventListener("keydown", (e) => {
            // 텍스트 입력 중일 때는 무시
            const activeTag = document.activeElement.tagName.toUpperCase();
            if (activeTag === "INPUT" || activeTag === "TEXTAREA") return;

            const graph = app.graph;
            if (!graph) return;

            const focusNodes = graph.findNodesByType("DINKI_Auto_Focus");
            if (!focusNodes || focusNodes.length === 0) return;

            // 모든 Auto Focus 노드에 대해 단축키 검사
            focusNodes.forEach(node => {
                const enableWidget = node.widgets[0];   // enable
                const shortcutWidget = node.widgets[1]; // shortcut_key

                if (shortcutWidget && shortcutWidget.value.toLowerCase() === e.key.toLowerCase()) {
                    // 값 토글 (True <-> False)
                    enableWidget.value = !enableWidget.value;
                    
                    // 시각적 피드백 (노드 업데이트)
                    node.setDirtyCanvas(true, true);
                    
                    // (선택) 상태 변경 알림 로그
                    // console.log(`Auto Focus ${enableWidget.value ? "Enabled" : "Disabled"}`);
                }
            });
        });

        // --------------------------------------------------------
        // 2. 노드 선택 시 이동 로직
        // --------------------------------------------------------
        const originalOnSelectionChange = LGraphCanvas.prototype.processNodeSelected;
        const canvas = app.canvas;
        const originalSelectionChange = canvas.onSelectionChange;
        
        canvas.onSelectionChange = function(nodes) {
            if (originalSelectionChange) {
                originalSelectionChange.apply(this, arguments);
            }

            // 활성화된 Auto Focus 노드 찾기
            const graph = app.graph;
            if (!graph) return;
            
            const focusNodes = graph.findNodesByType("DINKI_Auto_Focus");
            if (!focusNodes || focusNodes.length === 0) return;

            let activeFocusNode = null;
            for (let node of focusNodes) {
                // [0]: enable
                if (node.widgets && node.widgets[0] && node.widgets[0].value === true) {
                    activeFocusNode = node;
                    break;
                }
            }

            if (!activeFocusNode) return;

            // 타겟 노드 확인
            const selected = Object.values(canvas.selected_nodes || {});
            if (selected.length === 0) return;
            
            const targetNode = selected[selected.length - 1];

            // 자기 자신 클릭 시 이동 안 함
            if (targetNode.id === activeFocusNode.id) return;

            // 설정값 가져오기
            const zoomLevel = activeFocusNode.widgets[2].value || 1.0;     // zoom_level
            const smoothness = activeFocusNode.widgets[3].value || 0.2;    // smoothness

            // 이동 시작
            startSmoothMove(canvas, targetNode, zoomLevel, smoothness);
        };
    }
});

/**
 * 부드러운 이동을 위한 애니메이션 함수
 */
function startSmoothMove(canvas, node, targetZoom, smoothness) {
    // 1. 목표 좌표 계산 (이전과 동일한 중앙 정렬 로직)
    const nodeCenterX = node.pos[0] + node.size[0] / 2;
    const nodeCenterY = node.pos[1] + node.size[1] / 2;
    
    // 실제 뷰포트 크기
    const rect = canvas.canvas.getBoundingClientRect();
    const visibleWidth = rect.width;
    const visibleHeight = rect.height;

    // 최종 목표 Offset
    const targetOffsetX = (visibleWidth / 2) / targetZoom - nodeCenterX;
    const targetOffsetY = (visibleHeight / 2) / targetZoom - nodeCenterY;

    // 2. 목표 상태 저장
    targetState = {
        x: targetOffsetX,
        y: targetOffsetY,
        scale: targetZoom,
        smoothness: smoothness
    };

    // 3. 애니메이션 루프 시작 (이미 돌고 있다면 타겟만 갱신됨)
    if (!isAnimating) {
        isAnimating = true;
        requestAnimationFrame(() => animateLoop(canvas));
    }
}

function animateLoop(canvas) {
    if (!targetState) {
        isAnimating = false;
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
