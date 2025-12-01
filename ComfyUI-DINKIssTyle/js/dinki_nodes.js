// ComfyUI/custom_nodes/ComfyUI-DINKIssTyle/js/dinki_nodes.js

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// 공통 헬퍼
function getWidget(node, name) {
  return node.widgets?.find(w => w.name === name);
}
function ensureLater(fn) {
  requestAnimationFrame(() => setTimeout(fn, 0));
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
      if (node.__dinki_live_attached) return;
      node.__dinki_live_attached = true;

      ensureLater(() => {
        const titleW = getWidget(node, "title");
        const textW  = getWidget(node, "text");
        const modeW  = getWidget(node, "mode");
        const sepW   = getWidget(node, "separator");
        if (!titleW || !textW) return;

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
              const res = await fetch("/get-csv-prompts");
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

          titleW.callback = async (value) => {
            const sepVal = sepW?.value ?? "\n";
            const sig = JSON.stringify([value, modeW?.value || "append", sepVal, textW.value]);
            if (node.__dinki_last_apply_sig === sig) return;
            node.__dinki_last_apply_sig = sig;

            if (origCb) origCb(value);

            try {
              const res = await fetch("/dinki/prompts");
              const map = await res.json();
              const picked = (map && value && map[value]) ? (map[value] || "") : "";
              const mode = modeW?.value || "append";
              let sep = sepVal;
              if (sep === "\\n") sep = "\n";
              if (sep === "\\n\\n") sep = "\n\n";
              if (!picked) return;

              if (mode === "replace") {
                textW.value = picked;
              } else if (mode === "append") {
                if (!textW.value) textW.value = picked;
                else textW.value = (sep && !textW.value.endsWith(sep))
                  ? textW.value + sep + picked
                  : textW.value + picked;
              }
              node.setDirtyCanvas(true);
            } catch (e) {
              console.error("DINKI Live fetch/prompts error:", e);
            } finally {
              setTimeout(() => { node.__dinki_last_apply_sig = null; }, 0);
            }
          };

          node.__dinki_live_cb_wrapped = true;
        }
      });
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
app.registerExtension({
    name: "DINKI.NodeSwitch",
    async nodeCreated(node, app) {
        if (node.comfyClass === "DINKI_Node_Switch") {
            
            const onWidgetChange = function () {
                try {
                    const idWidget = node.widgets.find(w => w.name === "node_ids");
                    const toggleWidget = node.widgets.find(w => w.name === "active");

                    if (!idWidget || !toggleWidget) return;

                    const idsText = idWidget.value;
                    const isActive = toggleWidget.value;

                    const ids = idsText.split(",").map(id => parseInt(id.trim())).filter(id => !isNaN(id));

                    app.graph._nodes.forEach(targetNode => {
                        if (ids.includes(targetNode.id)) {
                            if (isActive) {
                                if (targetNode.mode === 4) {
                                    targetNode.mode = 0;
                                }
                            } else {
                                targetNode.mode = 4;
                            }
                        }
                    });
                    
                    app.graph.setDirtyCanvas(true, true);

                } catch (error) {
                    console.error("DINKI Switch Error:", error);
                }
            };

            const idWidget = node.widgets.find(w => w.name === "node_ids");
            const toggleWidget = node.widgets.find(w => w.name === "active");

            if (idWidget) idWidget.callback = onWidgetChange;
            if (toggleWidget) toggleWidget.callback = onWidgetChange;
            
            setTimeout(onWidgetChange, 1000);
        }
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
