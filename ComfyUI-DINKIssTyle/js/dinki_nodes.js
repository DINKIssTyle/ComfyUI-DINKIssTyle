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
        // DINKI_PromptSelector 노드일 때만 이 로직을 적용
        if (nodeData.name === "DINKI_PromptSelector") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // 1. Python이 만든 원래 텍스트 위젯을 찾습니다.
                const originalWidget = this.widgets.find(w => w.name === "title");

                // 2. 새로운 드롭다운 위젯을 만듭니다.
                const comboWidget = this.addWidget(
                    "combo",
                    "title", // 이름은 같게 유지
                    "",      // 초기값
                    (value) => {
                        // 드롭다운 값이 바뀔 때마다 숨겨진 원래 위젯의 값을 업데이트
                        originalWidget.value = value;
                    },
                    { values: [] } // 필수 옵션
                );
                comboWidget.serialize = false; // 워크플로우에 이 위젯의 값은 저장하지 않음

                // 3. 원래 텍스트 위젯은 화면에서 완전히 숨깁니다.
                originalWidget.hidden = true;
                
                // 4. 새로고침 버튼을 추가합니다.
                const refreshButton = this.addWidget(
                    "button",
                    "🔄 Refresh Prompts",
                    null,
                    () => refreshPromptList(true) // 버튼 클릭 시 강제 새로고침
                );

                // 5. 프롬프트 목록을 가져와 드롭다운을 채우는 함수
                const refreshPromptList = async (force) => {
                    try {
                        // 현재 목록이 비어있거나, 강제 새로고침일 때만 API 호출
                        if (force || !comboWidget.options.values || comboWidget.options.values.length === 0) {
                            const response = await api.fetchApi('/get-csv-prompts');
                            const titles = await response.json();
                            
                            comboWidget.options.values = titles;
                            
                            // 현재 선택된 값이 새 목록에 없으면 첫 번째 항목으로 설정
                            if (!titles.includes(comboWidget.value) && titles.length > 0) {
                                comboWidget.value = titles[0];
                            } else if (titles.length === 0) {
                                comboWidget.value = "";
                            }
                        }
                    } catch (error) {
                        console.error("❌ Error refreshing DINKI prompt list:", error);
                    } finally {
                        // 드롭다운 콜백을 수동으로 호출하여 숨겨진 위젯 값 동기화
                        if (comboWidget.callback) {
                            comboWidget.callback(comboWidget.value);
                        }
                    }
                };

                // 노드가 처음 생성/로드될 때 목록을 한 번 불러옵니다.
                refreshPromptList(false);

                // 기존 위젯들을 재배치하여 올바른 순서를 유지합니다.
                this.widgets.splice(this.widgets.indexOf(originalWidget), 1); // 원래 위젯 제거
                this.widgets.splice(0, 0, comboWidget); // 드롭다운을 맨 위에 추가
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

    // 같은 노드 정의에 중복 패치 금지
    if (nodeType.prototype.__dinki_live_patched) return;
    nodeType.prototype.__dinki_live_patched = true;

    async function attach(node) {
      // 같은 노드 인스턴스에 중복 attach 금지
      if (node.__dinki_live_attached) return;
      node.__dinki_live_attached = true;

      ensureLater(() => {
        const titleW = getWidget(node, "title");
        const textW  = getWidget(node, "text");
        const modeW  = getWidget(node, "mode");
        const sepW   = getWidget(node, "separator");
        if (!titleW || !textW) return;

        // Clear 버튼(중복 생성 방지)
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

        // Refresh 버튼(중복 생성 방지)
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

              // 프로그램적으로 값 바꾼 뒤 한 번만 반영
              if (titleW.callback) titleW.callback(titleW.value);
              node.setDirtyCanvas(true);
            } catch (e) {
              console.error("DINKI Live refresh error:", e);
            }
          });
          node.__dinki_live_refresh_added = true;
        }

        // 콤보 콜백: 중복 래핑 방지
        if (!node.__dinki_live_cb_wrapped) {
          const origCb = titleW.callback;

          titleW.callback = async (value) => {
            // 같은 시그니처로 중복 호출되면 스킵(안전장치)
            const sepVal = sepW?.value ?? "\n";
            const sig = JSON.stringify([value, modeW?.value || "append", sepVal, textW.value]);
            if (node.__dinki_last_apply_sig === sig) return;
            node.__dinki_last_apply_sig = sig;

            // 기존 콜백 먼저/나중 어떤 쪽이든 상관없지만, 2중 호출만 예방되면 OK
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
              // 다음 정상 선택에서 다시 적용될 수 있도록 시그니처 갱신
              // 단, 즉시 동일 값 재호출을 막기 위해 약간 지연 후 해제
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
  // 옵션 목록에서 "-- None --"가 있으면 그걸로
  const noneIdx = (w.options || []).indexOf("-- None --");
  if (noneIdx >= 0) {
    w.value = "-- None --";
  } else {
    // 없으면 첫 항목으로
    w.value = (w.options && w.options[0]) || w.value;
  }
  // UI 갱신
  if (w.callback) try { w.callback(w.value); } catch (e) {}
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "DINKI.PromptSelector.AutoReset",
  async setup() {
    api.addEventListener("executedNode", ({ detail }) => {
      const { node } = detail || {};
      if (!node) return;
      // 대상 노드만
      const targetNames = ["DINKI_PromptSelector", "DINKI_PromptSelectorLive"];
      if (!targetNames.includes(node?.comfyClass)) return;
      resetTitleWidget(node);
    });
  }
});


// ============================================================
// 4. [NEW] DINKI Node Switch Logic
// ============================================================
app.registerExtension({
    name: "DINKI.NodeSwitch",
    async nodeCreated(node, app) {
        // DINKI_Node_Switch 클래스일 때만 동작
        if (node.comfyClass === "DINKI_Node_Switch") {
            
            // 위젯 값 변경 시 실행될 함수
            const onWidgetChange = function () {
                try {
                    const idWidget = node.widgets.find(w => w.name === "node_ids");
                    const toggleWidget = node.widgets.find(w => w.name === "active");

                    if (!idWidget || !toggleWidget) return;

                    const idsText = idWidget.value;
                    const isActive = toggleWidget.value; // On=True, Off=False

                    // 쉼표로 구분된 ID 파싱
                    const ids = idsText.split(",").map(id => parseInt(id.trim())).filter(id => !isNaN(id));

                    // 그래프 내의 모든 노드를 순회
                    app.graph._nodes.forEach(targetNode => {
                        if (ids.includes(targetNode.id)) {
                            // ComfyUI Node Modes: 0: Always, 2: Mute, 4: Bypass
                            
                            if (isActive) {
                                // On 상태: 현재 Bypass(4)라면 Always(0)로 변경
                                if (targetNode.mode === 4) {
                                    targetNode.mode = 0;
                                }
                            } else {
                                // Off 상태: Bypass(4)로 변경
                                targetNode.mode = 4;
                            }
                        }
                    });
                    
                    // 캔버스 다시 그리기
                    app.graph.setDirtyCanvas(true, true);

                } catch (error) {
                    console.error("DINKI Switch Error:", error);
                }
            };

            // 위젯 찾아서 콜백 연결
            const idWidget = node.widgets.find(w => w.name === "node_ids");
            const toggleWidget = node.widgets.find(w => w.name === "active");

            if (idWidget) {
                idWidget.callback = onWidgetChange;
            }
            if (toggleWidget) {
                toggleWidget.callback = onWidgetChange;
            }
            
            // 초기 로딩 시 상태 동기화 (약간의 지연 후)
            setTimeout(onWidgetChange, 1000);
        }
    }
});


// ============================================================
// 4. DINKI Color LUT Upload Logic
// ============================================================
app.registerExtension({
	name: "DINKIssTyle.ColorLUT.Upload",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "DINKI_Color_Lut") {
			
			// 노드가 생성될 때 실행되는 함수
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				const node = this;
				// "Upload .cube" 버튼 위젯 추가
				const uploadWidget = this.addWidget("button", "Upload .cube", "Upload", () => {
					// 숨겨진 파일 입력창 생성 및 클릭
					const fileInput = document.createElement("input");
					Object.assign(fileInput, {
						type: "file",
						accept: ".cube",
						style: "display: none",
						onchange: async () => {
							if (fileInput.files.length > 0) {
								await uploadFile(fileInput.files[0]);
							}
						},
					});
					document.body.appendChild(fileInput);
					fileInput.click();
					document.body.removeChild(fileInput);
				});

				// 파일 업로드 처리 함수
				async function uploadFile(file) {
					try {
						const body = new FormData();
						body.append("image", file); // ComfyUI API는 키 이름을 'image'로 받음
						body.append("subfolder", "luts"); // input/luts 폴더 지정
						body.append("type", "input");
						body.append("overwrite", "true");

						// ComfyUI 업로드 API 호출
						const resp = await api.fetchApi("/upload/image", {
							method: "POST",
							body,
						});

						if (resp.status === 200) {
							const data = await resp.json();
							const filename = data.name;

							// lut_name 위젯 찾기
							const lutWidget = node.widgets.find((w) => w.name === "lut_name");
							if (lutWidget) {
								// 리스트에 없으면 추가 (옵션 갱신 시늉)
								if (!lutWidget.options.values.includes(filename)) {
									lutWidget.options.values.push(filename);
								}
								// 값 선택
								lutWidget.value = filename;
                                
                                // 노드 그래프 업데이트 알림
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

                this.previewImage.onload = () => {
                    app.graph.setDirtyCanvas(true);
                };

                // --- Preview Request Logic ---
                const lutWidget = this.widgets.find((w) => w.name === "lut_name");
                const strengthWidget = this.widgets.find((w) => w.name === "strength");

                const requestPreview = async () => {
                    const lutName = lutWidget.value;
                    const strength = strengthWidget.value;

                    try {
                        const resp = await api.fetchApi("/dinki/preview_lut", {
                            method: "POST",
                            body: JSON.stringify({ 
                                lut_name: lutName,
                                strength: strength 
                            }),
                        });

                        if (resp.status === 200) {
                            const blob = await resp.blob();
                            
                            if (node.previewUrl) {
                                URL.revokeObjectURL(node.previewUrl);
                            }

                            const url = URL.createObjectURL(blob);
                            node.previewUrl = url; 
                            node.previewImage.src = url;

                        } else if (resp.status === 400) {
                            // 캐시 없음 (정상)
                        }
                    } catch (e) {
                        console.error("DINKI LUT Preview Error:", e);
                    }
                };

                if (lutWidget) lutWidget.callback = requestPreview;
                if (strengthWidget) strengthWidget.callback = requestPreview;

                // -----------------------------------------------------------
                // [수정됨] ID 비교 로직 완화 (=== -> ==)
                // -----------------------------------------------------------
                api.addEventListener("executed", ({ detail }) => {
                    // detail.node는 문자열일 수 있고, node.id는 숫자일 수 있음
                    // 따라서 == (느슨한 비교)를 사용해야 함
                    if (detail?.node == node.id) {
                        requestPreview();
                    }
                });
                // -----------------------------------------------------------

                // --- Upload Button ---
                this.addWidget("button", "Upload .cube", "Upload", () => {
                    const fileInput = document.createElement("input");
                    Object.assign(fileInput, {
                        type: "file",
                        accept: ".cube",
                        style: "display: none",
                        onchange: async () => {
                            if (fileInput.files.length > 0) {
                                await uploadFile(fileInput.files[0]);
                            }
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
                                requestPreview(); 
                            }
                            alert(`Uploaded: ${filename}`);
                        } else {
                            alert("Upload failed: " + resp.statusText);
                        }
                    } catch (error) {
                        alert("Error: " + error);
                    }
                }

                return r;
            };

            // --- Context Menu ---
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function (_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }
                if (this.previewUrl) {
                    options.push(
                        {
                            content: "Open Preview Image",
                            callback: () => {
                                window.open(this.previewUrl, "_blank");
                            },
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

            // --- Draw ---
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) onDrawForeground.apply(this, arguments);

                if (this.previewImage && this.previewImage.src) {
                    const w = this.size[0];
                    const h = this.size[1];
                    const headerHeight = 50; 
                    const drawH = h - headerHeight - 10;
                    
                    if (drawH > 0) {
                        const imgW = this.previewImage.width;
                        const imgH = this.previewImage.height;
                        const ratio = Math.min(w / imgW, drawH / imgH);
                        
                        const finalW = imgW * ratio;
                        const finalH = imgH * ratio;
                        const x = (w - finalW) / 2;
                        const y = headerHeight + (drawH - finalH) / 2;

                        ctx.save();
                        ctx.drawImage(this.previewImage, x, y + 10, finalW, finalH);
                        ctx.strokeStyle = "#555";
                        ctx.lineWidth = 1;
                        ctx.strokeRect(x, y + 10, finalW, finalH);
                        ctx.restore();
                    }
                }
            };
        }
    },
});