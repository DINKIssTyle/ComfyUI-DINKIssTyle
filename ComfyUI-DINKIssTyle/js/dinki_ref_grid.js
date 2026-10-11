// ComfyUI-DINKIssTyle/js/dinki_ref_grid.js
// Interactive Multi-Image Grid Widget for DINKI_Create_Image_Ref (DKST PS)

import { app } from "../../scripts/app.js";

function createElement(tag, styles = {}, attrs = {}) {
    const el = document.createElement(tag);
    Object.assign(el.style, styles);
    for (const [k, v] of Object.entries(attrs)) {
        if (k.startsWith("on") && typeof v === "function") {
            el.addEventListener(k.substring(2).toLowerCase(), v);
        } else {
            el.setAttribute(k, v);
        }
    }
    return el;
}

function parseItems(value) {
    if (!value) return [];
    try {
        const parsed = typeof value === "string" ? JSON.parse(value) : value;
        if (!Array.isArray(parsed)) return [];
        return parsed.filter(item => item && typeof item === "object" && item.filename);
    } catch {
        return [];
    }
}

app.registerExtension({
    name: "DINKI.ImageRefGrid",
    getCustomWidgets() {
        return {
            DKST_IMAGE_REF_GRID(node, inputName, inputData) {
                let items = parseItems(inputData?.[1]?.default || "[]");
                let thumbSize = 80; // default 80px
                let widget;

                const root = createElement("div", {
                    boxSizing: "border-box",
                    width: "100%",
                    minHeight: "140px",
                    padding: "8px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "8px",
                    background: "rgba(20, 20, 24, 0.75)",
                    borderRadius: "8px",
                    border: "1px solid rgba(255, 255, 255, 0.12)",
                    color: "var(--fg-color, #e0e0e0)",
                    fontSize: "12px",
                    fontFamily: "sans-serif",
                    backdropFilter: "blur(6px)",
                });

                // Hidden file input for multi-upload
                const fileInput = createElement("input", { display: "none" }, {
                    type: "file",
                    multiple: "true",
                    accept: "image/*",
                });
                root.appendChild(fileInput);

                // --- Helper Functions ---
                function syncValue() {
                    const jsonStr = JSON.stringify(items.map(it => ({
                        filename: it.filename,
                        subfolder: it.subfolder || "",
                        type: it.type || "input",
                    })));
                    if (widget) {
                        widget.value = jsonStr;
                    }
                    node.onWidgetChanged?.(inputName, jsonStr, null, widget);
                    node.graph?.incrementVersion?.();
                    node.setDirtyCanvas?.(true, true);
                    ensureFits();
                }

                function ensureFits() {
                    requestAnimationFrame(() => {
                        const computed = node.computeSize?.();
                        if (computed && node.size) {
                            const newHeight = Math.max(node.size[1], computed[1]);
                            node.setSize?.([node.size[0], newHeight]);
                        }
                        node.setDirtyCanvas?.(true, true);
                    });
                }

                async function uploadFiles(files) {
                    if (!files || files.length === 0) return;
                    for (const file of Array.from(files)) {
                        try {
                            const formData = new FormData();
                            formData.append("image", file);
                            formData.append("subfolder", "ref_images");
                            formData.append("overwrite", "true");

                            const resp = await fetch("/upload/image", {
                                method: "POST",
                                body: formData,
                            });
                            if (resp.ok) {
                                const data = await resp.json();
                                items.push({
                                    filename: data.name,
                                    subfolder: data.subfolder || "ref_images",
                                    type: data.type || "input",
                                    selected: false,
                                });
                            }
                        } catch (err) {
                            console.error("[DKST Ref] Upload error:", err);
                        }
                    }
                    render();
                    syncValue();
                }

                fileInput.addEventListener("change", (e) => {
                    uploadFiles(e.target.files);
                    fileInput.value = "";
                });

                // Drag & Drop
                root.addEventListener("dragover", (e) => {
                    e.preventDefault();
                    root.style.borderColor = "#4a9eff";
                });
                root.addEventListener("dragleave", (e) => {
                    e.preventDefault();
                    root.style.borderColor = "rgba(255, 255, 255, 0.12)";
                });
                root.addEventListener("drop", (e) => {
                    e.preventDefault();
                    root.style.borderColor = "rgba(255, 255, 255, 0.12)";
                    if (e.dataTransfer?.files?.length) {
                        uploadFiles(e.dataTransfer.files);
                    }
                });

                // --- UI Controls Bar ---
                const toolbar = createElement("div", {
                    display: "flex",
                    flexWrap: "wrap",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: "6px",
                    paddingBottom: "4px",
                    borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
                });

                const leftBtns = createElement("div", { display: "flex", alignItems: "center", gap: "6px" });
                const addBtn = createElement("button", {
                    background: "#2563eb",
                    color: "#fff",
                    border: "none",
                    borderRadius: "4px",
                    padding: "4px 8px",
                    fontSize: "11px",
                    fontWeight: "600",
                    cursor: "pointer",
                });
                addBtn.textContent = "+ Add Images";
                addBtn.type = "button";
                addBtn.addEventListener("click", () => fileInput.click());

                const removeBtn = createElement("button", {
                    background: "rgba(239, 68, 68, 0.2)",
                    color: "#f87171",
                    border: "1px solid rgba(239, 68, 68, 0.4)",
                    borderRadius: "4px",
                    padding: "4px 8px",
                    fontSize: "11px",
                    cursor: "pointer",
                });
                removeBtn.textContent = "Remove Selected";
                removeBtn.type = "button";
                removeBtn.addEventListener("click", () => {
                    const before = items.length;
                    items = items.filter(it => !it.selected);
                    if (items.length !== before) {
                        render();
                        syncValue();
                    }
                });

                const clearBtn = createElement("button", {
                    background: "transparent",
                    color: "#9ca3af",
                    border: "1px solid rgba(255, 255, 255, 0.15)",
                    borderRadius: "4px",
                    padding: "4px 6px",
                    fontSize: "11px",
                    cursor: "pointer",
                });
                clearBtn.textContent = "Clear";
                clearBtn.type = "button";
                clearBtn.addEventListener("click", () => {
                    if (items.length > 0 && confirm("Clear all reference images?")) {
                        items = [];
                        render();
                        syncValue();
                    }
                });

                leftBtns.append(addBtn, removeBtn, clearBtn);

                // Grid Size Control
                const sizeControl = createElement("div", { display: "flex", alignItems: "center", gap: "4px" });
                const sizeLabel = createElement("span", { fontSize: "11px", color: "#9ca3af" });
                sizeLabel.textContent = "Size:";
                const sizeSlider = createElement("input", {
                    width: "60px",
                    cursor: "pointer",
                    accentColor: "#3b82f6",
                }, {
                    type: "range",
                    min: "50",
                    max: "130",
                    step: "5",
                    value: String(thumbSize),
                });
                sizeSlider.addEventListener("input", (e) => {
                    thumbSize = Number(e.target.value);
                    updateGridSizing();
                    ensureFits();
                });
                sizeControl.append(sizeLabel, sizeSlider);

                toolbar.append(leftBtns, sizeControl);
                root.appendChild(toolbar);

                // --- Grid Container ---
                const gridContainer = createElement("div", {
                    display: "grid",
                    gridTemplateColumns: `repeat(auto-fill, minmax(${thumbSize}px, 1fr))`,
                    gap: "6px",
                    maxHeight: "320px",
                    overflowY: "auto",
                    padding: "4px",
                });
                root.appendChild(gridContainer);

                function updateGridSizing() {
                    gridContainer.style.gridTemplateColumns = `repeat(auto-fill, minmax(${thumbSize}px, 1fr))`;
                    gridContainer.querySelectorAll(".dkst-ref-card").forEach(card => {
                        card.style.height = `${thumbSize}px`;
                    });
                }

                function render() {
                    gridContainer.replaceChildren();

                    if (items.length === 0) {
                        const emptyNotice = createElement("div", {
                            gridColumn: "1 / -1",
                            padding: "24px 8px",
                            textAlign: "center",
                            color: "#6b7280",
                            fontSize: "12px",
                            border: "1px dashed rgba(255, 255, 255, 0.15)",
                            borderRadius: "6px",
                        });
                        emptyNotice.innerHTML = "📸 Drop images here or click <b>+ Add Images</b>";
                        gridContainer.appendChild(emptyNotice);
                        return;
                    }

                    items.forEach((item, index) => {
                        const card = createElement("div", {
                            position: "relative",
                            width: "100%",
                            height: `${thumbSize}px`,
                            borderRadius: "6px",
                            overflow: "hidden",
                            border: item.selected ? "2px solid #3b82f6" : "1px solid rgba(255, 255, 255, 0.15)",
                            boxSizing: "border-box",
                            cursor: "pointer",
                            background: "#18181b",
                        });
                        card.className = "dkst-ref-card";

                        // Thumbnail image
                        const imgUrl = `/view?filename=${encodeURIComponent(item.filename)}&subfolder=${encodeURIComponent(item.subfolder || "")}&type=${item.type || "input"}`;
                        const img = createElement("img", {
                            width: "100%",
                            height: "100%",
                            objectFit: "cover",
                            display: "block",
                            userSelect: "none",
                            pointerEvents: "none",
                        }, {
                            src: imgUrl,
                            loading: "lazy",
                        });

                        // Select overlay/checkbox
                        const checkIndicator = createElement("div", {
                            position: "absolute",
                            top: "3px",
                            left: "3px",
                            width: "14px",
                            height: "14px",
                            borderRadius: "3px",
                            border: "1px solid rgba(255, 255, 255, 0.8)",
                            background: item.selected ? "#3b82f6" : "rgba(0, 0, 0, 0.4)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: "10px",
                            color: "#fff",
                            fontWeight: "bold",
                            pointerEvents: "none",
                        });
                        if (item.selected) {
                            checkIndicator.textContent = "✓";
                        }

                        // Delete 'x' button
                        const delBtn = createElement("button", {
                            position: "absolute",
                            top: "3px",
                            right: "3px",
                            width: "16px",
                            height: "16px",
                            borderRadius: "50%",
                            background: "rgba(0, 0, 0, 0.65)",
                            color: "#ef4444",
                            border: "none",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: "11px",
                            padding: "0",
                            lineHeight: "1",
                        }, { title: "Remove this image" });
                        delBtn.textContent = "×";
                        delBtn.type = "button";
                        delBtn.addEventListener("click", (e) => {
                            e.stopPropagation();
                            items.splice(index, 1);
                            render();
                            syncValue();
                        });

                        // Card click -> toggle select
                        card.addEventListener("click", () => {
                            item.selected = !item.selected;
                            render();
                        });

                        card.append(img, checkIndicator, delBtn);
                        gridContainer.appendChild(card);
                    });
                }

                // Initial render
                render();

                widget = node.addDOMWidget(inputName, "DKST_IMAGE_REF_GRID", root, {
                    getValue() {
                        return JSON.stringify(items.map(it => ({
                            filename: it.filename,
                            subfolder: it.subfolder || "",
                            type: it.type || "input",
                        })));
                    },
                    setValue(val) {
                        items = parseItems(val);
                        render();
                        ensureFits();
                    },
                });

                return { widget };
            },
        };
    },
});
