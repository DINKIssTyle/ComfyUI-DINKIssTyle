import { app } from "../../scripts/app.js";

const SLIDER_MIN = -3;
const SLIDER_MAX = 3;

function sliderValue(strength) {
    return String(Math.max(SLIDER_MIN, Math.min(SLIDER_MAX, strength)));
}

function element(tag, styles = {}) {
    const item = document.createElement(tag);
    Object.assign(item.style, styles);
    return item;
}

function decodeRows(value) {
    try {
        const parsed = JSON.parse(value);
        if (!Array.isArray(parsed)) return [];
        return parsed.filter(row => row && typeof row === "object" && !Array.isArray(row))
            .map(row => ({
                name: typeof row.name === "string" ? row.name : "None",
                enabled: row.enabled !== false,
                strength_model: Number.isFinite(Number(row.strength_model))
                    ? Number(row.strength_model) : 1,
            }));
    } catch {
        return [];
    }
}

app.registerExtension({
    name: "DINKI.MultiLoRALoader",
    getCustomWidgets() {
        return {
            DKST_LORA_STACK(node, inputName, inputData) {
                const names = inputData?.[1]?.lora_names || ["None"];
                const options = [...new Set(["None", ...names])];
                let rows = [{ name: "None", enabled: true, strength_model: 1 }];
                let widget;
                const root = element("div", {
                    boxSizing: "border-box", width: "100%", padding: "4px 8px",
                    display: "flex", flexDirection: "column", gap: "6px",
                    color: "var(--fg-color, #ddd)", fontSize: "12px",
                });

                const controlStyle = {
                    boxSizing: "border-box", minHeight: "25px", borderRadius: "5px",
                    border: "1px solid var(--border-color, #555)",
                    color: "var(--fg-color, #ddd)", background: "var(--comfy-input-bg, #333)",
                };

                function ensureFits() {
                    requestAnimationFrame(() => {
                        const size = node.computeSize?.();
                        if (size && node.size && node.size[1] < size[1]) {
                            node.setSize?.([node.size[0], size[1]]);
                        }
                        node.setDirtyCanvas?.(true, true);
                    });
                }

                function changed(needsLayout = false) {
                    widget?.callback?.(widget.value);
                    node.setDirtyCanvas?.(true, true);
                    if (needsLayout) ensureFits();
                }

                function render() {
                    root.replaceChildren();
                    rows.forEach((row, index) => {
                        const group = element("div", {
                            border: "1px solid var(--border-color, #555)", borderRadius: "6px",
                            padding: "5px", display: "flex", flexDirection: "column", gap: "5px",
                        });
                        const top = element("div", { display: "flex", gap: "5px" });
                        const remove = element("button", { ...controlStyle, cursor: "pointer", flex: "0 0 27px" });
                        remove.type = "button";
                        remove.textContent = "×";
                        remove.title = "Remove LoRA";
                        remove.setAttribute("aria-label", `Remove LoRA ${index + 1}`);
                        remove.addEventListener("click", () => {
                            rows.splice(index, 1);
                            render();
                            changed(true);
                        });
                        const select = element("select", { ...controlStyle, width: "100%", minWidth: "0" });
                        select.setAttribute("aria-label", `LoRA ${index + 1}`);
                        for (const name of [...new Set([...options, row.name])]) {
                            const option = document.createElement("option");
                            option.value = name;
                            option.textContent = name;
                            select.appendChild(option);
                        }
                        select.value = row.name;
                        select.addEventListener("change", () => {
                            row.name = select.value;
                            changed();
                        });
                        top.append(remove, select);

                        const bottom = element("div", { display: "flex", gap: "7px", alignItems: "center" });
                        const toggleLabel = element("label", { display: "flex", alignItems: "center", gap: "3px", flex: "0 0 auto" });
                        const toggle = document.createElement("input");
                        toggle.type = "checkbox";
                        toggle.checked = row.enabled;
                        toggle.addEventListener("change", () => {
                            row.enabled = toggle.checked;
                            changed();
                        });
                        toggleLabel.append(toggle, document.createTextNode("On"));
                        const slider = element("input", { flex: "1 1 auto", minWidth: "0", cursor: "pointer" });
                        slider.type = "range";
                        slider.min = String(SLIDER_MIN);
                        slider.max = String(SLIDER_MAX);
                        slider.step = "0.01";
                        slider.value = sliderValue(row.strength_model);
                        slider.title = "strength_model slider (-3 to 3)";
                        slider.setAttribute("aria-label", `LoRA ${index + 1} strength_model slider`);
                        slider.addEventListener("input", () => {
                            row.strength_model = Math.max(SLIDER_MIN, Math.min(SLIDER_MAX, Number(slider.value)));
                            strength.value = String(row.strength_model);
                            changed();
                        });
                        const strength = element("input", { ...controlStyle, width: "75px", padding: "0 4px" });
                        strength.type = "number";
                        strength.min = "-100";
                        strength.max = "100";
                        strength.step = "0.01";
                        strength.value = String(row.strength_model);
                        strength.title = "strength_model";
                        strength.setAttribute("aria-label", `LoRA ${index + 1} strength_model number`);
                        strength.addEventListener("input", () => {
                            if (strength.value === "") return;
                            const value = Number(strength.value);
                            if (!Number.isFinite(value) || value < -100 || value > 100) return;
                            row.strength_model = value;
                            slider.value = sliderValue(value);
                            changed();
                        });
                        strength.addEventListener("change", () => {
                            const value = Number(strength.value);
                            if (strength.value === "" || !Number.isFinite(value) || value < -100 || value > 100) {
                                strength.value = String(row.strength_model);
                                return;
                            }
                            row.strength_model = value;
                            slider.value = sliderValue(value);
                            changed();
                        });
                        bottom.append(toggleLabel, slider, strength);
                        group.append(top, bottom);
                        root.appendChild(group);
                    });

                    const add = element("button", { ...controlStyle, cursor: "pointer", width: "100%" });
                    add.type = "button";
                    add.textContent = "+ Add LoRA";
                    add.addEventListener("click", () => {
                        rows.push({ name: "None", enabled: true, strength_model: 1 });
                        render();
                        changed(true);
                    });
                    root.appendChild(add);
                }

                widget = node.addDOMWidget(inputName, "dkst-lora-stack", root, {
                    getValue: () => JSON.stringify(rows),
                    setValue: value => {
                        rows = decodeRows(value);
                        render();
                        ensureFits();
                    },
                    getMinHeight: () => 43 + rows.length * 77,
                    getHeight: () => `${43 + rows.length * 77}px`,
                    margin: 4,
                });
                render();
                ensureFits();
                return { widget };
            },
        };
    },
});
