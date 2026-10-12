import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const PREFIX = "DKST.Monitor.";
const METRICS = [
    { key: "cpu", id: "CPU", label: "CPU" },
    { key: "ram", id: "RAM", label: "RAM" },
    { key: "gpu", id: "GPU", label: "GPU" },
    { key: "vram", id: "VRAM", label: "VRAM" },
    { key: "temperature", id: "Temperature", label: "GPU Temp" },
];
const valid = value => typeof value === "number" && Number.isFinite(value) && value >= 0;
const percent = value => valid(value) ? `${Math.round(value)}%` : "—";
const usage = value => valid(value) ? Math.max(0, Math.min(100, value)) : null;
const memoryUsage = (used, total) => valid(used) && valid(total) && total > 0
    ? usage(100 * used / total) : null;
const usageColor = value => value >= 85 ? "#ed6262"
    : value >= 70 ? "#ef9b43" : value >= 50 ? "#e8ca46" : "#49c777";
function memoryText(used, total, divisor) {
    return valid(used) && valid(total) && total > 0
        ? `${(used / divisor).toFixed(1)}/${(total / divisor).toFixed(1)} GiB` : "—";
}

export function monitorValues(data, index, ramPercent = false, vramPercent = false) {
    const gpu = data?.gpus?.find(item => item.index === index);
    const memory = (used, total, divisor, asPercent) => asPercent
        ? (valid(used) && valid(total) && total > 0 ? percent(100 * used / total) : "—")
        : memoryText(used, total, divisor);
    return {
        cpu: percent(data?.cpu_percent),
        ram: memory(data?.ram?.used_bytes, data?.ram?.total_bytes, 1024 ** 3, ramPercent),
        gpu: percent(gpu?.utilization),
        temperature: valid(gpu?.temperature) ? `${Math.round(gpu.temperature)}°C` : "—",
        vram: memory(gpu?.memory_used_mib, gpu?.memory_total_mib, 1024, vramPercent),
        name: gpu?.name ?? `GPU ${index} unavailable`,
    };
}

let monitor;
const changed = () => monitor?.restart();

function readSetting(key, fallback) {
    return app.extensionManager?.setting?.get(PREFIX + key)
        ?? app.ui?.settings?.getSettingValue(PREFIX + key, fallback) ?? fallback;
}

export function metricLayout(value) {
    const legacy = METRICS.map((metric, index) => {
        const order = Number(readSetting("Order" + metric.id, index + 1));
        return { ...metric, index, order: Number.isFinite(order) ? order : index + 1 };
    }).sort((a, b) => a.order - b.order || a.index - b.index);
    const result = [];
    for (const item of Array.isArray(value) ? value : []) {
        if (!METRICS.some(metric => metric.id === item?.id) || result.some(row => row.id === item.id)) continue;
        result.push({id: item.id, visible: item.visible !== false});
    }
    for (const metric of legacy) {
        if (!result.some(row => row.id === metric.id)) result.push({id: metric.id, visible: Boolean(readSetting("Show" + metric.id, true))});
    }
    return result;
}

export function renderMetricList(name, setter, value) {
    const root = document.createElement("div");
    root.className = "dkst-monitor-editor";
    root.style.cssText = "width:100%;min-width:220px;display:grid;gap:8px";
    const hint = document.createElement("div");
    hint.textContent = "Drag to reorder · Check to show";
    hint.style.cssText = "font-size:12px;opacity:.7";
    const list = document.createElement("div");
    list.setAttribute("aria-label", name);
    list.setAttribute("role", "list");
    list.style.cssText = "display:grid;gap:6px";
    const state = metricLayout(value);
    const commit = () => setter(Array.from(list.children, row => ({
        id: row.dataset.metric, visible: row.querySelector("input").checked,
    })));
    for (const item of state) {
        const metric = METRICS.find(metric => metric.id === item.id);
        const row = document.createElement("div");
        row.dataset.metric = item.id;
        row.setAttribute("role", "listitem");
        row.style.cssText = "display:flex;align-items:center;gap:12px;padding:8px 12px;border:1px solid var(--border-color,#555);border-radius:6px;background:var(--comfy-input-bg,#292929)";
        const handle = document.createElement("button");
        handle.type = "button";
        handle.textContent = "⠿";
        handle.title = "Drag to reorder. Arrow keys also move this item.";
        handle.setAttribute("aria-label", "Reorder " + metric.label);
        handle.style.cssText = "cursor:grab;touch-action:none;color:inherit;background:none;border:0;font-size:20px;padding:0 4px";
        let dragging = false;
        handle.addEventListener("pointerdown", event => {
            if (event.button !== 0) return;
            event.preventDefault();
            dragging = true;
            handle.setPointerCapture(event.pointerId);
            row.style.opacity = ".6";
        });
        handle.addEventListener("pointermove", event => {
            if (!dragging) return;
            const target = document.elementFromPoint(event.clientX, event.clientY)?.closest("[data-metric]");
            if (!target || target === row || target.parentElement !== list) return;
            const rect = target.getBoundingClientRect();
            list.insertBefore(row, event.clientY < rect.top + rect.height / 2 ? target : target.nextSibling);
            handle.setPointerCapture(event.pointerId);
        });
        const finish = event => {
            if (!dragging) return;
            dragging = false;
            row.style.opacity = "";
            if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
            commit();
        };
        handle.addEventListener("pointerup", finish);
        handle.addEventListener("pointercancel", finish);
        handle.addEventListener("keydown", event => {
            if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
            event.preventDefault();
            const target = event.key === "ArrowUp" ? row.previousElementSibling : row.nextElementSibling;
            if (!target) return;
            list.insertBefore(row, event.key === "ArrowUp" ? target : target.nextSibling);
            commit();
        });
        const label = document.createElement("label");
        label.style.cssText = "display:flex;align-items:center;gap:10px;flex:1;cursor:pointer";
        const check = document.createElement("input");
        check.type = "checkbox";
        check.checked = item.visible;
        check.setAttribute("aria-label", "Show " + metric.label);
        check.addEventListener("change", commit);
        const caption = document.createElement("span");
        caption.textContent = metric.label;
        label.append(check, caption);
        row.append(handle, label);
        list.append(row);
    }
    root.append(hint, list);
    return root;
}

app.registerExtension({
    name: "DINKI.SystemMonitor",
    settings: [
        { id: PREFIX + "Enabled", name: "Show system monitor", type: "boolean", defaultValue: true, onChange: changed },
        { id: PREFIX + "Placement", name: "Monitor placement", type: "combo", defaultValue: "Floating",
            options: ["Toolbar", "Floating"], onChange: changed },
        { id: PREFIX + "Position", name: "Floating monitor position", type: "hidden", defaultValue: "" },
        { id: PREFIX + "RAMPercent", name: "Show RAM as percentage", type: "boolean", defaultValue: true, onChange: changed },
        { id: PREFIX + "VRAMPercent", name: "Show VRAM as percentage", type: "boolean", defaultValue: true, onChange: changed },
        { id: PREFIX + "Graph", name: "Show usage bars", type: "boolean", defaultValue: false, onChange: changed },
        { id: PREFIX + "Color", name: "Color by usage", type: "boolean", defaultValue: false, onChange: changed },
        { id: PREFIX + "Interval", name: "Refresh interval (seconds)", type: "slider", defaultValue: 2,
            attrs: { min: 1, max: 30, step: 1 }, onChange: changed },
        { id: PREFIX + "GPU", name: "NVIDIA GPU index", type: "number", defaultValue: 0,
            attrs: { min: 0, maxFractionDigits: 0 },
            tooltip: "Physical GPU index shown by nvidia-smi, independent of CUDA_VISIBLE_DEVICES.", onChange: changed },
        { id: PREFIX + "Layout", name: "Monitor items", type: renderMetricList,
            defaultValue: null, onChange: changed },
        ...METRICS.flatMap((metric, index) => [
            { id: PREFIX + "Show" + metric.id, name: "Show " + metric.label,
                type: "hidden", defaultValue: true, onChange: changed },
            { id: PREFIX + "Order" + metric.id, name: metric.label + " display order",
                type: "hidden", defaultValue: index + 1,
                onChange: changed },
        ]),
    ],
    setup() {
        monitor?.destroy();
        monitor = new SystemMonitor();
        monitor.restart();
    },
});

export class SystemMonitor {
    constructor() {
        this.generation = 0;
        this.element = document.createElement("div");
        this.element.id = "dkst-system-monitor";
        this.element.setAttribute("role", "group");
        this.element.setAttribute("aria-label", "DKST server system monitor");
        this.handle = document.createElement("button");
        this.handle.textContent = "⠿";
        this.handle.title = "Drag to move monitor";
        this.handle.setAttribute("aria-label", "Drag to move monitor");
        this.handle.className = "dkst-monitor-handle";
        this.handle.addEventListener("pointerdown", event => this.startDrag(event));
        this.handle.addEventListener("pointermove", event => this.moveDrag(event));
        this.handle.addEventListener("pointerup", event => this.endDrag(event));
        this.handle.addEventListener("pointercancel", event => this.endDrag(event));
        this.dock = document.createElement("button");
        this.dock.textContent = "↗";
        this.dock.addEventListener("click", () => void this.save("Placement", this.setting("Placement", "Floating") === "Floating" ? "Toolbar" : "Floating"));
        this.cells = {};
        this.items = {};
        this.bars = {};
        for (const {key, label} of METRICS) {
            const cell = document.createElement("span");
            const line = document.createElement("span");
            line.className = "dkst-monitor-line";
            const caption = document.createElement("span");
            caption.className = "dkst-monitor-label";
            caption.textContent = label;
            const value = document.createElement("span");
            value.className = "dkst-monitor-value";
            value.dataset.metric = key;
            value.textContent = "—";
            line.append(caption, value);
            cell.append(line);
            if (key !== "temperature") {
                const track = document.createElement("span");
                track.className = "dkst-monitor-track";
                const fill = document.createElement("span");
                fill.className = "dkst-monitor-fill";
                track.append(fill);
                cell.append(track);
                this.bars[key] = fill;
            }
            this.element.append(cell);
            this.cells[key] = value;
            this.items[key] = cell;
        }
        if (!document.getElementById("dkst-monitor-style")) {
            const style = document.createElement("style");
            style.id = "dkst-monitor-style";
            style.textContent = `
                #dkst-system-monitor { display:flex; align-items:center; gap:12px; padding:4px 8px;
                    min-width:0; max-width:min(620px,60vw); overflow-x:auto; flex:0 1 auto;
                    color:var(--fg-color,#ddd); font:11px/1.5 sans-serif; font-variant-numeric:tabular-nums; }
                #dkst-system-monitor > span { display:flex; flex:none; flex-direction:column; gap:2px; white-space:nowrap; }
                #dkst-system-monitor .dkst-monitor-line { display:flex; gap:4px; align-items:center; }
                #dkst-system-monitor .dkst-monitor-label { opacity:.6; }
                #dkst-system-monitor .dkst-monitor-track { height:4px; border-radius:999px;
                    overflow:hidden; background:color-mix(in srgb, currentColor 18%, transparent); }
                #dkst-system-monitor[data-graph="false"] .dkst-monitor-track { display:none; }
                #dkst-system-monitor .dkst-monitor-fill { display:block; height:100%; width:0;
                    border-radius:inherit; background:var(--dkst-usage-color,var(--p-primary-color,#91b9ef)); }
                #dkst-system-monitor .dkst-monitor-value { display:inline-block; flex:none; width:5ch;
                    font-family:ui-monospace,Consolas,monospace; text-align:right; overflow:hidden; text-overflow:ellipsis; }
                #dkst-system-monitor .dkst-monitor-value[data-metric="temperature"] { width:6ch; }
                #dkst-system-monitor .dkst-monitor-value[data-format="capacity"] { width:19ch; }
                #dkst-system-monitor[data-offline="true"] { opacity:.55; }
                #dkst-system-monitor button { flex:none; border:0; border-radius:4px; background:transparent;
                    color:inherit; padding:2px 4px; cursor:pointer; font:inherit; }
                #dkst-system-monitor button:hover { background:var(--comfy-input-bg,#444); }
                #dkst-system-monitor .dkst-monitor-handle { cursor:grab; touch-action:none; }
                #dkst-system-monitor[data-floating="true"] { position:fixed; z-index:1100;
                    max-width:calc(100vw - 16px); box-sizing:border-box; border-radius:8px;
                    background:var(--comfy-menu-bg,#252527); border:1px solid var(--border-color,#555);
                    box-shadow:0 4px 16px #0005; }
                @media(max-width:900px) { #dkst-system-monitor { gap:7px; max-width:45vw; font-size:10px; } }
            `;
            document.head.append(style);
        }
        this.visibility = () => this.restart();
        document.addEventListener("visibilitychange", this.visibility);
        this.unload = () => this.destroy();
        window.addEventListener("pagehide", this.unload, { once: true });
        this.resize = () => { if (this.element.dataset.floating === "true") this.position(); };
        window.addEventListener("resize", this.resize);
    }

    setting(key, fallback) {
        return readSetting(key, fallback);
    }

    mount() {
        if (this.drag) return;
        const floating = this.setting("Placement", "Floating") === "Floating";
        this.element.dataset.floating = String(floating);
        this.dock.textContent = floating ? "↥" : "↗";
        this.dock.title = floating ? "Dock in top toolbar" : "Float monitor";
        this.dock.setAttribute("aria-label", this.dock.title);
        if (floating) {
            if (this.element.parentElement !== document.body) document.body.append(this.element);
            this.position();
            return;
        }
        this.element.style.left = "";
        this.element.style.top = "";
        // Current Vue topbar, then the compatibility menu used by older frontends.
        const actions = document.querySelector('[data-testid="action-bar-buttons"]');
        const host = actions?.parentElement ?? app.menu?.element;
        if (host?.isConnected && this.element.parentElement !== host) host.append(this.element);
    }

    async save(key, value, refresh = true) {
        try {
            if (app.extensionManager?.setting?.set) await app.extensionManager.setting.set(PREFIX + key, value);
            else await app.ui.settings.setSettingValue(PREFIX + key, value);
        } catch (error) {
            console.warn("DKST monitor setting could not be saved", error);
            this.element.title = "Monitor setting could not be saved. Please try again.";
        }
        if (refresh) this.restart();
    }

    position(point) {
        if (!point) {
            try { point = JSON.parse(this.setting("Position", "")); } catch { /* Default position. */ }
        }
        const rect = this.element.getBoundingClientRect();
        const x = Math.max(8, Math.min(Number.isFinite(point?.x) ? point.x : 24, window.innerWidth - rect.width - 8));
        const y = Math.max(8, Math.min(Number.isFinite(point?.y) ? point.y : 64, window.innerHeight - rect.height - 8));
        this.element.style.left = `${x}px`;
        this.element.style.top = `${y}px`;
        return {x, y};
    }

    startDrag(event) {
        if (event.button !== 0) return;
        event.preventDefault();
        const rect = this.element.getBoundingClientRect();
        this.drag = { id: event.pointerId, x: event.clientX, y: event.clientY,
            offsetX: event.clientX - rect.left, offsetY: event.clientY - rect.top, moved: false };
        this.handle.setPointerCapture(event.pointerId);
    }

    moveDrag(event) {
        if (!this.drag || this.drag.id !== event.pointerId) return;
        if (!this.drag.moved && Math.hypot(event.clientX - this.drag.x, event.clientY - this.drag.y) < 4) return;
        this.drag.moved = true;
        this.element.dataset.floating = "true";
        if (this.element.parentElement !== document.body) {
            document.body.append(this.element);
            this.handle.setPointerCapture(event.pointerId);
        }
        this.drag.point = this.position({x: event.clientX - this.drag.offsetX, y: event.clientY - this.drag.offsetY});
    }

    async endDrag(event) {
        if (!this.drag || this.drag.id !== event.pointerId) return;
        const drag = this.drag;
        this.drag = null;
        if (this.handle.hasPointerCapture(event.pointerId)) this.handle.releasePointerCapture(event.pointerId);
        if (!drag.moved) return;
        await this.save("Position", JSON.stringify(drag.point), false);
        await this.save("Placement", "Floating");
    }

    layout() {
        this.updateValueWidths();
        this.element.dataset.graph = String(Boolean(this.setting("Graph", false)));
        const visible = metricLayout(this.setting("Layout", null)).filter(item => item.visible)
            .map(item => METRICS.find(metric => metric.id === item.id));
        this.element.replaceChildren(this.handle, ...visible.map(metric => this.items[metric.key]), this.dock);
        return visible.length > 0;
    }

    restart() {
        this.generation++;
        clearTimeout(this.timer);
        this.controller?.abort();
        const hasVisibleMetrics = this.layout();
        if (!this.setting("Enabled", true) || !hasVisibleMetrics) {
            this.element.remove();
            return;
        }
        this.mount();
        if (!document.hidden) void this.poll(this.generation);
    }

    render(data, error) {
        this.updateValueWidths();
        const index = Math.max(0, Math.floor(Number(this.setting("GPU", 0)) || 0));
        const values = monitorValues(data, index, this.setting("RAMPercent", true), this.setting("VRAMPercent", true));
        for (const [key, cell] of Object.entries(this.cells)) cell.textContent = values[key];
        const gpu = data?.gpus?.find(item => item.index === index);
        const levels = {
            cpu: usage(data?.cpu_percent),
            ram: memoryUsage(data?.ram?.used_bytes, data?.ram?.total_bytes),
            gpu: usage(gpu?.utilization),
            vram: memoryUsage(gpu?.memory_used_mib, gpu?.memory_total_mib),
            temperature: usage(gpu?.temperature),
        };
        const colored = Boolean(this.setting("Color", false));
        for (const [key, cell] of Object.entries(this.cells)) {
            const level = levels[key];
            const fill = this.bars[key];
            if (fill) fill.style.width = `${level ?? 0}%`;
            const color = colored && level !== null ? usageColor(level) : "";
            cell.style.color = color;
            if (fill) fill.style.backgroundColor = color;
        }
        this.element.dataset.offline = String(Boolean(error));
        const devices = (data?.gpus ?? []).map(gpu => `GPU ${gpu.index}: ${gpu.name}`).join("\n");
        this.element.title = ["ComfyUI server • " + values.name, devices, error, ...(data?.errors ?? [])].filter(Boolean).join("\n");
    }

    updateValueWidths() {
        for (const [key, setting] of [["ram", "RAMPercent"], ["vram", "VRAMPercent"]]) {
            this.cells[key].dataset.format = this.setting(setting, true) ? "percent" : "capacity";
        }
    }

    async poll(generation) {
        this.mount();
        const controller = new AbortController();
        this.controller = controller;
        const timeout = setTimeout(() => controller.abort(), 6000);
        try {
            const response = await api.fetchApi("/dinki/monitor", { signal: controller.signal, cache: "no-store" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (generation === this.generation) this.render(data);
        } catch (error) {
            if (generation === this.generation) this.render(null, "Monitor unavailable — retrying");
        } finally {
            clearTimeout(timeout);
            if (generation === this.generation && !document.hidden && this.setting("Enabled", true)) {
                const seconds = Math.min(30, Math.max(1, Number(this.setting("Interval", 2)) || 2));
                this.timer = setTimeout(() => void this.poll(generation), seconds * 1000);
            }
        }
    }

    destroy() {
        this.generation++;
        clearTimeout(this.timer);
        this.controller?.abort();
        document.removeEventListener("visibilitychange", this.visibility);
        window.removeEventListener("pagehide", this.unload);
        window.removeEventListener("resize", this.resize);
        this.element.remove();
    }
}
