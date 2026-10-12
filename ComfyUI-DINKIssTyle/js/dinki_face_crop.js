import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { bindScrollWheel, createScrollControls } from "./dinki_scroll.js";

const ACTIVE = new Set(["queued", "processing", "cancel_requested"]);
const endpoint = "/dinki/face_crop";

function element(tag, styles = {}, text = "") {
    const el = document.createElement(tag);
    Object.assign(el.style, styles);
    el.textContent = text;
    return el;
}

async function request(path, data) {
    const response = await api.fetchApi(endpoint + path, data === undefined ? undefined : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Face export request failed (${response.status}).`);
    return result;
}

app.registerExtension({
    name: "DINKI.FaceCropExport",
    nodeCreated(node) {
        if (node.comfyClass !== "DINKI_Face_Crop_Export") return;
        const jobWidget = node.widgets.find(w => w.name === "result_job_id");
        if (!jobWidget) return;
        jobWidget.type = "converted-widget";
        jobWidget.computeSize = () => [0, -4];
        jobWidget.serializeValue = () => jobWidget.value;
        const value = name => node.widgets.find(w => w.name === name)?.value;
        const root = element("div", {
            boxSizing: "border-box", padding: "8px", width: "100%", height: "100%", minWidth: "0", overflow: "hidden",
            background: "#19191d", color: "#eee", borderRadius: "8px",
            display: "flex", flexDirection: "column", gap: "8px", font: "12px sans-serif",
        });
        const toolbar = element("div", { display: "flex", flexWrap: "wrap", gap: "6px", flexShrink: "0" });
        const button = label => {
            const b = element("button", { cursor: "pointer", padding: "5px 9px" }, label);
            b.type = "button"; toolbar.append(b); return b;
        };
        const execute = button("Execute");
        const cancel = button("Cancel");
        const refresh = button("Refresh Results");
        const manifest = element("a", { color: "#9cc6ff", alignSelf: "center" }, "Download Manifest");
        manifest.target = "_blank"; manifest.rel = "noopener"; manifest.style.display = "none";
        toolbar.append(manifest);
        const status = element("div", { whiteSpace: "pre-wrap", overflowWrap: "anywhere", userSelect: "text", maxHeight: "100px", overflowY: "auto", flexShrink: "0" },
            "Add photos, set options, then Execute. Queue/Run outputs the most recent export.");
        bindScrollWheel(status);
        const viewport = element("div", { height: "260px", minHeight: "220px", flex: "1 1 auto", overflowY: "scroll", overflowX: "hidden", scrollbarGutter: "stable" });
        const gallery = element("div", { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: "8px" });
        viewport.append(gallery);
        const scroll = createScrollControls(viewport, "y", "Face crop results");
        scroll.element.style.flexShrink = "0";
        root.append(toolbar, status, viewport, scroll.element);
        for (const name of ["pointerdown", "mousedown", "click"]) root.addEventListener(name, event => event.stopPropagation());
        const widget = node.addDOMWidget("face_crop_export", "DKST_FACE_CROP_EXPORT", root, {
            serialize: false, hideOnZoom: false, getMinHeight: () => 460, getHeight: () => 520, getMaxHeight: () => 2000,
        });
        widget.serialize = false;
        node.setSize?.([Math.max(520, node.size?.[0] || 520), Math.max(1200, node.size?.[1] || 1200)]);
        let state = null, busy = false, polling = false, disposed = false, renderedJob = "";
        const rendered = new Set();
        const controls = () => {
            const matching = state?.id === jobWidget.value;
            execute.disabled = busy || (matching && ACTIVE.has(state.status));
            cancel.disabled = busy || !matching || !ACTIVE.has(state.status) || state.status === "cancel_requested";
            refresh.disabled = busy || !jobWidget.value;
        };
        function setJob(id) {
            jobWidget.value = id;
            node.onWidgetChanged?.("result_job_id", id, null, jobWidget);
            node.graph?.incrementVersion?.();
            node.setDirtyCanvas?.(true, true);
        }
        function render(next) {
            if (disposed || next.id !== jobWidget.value) return;
            state = next;
            status.textContent = `${next.status} · ${next.processed}/${next.total} images/cells\n${next.results.length} crops · ${next.skipped.length} skipped\n${next.output_dir}`;
            if (next.error) status.textContent += `\n${next.error}`;
            if (next.current_file) status.textContent += `\nProcessing: ${next.current_file}`;
            if (next.skipped.length) {
                const reasons = [...new Set(next.skipped.map(s => s.reason))];
                status.textContent += `\nSkipped: ${reasons.slice(0, 3).join("; ")}. Details in manifest.`;
            }
            manifest.href = api.apiURL(`${endpoint}/manifest?id=${encodeURIComponent(next.id)}`);
            manifest.style.display = "inline";
            if (renderedJob !== next.id) {
                gallery.replaceChildren(); rendered.clear(); renderedJob = next.id; viewport.scrollTop = 0;
            }
            for (const crop of next.results) {
                if (rendered.has(crop.filename)) continue;
                const card = element("a", { display: "flex", flexDirection: "column", color: "#ddd", textDecoration: "none", minWidth: "0" });
                card.href = api.apiURL(`${endpoint}/crop?id=${encodeURIComponent(next.id)}&file=${encodeURIComponent(crop.filename)}`);
                card.target = "_blank"; card.rel = "noopener";
                const image = element("img", { width: "100%", height: "140px", objectFit: "contain", background: "#101014" });
                image.src = card.href; image.alt = crop.filename; image.loading = "lazy";
                image.addEventListener("load", scroll.refresh);
                card.append(image, element("span", { overflowWrap: "anywhere" }, `${crop.filename}\n${crop.width} × ${crop.height}`));
                gallery.append(card); rendered.add(crop.filename);
            }
            scroll.refresh(); controls();
        }
        async function update(force = false) {
            const id = jobWidget.value;
            if (disposed || polling || busy || !id || (!force && state?.id === id && !ACTIVE.has(state.status))) return;
            polling = true;
            try { render(await request(`/jobs?id=${encodeURIComponent(id)}`)); }
            catch (error) { if (!disposed && id === jobWidget.value) status.textContent = error.message; }
            finally { polling = false; controls(); }
        }
        execute.addEventListener("click", async () => {
            if (busy || execute.disabled || disposed) return;
            busy = true; controls(); status.textContent = "Preparing face detector…";
            try {
                const data = { action: "start" };
                for (const name of ["image_gallery", "output_folder", "filename_prefix", "face_selection", "crop_shape",
                    "padding_percent", "limit_size", "max_size", "grid_split", "confidence"]) data[name] = value(name);
                const next = await request("/jobs", data);
                if (!disposed) { setJob(next.id); render(next); }
            } catch (error) { if (!disposed) status.textContent = error.message; }
            finally { busy = false; controls(); }
        });
        cancel.addEventListener("click", async () => {
            if (busy || cancel.disabled || disposed) return;
            const id = jobWidget.value;
            busy = true; controls();
            try { render(await request("/jobs", { action: "cancel", id })); }
            catch (error) { if (!disposed && id === jobWidget.value) status.textContent = error.message; }
            finally { busy = false; controls(); }
        });
        refresh.addEventListener("click", () => update(true));
        const timer = setInterval(() => update(), 1000);
        const previousConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const result = previousConfigure?.apply(this, args);
            state = null; controls(); update(true); return result;
        };
        const previousRemoved = node.onRemoved;
        node.onRemoved = function (...args) {
            disposed = true; clearInterval(timer); scroll.dispose();
            return previousRemoved?.apply(this, args);
        };
        controls(); update();
    },
});
