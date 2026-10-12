import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const endpoint = "/dinki/qwen21";
const activeStates = new Set(["starting", "latents", "text", "training", "pause_requested", "cancel_requested"]);

async function request(path, data, base = endpoint) {
    const response = await api.fetchApi(base + path, data === undefined ? {} : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
    });
    const result = await response.json();
    if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
    return result;
}

function el(tag, text = "") {
    const element = document.createElement(tag);
    element.textContent = text;
    return element;
}

function nativeScroll(element) {
    element.style.overscrollBehavior = "contain";
    element.addEventListener("wheel", event => event.stopPropagation(), { passive: true });
    element.addEventListener("pointerdown", event => event.stopPropagation());
}

async function viewFullLog(id) {
    if (!id) throw new Error("Select a training job first.");
    const dialog = el("dialog");
    Object.assign(dialog.style, { width: "min(1100px, 90vw)", maxHeight: "90vh", background: "#202024", color: "#eee" });
    const kind = el("select");
    for (const name of ["train", "worker"]) {
        const option = el("option", name === "train" ? "Training log" : "Worker log");
        option.value = name;
        kind.append(option);
    }
    kind.value = "train";
    const refresh = el("button", "Refresh");
    const download = el("a", "Download Log");
    const close = el("button", "Close");
    const log = el("pre");
    Object.assign(log.style, { height: "65vh", overflow: "auto", whiteSpace: "pre-wrap", overflowWrap: "anywhere", userSelect: "text" });
    log.tabIndex = 0;
    nativeScroll(log);
    let sequence = 0;
    const load = async () => {
        const version = ++sequence;
        const path = `${endpoint}/log?id=${encodeURIComponent(id)}&kind=${kind.value}`;
        download.href = api.apiURL(path + "&download=1");
        log.textContent = "Loading the complete log…";
        try {
            const response = await api.fetchApi(path);
            if (!response.ok) throw new Error((await response.json()).error || `HTTP ${response.status}`);
            const text = await response.text();
            if (version === sequence) { log.textContent = text || "The log is empty."; log.scrollTop = 0; }
        } catch (error) { if (version === sequence) log.textContent = error.message; }
    };
    refresh.type = close.type = "button";
    refresh.onclick = kind.onchange = load;
    close.onclick = () => dialog.close();
    dialog.addEventListener("close", () => { sequence++; dialog.remove(); }, { once: true });
    dialog.append(el("h3", "Qwen 2.1 · Complete Log"), kind, refresh, download, close, log);
    document.body.append(dialog);
    dialog.showModal();
    await load();
}

async function configureModels() {
    const settings = await request("/settings");
    let catalog;
    try { catalog = await request("/models"); }
    catch (error) { catalog = { choices: {}, warnings: [error.message] }; }
    const dialog = el("dialog");
    Object.assign(dialog.style, { width: "min(760px, 90vw)", background: "#202024", color: "#eee", borderRadius: "8px" });
    dialog.append(el("h3", "DKST · Qwen Image 2.1 Training"));
    dialog.append(el("p", "Select models registered in ComfyUI. Install the separate training environment first. Paths are stored on the server."));
    const fields = {};
    const labels = { python: "Training Python (.venv/Scripts/python.exe)", dit: "Qwen Image 2.1 base DiT",
        vae: "Qwen Image 2.1 VAE (ComfyUI / Diffusers)", text_encoder: "Qwen3-VL-8B text encoder",
        training_adapter: "Fizgig Qwen 2.1 training adapter", turbo_lora: "Qwen 2.1 Turbo LoRA (optional previews)" };
    for (const [key, label] of Object.entries(labels)) {
        const row = el("label", label);
        row.style.display = "block";
        row.style.margin = "10px 0";
        const select = el("select");
        select.style.width = "100%";
        const input = el("input");
        input.type = "text";
        input.value = settings[key] || "";
        input.placeholder = "Absolute path on the ComfyUI server";
        input.style.width = "100%";
        const populate = (value) => {
            input.value = value;
            select.replaceChildren();
            const add = (label, path) => {
                const option = el("option", label);
                option.value = path;
                option.title = path;
                select.append(option);
            };
            add(key === "turbo_lora" ? "None (optional)" : "Select…", "");
            const entries = catalog.choices[key] || [];
            for (const entry of entries) add(entry.label, entry.path);
            if (value && !entries.some(entry => entry.path === value)) add(`Saved/custom: ${value}`, value);
            add("Enter a server path manually…", "__manual__");
            select.value = value || (key === "python" ? entries[0]?.path || "" : "");
            select.onchange();
        };
        select.onchange = () => {
            input.hidden = select.value !== "__manual__";
            if (!input.hidden) input.focus?.();
            else input.value = select.value;
        };
        populate(settings[key] || "");
        row.append(select, input);
        fields[key] = { value: () => select.value === "__manual__" ? input.value.trim() : select.value, populate };
        dialog.append(row);
    }
    const message = el("pre");
    message.style.whiteSpace = "pre-wrap";
    const save = el("button", "Save");
    const check = el("button", "Check Runtime");
    const close = el("button", "Close");
    const refresh = el("button", "Refresh Models");
    for (const button of [save, check, refresh, close]) { button.type = "button"; button.style.marginRight = "8px"; }
    const values = () => Object.fromEntries(Object.entries(fields).map(([k, v]) => [k, v.value()]));
    message.textContent = (catalog.warnings || []).join("\n");
    refresh.onclick = async () => {
        refresh.disabled = true;
        try {
            const selected = values();
            catalog = await request("/models");
            for (const [key, field] of Object.entries(fields)) field.populate(selected[key]);
            message.textContent = (catalog.warnings || []).join("\n") || "Model list refreshed.";
        } catch (error) { message.textContent = error.message; }
        finally { refresh.disabled = false; }
    };
    save.onclick = async () => {
        save.disabled = true;
        try { await request("/settings", values()); message.textContent = "Settings saved."; }
        catch (error) { message.textContent = error.message; }
        finally { save.disabled = false; }
    };
    check.onclick = async () => {
        check.disabled = true;
        message.textContent = "Checking the training environment…";
        try {
            await request("/settings", values());
            const result = await request("/check", {});
            message.textContent = JSON.stringify(result, null, 2);
        } catch (error) { message.textContent = error.message; }
        finally { check.disabled = false; }
    };
    close.onclick = () => dialog.close();
    dialog.addEventListener("close", () => dialog.remove(), { once: true });
    dialog.append(save, check, refresh, close, message);
    document.body.append(dialog);
    dialog.showModal();
}

app.registerExtension({
    name: "DINKI.Qwen21Training",
    setup() {
        app.ui.settings.addSetting({ id: "DKST.Training.Qwen21.Models", name: "Qwen 2.1 LoRA Training · Models & Runtime",
            type: () => {
                const button = el("button", "Configure Models & Runtime");
                button.type = "button";
                button.onclick = () => configureModels().catch(error => app.extensionManager?.toast?.add({
                    severity: "error", summary: "Qwen Training", detail: error.message }));
                return button;
            }, defaultValue: "", category: ["DKST", "Training", "Qwen 2.1"],
        });
        // Generation submitted through this UI must not race the training GPU.
        const queuePrompt = app.queuePrompt;
        app.queuePrompt = async function (...args) {
            const state = await request("/training");
            if (state.active) throw new Error("Pause or finish Qwen LoRA training before running a generation workflow.");
            return queuePrompt.apply(this, args);
        };
    },
    nodeCreated(node) {
        if (node.comfyClass !== "DINKI_Qwen21_LoRA_Trainer") return;
        node.properties ||= {};
        const aliasWidget = node.widgets.find(w => w.name === "caption_connection_alias");
        const modelWidget = node.widgets.find(w => w.name === "caption_model");
        if (aliasWidget) aliasWidget.label = "Caption LLM alias (optional)";
        if (modelWidget) modelWidget.label = "Caption model (optional)";
        const root = el("div");
        Object.assign(root.style, { padding: "8px", color: "#ddd", background: "#18181b", borderRadius: "6px" });
        const toolbar = el("div");
        Object.assign(toolbar.style, { display: "flex", flexWrap: "wrap", gap: "6px" });
        const status = el("pre", "Configure models, add photos and captions, then Start Training.");
        Object.assign(status.style, { whiteSpace: "pre-wrap", fontSize: "12px", maxHeight: "100px", overflow: "auto" });
        const history = el("select");
        history.style.width = "100%";
        const checkpoints = el("pre");
        Object.assign(checkpoints.style, { whiteSpace: "pre-wrap", fontSize: "11px", maxHeight: "90px", overflow: "auto" });
        const samples = el("div");
        Object.assign(samples.style, { display: "flex", overflowX: "auto", gap: "4px" });
        const log = el("pre");
        Object.assign(log.style, { whiteSpace: "pre-wrap", maxHeight: "170px", overflow: "auto", fontSize: "10px" });
        for (const element of [status, checkpoints, samples, log]) {
            nativeScroll(element);
            element.tabIndex = 0;
        }
        for (const element of [status, checkpoints, log]) element.style.userSelect = "text";
        const captionStatus = el("pre", "Optional LLM: select an alias and enter a vision model name, then Auto Caption to fill empty descriptions.");
        Object.assign(captionStatus.style, { whiteSpace: "pre-wrap", fontSize: "11px" });
        let removed = false;
        let busy = false;
        let polling = false;
        let captioning = false;
        let stopCaptions = false;
        let aliasSequence = 0;
        let current;
        const buttons = {};
        const values = () => Object.fromEntries(node.widgets.filter(w => w.name !== "training_status").map(w => [w.name, w.value]));

        function updateButtons() {
            const active = activeStates.has(current?.status);
            buttons.start.disabled = busy || captioning || active;
            buttons.pause.disabled = busy || !active || current?.status === "pause_requested";
            buttons.cancel.disabled = busy || !active;
            buttons.resume.disabled = busy || captioning || !current || active || current.status === "completed";
            buttons.caption.disabled = busy || captioning || active;
            buttons.stopCaptions.disabled = !captioning || stopCaptions;
            buttons.aliases.disabled = busy || captioning;
        }

        async function loadAliases() {
            const sequence = ++aliasSequence;
            const result = await request("/aliases", undefined, "/dinki/llm");
            if (removed || sequence !== aliasSequence) return;
            const widget = node.widgets.find(w => w.name === "caption_connection_alias");
            if (!widget) return;
            const choices = ["-- Select connection --", ...result.aliases];
            // Retain a missing saved alias, allowing users to select a replacement.
            if (widget.value && !choices.includes(widget.value)) choices.push(widget.value);
            widget.options ||= {};
            widget.options.values = choices;
            if (widget._state?.options) widget._state.options.values = choices;
            node.setDirtyCanvas?.(true, true);
        }

        async function autoCaption() {
            if (busy || captioning || activeStates.has(current?.status)) return;
            const options = values();
            if (!options.caption_connection_alias || options.caption_connection_alias === "-- Select connection --") {
                captionStatus.textContent = "Select an optional LLM alias from Settings > DKST > LLM first.";
                return;
            }
            if (!String(options.caption_model || "").trim()) {
                captionStatus.textContent = "Enter the exact name of an image-capable LLM in caption_model.";
                return;
            }
            const gallery = node.widgets.find(w => w.name === "image_gallery");
            if (!gallery?.dkstCaptionTargets) throw new Error("Reload the browser to update the caption gallery.");
            const targets = gallery.dkstCaptionTargets();
            if (!targets.length) { captionStatus.textContent = "No empty captions. Existing descriptions are preserved."; return; }
            captioning = true;
            stopCaptions = false;
            updateButtons();
            let filled = 0, skipped = 0;
            try {
                for (let index = 0; index < targets.length; index++) {
                    if (removed || stopCaptions) break;
                    captionStatus.textContent = `Generating caption ${index + 1}/${targets.length}…`;
                    const result = await request("/caption", { image: targets[index].image,
                        caption_connection_alias: options.caption_connection_alias, caption_model: options.caption_model });
                    if (removed) break;
                    if (targets[index].apply(result.caption)) filled++;
                    else skipped++;
                }
                if (!removed) captionStatus.textContent = `${stopCaptions ? "Stopped" : "Finished"}: ${filled} captions added` +
                    (skipped ? `, ${skipped} edited/removed photos skipped.` : ".") + " Review descriptions before training.";
            } catch (error) {
                if (!removed) captionStatus.textContent = `${filled} captions added. ${error.message}`;
            } finally {
                captioning = false;
                if (!removed) updateButtons();
            }
        }

        function render(state) {
            current = state;
            updateButtons();
            status.textContent = `${state.project_name} · ${state.status}\n${state.num_images} photos` +
                (state.epoch ? ` · Epoch ${state.epoch}/${state.epochs} · Step ${state.step} · Loss ${state.loss}` : "") +
                (state.error ? `\n${state.error}` : "") + (state.lora_path ? `\nLoRA: ${state.lora_path}` : "");
            log.textContent = state.log || "";
            checkpoints.textContent = `Output: ${state.output_dir}\n${(state.checkpoints || []).join("\n")}`;
            const names = state.id + ":" + (state.samples || []).join("|");
            if (samples.dataset.names !== names) {
                samples.dataset.names = names;
                samples.replaceChildren();
                for (const name of state.samples || []) {
                    const image = el("img");
                    image.src = api.apiURL(`${endpoint}/sample?id=${encodeURIComponent(state.id)}&file=${encodeURIComponent(name)}`);
                    image.alt = name;
                    image.style.height = "120px";
                    samples.append(image);
                }
            }
            node.setDirtyCanvas?.(true, true);
        }

        async function refresh() {
            if (polling || removed) return;
            polling = true;
            try {
                const id = node.properties.dkst_qwen21_job_id;
                if (id) render(await request(`/training?id=${encodeURIComponent(id)}`));
            } catch (error) { status.textContent = error.message; }
            finally { polling = false; }
        }

        async function loadHistory() {
            const project = values().project_name;
            const result = await request(`/training?project=${encodeURIComponent(project)}`);
            history.replaceChildren(el("option", "Select a training job"));
            history.firstChild.value = "";
            for (const job of result.jobs) {
                const option = el("option", `${new Date(job.created * 1000).toLocaleString()} · ${job.status} · ${job.id.slice(0, 8)}`);
                option.value = job.id;
                history.append(option);
            }
            if (!node.properties.dkst_qwen21_job_id && result.jobs.length) node.properties.dkst_qwen21_job_id = result.jobs[0].id;
            history.value = node.properties.dkst_qwen21_job_id || "";
            await refresh();
        }

        async function act(action) {
            if (busy || captioning) return;
            busy = true;
            Object.values(buttons).forEach(button => { button.disabled = true; });
            status.textContent = action === "start" ? "Preparing the photo dataset…" : "Updating training job…";
            try {
                const data = action === "start" ? { ...values(), action } : { action, id: node.properties.dkst_qwen21_job_id };
                const state = await request("/training", data);
                node.properties.dkst_qwen21_job_id = state.id;
                node.graph?.incrementVersion?.();
                current = state;
            } catch (error) { status.textContent = error.message; current = undefined; }
            finally {
                busy = false;
                Object.values(buttons).forEach(button => { button.disabled = false; });
                if (current) render(current);
                else updateButtons();
            }
        }

        const addButton = (key, label, callback) => {
            const button = el("button", label);
            button.type = "button";
            button.onclick = () => Promise.resolve(callback()).catch(error => { status.textContent = error.message; });
            toolbar.append(button);
            buttons[key] = button;
        };
        addButton("configure", "Configure Models", configureModels);
        addButton("start", "Start Training", () => act("start"));
        addButton("pause", "Pause", () => act("pause"));
        addButton("resume", "Resume", () => act("resume"));
        addButton("cancel", "Stop", () => act("cancel"));
        addButton("refresh", "Refresh Jobs", loadHistory);
        addButton("fullLog", "View Full Log", () => viewFullLog(node.properties.dkst_qwen21_job_id));
        addButton("aliases", "Refresh LLM Aliases", () => loadAliases().catch(error => { captionStatus.textContent = error.message; }));
        addButton("caption", "Auto Caption (Empty)", autoCaption);
        addButton("stopCaptions", "Stop Captioning", () => {
            stopCaptions = true;
            captionStatus.textContent = "Stopping after the current photo…";
            updateButtons();
        });
        updateButtons();
        history.onchange = () => {
            node.properties.dkst_qwen21_job_id = history.value;
            node.graph?.incrementVersion?.();
            refresh();
        };
        root.append(toolbar, status, history, samples, checkpoints, log, captionStatus);
        const widget = node.addDOMWidget("training_status", "DKST_QWEN_TRAIN_STATUS", root, { serialize: false });
        widget.serialize = false;
        if (node.size[0] < 520) node.setSize([520, node.size[1]]);
        const interval = setInterval(refresh, 3000);
        const oldRemoved = node.onRemoved;
        node.onRemoved = function (...args) {
            removed = true;
            stopCaptions = true;
            clearInterval(interval);
            return oldRemoved?.apply(this, args);
        };
        const oldConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const result = oldConfigure?.apply(this, args);
            loadAliases().catch(error => { captionStatus.textContent = error.message; });
            return result;
        };
        setTimeout(() => {
            if (removed) return;
            loadHistory().catch(error => { status.textContent = error.message; });
            loadAliases().catch(error => { captionStatus.textContent = error.message; });
        }, 0);
    },
});
