import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { bindScrollWheel, createScrollControls } from "./dinki_scroll.js";

export function registerTrainingUI(config) {
    const endpoint = config.endpoint;
    const title = config.title;
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
        bindScrollWheel(log);
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
        dialog.append(el("h3", `${title} · Complete Log`), kind, refresh, download, close, log);
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
        Object.assign(dialog.style, { width: "min(760px, 90vw)", maxHeight: "90vh", overflow: "auto",
            background: "#202024", color: "#eee", borderRadius: "8px" });
        dialog.append(el("h3", `DKST · ${title} Training`));
        dialog.append(el("p", "Select models registered in ComfyUI. Prepare Runtime installs a separate CUDA environment on the server (Windows/Linux, NVIDIA GPU; downloads several GB). Python 3.12 is recommended. Paths are stored on the server."));
        const fields = {};
        const labels = config.modelLabels;
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
                add(config.optionalModels.includes(key) ? "None (optional)" : "Select…", "");
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
        Object.assign(message.style, { whiteSpace: "pre-wrap", maxHeight: "40vh", overflow: "auto", overflowWrap: "anywhere" });
        bindScrollWheel(message);
        const save = el("button", "Save");
        const check = el("button", "Check Runtime");
        const prepare = el("button", "Prepare Runtime");
        const close = el("button", "Close");
        const refresh = el("button", "Refresh Models");
        for (const button of [save, check, prepare, refresh, close]) { button.type = "button"; button.style.marginRight = "8px"; }
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
        let setupTimer, setupPolling = false, closed = false, selectPrepared = false;
        const showSetup = async (state) => {
            if (closed) return;
            const installing = state.status === "installing";
            if (installing) selectPrepared = true;
            prepare.disabled = check.disabled = installing;
            if (state.status !== "idle") message.textContent = state.message + (state.log ? "\n\n" + state.log : "");
            if (installing && !setupTimer) setupTimer = setInterval(pollSetup, 2000);
            if (!installing && setupTimer) { clearInterval(setupTimer); setupTimer = undefined; }
            if (state.status === "ready") {
                const selected = values();
                const python = selectPrepared || !selected.python ? state.python : selected.python;
                catalog = await request("/models");
                if (closed) return;
                for (const [key, field] of Object.entries(fields)) field.populate(key === "python" ? python : selected[key]);
                selectPrepared = false;
            }
        };
        const pollSetup = async () => {
            if (setupPolling || closed) return;
            setupPolling = true;
            try { await showSetup(await request("/runtime/setup")); }
            catch (error) {
                if (!closed) message.textContent = error.message;
                if (setupTimer) clearInterval(setupTimer);
                setupTimer = undefined;
                prepare.disabled = check.disabled = false;
            } finally { setupPolling = false; }
        };
        prepare.onclick = async () => {
            selectPrepared = true;
            prepare.disabled = check.disabled = true;
            message.textContent = "Preparing the separate training environment…";
            try { await showSetup(await request("/runtime/setup", {})); }
            catch (error) { if (!closed) message.textContent = error.message; prepare.disabled = check.disabled = false; }
        };
        close.onclick = () => dialog.close();
        dialog.addEventListener("close", () => {
            closed = true;
            if (setupTimer) clearInterval(setupTimer);
            dialog.remove();
        }, { once: true });
        dialog.append(save, check, prepare, refresh, close, message);
        document.body.append(dialog);
        dialog.showModal();
        await pollSetup();
    }

    app.registerExtension({
        name: config.extensionName,
        setup() {
            app.ui.settings.addSetting({ id: config.settingId, name: `${title} LoRA Training · Models & Runtime`,
                type: () => {
                    const button = el("button", "Configure Models & Runtime");
                    button.type = "button";
                    button.onclick = () => configureModels().catch(error => app.extensionManager?.toast?.add({
                        severity: "error", summary: `${title} Training`, detail: error.message }));
                    return button;
                }, defaultValue: "", category: ["DKST", "Training", title],
            });
            // Generation submitted through this UI must not race the training GPU.
            if (!app.dkstTrainingQueueGuardInstalled) {
                app.dkstTrainingQueueGuardInstalled = true;
                const queuePrompt = app.queuePrompt;
                app.queuePrompt = async function (...args) {
                    const state = await request("", undefined, "/dinki/training");
                    if (state.active) throw new Error("Pause or finish LoRA training before running a generation workflow.");
                    return queuePrompt.apply(this, args);
                };
            }
        },
        nodeCreated(node) {
            if (node.comfyClass !== config.nodeClass) return;
            node.properties ||= {};
            const aliasWidget = node.widgets.find(w => w.name === "caption_connection_alias");
            const modelWidget = node.widgets.find(w => w.name === "caption_model");
            if (aliasWidget) aliasWidget.label = "Caption LLM alias (optional)";
            if (modelWidget) modelWidget.label = "Caption model (optional)";
            const root = el("div");
            Object.assign(root.style, { padding: "8px", color: "#ddd", background: "#18181b", borderRadius: "6px",
                width: "100%", minWidth: "0", maxWidth: "100%", boxSizing: "border-box", overflow: "hidden" });
            const toolbar = el("div");
            Object.assign(toolbar.style, { display: "flex", flexWrap: "wrap", gap: "6px" });
            const status = el("pre", "Configure models, add photos and captions, then Start Training.");
            Object.assign(status.style, { whiteSpace: "pre-wrap", fontSize: "12px", maxHeight: "100px", overflow: "auto" });
            const history = el("select");
            history.style.width = "100%";
            const checkpoints = el("pre");
            Object.assign(checkpoints.style, { whiteSpace: "pre-wrap", fontSize: "11px", maxHeight: "90px", overflow: "auto" });
            const samples = el("div");
            samples.className = "dkst-sample-viewport";
            Object.assign(samples.style, { display: "flex", flexWrap: "nowrap", overflowX: "auto", overflowY: "hidden",
                width: "100%", minWidth: "0", gap: "4px" });
            const sampleScroll = createScrollControls(samples, "x", "Sample images");
            const samplePanel = el("div");
            Object.assign(samplePanel.style, { width: "100%", minWidth: "0" });
            samplePanel.hidden = true;
            samplePanel.append(samples, sampleScroll.element);
            const log = el("pre");
            Object.assign(log.style, { whiteSpace: "pre-wrap", maxHeight: "170px", overflow: "auto", fontSize: "10px" });
            for (const element of [status, checkpoints, log]) {
                bindScrollWheel(element);
                element.tabIndex = 0;
            }
            samples.tabIndex = 0;
            for (const element of [status, checkpoints, log]) {
                element.style.userSelect = "text";
                element.style.overflowWrap = "anywhere";
            }
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
                    const sameJob = samples.dataset.job === state.id;
                    const scrollLeft = sameJob ? samples.scrollLeft || 0 : 0;
                    const existing = sameJob ? new Map(Array.from(samples.children, image => [image.alt, image])) : new Map();
                    samples.dataset.job = state.id;
                    samples.dataset.names = names;
                    const images = (state.samples || []).map(name => {
                        const image = existing.get(name) || el("img");
                        if (!existing.has(name)) {
                            image.src = api.apiURL(`${endpoint}/sample?id=${encodeURIComponent(state.id)}&file=${encodeURIComponent(name)}`);
                            image.alt = name;
                            image.title = name;
                            image.draggable = false;
                            Object.assign(image.style, { height: "120px", flex: "0 0 auto", maxWidth: "none", objectFit: "contain" });
                            image.addEventListener("load", sampleScroll.refresh);
                        }
                        return image;
                    });
                    samples.replaceChildren(...images);
                    samplePanel.hidden = images.length === 0;
                    samples.scrollLeft = scrollLeft;
                    requestAnimationFrame(sampleScroll.refresh);
                }
                node.setDirtyCanvas?.(true, true);
            }

            async function refresh() {
                if (polling || removed) return;
                polling = true;
                try {
                    const id = node.properties[config.jobProperty];
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
                if (!node.properties[config.jobProperty] && result.jobs.length) node.properties[config.jobProperty] = result.jobs[0].id;
                history.value = node.properties[config.jobProperty] || "";
                await refresh();
            }

            async function act(action) {
                if (busy || captioning) return;
                busy = true;
                Object.values(buttons).forEach(button => { button.disabled = true; });
                status.textContent = action === "start" ? "Preparing the photo dataset…" : "Updating training job…";
                try {
                    const data = action === "start" ? { ...values(), action } : { action, id: node.properties[config.jobProperty] };
                    const state = await request("/training", data);
                    node.properties[config.jobProperty] = state.id;
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
            addButton("fullLog", "View Full Log", () => viewFullLog(node.properties[config.jobProperty]));
            addButton("aliases", "Refresh LLM Aliases", () => loadAliases().catch(error => { captionStatus.textContent = error.message; }));
            addButton("caption", "Auto Caption (Empty)", autoCaption);
            addButton("stopCaptions", "Stop Captioning", () => {
                stopCaptions = true;
                captionStatus.textContent = "Stopping after the current photo…";
                updateButtons();
            });
            updateButtons();
            history.onchange = () => {
                node.properties[config.jobProperty] = history.value;
                node.graph?.incrementVersion?.();
                refresh();
            };
            root.append(toolbar, status, history, samplePanel, checkpoints, log, captionStatus);
            const widget = node.addDOMWidget("training_status", config.statusType, root, { serialize: false });
            widget.serialize = false;
            if (node.size[0] < 520) node.setSize([520, node.size[1]]);
            const interval = setInterval(refresh, 3000);
            const oldRemoved = node.onRemoved;
            node.onRemoved = function (...args) {
                removed = true;
                stopCaptions = true;
                clearInterval(interval);
                sampleScroll.dispose();
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

}
