import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

function promptForNode(prompt, nodeId) {
    const source = prompt.output;
    const selected = {};
    const visit = (id) => {
        id = String(id);
        if (selected[id]) return;
        const entry = source[id];
        if (!entry) throw new Error(`Node not found in the prompt: ${id}`);
        selected[id] = entry;
        for (const value of Object.values(entry.inputs ?? {})) {
            if (Array.isArray(value) && value.length === 2 && source[String(value[0])]) {
                visit(value[0]);
            }
        }
    };
    visit(nodeId);
    prompt.output = selected;
    return prompt;
}

async function runNodeAction(node, inputs, button, onResult) {
    if (button._dinkiBusy) return;
    button._dinkiBusy = true;
    const originalLabel = button.name;
    button.name = "Working…";
    node.setDirtyCanvas(true, true);

    let timer;
    let promptId;
    let finished = false;
    const requestId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
    const cleanup = () => {
        clearTimeout(timer);
        api.removeEventListener("executed", onExecuted);
        api.removeEventListener("execution_error", onError);
        api.removeEventListener("execution_success", onSuccess);
        button._dinkiBusy = false;
        button.name = originalLabel;
        node.setDirtyCanvas(true, true);
    };
    const onExecuted = ({ detail }) => {
        if (String(detail?.node) !== String(node.id)) return;
        if (detail?.output?.request_id?.[0] !== requestId) return;
        finished = true;
        cleanup();
        onResult(detail.output);
    };
    const onError = ({ detail }) => {
        if (!promptId || detail?.prompt_id !== promptId) return;
        if (finished) return;
        finished = true;
        cleanup();
        alert(`VoxCPM2 action failed: ${detail?.exception_message ?? "Execution error"}`);
    };
    const onSuccess = ({ detail }) => {
        if (!promptId || detail?.prompt_id !== promptId) return;
        if (finished) return;
        finished = true;
        cleanup();
        alert("No VoxCPM2 result was received. Check the ComfyUI execution history.");
    };

    api.addEventListener("executed", onExecuted);
    api.addEventListener("execution_error", onError);
    api.addEventListener("execution_success", onSuccess);
    timer = setTimeout(() => {
        if (finished) return;
        finished = true;
        cleanup();
        alert("The VoxCPM2 action is taking a long time. Check the ComfyUI execution history.");
    }, 30 * 60 * 1000);

    try {
        const prompt = promptForNode(await app.graphToPrompt(), node.id);
        prompt.output[String(node.id)].inputs = {
            ...prompt.output[String(node.id)].inputs,
            ...inputs,
            request_id: requestId,
        };
        const response = await api.queuePrompt(0, prompt);
        if (!response?.prompt_id) throw new Error("Could not queue the prompt.");
        promptId = response.prompt_id;
    } catch (error) {
        if (!finished) {
            finished = true;
            cleanup();
            alert(`VoxCPM2 action failed: ${error?.message ?? error}`);
        }
    }
}

app.registerExtension({
    name: "DINKI.VoxCPM2.Controls",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const isManager = nodeData.name === "DKST_VoxCPM2_Downloader";
        const isReference = nodeData.name === "DKST_VoxCPM2_ReferenceAudio";
        if (!isManager && !isReference) return;

        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = previous?.apply(this, args);
            const node = this;
            const internalNames = isManager
                ? ["download_action", "request_id"]
                : ["transcribe_action", "request_id"];
            node.widgets = (node.widgets ?? []).filter((widget) => !internalNames.includes(widget.name));

            if (isManager) {
                const voxButton = node.addWidget("button", "Download VoxCPM2", null, () => {
                    const button = voxButton;
                    runNodeAction(node, { download_action: "voxcpm2" }, button,
                        (output) => alert(output.status?.[0] ?? "VoxCPM2 download complete"));
                }, { serialize: false });
                const whisperButton = node.addWidget("button", "Download Whisper", null, () => {
                    const button = whisperButton;
                    runNodeAction(node, { download_action: "whisper" }, button,
                        (output) => alert(output.status?.[0] ?? "Whisper download complete"));
                }, { serialize: false });
            } else {
                const fileWidget = node.widgets?.find((item) => item.name === "voice_file");
                const previewWidget = node.widgets?.find((item) => item.name === "transcript_preview");
                if (previewWidget) {
                    previewWidget.options ??= {};
                    previewWidget.options.read_only = true;
                    if (previewWidget.inputEl) previewWidget.inputEl.readOnly = true;
                }
                let selectionVersion = 0;
                let voiceListVersion = 0;
                const setTranscript = (value) => {
                    if (!previewWidget) return;
                    previewWidget.value = value;
                    previewWidget.callback?.(value);
                    node.setDirtyCanvas(true, true);
                };
                const refreshTranscript = async () => {
                    const name = String(fileWidget?.value ?? "");
                    const version = ++selectionVersion;
                    if (!name) {
                        setTranscript("");
                        return;
                    }
                    const response = await api.fetchApi(`/dkst/voxcpm2/transcript?name=${encodeURIComponent(name)}`);
                    const payload = await response.json();
                    if (!response.ok) throw new Error(payload.error ?? "Could not load the transcript.");
                    if (version === selectionVersion && name === fileWidget?.value) {
                        setTranscript(payload.transcript ?? "");
                    }
                };
                const refreshVoices = async (preferredName) => {
                    const version = ++voiceListVersion;
                    const response = await api.fetchApi("/dkst/voxcpm2/voices");
                    const payload = await response.json();
                    if (!response.ok) throw new Error(payload.error ?? "Could not list voice files.");
                    if (!fileWidget || version !== voiceListVersion) return;
                    const files = payload.files ?? [];
                    fileWidget.options ??= {};
                    fileWidget.options.values = files.length ? files : [""];
                    const current = preferredName ?? fileWidget.value;
                    fileWidget.value = files.includes(current) ? current : (files[0] ?? "");
                    node.setDirtyCanvas(true, true);
                    await refreshTranscript();
                };
                if (fileWidget) {
                    const previousCallback = fileWidget.callback;
                    fileWidget.callback = function (...args) {
                        previousCallback?.apply(this, args);
                        refreshTranscript().catch((error) => alert(error.message));
                    };
                }
                const configured = node.onConfigure;
                node.onConfigure = function (...args) {
                    const result = configured?.apply(this, args);
                    queueMicrotask(() => refreshVoices().catch((error) => alert(error.message)));
                    return result;
                };
                queueMicrotask(() => refreshVoices().catch((error) => alert(error.message)));

                node.addWidget("button", "Upload Voice", null, () => {
                    const picker = document.createElement("input");
                    picker.type = "file";
                    picker.accept = ".wav,.mp3,.flac,.ogg,.m4a,.aac,.opus,audio/*";
                    picker.style.display = "none";
                    picker.onchange = async () => {
                        const file = picker.files?.[0];
                        picker.remove();
                        if (!file) return;
                        try {
                            const form = new FormData();
                            form.append("file", file);
                            const response = await api.fetchApi("/dkst/voxcpm2/upload-voice", {
                                method: "POST", body: form,
                            });
                            const payload = await response.json();
                            if (!response.ok) throw new Error(payload.error ?? "Voice upload failed.");
                            await refreshVoices(payload.name);
                        } catch (error) {
                            alert(`Voice upload failed: ${error.message}`);
                        }
                    };
                    document.body.appendChild(picker);
                    picker.click();
                }, { serialize: false });
                node.addWidget("button", "Refresh Voices", null, () => {
                    refreshVoices().catch((error) => alert(error.message));
                }, { serialize: false });
                const transcriptButton = node.addWidget("button", "Transcribe Reference (Whisper)", null, () => {
                    const button = transcriptButton;
                    runNodeAction(node, { transcribe_action: true }, button,
                        (output) => setTranscript(output.transcript?.[0] ?? ""));
                }, { serialize: false });
            }
            return result;
        };
    },
});
