const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "..", "ComfyUI-DINKIssTyle_VoxCPM2", "web", "voxcpm2.js"), "utf8")
    .replace(/^import .*;\n/gm, "");
const extensions = [];
const listeners = new Map();
const queued = [];
const alerts = [];
let currentPrompt;
let voiceFiles = ["voice.wav"];
const document = {
    body: { appendChild() {} },
    createElement() {
        return {
            style: {}, files: [{ name: "uploaded.wav" }], remove() {},
            click() { this.onchange?.(); },
        };
    },
};
class FormDataMock { append() {} }
const app = {
    registerExtension(extension) { extensions.push(extension); },
    async graphToPrompt() { return structuredClone(currentPrompt); },
};
const api = {
    async fetchApi(url, options) {
        const data = url === "/dkst/voxcpm2/voices"
            ? { files: voiceFiles }
            : url.startsWith("/dkst/voxcpm2/transcript?")
                ? { transcript: "existing transcript" }
                : { name: "uploaded.wav", files: (voiceFiles = [...voiceFiles, "uploaded.wav"]) };
        assert.ok(!options || options.method === "POST");
        return { ok: true, async json() { return data; } };
    },
    addEventListener(name, callback) {
        if (!listeners.has(name)) listeners.set(name, new Set());
        listeners.get(name).add(callback);
    },
    removeEventListener(name, callback) { listeners.get(name)?.delete(callback); },
    async queuePrompt(_, prompt) {
        queued.push(prompt);
        const nodeId = Object.keys(prompt.output).at(-1);
        queueMicrotask(() => {
            const output = { request_id: [prompt.output[nodeId].inputs.request_id] };
            if (prompt.output[nodeId].inputs.transcribe_action) output.transcript = ["recognized transcript"];
            else output.status = ["Download complete"];
            for (const callback of listeners.get("executed") ?? []) {
                callback({ detail: { node: nodeId, output } });
            }
        });
        return { prompt_id: "prompt-1" };
    },
};
vm.runInNewContext(source, {
    app, api, alert: (message) => alerts.push(message),
    setTimeout, clearTimeout, crypto: { randomUUID: () => "request-1" },
    console, queueMicrotask, document, FormData: FormDataMock,
});

function createNode(type, id) {
    class Node {
        constructor() {
            this.id = id;
            this.widgets = [
                { name: "voice_file", value: "voice.wav", options: { values: ["voice.wav"] } },
                { name: "transcript_preview", value: "", options: {} },
            ];
        }
        addWidget(kind, name, value, callback) {
            const widget = { kind, name, value, callback };
            this.widgets.push(widget);
            return widget;
        }
        setDirtyCanvas() {}
    }
    extensions[0].beforeRegisterNodeDef(Node, { name: type });
    const node = new Node();
    node.onNodeCreated();
    return node;
}

(async () => {
    const manager = createNode("DKST_VoxCPM2_Downloader", 1);
    currentPrompt = { output: {
        "1": { class_type: "DKST_VoxCPM2_Downloader", inputs: { voxcpm_model: "VoxCPM2", whisper_model: "base" } },
        "99": { class_type: "UnrelatedOutput", inputs: {} },
    }, workflow: {} };
    await manager.widgets.find((widget) => widget.name === "Download VoxCPM2").callback();
    await new Promise(setImmediate);
    assert.deepEqual(Object.keys(queued[0].output), ["1"]);
    assert.equal(queued[0].output["1"].inputs.download_action, "voxcpm2");
    assert.deepEqual(alerts, ["Download complete"]);

    const reference = createNode("DKST_VoxCPM2_ReferenceAudio", 2);
    await new Promise(setImmediate);
    assert.equal(reference.widgets.find((widget) => widget.name === "transcript_preview").value,
        "existing transcript");
    currentPrompt = { output: {
        "1": { class_type: "DKST_VoxCPM2_Downloader", inputs: {} },
        "2": { class_type: "DKST_VoxCPM2_ReferenceAudio", inputs: { whisper_model_path: ["1", 1], voice_file: "voice.wav" } },
        "99": { class_type: "UnrelatedOutput", inputs: {} },
    }, workflow: {} };
    await reference.widgets.find((widget) => widget.name === "Transcribe Reference (Whisper)").callback();
    await new Promise(setImmediate);
    assert.deepEqual(Object.keys(queued[1].output).sort(), ["1", "2"]);
    assert.equal(queued[1].output["2"].inputs.transcribe_action, true);
    assert.equal(reference.widgets.find((widget) => widget.name === "transcript_preview").value,
        "recognized transcript");
    reference.widgets.find((widget) => widget.name === "Upload Voice").callback();
    await new Promise(setImmediate);
    assert.equal(reference.widgets.find((widget) => widget.name === "voice_file").value, "uploaded.wav");
    console.log("VoxCPM2 frontend button checks passed");
})().catch((error) => { console.error(error); process.exitCode = 1; });
