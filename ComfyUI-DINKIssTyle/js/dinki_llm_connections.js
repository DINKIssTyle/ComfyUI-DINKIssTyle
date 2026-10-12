import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const TYPES = new Set(["DINKI_LMStudio", "DINKI LM Studio Assistant"]);
const NONE = "-- Select connection --";
const PRIVATE = new Set(["ip_address", "port", "api_key"]);
let choices = [NONE];
let refreshSequence = 0;
const nodes = new Set();
const migrations = new WeakMap();
const serializedMigrations = new WeakMap();

async function request(path, payload) {
    const response = await api.fetchApi("/dinki/llm/" + path, payload === undefined ? {} : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "DKST LLM settings request failed.");
    return data;
}

function report(message) {
    if (app.extensionManager?.toast?.add) {
        app.extensionManager.toast.add({ severity: "warn", summary: "DKST LLM", detail: message, life: 10000 });
    } else if (app.ui?.dialog?.show) app.ui.dialog.show(message);
    else console.warn(message);
}

function combo(node) {
    return node.widgets?.find(widget => widget.name === "connection_alias");
}

function setAlias(node, value) {
    const widget = combo(node);
    if (!widget) return;
    widget.value = value;
    widget.options?.setValue?.(value);
    updateChoices(node);
}

function updateChoices(node) {
    const widget = combo(node);
    if (!widget) return;
    // Keep a missing saved alias visible; never silently switch to another server.
    const values = choices.includes(widget.value) || !widget.value
        ? [...choices] : [...choices, widget.value];
    if (widget.options) Object.assign(widget.options, { values });
    else widget.options = { values };
    if (widget._state?.options) Object.assign(widget._state.options, { values });
    node.graph?.incrementVersion?.();
    node.setDirtyCanvas?.(true, true);
}

async function refreshAliases() {
    const sequence = ++refreshSequence;
    const data = await request("aliases");
    if (sequence !== refreshSequence) return;
    choices = [NONE, ...data.aliases];
    for (const node of nodes) updateChoices(node);
}

function scrubFields(value, seen = new WeakSet(), depth = 0) {
    if (!value || typeof value !== "object" || seen.has(value) || depth > 32) return;
    // Serialized properties can include runtime widget/graph references in
    // some frontend versions. Never enumerate class instances such as LinkMap.
    const prototype = Object.getPrototypeOf(value);
    if (!Array.isArray(value) && prototype && prototype.constructor?.name !== "Object") return;
    seen.add(value);
    for (const key of Object.keys(value)) {
        if (PRIVATE.has(key)) delete value[key];
        else if (!["_widget", "graph", "_graph", "node", "_node", "canvas"].includes(key)) {
            scrubFields(value[key], seen, depth + 1);
        }
    }
}

function scrubNodeFields(info) {
    for (const name of PRIVATE) delete info[name];
    scrubFields(info.properties);
    scrubFields(info.widgets_values_named);
    if (!Array.isArray(info.widgets_values)) scrubFields(info.widgets_values);
}

function legacyConnection(info) {
    if (info.properties?.dkstLLMVersion === 2) return null;
    const linked = Boolean(info.inputs?.some(input => PRIVATE.has(input.name) && input.link != null));
    const values = info.widgets_values;
    // The original node has 10 generation widgets, plus an optional seed
    // control widget, followed by address/port and (in newer versions) API key.
    for (const count of Array.isArray(values) ? [3, 2] : []) {
        const prefix = values.length - count;
        if (prefix !== 10 && prefix !== 11) continue;
        const host = values[prefix], port = values[prefix + 1];
        if (typeof host !== "string" || !Number.isInteger(port)) continue;
        if (count === 3 && typeof values[prefix + 2] !== "string") continue;
        return { ip_address: host, port, api_key: count === 3 ? values[prefix + 2] : "", prefix, linked };
    }
    const named = info.widgets_values_named ?? (!Array.isArray(values) ? values : null);
    for (const source of [named, info, info.properties]) {
        if (source && Object.hasOwn(source, "ip_address")) {
            if (Array.isArray(values) && values.length) {
                return { unsupported: true, linked };
            }
            return { ip_address: source.ip_address, port: source.port ?? 1234, api_key: source.api_key ?? "", linked };
        }
    }
    if (linked || (Array.isArray(values) && values.length > 12)) return { unsupported: true, linked };
    return null;
}

function replaceLegacy(info, legacy, alias = NONE) {
    if (legacy?.prefix !== undefined) {
        info.widgets_values = [...info.widgets_values.slice(0, legacy.prefix), alias];
    } else if (legacy?.unsupported && Array.isArray(info.widgets_values)) {
        // Unknown positional layouts cannot safely be assigned to the new
        // widgets; preserve named generation values and use defaults otherwise.
        info.widgets_values = [];
    } else if (info.widgets_values && !Array.isArray(info.widgets_values)) {
        info.widgets_values.connection_alias = alias;
    }
    info.widgets_values_named ??= {};
    info.widgets_values_named.connection_alias = alias;
    info.inputs = info.inputs?.filter(input => !PRIVATE.has(input.name));
    scrubNodeFields(info);
    info.properties ??= {};
    info.properties.dkstLLMVersion = 2;
}

function graphContainers(data) {
    const found = [];
    const seen = new WeakSet();
    const visit = graph => {
        if (!graph || typeof graph !== "object" || seen.has(graph)) return;
        seen.add(graph);
        if (Array.isArray(graph.nodes)) {
            found.push(graph);
            for (const node of graph.nodes) visit(node.subgraph);
        }
        for (const child of graph.definitions?.subgraphs ?? []) visit(child);
        for (const child of Array.isArray(graph.subgraphs) ? graph.subgraphs : []) visit(child);
    };
    visit(data);
    return found;
}

function serializedNodes(data) {
    return graphContainers(data).flatMap(graph => graph.nodes.filter(node => TYPES.has(node.type)));
}

function detachPrivateInputs(data) {
    for (const value of graphContainers(data)) {
        const removedSlots = new Map();
        for (const node of value.nodes) {
            if (!TYPES.has(node.type)) continue;
            const slots = (node.inputs ?? []).flatMap((input, index) => PRIVATE.has(input.name) ? [index] : []);
            if (slots.length) removedSlots.set(String(node.id), slots);
        }
        const removedLinks = new Set();
        const keep = link => {
            const [id, target, slot] = Array.isArray(link)
                ? [link[0], link[3], link[4]] : [link.id, link.target_id, link.target_slot];
            const slots = removedSlots.get(String(target));
            if (!slots) return true;
            if (slots.includes(slot)) { removedLinks.add(String(id)); return false; }
            const adjusted = slot - slots.filter(index => index < slot).length;
            if (Array.isArray(link)) link[4] = adjusted;
            else link.target_slot = adjusted;
            return true;
        };
        if (Array.isArray(value.links)) value.links = value.links.filter(keep);
        else if (value.links && typeof value.links === "object") {
            for (const [id, link] of Object.entries(value.links)) {
                if (!keep(link)) delete value.links[id];
            }
        }
        for (const node of value.nodes) {
            if (removedSlots.has(String(node.id))) node.inputs = node.inputs.filter(input => !PRIVATE.has(input.name));
            for (const output of node.outputs ?? []) {
                if (output.links) output.links = output.links.filter(id => !removedLinks.has(String(id)));
            }
        }
    }
}

export function migrateWorkflow(data) {
    const entries = serializedNodes(data).map(info => ({ info, legacy: legacyConnection(info) }));
    // Only normalize JSON here. Network/storage failures must never abort or
    // delay graph configuration (including workflows without an LLM node).
    detachPrivateInputs(data);
    for (const { info, legacy } of entries) {
        if (!legacy) continue;
        serializedMigrations.set(info, legacy);
        replaceLegacy(info, legacy);
    }
}

export function renderConnectionManager() {
    const root = document.createElement("div");
    root.style.cssText = "display:grid;gap:10px;min-width:260px;width:100%";
    const hint = document.createElement("div");
    hint.textContent = "Connections are shared on this ComfyUI server. Workflows save only the alias. Stored API keys are never displayed.";
    const select = document.createElement("select");
    select.setAttribute("aria-label", "Registered LLM connections");
    const status = document.createElement("div");
    status.setAttribute("role", "status");
    const fields = {};
    const field = (name, label, type = "text", value = "") => {
        const row = document.createElement("label");
        row.style.cssText = "display:grid;gap:4px";
        const caption = document.createElement("span");
        caption.textContent = label;
        const input = document.createElement("input");
        input.type = type;
        input.value = value;
        input.setAttribute("aria-label", label);
        input.autocomplete = "off";
        row.append(caption, input);
        fields[name] = input;
        return row;
    };
    const alias = field("alias", "Alias");
    fields.alias.maxLength = 80;
    const host = field("ip_address", "IP address / hostname", "text", "127.0.0.1");
    const port = field("port", "Port", "number", "1234");
    fields.port.min = "1";
    fields.port.max = "65535";
    const protocol = document.createElement("select");
    protocol.setAttribute("aria-label", "Protocol");
    for (const value of ["http", "https"]) {
        const option = document.createElement("option");
        option.value = value; option.textContent = value.toUpperCase(); protocol.append(option);
    }
    protocol.value = "http";
    const key = field("api_key", "API key (leave blank to keep stored key)", "password");
    fields.api_key.autocomplete = "new-password";
    const clearKey = document.createElement("label");
    const clear = document.createElement("input");
    clear.type = "checkbox";
    clear.setAttribute("aria-label", "Remove stored API key");
    const clearLabel = document.createElement("span");
    clearLabel.textContent = "Remove stored API key";
    clearKey.append(clear, clearLabel);
    const buttons = document.createElement("div");
    buttons.style.cssText = "display:flex;gap:8px;flex-wrap:wrap";
    let profiles = [], selected = "", busy = false;
    const controls = [];
    const showSelection = () => {
        selected = select.value;
        const profile = profiles.find(item => item.alias === selected);
        fields.alias.value = profile?.alias ?? "";
        fields.alias.readOnly = Boolean(profile);
        fields.ip_address.value = profile?.ip_address ?? "127.0.0.1";
        fields.port.value = String(profile?.port ?? 1234);
        protocol.value = profile?.protocol ?? "http";
        fields.api_key.value = "";
        fields.api_key.placeholder = profile?.has_api_key ? "API key is stored" : "No API key";
        clear.checked = false;
        status.textContent = "";
    };
    const load = async keep => {
        const data = await request("connections");
        profiles = data.connections;
        select.replaceChildren();
        const fresh = document.createElement("option");
        fresh.value = ""; fresh.textContent = "New connection"; select.append(fresh);
        for (const profile of profiles) {
            const option = document.createElement("option");
            option.value = profile.alias; option.textContent = profile.alias; select.append(option);
        }
        select.value = profiles.some(item => item.alias === keep) ? keep : "";
        showSelection();
    };
    const button = (label, action) => {
        const element = document.createElement("button");
        element.type = "button"; element.textContent = label;
        element.addEventListener("click", async () => {
            if (busy) return;
            busy = true;
            for (const control of controls) control.disabled = true;
            status.textContent = "Working…";
            try { await action(); }
            catch (error) { status.textContent = error.message; }
            finally {
                busy = false;
                for (const control of controls) control.disabled = false;
            }
        });
        controls.push(element); buttons.append(element);
    };
    button("Save", async () => {
        const payload = { action: "save", alias: fields.alias.value.trim(),
            ip_address: fields.ip_address.value.trim(), port: Number(fields.port.value),
            protocol: protocol.value, overwrite: Boolean(selected) };
        if (clear.checked) payload.api_key = "";
        else if (fields.api_key.value) payload.api_key = fields.api_key.value;
        const { alias } = await request("connections", payload);
        fields.api_key.value = "";
        await refreshAliases();
        await load(alias);
        status.textContent = "Connection saved. Select its alias on the LLM node.";
    });
    button("Test saved connection", async () => {
        if (!selected) throw new Error("Save a connection first.");
        const result = await request("connections", { action: "test", alias: selected });
        status.textContent = result.message;
    });
    button("Delete", async () => {
        if (!selected) throw new Error("Select a saved connection first.");
        await request("connections", { action: "delete", alias: selected });
        await refreshAliases();
        await load("");
        status.textContent = "Connection deleted. Nodes retaining this alias must select another connection.";
    });
    button("Refresh", async () => { await load(selected); await refreshAliases(); });
    controls.push(select, protocol, clear, ...Object.values(fields));
    select.addEventListener("change", showSelection);
    root.append(hint, select, alias, host, port, protocol, key, clearKey, buttons, status);
    load("").catch(error => { status.textContent = error.message; });
    // The ComfyUI setting setter is intentionally unused: no credentials are
    // copied into its browser/general settings store.
    return root;
}

app.registerExtension({
    name: "DINKI.LLMConnections",
    settings: [{ id: "DKST.LLM.Connections", name: "LM Studio connections",
        category: ["DKST", "LLM", "Connections"], type: renderConnectionManager, defaultValue: null }],
    beforeConfigureGraph(data) { migrateWorkflow(data); },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TYPES.has(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = created?.apply(this, arguments);
            nodes.add(this);
            const widget = combo(this);
            if (widget) widget.label = "Connection alias";
            updateChoices(this);
            const refresh = this.addWidget("button", "Refresh connections", null,
                () => refreshAliases().catch(error => report(error.message)), { serialize: false });
            refresh.serialize = false;
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            nodes.delete(this);
            migrations.delete(this);
            return removed?.apply(this, arguments);
        };
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            // Also cover pasted nodes / undo, which do not invoke the graph hook.
            nodes.add(this);
            migrations.delete(this);
            let legacy;
            const prepared = serializedMigrations.get(info);
            serializedMigrations.delete(info);
            legacy = prepared ?? legacyConnection(info);
            const values = legacy?.prefix !== undefined ? info.widgets_values.slice(0, legacy.prefix) : null;
            if (legacy && !prepared) replaceLegacy(info, legacy);
            const result = configure?.apply(this, arguments);
            const named = info.widgets_values_named ?? (!Array.isArray(info.widgets_values) ? info.widgets_values : null);
            if (named && typeof named === "object") {
                for (const widget of this.widgets ?? []) {
                    if (widget.serialize !== false && Object.hasOwn(named, widget.name)) widget.value = named[widget.name];
                }
            }
            if (values) {
                const stored = this.widgets.filter(widget => widget.serialize !== false);
                values.forEach((value, index) => { if (stored[index]) stored[index].value = value; });
                setAlias(this, NONE);
            }
            // Never retain old fields in a live node's properties.
            scrubFields(this.properties);
            // configure() may have already installed the serialized input list.
            for (let index = (this.inputs?.length ?? 0) - 1; index >= 0; index--) {
                if (!PRIVATE.has(this.inputs[index].name)) continue;
                if (this.removeInput) this.removeInput(index);
                else this.inputs.splice(index, 1);
            }
            if (legacy?.linked || legacy?.unsupported) {
                setAlias(this, NONE);
                report(legacy.linked
                    ? "Old LLM connection links were disconnected. Register a connection in Settings > DKST > LLM and select its alias. Credential source nodes may still contain secrets; remove them before sharing the workflow."
                    : "This old LLM widget layout needs manual setup. Register a connection in Settings > DKST > LLM and select its alias. Unknown positional widget values were reset; named generation settings were retained.");
            } else if (legacy) {
                const pending = request("connections", { action: "migrate", ...legacy }).then(async ({ alias }) => {
                    if (migrations.get(this) !== pending) return;
                    if (combo(this)?.value === NONE) setAlias(this, alias);
                    // Alias list refresh is optional once registration succeeded.
                    await refreshAliases().catch(() => {});
                }).catch(error => {
                    if (migrations.get(this) !== pending) return;
                    report("The workflow was opened, but its LLM connection could not be migrated. Register a connection in Settings > DKST > LLM and select its alias. Previously saved files are unchanged.");
                }).finally(() => {
                    if (migrations.get(this) === pending) migrations.delete(this);
                });
                migrations.set(this, pending);
                // Graph loading never awaits this request; exports await it.
                pending.catch(() => {});
            }
            updateChoices(this);
            return result;
        };
        const serialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(info) {
            const result = serialize?.apply(this, arguments);
            scrubNodeFields(info);
            info.inputs = info.inputs?.filter(input => !PRIVATE.has(input.name));
            info.properties ??= {};
            info.properties.dkstLLMVersion = 2;
            return result;
        };
    },
    setup() {
        refreshAliases().catch(error => report(error.message));
        const exportPrompt = app.graphToPrompt;
        if (exportPrompt) app.graphToPrompt = async function(...args) {
            await Promise.all([...nodes].filter(node => node.graph).map(node => migrations.get(node)));
            return exportPrompt.apply(this, args);
        };
        // API JSON is loaded without beforeConfigureGraph in the frontend.
        const loadApi = app.loadApiJson;
        if (loadApi) app.loadApiJson = async function(data, ...args) {
            const cleaned = structuredClone(data);
            const legacy = new Map();
            for (const [id, node] of Object.entries(cleaned)) {
                if (!TYPES.has(node.class_type)) continue;
                const inputs = node.inputs ?? {};
                if ([...PRIVATE].some(name => Object.hasOwn(inputs, name))) {
                    legacy.set(id, { ip_address: inputs.ip_address ?? "127.0.0.1",
                        port: inputs.port ?? 1234, api_key: inputs.api_key ?? "" });
                    // API imports have no serialized node identity. Open them
                    // immediately with a safe alias and migrate after loading.
                    inputs.connection_alias = NONE;
                }
                for (const name of PRIVATE) delete inputs[name];
            }
            const result = await loadApi.call(this, cleaned, ...args);
            for (const [id, profile] of legacy) {
                if (Object.values(profile).some(value => Array.isArray(value))) {
                    report("The workflow was opened with its old LLM connection links removed. Register a connection in Settings > DKST > LLM and select its alias. Remove credential source nodes before sharing.");
                    continue;
                }
                const node = (app.rootGraph ?? app.graph)?.getNodeById?.(id);
                if (!node) continue;
                node.onConfigure?.({ properties: {}, widgets_values_named: profile });
            }
            return result;
        };
    },
});
