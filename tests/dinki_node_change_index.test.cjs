const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_nodes.js'), 'utf8');

function fixture(count, { active = 1, mode = 'Bypass', groups, labels = [] } = {}) {
    let extension, tick, changes = 0, redraws = 0;
    const app = { registerExtension(ext) { if (ext.name === `DINKI.NodeChange${count}`) extension = ext; } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), {
        app, api: {}, queueMicrotask, setInterval(fn) {
            if (fn.name === 'syncSubgraphs') tick = fn;
            return 1;
        },
    });
    const targets = Array.from({ length: count }, (_, index) => ({ id: `branch-${index + 1}`, mode: 2 }));
    targets.push({ id: 'untouched', mode: 2 });
    const graph = { _nodes: targets.slice(), change() { changes++; }, setDirtyCanvas() {} };
    const node = { id: 'control', mode: 0, comfyClass: `DINKI_Node_Change_${count}`, graph,
        widgets: [
            ...Array.from({ length: count }, (_, index) => ({ name: `node_ids_${index + 1}`,
                value: groups?.[index] ?? `branch-${index + 1}` })),
            { name: 'active', value: active }, { name: 'disable_mode', value: mode },
            ...Array.from({ length: count }, (_, index) => ({ name: `group_${index + 1}_label`,
                value: labels[index] ?? `Group ${index + 1}` })),
        ], outputs: [], properties: {}, size: [310, 110],
        setDirtyCanvas() { redraws++; }, addOutput(name, type) { this.outputs.push({ name, type, links: [] }); },
    };
    graph._nodes.push(node);
    app.graph = app.rootGraph = graph;
    const widget = name => node.widgets.find(widget => widget.name === name);
    return { app, graph, node, targets, extension, widget, tick: () => tick(),
        changes: () => changes, redraws: () => redraws };
}

function promote(inner, node, widgetName, hostName, value) {
    const input = { name: widgetName, widget: { name: widgetName } };
    node.inputs ??= [];
    node.inputs.push(input);
    inner.inputNode ??= { slots: [] };
    inner.links ??= {};
    const id = Object.keys(inner.links).length + 1;
    inner.links[id] = { resolve: () => ({ inputNode: node, input }) };
    inner.inputNode.slots.push({ name: hostName, linkIds: [id] });
    const widget = { name: hostName, value };
    return { subgraph: inner, inputs: [{ name: hostName, _widget: widget }], widgets: [widget] };
}

for (const count of [3, 4]) {
    test(`${count} groups: every classic and Nodes 2.0 selection supports Bypass and Mute`, () => {
        for (const mode of ['Bypass', 'Mute']) {
            const f = fixture(count, { mode });
            f.extension.nodeCreated(f.node);
            const active = f.widget('active');
            // Notifications may arrive before the store commits active.value.
            for (const selected of Array.from({ length: count }, (_, index) => index + 1)) {
                f.node.onWidgetChanged('active', selected);
                assert.deepEqual(f.targets.map(target => target.mode),
                    [...Array.from({ length: count }, (_, index) => index + 1 === selected ? 0 : mode === 'Mute' ? 2 : 4), 2]);
                assert.equal(active.label, `active · Group ${selected}`);
            }
            f.node.onWidgetChanged('active', 1);
            active.callback(count);
            assert.equal(f.targets[count - 1].mode, 0);
            assert.equal(active.value, 1, 'do not replace the framework-owned value');
            const changes = f.changes(), redraws = f.redraws();
            active.callback(count);
            assert.equal(f.changes(), changes);
            assert.equal(f.redraws(), redraws);
        }
    });

    test(`${count} groups: selected IDs win over partial and full shared membership`, () => {
        const groups = [' control, branch-1, partial, common, branch-1, missing, ',
            'branch-2, partial, common', ...Array.from({ length: count - 2 }, (_, index) => `branch-${index + 3}, common`)];
        const f = fixture(count, { groups });
        const partial = { id: 'partial', mode: 2 }, common = { id: 'common', mode: 2 };
        f.graph._nodes.push(partial, common);
        f.extension.nodeCreated(f.node);
        for (const active of [1, 2, count]) {
            f.node.onWidgetChanged('active', active);
            assert.equal(partial.mode, active === count ? 4 : 0);
            assert.equal(common.mode, 0);
            assert.equal(f.node.mode, 0);
        }
        f.widget(`node_ids_${count}`).value = '';
        f.node.onWidgetChanged('active', count);
        assert.equal(common.mode, 4);
        assert.deepEqual(f.targets.map(target => target.mode), [...Array(count - 1).fill(4), 0, 2],
            'an ID removed from every group keeps its current mode');
        f.widget('active').value = count;
        f.node.onWidgetChanged(`node_ids_${count}`, 'untouched');
        assert.equal(f.targets[count].mode, 0);
    });

    test(`${count} groups: invalid indices leave target modes unchanged`, () => {
        const f = fixture(count);
        f.extension.nodeCreated(f.node);
        f.node.onWidgetChanged('active', count);
        const modes = f.targets.map(target => target.mode), changes = f.changes();
        for (const active of [0, -1, count + 1, true, false, '2', 2.5, null, undefined, NaN, Infinity]) {
            f.node.onWidgetChanged('active', active);
            assert.deepEqual(f.targets.map(target => target.mode), modes);
            assert.equal(f.changes(), changes);
        }
        f.node.graph = null;
        f.node.onWidgetChanged('active', 1);
        assert.equal(f.changes(), changes);
    });

    test(`${count} groups: restores only the INT output and preserves saved links`, async () => {
        const f = fixture(count, { active: count });
        f.node.onConfigure = info => { f.node.outputs = info.outputs; return 42; };
        f.extension.nodeCreated(f.node);
        assert.deepEqual(f.node.outputs, [{ name: 'index', type: 'INT', links: [] }]);
        const saved = [{ name: 'index', type: 'INT', links: [123] }];
        assert.equal(f.node.onConfigure({ outputs: saved }), 42);
        await Promise.resolve();
        assert.equal(f.node.outputs, saved);
        assert.equal(f.targets[count - 1].mode, 0);
        f.node.onConfigure({ outputs: [] });
        await Promise.resolve();
        assert.deepEqual(f.node.outputs, [{ name: 'index', type: 'INT', links: [] }]);
    });

    test(`${count} groups: labels and store-only root updates retain the numeric value`, () => {
        const f = fixture(count, { active: count, labels: Array(count).fill(' Custom ') });
        f.extension.nodeCreated(f.node);
        f.extension.setup();
        assert.equal(f.widget('active').label, 'active · Custom');
        const changes = f.changes();
        f.node.onWidgetChanged(`group_${count}_label`, ' Portrait ');
        assert.equal(f.widget('active').label, 'active · Portrait');
        assert.equal(f.changes(), changes);
        f.widget(`group_${count}_label`).value = '';
        f.tick();
        assert.equal(f.widget('active').label, `active · Group ${count}`);
        assert.equal(f.widget('active').value, count);
        f.widget('active').value = 2;
        f.tick();
        assert.equal(f.targets[1].mode, 0);
        assert.equal(f.targets[count - 1].mode, 4);
        const redraws = f.redraws(), after = f.changes();
        f.tick();
        assert.equal(f.redraws(), redraws);
        assert.equal(f.changes(), after);
    });

    test(`${count} groups: nested promoted indices and labels use the outermost value`, () => {
        const f = fixture(count);
        f.extension.nodeCreated(f.node);
        const inner = promote(f.graph, f.node, 'active', 'inner group', 1);
        const middle = { _nodes: [inner] };
        const outer = promote(middle, inner, 'inner group', 'outer group', count);
        f.app.rootGraph = { _nodes: [outer] };
        f.extension.setup();
        assert.equal(f.targets[count - 1].mode, 0);
        assert.equal(outer.widgets[0].label, `active · Group ${count}`);
        assert.equal(f.widget('active').value, 1);
        assert.equal(inner.widgets[0].value, 1);
        outer.widgets[0].value = 2;
        f.tick();
        assert.equal(f.targets[1].mode, 0);
        f.widget('group_2_label').value = 'Second branch';
        f.tick();
        assert.equal(outer.widgets[0].label, 'active · Second branch');
        const changes = f.changes();
        f.tick();
        assert.equal(f.changes(), changes);
        f.targets[1].mode = 4;
        f.extension.afterConfigureGraph();
        assert.equal(f.targets[1].mode, 0, 'workflow reload reapplies promoted selection');
        const root = f.app.rootGraph;
        f.app.rootGraph = { _nodes: [] };
        f.tick();
        f.targets[1].mode = 4;
        f.app.rootGraph = root;
        f.tick();
        assert.equal(f.targets[1].mode, 0, 'returning to the tab reasserts selection');
    });

    test(`${count} groups: legacy proxy promotion respects graph scope and configuration`, () => {
        for (const overlay of [false, true]) {
            const f = fixture(count);
            const widget = { name: 'renamed group', value: count };
            if (overlay) widget._overlay = { isProxyWidget: true, nodeId: 'control', widgetName: 'active' };
            const host = { subgraph: f.graph, widgets: [widget],
                properties: overlay ? {} : { proxyWidgets: [['control', 'active']] } };
            const outside = { id: 'branch-1', mode: 2 };
            f.app.graph = { _nodes: [outside] };
            f.app.rootGraph = { _nodes: [host, outside] };
            f.app.configuringGraph = true;
            f.extension.setup();
            f.tick();
            assert.equal(f.changes(), 0);
            f.app.configuringGraph = false;
            f.tick();
            assert.equal(f.targets[count - 1].mode, 0);
            assert.equal(outside.mode, 2);
            assert.equal(widget.value, count);
        }
    });

    test(`${count} groups: expanded controls return to the saved compact size`, () => {
        const f = fixture(count);
        f.node.showAdvanced = false;
        f.node.computeSize = () => [310, 350];
        f.node.setSize = function(size) { this.size = [...size]; this.onResize?.(size); };
        f.node.toggleAdvanced = function() {
            this.showAdvanced = !this.showAdvanced;
            this.setSize(this.computeSize());
        };
        f.extension.nodeCreated(f.node);
        f.node.toggleAdvanced();
        assert.equal(f.node.size[1], 350);
        f.node.toggleAdvanced();
        assert.deepEqual(f.node.size, [310, 110]);
        const info = {};
        f.node.onSerialize(info);
        assert.deepEqual(Array.from(info.size), [310, 110]);
        assert.deepEqual(Array.from(info.properties.dkstNodeChangeCompactSize), [310, 110]);
    });
}
