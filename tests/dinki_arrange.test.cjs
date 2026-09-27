const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_arrange.js'), 'utf8');

function fixture({ classic = false, subgraph = false } = {}) {
    let extension;
    const history = [];
    const root = {
        beforeChange() { history.push(['before', positions()]); },
        afterChange() { history.push(['after', positions()]); },
    };
    const graph = { _nodes: [], versions: 0, incrementVersion() { this.versions++; } };
    Object.assign(graph, subgraph ? { rootGraph: root } : root);
    const positions = () => graph._nodes.map(node => [...node.pos]);
    const canvas = { graph, redraws: 0, selected_nodes: {}, setDirty() { this.redraws++; },
        select(item) { this.selectedItems.add(item); return 'selected'; },
        deselect(item) { this.selectedItems.delete(item); },
        deselectAll() { this.selectedItems.clear(); },
    };
    if (!classic) canvas.selectedItems = new Set();
    else for (const name of ['select', 'deselect', 'deselectAll']) delete canvas[name];
    const app = { graph: subgraph ? { _nodes: [] } : graph, canvas,
        registerExtension(value) { extension = value; } };
    const document = { createElement(tag) { return {
        tag, style: {}, children: [], listeners: {}, attrs: {},
        append(...children) { this.children.push(...children); },
        addEventListener(event, handler) { this.listeners[event] = handler; },
        setAttribute(name, value) { this.attrs[name] = value; },
    }; } };
    vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ''), { app, document, queueMicrotask });
    const makeNode = (id, x, y, w, h, extra = {}) => {
        const node = { id, graph, pos: [x, y], size: [w, h], flags: {},
            getBounding() {
                return [this.pos[0], this.pos[1] - 30, this.size[0], this.size[1] + 30];
            }, ...extra };
        graph._nodes.push(node);
        return node;
    };
    const control = makeNode(0, 900, 900, 320, 230, {
        comfyClass: 'DINKI_Arrange', widgets: [], expandToFitContent() {},
        addDOMWidget(name, type, element, options) {
            const widget = { name, type, element, options };
            this.widgets.push(widget);
            return widget;
        },
    });
    extension.setup();
    extension.nodeCreated(control);
    const widget = control.widgets[0];
    const status = widget.element.children.at(-1);
    const buttons = widget.element.children.slice(0, 2).flatMap(section => section.children[1].children);
    const select = (...nodes) => {
        if (classic) canvas.selected_nodes = Object.fromEntries(nodes.map(node => [node.id, node]));
        else canvas.selectedItems = new Set(nodes);
        control.dkstArrangePanel.update();
    };
    const click = title => {
        const button = buttons.find(button => button.textContent === title);
        assert.equal(button.disabled, false, `${title} should be enabled`);
        button.listeners.click();
    };
    return { app, graph, canvas, control, extension, widget, makeNode, select, click, buttons, status, history };
}

for (const [action, coordinate] of [
    ['Left', box => box[0]], ['Center', box => box[0] + box[2] / 2],
    ['Right', box => box[0] + box[2]], ['Top', box => box[1]],
    ['Middle', box => box[1] + box[3] / 2], ['Bottom', box => box[1] + box[3]],
]) {
    test(`${action} aligns visible bounds of unequal-size nodes in one undo transaction`, () => {
        const f = fixture();
        const nodes = [f.makeNode(1, -100, -20, 80, 60),
            f.makeNode(2, 90, 150, 220, 100), f.makeNode(3, 20, 55, 140, 230)];
        const before = nodes.map(node => [...node.pos]);
        const sizes = nodes.map(node => [...node.size]);
        f.select(...nodes, f.control);
        const selection = [...f.canvas.selectedItems];
        f.click(action);
        const coordinates = nodes.map(node => coordinate(node.getBounding()));
        assert.ok(coordinates.every(value => Math.abs(value - coordinates[0]) < 0.0001));
        const unchangedAxis = ['Left', 'Center', 'Right'].includes(action) ? 1 : 0;
        nodes.forEach((node, i) => {
            assert.equal(node.pos[unchangedAxis], before[i][unchangedAxis]);
            assert.deepEqual(node.size, sizes[i]);
        });
        assert.deepEqual([...f.canvas.selectedItems], selection);
        assert.deepEqual(f.control.pos, [900, 900]);
        assert.deepEqual(f.history.map(entry => entry[0]), ['before', 'after']);
        assert.deepEqual(f.history[0][1].slice(1), before);
        assert.equal(f.graph.versions, 1);
        f.click(action);
        assert.equal(f.history.length, 2, 'an unchanged layout must not create another undo entry');
        assert.equal(f.status.textContent, 'Already arranged.');
    });
}

for (const [action, axis, length] of [['Horizontally', 0, 2], ['Vertically', 1, 3]]) {
    test(`${action} equalizes edge gaps and preserves endpoints regardless of selection order`, () => {
        const f = fixture({ classic: true, subgraph: true });
        const nodes = [f.makeNode(1, 0, 0, 80, 40), f.makeNode(2, 150, 180, 140, 90),
            f.makeNode(3, 700, 800, 60, 70)];
        const start = nodes[0].getBounding()[axis];
        const last = nodes[2].getBounding();
        const end = last[axis] + last[length];
        const untouched = nodes.map(node => node.pos[1 - axis]);
        f.select(nodes[2], nodes[0], nodes[1]);
        f.click(action);
        const boxes = nodes.map(node => node.getBounding());
        const gaps = [boxes[1][axis] - boxes[0][axis] - boxes[0][length],
            boxes[2][axis] - boxes[1][axis] - boxes[1][length]];
        assert.equal(gaps[0], gaps[1]);
        assert.equal(boxes[0][axis], start);
        assert.equal(boxes[2][axis] + boxes[2][length], end);
        assert.deepEqual(nodes.map(node => node.pos[1 - axis]), untouched);
        assert.deepEqual(f.history.map(entry => entry[0]), ['before', 'after']);
    });
}

test('overlapping selections expand without negative gaps', () => {
    const f = fixture();
    const nodes = [f.makeNode(1, 0, 0, 160, 60), f.makeNode(2, 40, 0, 100, 60),
        f.makeNode(3, 80, 0, 200, 60)];
    f.select(...nodes);
    f.click('Horizontally');
    assert.equal(nodes[0].pos[0], 0);
    assert.equal(nodes[1].pos[0] - nodes[0].pos[0], nodes[0].size[0]);
    assert.equal(nodes[2].pos[0] - nodes[1].pos[0], nodes[1].size[0]);
});

test('Evenly builds a nonoverlapping grid at the selection origin, using spatial order', () => {
    const f = fixture();
    const nodes = [f.makeNode(1, 20, 0, 80, 60), f.makeNode(2, 40, 5, 200, 120),
        f.makeNode(3, 60, 10, 130, 70), f.makeNode(4, 10, 20, 100, 160),
        f.makeNode(5, 50, 25, 160, 80)];
    f.select(...nodes.toReversed());
    f.click('Evenly');
    assert.equal(nodes[0].pos[0], 10);
    assert.equal(nodes[0].getBounding()[1], -30);
    assert.equal(nodes[1].pos[0] - nodes[0].pos[0], 240);
    assert.equal(nodes[2].pos[0] - nodes[1].pos[0], 240);
    assert.equal(nodes[3].pos[0], nodes[0].pos[0]);
    assert.equal(nodes[3].pos[1] - nodes[0].pos[1], 230);
    assert.equal(nodes[4].pos[0], nodes[1].pos[0]);
    assert.equal(nodes[3].pos[1], nodes[4].pos[1]);
});

test('visible bounds account for collapsed nodes and different title offsets', () => {
    const f = fixture();
    const first = f.makeNode(1, 10, 80, 200, 240);
    const collapsed = f.makeNode(2, 100, 200, 300, 400, {
        getBounding() { return [this.pos[0], this.pos[1] - 24, 100, 24]; },
    });
    f.select(first, collapsed);
    f.click('Bottom');
    assert.equal(first.getBounding()[1] + first.getBounding()[3],
        collapsed.getBounding()[1] + collapsed.getBounding()[3]);
    f.click('Right');
    assert.equal(first.getBounding()[0] + first.getBounding()[2],
        collapsed.getBounding()[0] + collapsed.getBounding()[2]);
});

test('filters pinned nodes, groups, reroutes, invalid geometry, other Arrange nodes and stale selections', () => {
    const f = fixture();
    const a = f.makeNode(1, 0, 0, 100, 100), b = f.makeNode(2, 200, 100, 100, 100);
    const pinned = f.makeNode(3, 400, 50, 100, 100, { flags: { pinned: true } });
    const modernPinned = f.makeNode(4, 600, 80, 100, 100, { pinned: true });
    const otherControl = f.makeNode(5, 800, 80, 100, 100, { type: 'DINKI_Arrange' });
    const invalid = f.makeNode(6, NaN, 80, 100, 100);
    const group = { id: 9, pos: [-200, -200], size: [100, 100] };
    f.select(a, b, pinned, modernPinned, otherControl, invalid, group, { ...a }, f.control);
    assert.equal(f.status.textContent, '2 nodes selected');
    f.click('Left');
    assert.equal(b.pos[0], 0);
    assert.equal(pinned.pos[0], 400);
    assert.equal(modernPinned.pos[0], 600);
    assert.equal(otherControl.pos[0], 800);
    assert.deepEqual(group.pos, [-200, -200]);
    f.canvas.graph = { _nodes: [] };
    f.control.dkstArrangePanel.update();
    assert.ok(f.buttons.every(button => button.disabled));
});

test('selection updates button availability without selecting the controller and cleanup stops updates', async () => {
    const f = fixture();
    assert.ok(f.buttons.every(button => button.disabled));
    const a = f.makeNode(1, 0, 0, 100, 100), b = f.makeNode(2, 200, 100, 100, 100);
    assert.equal(f.canvas.select(a), 'selected');
    f.canvas.select(b);
    await Promise.resolve();
    assert.equal(f.status.textContent, '2 nodes selected');
    assert.equal(f.buttons.find(button => button.textContent === 'Left').disabled, false);
    assert.equal(f.buttons.find(button => button.textContent === 'Horizontally').disabled, true);
    const wrapped = f.canvas.select;
    f.extension.setup();
    assert.equal(f.canvas.select, wrapped);
    assert.equal(f.widget.options.selectOn.length, 0);
    assert.equal(f.widget.serialize, false);
    assert.equal(f.widget.options.serialize, false);
    let stopped = 0;
    for (const event of ['pointerdown', 'pointerup', 'click']) {
        f.widget.element.listeners[event]({ stopPropagation() { stopped++; } });
    }
    assert.equal(stopped, 3);
    assert.equal(f.canvas.selectedItems.has(f.control), false);
    f.control.onRemoved();
    f.canvas.deselectAll();
    await Promise.resolve();
    assert.equal(f.status.textContent, '2 nodes selected');
    assert.equal(f.control.dkstArrangePanel, undefined);
});

test('classic callbacks keep their context, arguments and return value', async () => {
    const f = fixture({ classic: true });
    delete f.canvas.__dkstArrangeAttached;
    let seen;
    f.canvas.onSelectionChange = function (...args) { seen = [this, ...args]; return 42; };
    f.extension.setup();
    assert.equal(f.canvas.onSelectionChange('selection', 'extra'), 42);
    await Promise.resolve();
    assert.deepEqual(seen, [f.canvas, 'selection', 'extra']);
});
