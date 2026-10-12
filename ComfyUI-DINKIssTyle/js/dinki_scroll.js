// DOM widgets share a canvas whose wheel handler can cancel native scrolling.
// Move the viewport ourselves even when an ancestor already prevented default.
function scrollMetrics(element, axis) {
    const horizontal = axis === "x";
    const position = horizontal ? "scrollLeft" : "scrollTop";
    const size = Number(element[horizontal ? "clientWidth" : "clientHeight"]) || 0;
    const extent = Number(element[horizontal ? "scrollWidth" : "scrollHeight"]) || 0;
    return { position, size, max: Math.max(0, extent - size) };
}

function moveScroll(element, axis, delta) {
    const { position, max } = scrollMetrics(element, axis);
    const before = Number(element[position]) || 0;
    element[position] = Math.max(0, Math.min(max, before + delta));
    return element[position] !== before;
}

export function bindScrollWheel(element, axis = "y", fallback) {
    element.style.overscrollBehavior = "contain";
    element.addEventListener("wheel", event => {
        if (event.ctrlKey || event.metaKey) return;
        const { size } = scrollMetrics(element, axis);
        const scale = event.deltaMode === 1 ? 20 : event.deltaMode === 2 ? size : 1;
        const delta = axis === "x" && Math.abs(event.deltaX || 0) > Math.abs(event.deltaY || 0)
            ? event.deltaX : event.deltaY || 0;
        event.preventDefault();
        event.stopPropagation();
        if (!moveScroll(element, axis, delta * scale) && fallback) moveScroll(fallback, axis, delta * scale);
    }, { passive: false });
    for (const name of ["pointerdown", "mousedown"]) {
        element.addEventListener(name, event => event.stopPropagation());
    }
}

export function createScrollControls(viewport, axis, label) {
    const row = document.createElement("div");
    row.className = "dkst-scroll-controls";
    Object.assign(row.style, { display: "flex", alignItems: "center", gap: "6px", width: "100%", minWidth: "0" });
    const previous = document.createElement("button");
    const next = document.createElement("button");
    const slider = document.createElement("input");
    previous.type = next.type = "button";
    previous.textContent = axis === "x" ? "◀" : "▲";
    next.textContent = axis === "x" ? "▶" : "▼";
    previous.title = `${label}: ${axis === "x" ? "left" : "up"}`;
    next.title = `${label}: ${axis === "x" ? "right" : "down"}`;
    previous.setAttribute("aria-label", previous.title);
    next.setAttribute("aria-label", next.title);
    for (const button of [previous, next]) {
        Object.assign(button.style, { flex: "0 0 auto", minWidth: "28px", minHeight: "24px", cursor: "pointer" });
    }
    slider.type = "range";
    slider.min = "0";
    slider.max = "1000";
    slider.step = "1";
    slider.setAttribute("aria-label", `${label}: scroll position`);
    slider.title = `${label}: drag to scroll`;
    Object.assign(slider.style, { flex: "1 1 0", width: "0", minWidth: "0", accentColor: "#3b82f6", cursor: "pointer" });
    const refresh = () => {
        const { position, max } = scrollMetrics(viewport, axis);
        const offset = Number(viewport[position]) || 0;
        slider.value = String(max ? Math.round(offset / max * 1000) : 0);
        slider.disabled = max === 0;
        previous.disabled = offset <= 0;
        next.disabled = offset >= max - 1;
    };
    const step = direction => {
        moveScroll(viewport, axis, direction * Math.max(80, scrollMetrics(viewport, axis).size * 0.8));
        refresh();
    };
    previous.addEventListener("click", () => step(-1));
    next.addEventListener("click", () => step(1));
    slider.addEventListener("input", () => {
        const { position, max } = scrollMetrics(viewport, axis);
        viewport[position] = max * Number(slider.value) / 1000;
        refresh();
    });
    for (const name of ["pointerdown", "mousedown", "click"]) row.addEventListener(name, event => event.stopPropagation());
    viewport.addEventListener("scroll", refresh);
    bindScrollWheel(viewport, axis);
    row.append(previous, slider, next);
    const observer = typeof ResizeObserver === "function" ? new ResizeObserver(refresh) : null;
    observer?.observe(viewport);
    refresh();
    return { element: row, refresh, dispose: () => observer?.disconnect() };
}
