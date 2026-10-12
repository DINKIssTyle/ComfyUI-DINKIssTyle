import { app } from "/scripts/app.js";

const PREFIX = "DKST.Appearance.";

export const ICON_OPTIONS = ["Default", "Lock", "Circle"];
export const COLOR_OPTIONS = ["Red", "Orange", "Yellow", "Blue", "Green", "Purple", "White", "Gray", "Black"];

export const COLOR_PALETTE = {
    "Red": { fill: "#ef4444", contrast: "#ffffff" },
    "Orange": { fill: "#f97316", contrast: "#ffffff" },
    "Yellow": { fill: "#eab308", contrast: "#18181b" },
    "Blue": { fill: "#3b82f6", contrast: "#ffffff" },
    "Green": { fill: "#22c55e", contrast: "#18181b" },
    "Purple": { fill: "#a855f7", contrast: "#ffffff" },
    "White": { fill: "#ffffff", contrast: "#18181b" },
    "Gray": { fill: "#9ca3af", contrast: "#18181b" },
    "Black": { fill: "#18181b", contrast: "#ffffff" },
};

const COLOR_ALIAS = {
    "red": "Red", "빨강": "Red",
    "orange": "Orange", "주황": "Orange",
    "yellow": "Yellow", "노랑": "Yellow",
    "blue": "Blue", "파랑": "Blue",
    "green": "Green", "초록": "Green",
    "purple": "Purple", "보라": "Purple",
    "white": "White", "흰색": "White",
    "gray": "Gray", "grey": "Gray", "회색": "Gray",
    "black": "Black", "검정색": "Black", "검정": "Black",
};

const ICON_ALIAS = {
    "default": "Default", "pin": "Default", "기본": "Default",
    "lock": "Lock", "자물쇠": "Lock",
    "circle": "Circle", "dot": "Circle", "동그라미": "Circle",
};

export function normalizeIcon(value) {
    if (!value || typeof value !== "string") return "Default";
    const trimmed = value.trim();
    return ICON_ALIAS[trimmed.toLowerCase()] ?? ICON_ALIAS[trimmed] ?? "Default";
}

export function normalizeColor(value) {
    if (!value || typeof value !== "string") return "Red";
    const trimmed = value.trim();
    return COLOR_ALIAS[trimmed.toLowerCase()] ?? COLOR_ALIAS[trimmed] ?? (COLOR_PALETTE[trimmed] ? trimmed : "Red");
}

export function createPinIconSvg(styleName, colorName) {
    const icon = normalizeIcon(styleName);
    const colorKey = normalizeColor(colorName);
    const palette = COLOR_PALETTE[colorKey] ?? COLOR_PALETTE["Red"];
    const fill = palette.fill;
    const contrast = palette.contrast;

    if (icon === "Lock") {
        // Cute, simple lock with soft rounded corners
        return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">` +
            `<path d="M7.5 10V7C7.5 4.51 9.51 2.5 12 2.5C14.49 2.5 16.5 4.51 16.5 7V10" fill="none" stroke="${fill}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>` +
            `<rect x="4.5" y="9.5" width="15" height="12" rx="4" ry="4" fill="${fill}"/>` +
            `<circle cx="12" cy="14" r="1.4" fill="${contrast}"/>` +
            `<path d="M11.3 14.5H12.7L13 17.2C13 17.6 12.6 18 12.2 18H11.8C11.4 18 11 17.6 11 17.2L11.3 14.5Z" fill="${contrast}"/>` +
            `<path d="M7 12C6 12.5 5.5 13.5 5.5 15" stroke="${contrast}" stroke-width="1.2" stroke-linecap="round" opacity="0.35"/>` +
            `</svg>`;
    }

    if (icon === "Circle") {
        // Minimal round badge indicator
        return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">` +
            `<circle cx="12" cy="12" r="9" fill="${fill}"/>` +
            `<circle cx="12" cy="12" r="4.5" fill="${contrast}"/>` +
            `<circle cx="12" cy="12" r="2" fill="${fill}" opacity="0.9"/>` +
            `</svg>`;
    }

    // Default: official ComfyUI pushpin icon
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" fill="none">` +
        `<path d="M7.13727 11.2201L6.27454 14.4398" stroke="#8a8a8a" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>` +
        `<path d="M6.28085 6.68385C6.21652 6.92342 6.08664 7.1403 5.9058 7.31009C5.72497 7.47989 5.50035 7.59587 5.25721 7.645L3.95569 7.91742C3.71254 7.96655 3.48793 8.08253 3.30709 8.25233C3.12626 8.42212 2.99637 8.639 2.93204 8.87857L2.80091 9.36797C2.75515 9.53876 2.7791 9.72073 2.86751 9.87385C2.95591 10.027 3.10153 10.1387 3.27231 10.1845L10.9997 12.255C11.1705 12.3008 11.3525 12.2768 11.5056 12.1884C11.6587 12.1 11.7705 11.9544 11.8162 11.7836L11.9474 11.2942C12.0114 11.0546 12.0074 10.8018 11.9357 10.5643C11.864 10.3269 11.7274 10.1141 11.5414 9.95001L10.5505 9.06333C10.3645 8.89921 10.2279 8.68646 10.1562 8.44899C10.0845 8.21153 10.0805 7.95877 10.1446 7.71913L10.7933 5.29787C10.8391 5.12709 10.9508 4.98148 11.1039 4.89307C11.2571 4.80466 11.439 4.78071 11.6098 4.82647C11.9514 4.91799 12.3153 4.87008 12.6216 4.69327C12.9278 4.51646 13.1513 4.22523 13.2428 3.88366C13.3343 3.54209 13.2864 3.17815 13.1096 2.8719C12.9328 2.56566 12.6416 2.34219 12.3 2.25067L7.1484 0.870299C6.80683 0.778775 6.44289 0.826689 6.13665 1.0035C5.8304 1.18031 5.60694 1.47154 5.51541 1.81311C5.42389 2.15468 5.4718 2.51862 5.64861 2.82487C5.82542 3.13111 6.11665 3.35458 6.45822 3.4461C6.62901 3.49186 6.77462 3.6036 6.86302 3.75672C6.95143 3.90984 6.97539 4.09181 6.92962 4.2626L6.28085 6.68385Z" fill="${fill}"/>` +
        `</svg>`;
}

const imageCache = new Map();

export function getPinIconImage(styleName, colorName) {
    const icon = normalizeIcon(styleName);
    const color = normalizeColor(colorName);
    const key = `${icon}:${color}`;
    if (imageCache.has(key)) return imageCache.get(key);

    if (typeof Image === "undefined") return null;

    try {
        const svg = createPinIconSvg(icon, color);
        const img = new Image();
        img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
        imageCache.set(key, img);
        return img;
    } catch {
        return null;
    }
}

export function readSetting(key, fallback) {
    return app.extensionManager?.setting?.get(PREFIX + key)
        ?? app.ui?.settings?.getSettingValue(PREFIX + key, fallback)
        ?? fallback;
}

export function isNodePinned(node) {
    return Boolean(node?.pinned ?? node?.flags?.pinned);
}

export function drawFallbackVector(ctx, icon, fill, contrast, x, y, size) {
    ctx.save();
    ctx.translate(x, y);
    const scale = size / 24;
    ctx.scale(scale, scale);

    if (icon === "Lock") {
        ctx.strokeStyle = fill;
        ctx.lineWidth = 2.8;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.arc(12, 7.5, 4.5, Math.PI, 0, false);
        ctx.lineTo(16.5, 10);
        ctx.moveTo(7.5, 10);
        ctx.lineTo(7.5, 7.5);
        ctx.stroke();

        ctx.fillStyle = fill;
        if (typeof ctx.roundRect === "function") {
            ctx.beginPath();
            ctx.roundRect(4.5, 9.5, 15, 12, 4);
            ctx.fill();
        } else {
            ctx.fillRect(4.5, 9.5, 15, 12);
        }

        ctx.fillStyle = contrast;
        ctx.beginPath();
        ctx.arc(12, 14, 1.4, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(11.3, 14.5);
        ctx.lineTo(12.7, 14.5);
        ctx.lineTo(13, 17.2);
        ctx.lineTo(11, 17.2);
        ctx.closePath();
        ctx.fill();
    } else if (icon === "Circle") {
        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.arc(12, 12, 9, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = contrast;
        ctx.beginPath();
        ctx.arc(12, 12, 4.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.arc(12, 12, 2, 0, Math.PI * 2);
        ctx.fill();
    } else {
        if (typeof Path2D !== "undefined") {
            const needle = new Path2D("M7.13727 11.2201L6.27454 14.4398");
            const head = new Path2D("M6.28085 6.68385C6.21652 6.92342 6.08664 7.1403 5.9058 7.31009C5.72497 7.47989 5.50035 7.59587 5.25721 7.645L3.95569 7.91742C3.71254 7.96655 3.48793 8.08253 3.30709 8.25233C3.12626 8.42212 2.99637 8.639 2.93204 8.87857L2.80091 9.36797C2.75515 9.53876 2.7791 9.72073 2.86751 9.87385C2.95591 10.027 3.10153 10.1387 3.27231 10.1845L10.9997 12.255C11.1705 12.3008 11.3525 12.2768 11.5056 12.1884C11.6587 12.1 11.7705 11.9544 11.8162 11.7836L11.9474 11.2942C12.0114 11.0546 12.0074 10.8018 11.9357 10.5643C11.864 10.3269 11.7274 10.1141 11.5414 9.95001L10.5505 9.06333C10.3645 8.89921 10.2279 8.68646 10.1562 8.44899C10.0845 8.21153 10.0805 7.95877 10.1446 7.71913L10.7933 5.29787C10.8391 5.12709 10.9508 4.98148 11.1039 4.89307C11.2571 4.80466 11.439 4.78071 11.6098 4.82647C11.9514 4.91799 12.3153 4.87008 12.6216 4.69327C12.9278 4.51646 13.1513 4.22523 13.2428 3.88366C13.3343 3.54209 13.2864 3.17815 13.1096 2.8719C12.9328 2.56566 12.6416 2.34219 12.3 2.25067L7.1484 0.870299C6.80683 0.778775 6.44289 0.826689 6.13665 1.0035C5.8304 1.18031 5.60694 1.47154 5.51541 1.81311C5.42389 2.15468 5.4718 2.51862 5.64861 2.82487C5.82542 3.13111 6.11665 3.35458 6.45822 3.4461C6.62901 3.49186 6.77462 3.6036 6.86302 3.75672C6.95143 3.90984 6.97539 4.09181 6.92962 4.2626L6.28085 6.68385Z");
            ctx.save();
            ctx.scale(24 / 16, 24 / 16);
            ctx.strokeStyle = "#8a8a8a";
            ctx.lineWidth = 1.3;
            ctx.stroke(needle);
            ctx.fillStyle = fill;
            ctx.fill(head);
            ctx.restore();
        } else {
            ctx.strokeStyle = fill;
            ctx.lineWidth = 2.4;
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(12, 14.5);
            ctx.lineTo(12, 21.5);
            ctx.stroke();

            ctx.fillStyle = fill;
            ctx.fillRect(8.5, 2, 7, 2.5);
            ctx.beginPath();
            ctx.moveTo(9.5, 4);
            ctx.lineTo(14.5, 4);
            ctx.lineTo(17, 13.5);
            ctx.lineTo(7, 13.5);
            ctx.closePath();
            ctx.fill();
        }
    }

    ctx.restore();
}

/**
 * Draws the custom pin icon directly onto the node's title bar.
 * In LiteGraph onDrawForeground, local (0, 0) is at the top of the node,
 * and the title bar occupies y = 0 to y = titleHeight.
 */
export function drawNodePinIcon(node, ctx, isForegroundHook = false) {
    if (!isNodePinned(node) || !ctx) return;

    const icon = normalizeIcon(readSetting("PinIcon", "Default"));
    const colorKey = normalizeColor(readSetting("PinIconColor", "Red"));
    const palette = COLOR_PALETTE[colorKey] ?? COLOR_PALETTE["Red"];

    const isCollapsed = Boolean(node.flags?.collapsed);
    const globalLG = globalThis.LiteGraph;
    const titleHeight = globalLG?.NODE_TITLE_HEIGHT ?? 30;
    const width = isCollapsed
        ? (node._collapsed_width ?? globalLG?.NODE_COLLAPSED_WIDTH ?? 140)
        : (node.size ? node.size[0] : 140);

    const size = 16;
    const marginX = 8;
    // Align with vertical center of title bar
    const marginY = Math.max(3, Math.round((titleHeight - size) / 2));

    let iconX;
    let iconY;

    if (isForegroundHook) {
        // Called inside onDrawForeground: (0, 0) is node top-left
        iconX = width - size - marginX;
        iconY = marginY;
    } else {
        // Called from canvas.drawNode: apply node.pos translation
        const posX = node.pos ? node.pos[0] : 0;
        const posY = node.pos ? node.pos[1] : 0;
        iconX = posX + width - size - marginX;
        iconY = posY + marginY;
    }

    ctx.save();

    // Erase the existing default pin glyph/area with the node title background color
    const titleColor = node.color || node.bgcolor || globalThis.LiteGraph?.NODE_DEFAULT_BOXCOLOR || "#252528";
    ctx.fillStyle = titleColor;
    if (typeof ctx.roundRect === "function") {
        ctx.beginPath();
        ctx.roundRect(iconX - 3, iconY - 3, size + 6, size + 6, 3);
        ctx.fill();
    } else {
        ctx.fillRect(iconX - 3, iconY - 3, size + 6, size + 6);
    }

    const img = getPinIconImage(icon, colorKey);
    if (img && img.complete && img.naturalWidth > 0) {
        ctx.drawImage(img, iconX, iconY, size, size);
    } else {
        drawFallbackVector(ctx, icon, palette.fill, palette.contrast, iconX, iconY, size);
    }

    ctx.restore();
}

export function getSvgDataUri(svg) {
    return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

/**
 * Replaces emoji pins (📌) or pin icon elements inside the DOM / Vue frontend.
 */
export function replaceDomPinIcons(overrideIcon, overrideColor) {
    if (typeof document === "undefined" || !document.body) return;

    const icon = normalizeIcon(overrideIcon ?? readSetting("PinIcon", "Default"));
    const colorKey = normalizeColor(overrideColor ?? readSetting("PinIconColor", "Red"));
    const svg = createPinIconSvg(icon, colorKey);
    const dataUri = getSvgDataUri(svg);
    const palette = COLOR_PALETTE[colorKey] ?? COLOR_PALETTE["Red"];

    // 1. Direct inline styles on all VueNodes pin indicators to guarantee immediate render
    try {
        const pinElements = document.querySelectorAll(
            '[data-testid="node-pin-indicator"], [data-testid*="pin-indicator"], i[class*="comfy--pin"], .icon-\\[comfy--pin\\], [data-icon="pin"]'
        );
        for (const el of pinElements) {
            el.style.setProperty("background-image", `url("${dataUri}")`, "important");
            el.style.setProperty("background-size", "contain", "important");
            el.style.setProperty("background-repeat", "no-repeat", "important");
            el.style.setProperty("background-position", "center", "important");
            el.style.setProperty("background-color", "transparent", "important");
            el.style.setProperty("-webkit-mask", "none", "important");
            el.style.setProperty("-webkit-mask-image", "none", "important");
            el.style.setProperty("mask", "none", "important");
            el.style.setProperty("mask-image", "none", "important");
            el.style.setProperty("color", "transparent", "important");
            el.style.setProperty("display", "inline-block", "important");
            el.style.setProperty("width", "16px", "important");
            el.style.setProperty("height", "16px", "important");
            el.style.setProperty("font-size", "0", "important");
            el.setAttribute("data-dkst-pin", "true");
        }
    } catch {
        // Query selector safeguard
    }

    // 2. Scan and replace any text nodes containing 📌 in node headers (classic LiteGraph DOM)
    try {
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
        const textNodesToReplace = [];
        while (walker.nextNode()) {
            const node = walker.currentNode;
            if (node.nodeValue && node.nodeValue.includes("📌")) {
                textNodesToReplace.push(node);
            }
        }

        for (const textNode of textNodesToReplace) {
            const parent = textNode.parentElement;
            if (!parent) continue;
            textNode.nodeValue = textNode.nodeValue.replace(/📌/g, "").trim();

            let customSpan = parent.querySelector(".dkst-custom-pin-icon");
            if (!customSpan) {
                customSpan = document.createElement("span");
                customSpan.className = "dkst-custom-pin-icon";
                customSpan.style.display = "inline-flex";
                customSpan.style.alignItems = "center";
                customSpan.style.justifyContent = "center";
                customSpan.style.width = "16px";
                customSpan.style.height = "16px";
                customSpan.style.verticalAlign = "middle";
                customSpan.style.marginLeft = "4px";
                parent.appendChild(customSpan);
            }
            customSpan.innerHTML = svg;
        }
    } catch {
        // Non-browser or environment without createTreeWalker
    }

    // 3. Update existing custom icons
    try {
        const existing = document.querySelectorAll?.(".dkst-custom-pin-icon") ?? [];
        for (const el of existing) {
            el.innerHTML = svg;
        }
    } catch {
        // Safe fallback
    }

    // 4. Update any CSS-based icons (PrimeIcons / Lucide) in node headers
    try {
        const iconElements = document.querySelectorAll?.(".pi-thumbtack, [class*='thumbtack'], [data-icon='pin'], [data-testid*='pin']") ?? [];
        for (const el of iconElements) {
            el.style.color = palette.fill;
            el.style.fill = palette.fill;
        }
    } catch {
        // Safe fallback
    }
}

/**
 * Updates DOM style tag for CSS tokens, variables, and VueNodes background SVGs
 */
export function updateDomStyle(overrideIcon, overrideColor) {
    if (typeof document === "undefined") return;

    const icon = normalizeIcon(overrideIcon ?? readSetting("PinIcon", "Default"));
    const colorKey = normalizeColor(overrideColor ?? readSetting("PinIconColor", "Red"));
    const palette = COLOR_PALETTE[colorKey] ?? COLOR_PALETTE["Red"];
    const svg = createPinIconSvg(icon, colorKey);
    const dataUri = getSvgDataUri(svg);

    let style = document.getElementById("dkst-pin-appearance-style");
    if (!style) {
        style = document.createElement("style");
        style.id = "dkst-pin-appearance-style";
        document.head?.appendChild(style);
    }

    if (style) {
        style.textContent = `
            :root {
                --handle-color: ${palette.fill} !important;
                --pin-color: ${palette.fill} !important;
                --color-gold-600: ${palette.fill} !important;
                --dkst-pin-color: ${palette.fill} !important;
                --node-pin-color: ${palette.fill} !important;
                --comfy-pin-color: ${palette.fill} !important;
            }
            [data-testid="node-pin-indicator"],
            [data-testid*="pin-indicator"],
            i.icon-\\[comfy--pin\\],
            i[class*="icon-[comfy--pin]"],
            i[class*="comfy--pin"],
            .node-pin-indicator,
            .comfy-node-pin,
            .node-pin-icon,
            .dkst-custom-pin-icon {
                display: inline-block !important;
                width: 16px !important;
                height: 16px !important;
                min-width: 16px !important;
                min-height: 16px !important;
                background-image: url("${dataUri}") !important;
                background-size: contain !important;
                background-repeat: no-repeat !important;
                background-position: center !important;
                background-color: transparent !important;
                -webkit-mask: none !important;
                -webkit-mask-image: none !important;
                mask: none !important;
                mask-image: none !important;
                color: transparent !important;
                font-size: 0 !important;
                vertical-align: middle !important;
            }
            [data-testid="node-pin-indicator"] svg,
            [data-testid="node-pin-indicator"]::before,
            [data-testid="node-pin-indicator"]::after,
            .comfy-node-pin svg,
            .node-pin-icon svg {
                display: none !important;
            }
        `;
    }

    replaceDomPinIcons(icon, colorKey);
}

/**
 * Hooks onDrawForeground for a given nodeType prototype
 */
export function hookNodeForeground(nodeType) {
    if (!nodeType?.prototype || nodeType.prototype.__dkst_pin_fg_hooked) return;
    nodeType.prototype.__dkst_pin_fg_hooked = true;

    const origOnDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
        const result = origOnDrawForeground?.apply(this, arguments);
        if (isNodePinned(this) && ctx) {
            drawNodePinIcon(this, ctx, true);
        }
        return result;
    };
}

/**
 * Strips emoji 📌 from getTitle if present so our custom icon displays cleanly
 */
export function hookGetTitle(nodeProto) {
    if (!nodeProto || nodeProto.__dkst_get_title_hooked) return;
    nodeProto.__dkst_get_title_hooked = true;

    const origGetTitle = nodeProto.getTitle;
    if (typeof origGetTitle === "function") {
        nodeProto.getTitle = function () {
            const title = origGetTitle.apply(this, arguments);
            if (typeof title === "string" && title.includes("📌")) {
                return title.replace(/📌\s*/g, "");
            }
            return title;
        };
    }
}

let domObserver = null;

export function initDomObserver() {
    if (typeof MutationObserver === "undefined" || typeof document === "undefined" || !document.body) return;
    if (domObserver) return;

    let debounceTimer = null;
    domObserver = new MutationObserver(() => {
        if (debounceTimer) return;
        debounceTimer = setTimeout(() => {
            debounceTimer = null;
            replaceDomPinIcons();
        }, 16);
    });

    domObserver.observe(document.body, {
        childList: true,
        subtree: true,
        characterData: true,
    });
}

/**
 * Hooks LGraphNode prototype and LGraphCanvas.drawNode for comprehensive coverage
 */
export function installHooks() {
    const LGN = globalThis.LGraphNode;
    if (LGN?.prototype) {
        hookNodeForeground(LGN);
        hookGetTitle(LGN.prototype);
    }

    const LGC = globalThis.LGraphCanvas ?? app.canvas?.constructor;
    if (LGC?.prototype && !LGC.prototype.__dkst_pin_icon_hooked) {
        LGC.prototype.__dkst_pin_icon_hooked = true;
        const origDrawNode = LGC.prototype.drawNode;
        LGC.prototype.drawNode = function (node, ctx) {
            const result = origDrawNode?.apply(this, arguments);
            if (isNodePinned(node) && ctx && !node.__dkst_pin_fg_hooked) {
                drawNodePinIcon(node, ctx, false);
            }
            return result;
        };
    }
}

installHooks();
updateDomStyle();
initDomObserver();

app.registerExtension({
    name: "DINKI.Appearance",
    settings: [
        {
            id: PREFIX + "PinIcon",
            name: "Pin Icon Style",
            type: "combo",
            defaultValue: "Default",
            options: ICON_OPTIONS,
            tooltip: "Icon shape for pinned nodes (Default, Lock, Circle)",
            onChange(value) {
                updateDomStyle(value, undefined);
                app.canvas?.setDirty(true, true);
            },
        },
        {
            id: PREFIX + "PinIconColor",
            name: "Pin Icon Color",
            type: "combo",
            defaultValue: "Red",
            options: COLOR_OPTIONS,
            tooltip: "Color for pinned node icons (Red, Orange, Yellow, Blue, Green, Purple, White, Gray, Black)",
            onChange(value) {
                updateDomStyle(undefined, value);
                app.canvas?.setDirty(true, true);
            },
        },
    ],
    beforeRegisterNodeDef(nodeType) {
        hookNodeForeground(nodeType);
        if (nodeType?.prototype) hookGetTitle(nodeType.prototype);
    },
    nodeCreated(node) {
        hookNodeForeground(node.constructor);
        if (node.onDrawForeground && !node.__dkst_instance_fg_hooked) {
            node.__dkst_instance_fg_hooked = true;
            const orig = node.onDrawForeground;
            node.onDrawForeground = function (ctx) {
                const res = orig?.apply(this, arguments);
                if (isNodePinned(this) && ctx) {
                    drawNodePinIcon(this, ctx, true);
                }
                return res;
            };
        }
    },
    async setup() {
        installHooks();
        updateDomStyle();
        initDomObserver();

        // Refresh settings after backend settings load
        setTimeout(() => {
            updateDomStyle();
        }, 300);
        setTimeout(() => {
            updateDomStyle();
        }, 1000);
    },
});

