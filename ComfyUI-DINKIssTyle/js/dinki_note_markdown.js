let libraries;
function loadLibraries() {
    // .js is served as JavaScript even on systems without a .mjs MIME mapping.
    // A renderer load failure must not prevent the note extension registering.
    libraries ??= Promise.all([
        import("./vendor/marked/marked.esm.js"),
        import("./vendor/dompurify/purify.es.js"),
    ]).catch(error => {
        libraries = undefined;
        throw error;
    });
    return libraries;
}

// Restrict rendered notes to document content, including display-only GFM tasks.
const tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "br", "hr", "strong",
    "em", "del", "s", "ul", "ol", "li", "blockquote", "a", "img", "pre", "code",
    "table", "thead", "tbody", "tr", "th", "td", "input"];

export async function renderNoteMarkdown(preview, text, isCurrent = () => true) {
    const [{ marked }, { default: DOMPurify }] = await loadLibraries();
    if (!isCurrent()) return false;
    if (!DOMPurify.isSupported) {
        preview.textContent = text;
        preview.style.whiteSpace = "pre-wrap";
        return true;
    }
    preview.style.whiteSpace = "normal";
    preview.innerHTML = DOMPurify.sanitize(marked.parse(text, { gfm: true, async: false }), {
        ALLOWED_TAGS: tags,
        ALLOWED_ATTR: ["href", "src", "alt", "title", "start", "align", "type", "checked", "disabled"],
        ALLOW_DATA_ATTR: false,
        ALLOW_ARIA_ATTR: false,
    });
    for (const link of preview.querySelectorAll("a")) {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
    }
    for (const input of preview.querySelectorAll("input")) {
        input.type = "checkbox";
        input.disabled = true;
        input.tabIndex = -1;
    }
    for (const image of preview.querySelectorAll("img")) {
        image.loading = "lazy";
        image.referrerPolicy = "no-referrer";
    }
    return true;
}

export function installNoteMarkdownStyles() {
    if (document.getElementById("dkst-note-markdown-style")) return;
    const style = document.createElement("style");
    style.id = "dkst-note-markdown-style";
    style.textContent = `
        .dkst-note-preview { overflow: auto; overflow-wrap: anywhere; user-select: text; }
        .dkst-note-preview > :first-child { margin-top: 0; }
        .dkst-note-preview > :last-child { margin-bottom: 0; }
        .dkst-note-preview h1 { font-size: 1.7em; }
        .dkst-note-preview h2 { font-size: 1.4em; }
        .dkst-note-preview h3 { font-size: 1.2em; }
        .dkst-note-preview h4, .dkst-note-preview h5, .dkst-note-preview h6 { font-size: 1em; }
        .dkst-note-preview h1, .dkst-note-preview h2, .dkst-note-preview h3,
        .dkst-note-preview h4, .dkst-note-preview h5, .dkst-note-preview h6 { line-height: 1.3; margin: 1em 0 .5em; }
        .dkst-note-preview p, .dkst-note-preview ul, .dkst-note-preview ol { margin: .6em 0; }
        .dkst-note-preview ul, .dkst-note-preview ol { padding-left: 1.7em; }
        .dkst-note-preview ul { list-style: disc; }
        .dkst-note-preview ol { list-style: decimal; }
        .dkst-note-preview a { color: #91c9ff; text-decoration: underline; }
        .dkst-note-preview blockquote { margin: .8em 0; padding: 0 1em; border-left: 3px solid #777; color: #ccc; }
        .dkst-note-preview code { font-family: monospace; background: #151515; padding: .1em .3em; border-radius: 3px; }
        .dkst-note-preview pre { overflow-x: auto; padding: 10px; background: #151515; border-radius: 4px; white-space: pre; }
        .dkst-note-preview pre code { padding: 0; background: transparent; overflow-wrap: normal; }
        .dkst-note-preview table { display: block; max-width: 100%; overflow-x: auto; border-collapse: collapse; }
        .dkst-note-preview th, .dkst-note-preview td { border: 1px solid #555; padding: 6px 10px; }
        .dkst-note-preview th { background: #333; }
        .dkst-note-preview img { max-width: 100%; height: auto; }
        .dkst-note-preview hr { border: 0; border-top: 1px solid #555; margin: 1em 0; }
    `;
    document.head.appendChild(style);
}
