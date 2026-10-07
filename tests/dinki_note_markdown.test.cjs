const { test, before } = require('node:test');
const assert = require('node:assert/strict');
const { JSDOM } = require('jsdom');
const { pathToFileURL } = require('node:url');
const { join } = require('node:path');
let dom, renderNoteMarkdown, installNoteMarkdownStyles;

before(async () => {
    dom = new JSDOM('<!doctype html><html><head></head><body></body></html>');
    global.window = dom.window;
    global.document = dom.window.document;
    ({ renderNoteMarkdown, installNoteMarkdownStyles } = await import(pathToFileURL(
        join(__dirname, '../ComfyUI-DINKIssTyle/js/dinki_note_markdown.js'))));
});

async function render(text) {
    const preview = document.createElement('div');
    await renderNoteMarkdown(preview, text);
    return preview;
}

test('renders Markdown and GFM using bundled libraries', async () => {
    const preview = await render('# 한글 제목\n\n**bold** *italic* ~~deleted~~ `inline`\n\n' +
        '> quote\n\n- item\n- [x] done\n\n1. ordered\n\n' +
        '```js\nconst value = "<script>";\n```\n\n' +
        '| A | B |\n| --- | --- |\n| one | two |\n\n---\n\n' +
        '[link](https://example.com) ![picture](https://example.com/image.png)');
    assert.equal(preview.querySelector('h1').textContent, '한글 제목');
    for (const selector of ['strong', 'em', 'del', 'code', 'blockquote', 'ul', 'ol', 'pre code', 'table', 'hr']) {
        assert.ok(preview.querySelector(selector), selector);
    }
    const checkbox = preview.querySelector('input');
    assert.equal(checkbox.type, 'checkbox');
    assert.equal(checkbox.checked, true);
    assert.equal(checkbox.disabled, true);
    assert.equal(checkbox.tabIndex, -1);
    assert.equal(preview.querySelector('a').target, '_blank');
    assert.equal(preview.querySelector('a').rel, 'noopener noreferrer');
    assert.equal(preview.querySelector('img').getAttribute('alt'), 'picture');
    assert.equal(preview.querySelector('img').referrerPolicy, 'no-referrer');
    assert.match(preview.querySelector('pre').textContent, /<script>/);
    assert.equal(preview.querySelector('script'), null);
});

test('sanitizes scripts, unsafe URLs, event handlers and active HTML', async () => {
    const preview = await render('<script>alert(1)</script>\n\n' +
        '<img src="x" onerror="alert(1)">' +
        '<a href="javascript:alert(1)" onclick="alert(1)">bad</a>' +
        '<a href="data:text/html,test">data</a>' +
        '<iframe src="https://example.com"></iframe>' +
        '<svg onload="alert(1)"><a href="javascript:alert(1)">svg</a></svg>' +
        '<form id="note"><input type="text" name="text" value="evil"></form>' +
        '<p style="position:fixed" id="app" data-note="evil">text</p>');
    assert.equal(preview.querySelector('script,iframe,svg,form,style'), null);
    for (const element of preview.querySelectorAll('*')) {
        for (const attr of element.attributes) {
            assert.ok(!/^on/i.test(attr.name), attr.name);
            assert.ok(!['style', 'id', 'name', 'data-note'].includes(attr.name), attr.name);
        }
    }
    for (const link of preview.querySelectorAll('a')) {
        assert.equal(link.hasAttribute('href'), false);
    }
    assert.equal(preview.querySelector('input').type, 'checkbox');
    assert.equal(preview.querySelector('input').disabled, true);
});

test('refresh and empty notes remove earlier rendered content', async () => {
    const preview = await render('# Old');
    await renderNoteMarkdown(preview, '**New**');
    assert.equal(preview.querySelector('h1'), null);
    assert.equal(preview.querySelector('strong').textContent, 'New');
    await renderNoteMarkdown(preview, '');
    assert.equal(preview.innerHTML, '');
});

test('installs styles once and scopes rules to note previews', async () => {
    installNoteMarkdownStyles();
    installNoteMarkdownStyles();
    assert.equal(document.querySelectorAll('#dkst-note-markdown-style').length, 1);
    assert.match(document.querySelector('style').textContent, /\.dkst-note-preview pre/);
});

test('a cancelled async preview does not overwrite the current content', async () => {
    const preview = document.createElement('div');
    preview.textContent = 'current';
    const result = await renderNoteMarkdown(preview, '# stale', () => false);
    assert.equal(result, false);
    assert.equal(preview.textContent, 'current');
});
