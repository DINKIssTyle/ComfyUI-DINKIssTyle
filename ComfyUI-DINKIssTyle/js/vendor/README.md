# Note preview dependencies

Browser ESM distributions are bundled locally; no CDN or runtime npm install is required.

- Marked 18.1.0: https://github.com/markedjs/marked (MIT, `marked/LICENSE`).
  Source: https://registry.npmjs.org/marked/-/marked-18.1.0.tgz
  `lib/marked.esm.js` is included unchanged.
- DOMPurify 3.4.16: https://github.com/cure53/DOMPurify
  (Apache-2.0 OR MPL-2.0, `dompurify/LICENSE` and `dompurify/LICENSE-MPL`).
  Source: https://registry.npmjs.org/dompurify/-/dompurify-3.4.16.tgz
  `dist/purify.es.mjs` is stored as `purify.es.js` for JavaScript MIME compatibility across servers.

The modules have no extension-registration side effects. Preview loads them lazily so a dependency load failure cannot disable the note toolbar.

When upgrading, replace distributions and license files from the pinned package and run the note UI and markdown tests.
