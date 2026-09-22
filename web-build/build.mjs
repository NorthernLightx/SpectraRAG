// Production build of the SPA in web/: web-dist/ with the JSX compiled ahead of
// time and React's production UMD builds, so the browser no longer downloads
// @babel/standalone, transpiles ~180 KB of JSX on every load, and runs React's
// development build. web/ itself stays no-build for local work (`spectrarag
// serve` serves it as is).
//
// Each .jsx becomes a classic script with its top-level names intact, the same
// globals @babel/standalone produced by injecting script tags, so the files
// keep sharing components through the global scope.
//
//   npm ci --prefix web-build && node web-build/build.mjs [outDir]
import { cp, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transform } from "esbuild";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const src = path.join(root, "web");
const out = path.resolve(process.argv[2] ?? path.join(root, "web-dist"));

// Development UMD build -> production UMD build, each with its SRI hash.
const REACT_PROD = {
  "https://unpkg.com/react@18.3.1/umd/react.development.js": [
    "https://unpkg.com/react@18.3.1/umd/react.production.min.js",
    "sha384-DGyLxAyjq0f9SPpVevD6IgztCFlnMF6oW/XQGmfe+IsZ8TqEiDrcHkMLKI6fiB/Z",
  ],
  "https://unpkg.com/react-dom@18.3.1/umd/react-dom.development.js": [
    "https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js",
    "sha384-gTGxhz21lVGYNMcdJOyq01Edg0jhn/c22nsx0kyqP0TxaV5WVdsSH1fSDUf5YJj1",
  ],
};

const BABEL_TAG = /^\s*<script src="https:\/\/unpkg\.com\/@babel\/standalone@[^"]+"[^>]*><\/script>\r?\n/m;
const JSX_TAG = /<script type="text\/babel" src="(app\/[^"]+)\.jsx"><\/script>/g;

function fail(message) {
  console.error(`web build: ${message}`);
  process.exit(1);
}

await rm(out, { recursive: true, force: true });
await cp(src, out, { recursive: true });

const indexPath = path.join(out, "index.html");
let html = await readFile(indexPath, "utf8");

if (!BABEL_TAG.test(html)) fail("index.html no longer loads @babel/standalone where expected");
html = html.replace(BABEL_TAG, "");

for (const [dev, [prod, sri]] of Object.entries(REACT_PROD)) {
  const tag = new RegExp(`<script src="${dev.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"[^>]*></script>`);
  if (!tag.test(html)) fail(`index.html no longer loads ${dev}`);
  html = html.replace(tag, `<script src="${prod}" integrity="${sri}" crossorigin="anonymous"></script>`);
}

const entries = [...html.matchAll(JSX_TAG)].map((m) => m[1]);
if (entries.length === 0) fail("no text/babel scripts found in index.html");
for (const entry of entries) {
  const code = await readFile(path.join(src, `${entry}.jsx`), "utf8");
  // No format: the output stays a classic script and top-level names are kept.
  const result = await transform(code, {
    loader: "jsx",
    jsx: "transform",
    target: "es2020",
    sourcefile: `${entry}.jsx`,
  });
  await writeFile(path.join(out, `${entry}.js`), result.code);
  await rm(path.join(out, `${entry}.jsx`));
}
// defer keeps the order @babel/standalone ran them in: after parsing and after
// the deferred KaTeX scripts, one after another in document order.
html = html.replace(JSX_TAG, '<script defer src="$1.js"></script>');
if (html.includes("text/babel")) fail("a text/babel script survived the rewrite");

await writeFile(indexPath, html);
console.log(`web build: ${entries.length} scripts compiled into ${path.relative(root, out) || out}`);
