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
// The page gets a Content-Security-Policy meta tag. The visitor's OpenRouter key
// sits in browser storage (ADR 0031), so the policy limits which scripts run and
// where the page may send data. --api-base writes app/config.js for a frontend
// hosted apart from the API and admits that origin; --hosted hides the local
// Ollama option and drops localhost from the policy.
//
//   npm ci --prefix web-build
//   node web-build/build.mjs [outDir] [--api-base https://api.example] [--hosted]
import { cp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transform } from "esbuild";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const src = path.join(root, "web");

function fail(message) {
  console.error(`web build: ${message}`);
  process.exit(1);
}

const positional = [];
let apiBase = null;
let hosted = false;
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--api-base") apiBase = argv[++i] ?? fail("--api-base needs a URL");
  else if (argv[i] === "--hosted") hosted = true;
  else if (argv[i].startsWith("--")) fail(`unknown flag ${argv[i]}`);
  else positional.push(argv[i]);
}
const out = path.resolve(positional[0] ?? path.join(root, "web-dist"));

let apiOrigin = null;
if (apiBase !== null) {
  let url;
  try {
    url = new URL(apiBase);
  } catch {
    fail(`--api-base is not a URL: ${apiBase}`);
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") fail("--api-base must be http(s)");
  apiBase = apiBase.replace(/\/+$/, "");
  apiOrigin = url.origin;
}

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

// Stylesheet and font hosts: KaTeX's CSS and fonts on jsDelivr, the Google
// Fonts import in styles.css.
const STYLE_ORIGINS = ["https://cdn.jsdelivr.net", "https://fonts.googleapis.com"];
const FONT_ORIGINS = ["https://cdn.jsdelivr.net", "https://fonts.gstatic.com"];

const BABEL_TAG = /^\s*<script src="https:\/\/unpkg\.com\/@babel\/standalone@[^"]+"[^>]*><\/script>\r?\n/m;
const JSX_TAG = /<script type="text\/babel" src="(app\/[^"]+)\.jsx"><\/script>/g;
const CHARSET_TAG = /<meta charset="UTF-8" \/>\r?\n/;

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

// Every third-party script is allowed by its exact URL, not its host: a CDN host
// would admit any package published there.
const scriptUrls = [...html.matchAll(/<script[^>]*\ssrc="(https?:\/\/[^"]+)"/g)].map((m) => m[1]);
const stylesheetUrls = [...html.matchAll(/<link rel="stylesheet" href="(https?:\/\/[^"]+)"/g)].map((m) => m[1]);
const appDir = path.join(out, "app");
for (const file of (await readdir(appDir)).filter((f) => f.endsWith(".css"))) {
  const css = await readFile(path.join(appDir, file), "utf8");
  stylesheetUrls.push(...[...css.matchAll(/@import url\(['"]?(https?:\/\/[^'")]+)/g)].map((m) => m[1]));
}
for (const url of stylesheetUrls) {
  if (!STYLE_ORIGINS.includes(new URL(url).origin)) fail(`stylesheet origin missing from the policy: ${url}`);
}

const apiJs = await readFile(path.join(src, "app", "api.js"), "utf8");
function constOrigin(name) {
  const m = apiJs.match(new RegExp(`const ${name} = "([^"]+)"`));
  if (!m) fail(`app/api.js no longer defines ${name}`);
  return new URL(m[1]).origin;
}
const connect = ["'self'", apiOrigin, constOrigin("OPENROUTER_URL"), hosted ? null : constOrigin("OLLAMA_URL")];

const csp = [
  "default-src 'self'",
  `script-src 'self' ${scriptUrls.join(" ")}`,
  // KaTeX writes inline style attributes into the rendered math.
  `style-src 'self' 'unsafe-inline' ${STYLE_ORIGINS.join(" ")}`,
  `font-src 'self' data: ${FONT_ORIGINS.join(" ")}`,
  ["img-src 'self' data: blob:", apiOrigin].filter(Boolean).join(" "),
  `connect-src ${connect.filter(Boolean).join(" ")}`,
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join("; ");
if (!CHARSET_TAG.test(html)) fail("index.html no longer opens its head with the charset meta tag");
// The policy covers only what the document loads after the tag, so it goes first.
html = html.replace(CHARSET_TAG, (tag) => `${tag}  <meta http-equiv="Content-Security-Policy" content="${csp}" />\n`);

await writeFile(indexPath, html);

const config = [];
if (apiBase !== null) config.push(`window.SPECTRARAG_API_BASE=${JSON.stringify(apiBase)};`);
if (hosted) config.push("window.SPECTRARAG_HOSTED=true;");
if (config.length) await writeFile(path.join(appDir, "config.js"), `${config.join("\n")}\n`);

console.log(`web build: ${entries.length} scripts compiled into ${path.relative(root, out) || out}`);
