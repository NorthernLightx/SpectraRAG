/* APP SHELL — sidebar nav, theme toggle, tab routing */

const NAV = [
{ id: "chat", label: "Chat", icon: "chat" },
{ id: "inspection", label: "Inspection", icon: "inspect" },
{ id: "papers", label: "Papers", icon: "papers" },
{ id: "figures", label: "Figures", icon: "figures" },
{ id: "why", label: "Why multimodal?", icon: "why" }];


function Sidebar({ tab, setTab, theme, setTheme, layout, setLayout, stats, open }) {
  return (
    <aside className={"sidebar" + (open ? " open" : "")}>
      <div className="brand">
        <div className="brand-row">
          <div className="brand-mark"></div>
          <span className="brand-name">SpectraRAG</span>
        </div>
        <p className="brand-tag">Multimodal retrieval over {stats.papers || "the"} research papers. Each turn re-retrieves text <em>and</em> figures against the right context.</p>
      </div>

      <nav className="nav">
        <div className="nav-label">Workspace</div>
        {NAV.map((n) =>
        <button key={n.id} className={"nav-item" + (tab === n.id ? " active" : "")} onClick={() => setTab(n.id)}>
            <Icon name={n.icon} size={16} /> {n.label}
            {n.id === "papers" && stats.papers > 0 && <span className="count">{stats.papers}</span>}
            {n.id === "figures" && stats.figures > 0 && <span className="count">{stats.figures}</span>}
          </button>
        )}
      </nav>

      <div className="sidebar-spacer"></div>

      <div className="corpus-card">
        <h4>Index</h4>
        <div className="stat-row"><span className="k">Papers</span><span className="v">{stats.papers || "—"}</span></div>
        <div className="stat-row"><span className="k">Figures</span><span className="v">{stats.figures || "—"}</span></div>
        <hr className="divider" style={{ margin: "8px 0" }} />
        <div className="stat-row"><span className="k">Text</span><span className="v idx">bge-m3</span></div>
        <div className="stat-row"><span className="k">Embeddings</span><span className="v idx">1024-d</span></div>
      </div>

      <div className="sidebar-foot">
        <div className="theme-toggle" role="group" aria-label="Theme">
          <button className={theme === "light" ? "on" : ""} onClick={() => setTheme("light")} title="Light"><Icon name="sun" size={14} /></button>
          <button className={theme === "dark" ? "on" : ""} onClick={() => setTheme("dark")} title="Dark"><Icon name="moon" size={14} /></button>
        </div>
        <div className="foot-links">
          <a href="https://github.com/NorthernLightx/spectrarag" target="_blank" rel="noopener"><span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}><Icon name="github" size={13} /> GitHub</span></a>
          <a href="/docs" target="_blank" rel="noopener">API docs</a>
        </div>
      </div>
    </aside>);

}

const CRUMB = {
  chat: { t: "Chat", s: "Ask follow-up questions; each turn re-retrieves against the right context." },
  inspection: { t: "Inspection", s: "Trace a query through routing, retrieval, and reranking." },
  papers: { t: "Papers", s: "The 20-paper corpus, indexed by text and figure." },
  figures: { t: "Figures", s: "Every figure extracted from the corpus, searchable." },
  why: { t: "Why multimodal?", s: "Where text-only RAG breaks — and what visual retrieval recovers." }
};

const ACCENTS = {
  "#3b82f6": { a2: "#2563eb" },
  "#8b5cf6": { a2: "#7c3aed" },
  "#14b8a6": { a2: "#0d9488" },
  "#e0993a": { a2: "#c87f24" }
};
function hexToRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16 & 255}, ${n >> 8 & 255}, ${n & 255}, ${a})`;
}

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "#3b82f6",
  "density": "regular",
  "answerFont": "sans",
  "defaultLayout": "split"
} /*EDITMODE-END*/;

/* Two top-bar pills: model picker and key entry. Separate menus, because
   switching models is a mid-session action and key entry is one-time setup —
   one popover for both autofocused the password field on every model switch.
   The model menu groups the slate into free and premium. Every browser-direct
   call needs the visitor's key (OpenRouter requires auth even on :free
   models), so keyless rows hand off to the key menu; keyless chat itself runs
   on the server's demo path where the server picks the model (ADR 0027). */
function ConnectionControl({ apiKey, setApiKey, model, setModel, demoAvailable }) {
  const [menu, setMenu] = useState(null); // null | "model" | "key"
  const ref = useRef();
  const keyed = apiKey.trim().length > 0;
  const cur = window.RAG.MODELS.find((m) => m.id === model) || window.RAG.MODELS[0];
  const onDemo = !keyed && demoAvailable;
  const shortModel = onDemo ? "free (auto)" : cur.id.split("/").pop();
  const freeModels = window.RAG.MODELS.filter((m) => m.id.endsWith(":free"));
  const paidModels = window.RAG.MODELS.filter((m) => !m.id.endsWith(":free"));

  useEffect(() => {
    if (!menu) return;
    const onDown = (e) => {if (ref.current && !ref.current.contains(e.target)) setMenu(null);};
    const onEsc = (e) => {if (e.key === "Escape") setMenu(null);};
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onEsc);
    return () => {document.removeEventListener("mousedown", onDown);document.removeEventListener("keydown", onEsc);};
  }, [menu]);

  const toggle = (which) => setMenu(menu === which ? null : which);
  const modelRow = (m) =>
    <button key={m.id} className={"model-row" + (keyed && m.id === model ? " on" : "") + (keyed ? "" : " locked")}
      title={keyed ? undefined : "Needs your OpenRouter key"}
      onClick={() => (keyed ? setModel(m.id) : setMenu("key"))}>
      <span className="model-row-main">
        <span className="mono model-row-id">{m.id}</span>
        <span className="model-row-note">{m.note}</span>
      </span>
      {keyed && m.id === model && <Icon name="check" size={14} className="model-row-check" />}
    </button>;

  return (
    <div className="endpoint" ref={ref}>
      <button className={"endpoint-pill" + (menu === "model" ? " open" : "")} onClick={() => toggle("model")}>
        <span className="endpoint-model mono">{shortModel}</span>
        <Icon name="chevron" size={13} className="endpoint-caret" style={{ transform: menu === "model" ? "rotate(-90deg)" : "rotate(90deg)" }} />
      </button>
      <button className={"endpoint-pill" + (menu === "key" ? " open" : "")} onClick={() => toggle("key")}
        title={keyed ? "OpenRouter key set" : "Add your OpenRouter key"}>
        <span className={"endpoint-dot" + (keyed ? " on" : "")}></span>
        <Icon name="key" size={13} />
        <span className="endpoint-model endpoint-keylabel">{keyed ? "your key" : "add key"}</span>
      </button>

      {menu === "model" &&
      <div className="endpoint-pop rise">
          <div className="endpoint-pop-head">
            <span className="endpoint-pop-title"><Icon name="server" size={13} /> Model</span>
            <span className="endpoint-pop-sub mono">via OpenRouter</span>
          </div>
          <div className="model-list">
            {onDemo &&
            <button className="model-row on" onClick={() => setMenu(null)}>
              <span className="model-row-main">
                <span className="mono model-row-id">free (auto)</span>
                <span className="model-row-note">server picks a free vision model · daily cap</span>
              </span>
              <Icon name="check" size={14} className="model-row-check" />
            </button>
            }
            <div className="model-group-label">free · no charge to your key</div>
            {freeModels.map(modelRow)}
            <div className="model-group-label">premium · billed to your key</div>
            {paidModels.map(modelRow)}
          </div>
          {!keyed &&
          <button className="btn primary sm endpoint-cta" onClick={() => setMenu("key")}>
            Add your OpenRouter key to pick a model
          </button>
          }
        </div>
      }

      {menu === "key" &&
      <div className="endpoint-pop rise">
          <div className="endpoint-pop-head">
            <span className="endpoint-pop-title"><Icon name="key" size={13} /> OpenRouter API key</span>
            <span className="endpoint-pop-sub mono">stays in this browser</span>
          </div>
          <div className="endpoint-field">
            <input className="input" type="password" placeholder="sk-or-v1-…" value={apiKey}
          onChange={(e) => setApiKey(e.target.value)} autoFocus />
            <span className={"endpoint-keystat mono" + (keyed ? " ok" : "")}>
              <span className={"endpoint-dot" + (keyed ? " on" : "")}></span>
              {keyed ? "key stored locally · ready" : demoAvailable ? "no key · chat runs on the shared free demo model" : "add a key to run live queries"}
            </span>
            <span className="endpoint-keystat">
              Your key goes straight to OpenRouter, never to this server.{" "}
              <a href="https://openrouter.ai/settings/keys" target="_blank" rel="noopener">Create one</a>
            </span>
          </div>
        </div>
      }
    </div>);

}

/* Shown when the keyless demo hits its daily cap: the way to keep chatting is
   the visitor's own OpenRouter key. The key never touches the server — it
   lives in localStorage and goes browser-direct to OpenRouter. */
function KeyModal({ open, onSave, onClose }) {
  const [val, setVal] = useState("");
  useEffect(() => {
    if (!open) return;
    const onEsc = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [open, onClose]);
  if (!open) return null;
  const valid = val.trim().length > 0;
  const save = () => valid && onSave(val.trim());
  return (
    <div className="km-scrim" onClick={onClose}>
      <div className="km-card rise" role="dialog" aria-modal="true" aria-label="Add your OpenRouter key" onClick={(e) => e.stopPropagation()}>
        <h3>Today's free demo is used up</h3>
        <p>Answers here run on a shared free model with a daily cap, and it just ran out. Add your own OpenRouter key to keep chatting — and to switch to stronger models like GPT-4o or Claude.</p>
        <input className="input" type="password" placeholder="sk-or-v1-…" value={val} autoFocus
          onChange={(e) => setVal(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") save(); }} />
        <p className="km-note">Your key stays in this browser and goes straight to OpenRouter — it never touches this server. No key yet? <a href="https://openrouter.ai/settings/keys" target="_blank" rel="noopener">Creating one</a> takes about a minute.</p>
        <div className="km-actions">
          <button className="btn ghost" onClick={onClose}>Maybe later</button>
          <button className="btn primary" disabled={!valid} onClick={save}>Use my key</button>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [theme, setThemeRaw] = useState(() => localStorage.getItem("sr-theme") || "dark");
  const [tab, setTab] = useState(() => {
    // Deep-link support: /#inspection etc. (the legacy *.html pages redirect here).
    const h = (location.hash || "").replace(/^#/, "");
    const valid = ["chat", "inspection", "papers", "figures", "why"];
    return (valid.includes(h) && h) || localStorage.getItem("sr-tab") || "chat";
  });
  const [layout, setLayout] = useState(() => localStorage.getItem("sr-layout") || "split");
  const [navOpen, setNavOpen] = useState(false);
  const [model, setModel] = useState("openai/gpt-4o-mini");
  const [apiKey, setApiKeyRaw] = useState(() => localStorage.getItem("sr-key") || "");
  const setApiKey = (v) => {setApiKeyRaw(v);localStorage.setItem("sr-key", v);};
  const [settings, setSettings] = useState({ route: "auto", routingMode: "", topk: 5, paper: "" });
  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }));
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [papers, setPapers] = useState([]);
  const [figures, setFigures] = useState(null);
  const [pagesAvailable, setPagesAvailable] = useState(false);
  const [demoAvailable, setDemoAvailable] = useState(false);
  const [keyModalOpen, setKeyModalOpen] = useState(false);

  const setTheme = (th) => {setThemeRaw(th);localStorage.setItem("sr-theme", th);};
  useEffect(() => {document.documentElement.setAttribute("data-theme", theme);}, [theme]);
  useEffect(() => {localStorage.setItem("sr-tab", tab);}, [tab]);
  useEffect(() => {localStorage.setItem("sr-layout", layout);}, [layout]);

  // Real corpus data: the paper list (feeds the paper filter) and whether page
  // PNGs are mounted (gates vision generation). Best-effort; on failure the
  // defaults (empty list, no images) keep the UI working.
  useEffect(() => {
    window.RAG.loadPapers().then(setPapers);
    window.RAG.loadFigures().then(setFigures);
    window.RAG.loadHealth().then((h) => { setPagesAvailable(!!h.pages_available); setDemoAvailable(!!h.demo_available); });
  }, []);

  // apply tweaks → CSS
  useEffect(() => {
    const root = document.documentElement;
    const ac = ACCENTS[t.accent] || ACCENTS["#3b82f6"];
    root.style.setProperty("--accent", t.accent);
    root.style.setProperty("--accent-2", ac.a2);
    root.style.setProperty("--accent-soft", hexToRgba(t.accent, theme === "light" ? 0.10 : 0.14));
    root.style.setProperty("--accent-line", hexToRgba(t.accent, 0.34));
  }, [t.accent, theme]);
  useEffect(() => {document.documentElement.setAttribute("data-density", t.density);}, [t.density]);
  useEffect(() => {document.documentElement.setAttribute("data-answerfont", t.answerFont);}, [t.answerFont]);
  useEffect(() => {setLayout(t.defaultLayout); /* eslint-disable-next-line */}, [t.defaultLayout]);

  const crumb = CRUMB[tab];
  const stats = { papers: papers.length, figures: figures ? figures.length : 0 };
  // Tapping a nav item also dismisses the mobile drawer.
  const selectTab = (id) => { setTab(id); setNavOpen(false); };

  return (
    <div className="app">
      <Sidebar tab={tab} setTab={selectTab} theme={theme} setTheme={setTheme} layout={layout} setLayout={setLayout} stats={stats} open={navOpen} />
      {navOpen && <div className="nav-scrim" onClick={() => setNavOpen(false)}></div>}
      <main className="main">
        <div className="topbar">
          <button className="nav-burger" aria-label="Open menu" onClick={() => setNavOpen(true)}><Icon name="menu" size={18} /></button>
          <div>
            <div className="crumb"><b>{crumb.t}</b></div>
            <div className="topbar-sub">{crumb.s}</div>
          </div>
          <div className="topbar-right">
            {(tab === "chat" || tab === "inspection") &&
            <ConnectionControl apiKey={apiKey} setApiKey={setApiKey} model={model} setModel={setModel} demoAvailable={demoAvailable} />
            }
            {tab === "chat" &&
            <Segmented value={layout} onChange={setLayout}
            options={[{ value: "split", label: "split" }, { value: "single", label: "focus" }]} />
            }
            <span className="tag"><Icon name="layers" size={12} style={{ verticalAlign: -2, marginRight: 4 }} />1024-d embeddings</span>
          </div>
        </div>

        <div className="view">
          {tab === "chat" && <ChatView settings={settings} set={set} layout={layout} apiKey={apiKey} model={model} papers={papers} pagesAvailable={pagesAvailable} demoAvailable={demoAvailable} onNeedKey={() => setKeyModalOpen(true)} />}
          {tab === "inspection" && <InspectionView settings={settings} apiKey={apiKey} model={model} papers={papers} pagesAvailable={pagesAvailable} />}
          {tab === "papers" && <PapersView setTab={setTab} papers={papers} figures={figures} />}
          {tab === "figures" && <FiguresView figures={figures} />}
          {tab === "why" && <WhyView setTab={setTab} />}
        </div>
      </main>

      <KeyModal open={keyModalOpen} onClose={() => setKeyModalOpen(false)} onSave={(k) => { setApiKey(k); setKeyModalOpen(false); }} />

      <TweaksPanel>
        <TweakSection label="Brand" />
        <TweakColor label="Accent" value={t.accent}
        options={["#3b82f6", "#8b5cf6", "#14b8a6", "#e0993a"]}
        onChange={(v) => setTweak("accent", v)} />
        <TweakSection label="Layout" />
        <TweakRadio label="Chat default" value={t.defaultLayout}
        options={["split", "focus"]} onChange={(v) => setTweak("defaultLayout", v)} />
        <TweakRadio label="Density" value={t.density}
        options={["compact", "regular", "comfy"]} onChange={(v) => setTweak("density", v)} />
        <TweakSection label="Reading" />
        <TweakRadio label="Answer type" value={t.answerFont}
        options={["sans", "serif"]} onChange={(v) => setTweak("answerFont", v)} />
      </TweaksPanel>
    </div>);

}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);