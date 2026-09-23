/* APP SHELL: sidebar nav, theme toggle, tab routing */

const NAV = [
{ id: "chat", label: "Chat", icon: "chat" },
{ id: "inspection", label: "Inspection", icon: "inspect" },
{ id: "papers", label: "Papers", icon: "papers" },
{ id: "figures", label: "Figures", icon: "figures" },
{ id: "why", label: "Why multimodal?", icon: "why" }];


function Sidebar({ tab, setTab, theme, setTheme, stats, open }) {
  return (
    <aside className={"sidebar" + (open ? " open" : "")}>
      <div className="brand">
        <div className="brand-row">
          <div className="brand-mark"></div>
          <span className="brand-name">SpectraRAG</span>
        </div>
        <p className="brand-tag">Multimodal retrieval over {stats.papers || "the"} research papers.</p>
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

      <div className="sidebar-foot">
        <div className="theme-toggle" role="group" aria-label="Theme">
          <button className={theme === "light" ? "on" : ""} onClick={() => setTheme("light")} title="Light"><Icon name="sun" size={14} /></button>
          <button className={theme === "dark" ? "on" : ""} onClick={() => setTheme("dark")} title="Dark"><Icon name="moon" size={14} /></button>
        </div>
        <div className="foot-links">
          <a href="https://github.com/NorthernLightx/spectrarag" target="_blank" rel="noopener"><span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}><Icon name="github" size={13} /> GitHub</span></a>
          <a href={(window.SPECTRARAG_API_BASE || "") + "/docs"} target="_blank" rel="noopener">API docs</a>
        </div>
      </div>
    </aside>);

}

const CRUMB = {
  chat: { t: "Chat", s: "Answers cite the passages and pages they came from." },
  inspection: { t: "Inspection", s: "Trace a query through routing, retrieval, and reranking." },
  papers: { t: "Papers", s: "The 20-paper corpus, indexed by text and figure." },
  figures: { t: "Figures", s: "Every figure extracted from the corpus, searchable." },
  why: { t: "Why multimodal?", s: "Where text-only RAG breaks, and what visual retrieval recovers." }
};

/* Two top-bar pills: model picker and key entry. Separate menus, because
   switching models is a mid-session action and key entry is one-time setup.
   One popover for both autofocused the password field on every model switch.
   The model menu switches between the two providers (ADR 0031): OpenRouter
   (browser-direct with the visitor's key, required even on :free models, so
   keyless rows hand off to the key menu) and a local Ollama, whose vision
   models are listed live from /api/tags and need no key at all. */
/* Hosted deploys bake window.SPECTRARAG_HOSTED=true into config.js: the page
   is served from a public origin, so a visitor's local Ollama is unreachable
   (its CORS only allows localhost origins) and the provider toggle is hidden. */
const HOSTED = !!window.SPECTRARAG_HOSTED;

function ConnectionControl({ apiKey, setApiKey, rememberKey, setRememberKey, provider, setProvider, model, setModel }) {
  const [menu, setMenu] = useState(null); // null | "model" | "key"
  const [orList, setOrList] = useState(undefined); // undefined=unfetched, null=failed, []=vision models
  const [ollama, setOllama] = useState(null); // null=probing, {ok, models}
  const [pull, setPull] = useState(null); // null | { id, status?, pct?, error? }
  const [q, setQ] = useState("");
  const ref = useRef();
  const keyed = apiKey.trim().length > 0;
  const shortModel = provider === "ollama" ? (model || "pick a model") : model.split("/").pop();

  useEffect(() => {
    if (!menu) return;
    const onDown = (e) => {if (ref.current && !ref.current.contains(e.target)) setMenu(null);};
    const onEsc = (e) => {if (e.key === "Escape") setMenu(null);};
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onEsc);
    return () => {document.removeEventListener("mousedown", onDown);document.removeEventListener("keydown", onEsc);};
  }, [menu]);

  // Fetch the open pane's model list lazily: the OpenRouter catalog once per
  // page load, the Ollama probe on every open (it's local and instant, and the
  // user may have started Ollama since the last look).
  useEffect(() => {
    if (menu !== "model") return;
    if (provider === "openrouter" && orList === undefined) {
      window.RAG.loadOpenRouterModels().then(setOrList);
    }
    if (provider === "ollama") {
      window.RAG.loadOllamaModels(true).then(setOllama);
    }
  }, [menu, provider]);

  // Visitors who will generate on OpenRouter get its catalog up front, so a
  // withdrawn model is replaced before the first question rather than after it
  // fails. Keyless visitors make no request to OpenRouter.
  useEffect(() => {
    if (provider === "openrouter" && keyed && orList === undefined) window.RAG.loadOpenRouterModels().then(setOrList);
  }, [provider, keyed]);

  // A saved or default OpenRouter model the loaded catalog no longer lists
  // gives way to a pin it still lists.
  useEffect(() => {
    if (provider !== "openrouter") return;
    const next = window.RAG.usableModel(model, orList);
    if (next !== model) setModel(next);
  }, [provider, model, orList]);

  // Probe Ollama at mount too (all local calls): a persisted model that was
  // retired upstream since the last visit gets swapped for a usable one
  // before the first ask, not only when the menu happens to open.
  useEffect(() => {
    if (provider === "ollama") window.RAG.loadOllamaModels().then(setOllama);
  }, [provider]);

  // First Ollama pick is automatic: the provider is unusable without a model.
  // Also fires when the persisted pick is no longer listed (retired cloud
  // model, or removed with `ollama rm`). The first usable model takes over
  // instead of erroring at generation.
  useEffect(() => {
    if (provider !== "ollama" || !ollama || !ollama.ok || ollama.models.length === 0) return;
    if (model && ollama.models.some((m) => m.id === model)) return;
    setModel(ollama.models[0].id);
  }, [provider, model, ollama]);

  const toggle = (which) => setMenu(menu === which ? null : which);

  // One download at a time; the component outlives the popover, so a pull
  // keeps streaming (and the progress % stays current) if the menu closes.
  // On success the fresh tags list replaces the pane and the new model
  // becomes the selection.
  const startPull = (id) => {
    if (pull && !pull.error) return;
    setPull({ id, status: "starting", pct: null });
    window.RAG.pullOllamaModel(id, (p) => setPull({ id, status: p.status, pct: p.pct }))
      .then(() => window.RAG.loadOllamaModels(true))
      .then((res) => {
        setPull(null);
        setOllama(res);
        if (res.ok && res.models.some((m) => m.id === id)) setModel(id);
      })
      .catch((e) => setPull({ id, error: (e && e.message) || "pull failed, retry" }));
  };

  const orRow = (m) =>
    <button key={m.id} className={"model-row" + (keyed && m.id === model ? " on" : "") + (keyed ? "" : " locked")}
      title={keyed ? undefined : "Needs your OpenRouter key"}
      onClick={() => (keyed ? setModel(m.id) : setMenu("key"))}>
      <span className="model-row-main">
        <span className="mono model-row-id">{m.id}</span>
        <span className="model-row-note">{m.note}</span>
      </span>
      {keyed && m.id === model && <Icon name="check" size={14} className="model-row-check" />}
    </button>;

  const pinnedIds = new Set(window.RAG.PINNED.map((m) => m.id));
  const query = q.trim().toLowerCase();
  const fetched = (orList || [])
    .filter((m) => !pinnedIds.has(m.id))
    .filter((m) => !query || m.id.toLowerCase().includes(query) || m.name.toLowerCase().includes(query))
    .map((m) => ({ id: m.id, note: `${m.free ? "free · " : ""}${m.ctx ? Math.round(m.ctx / 1000) + "k ctx" : "vision"}` }));

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
            <span className="endpoint-pop-sub mono">{provider === "ollama" ? "localhost:11434" : "via OpenRouter"}</span>
          </div>
          {!HOSTED &&
          <div className="endpoint-provider">
            <Segmented value={provider} onChange={setProvider} options={[
              { value: "openrouter", label: "OpenRouter" },
              { value: "ollama", label: "Ollama (local)" },
            ]} />
          </div>}
          {provider === "openrouter" &&
          <React.Fragment>
            <div className="model-list">
              <div className="model-group-label"><span className="label-info">Suggested</span></div>
              {window.RAG.livePins(orList).map(orRow)}
              <div className="model-group-label">
                <span className="label-info">
                  {orList === undefined ? "Loading the full list…"
                    : orList === null ? "Couldn't load the full list, showing the shortlist"
                    : `All vision models · ${orList.length}`}
                </span>
              </div>
              {Array.isArray(orList) && orList.length > 0 &&
              <input className="input model-search" placeholder="Search models…" value={q}
                onChange={(e) => setQ(e.target.value)} />
              }
              {fetched.map(orRow)}
              {Array.isArray(orList) && query && fetched.length === 0 &&
              <div className="model-group-label"><span className="label-info">No match for “{q.trim()}”</span></div>
              }
            </div>
            {!keyed &&
            <button className="btn primary sm endpoint-cta" onClick={() => setMenu("key")}>
              Add your OpenRouter key to pick a model
            </button>
            }
          </React.Fragment>
          }
          {provider === "ollama" &&
          <div className="model-list">
            {ollama === null &&
            <div className="model-group-label"><span className="label-info">Looking for Ollama…</span></div>
            }
            {ollama && !ollama.ok &&
            <React.Fragment>
              <div className="model-group-label"><span className="label-info">Ollama isn't running at localhost:11434.</span></div>
              <button className="btn ghost sm endpoint-cta" onClick={() => { setOllama(null); window.RAG.loadOllamaModels(true).then(setOllama); }}>
                Retry
              </button>
            </React.Fragment>
            }
            {ollama && ollama.ok && ollama.models.length === 0 &&
            <div className="model-group-label"><span className="label-info">No vision models installed. Pull one below.</span></div>
            }
            {ollama && ollama.ok && ollama.models.length > 0 &&
            <React.Fragment>
              <div className="model-group-label"><span className="label-info">Local vision models · {ollama.models.length}</span></div>
              {ollama.models.map((m) =>
                <button key={m.id} className={"model-row" + (m.id === model ? " on" : "")}
                  onClick={() => setModel(m.id)}>
                  <span className="model-row-main">
                    <span className="mono model-row-id">{m.id}</span>
                    <span className="model-row-note">{m.note}</span>
                  </span>
                  {m.id === model && <Icon name="check" size={14} className="model-row-check" />}
                </button>
              )}
            </React.Fragment>
            }
            {ollama && ollama.ok && (() => {
              const installed = new Set(ollama.models.map((m) => m.id));
              const sugg = window.RAG.OLLAMA_SUGGESTED.filter((s) => !installed.has(s.id));
              if (sugg.length === 0) return null;
              return (
                <React.Fragment>
                  <div className="model-group-label"><span className="label-info">Pull a vision model</span></div>
                  {sugg.map((s) => {
                    const active = pull && pull.id === s.id;
                    const busyOther = pull && !pull.error && !active;
                    return (
                      <button key={s.id} className={"model-row" + (busyOther ? " locked" : "")}
                        title={active ? undefined : `Download ${s.id} into Ollama`}
                        onClick={() => !busyOther && startPull(s.id)}>
                        <span className="model-row-main">
                          <span className="mono model-row-id">{s.id}</span>
                          <span className="model-row-note">
                            {active
                              ? (pull.error || `${pull.status}${pull.pct != null ? ` · ${pull.pct}%` : ""}`)
                              : s.note}
                          </span>
                        </span>
                        {!active && <Icon name="plus" size={14} className="model-row-check" />}
                      </button>
                    );
                  })}
                </React.Fragment>
              );
            })()}
          </div>
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
              {keyed ? (rememberKey ? "key saved on this device · ready" : "key kept for this tab · ready") : "add a key to pick an OpenRouter model"}
            </span>
            <label className="endpoint-keystat">
              <input type="checkbox" checked={rememberKey} onChange={(e) => setRememberKey(e.target.checked)} />
              Remember on this device
            </label>
            {keyed && <button className="btn ghost sm" style={{ alignSelf: "flex-start" }} onClick={() => setApiKey("")}>Forget key</button>}
            <span className="endpoint-keystat">
              Chat sends your key straight to OpenRouter. Agentic search sends it to this server, which uses it for that request and does not keep it. Use a key with a credit limit.{" "}
              <a href="https://openrouter.ai/settings/keys" target="_blank" rel="noopener">Create one</a>
            </span>
          </div>
        </div>
      }
    </div>);

}

/* Shown when a turn needs the visitor's own OpenRouter key. Today that's
   agentic search, which runs server-side on it. The server uses the key for
   that request only; the browser keeps it (ADR 0031 amendment). */
const KEY_MODAL_COPY = {
  chat: {
    h: "Add a key for cited answers",
    p: "Search works without a key. With an OpenRouter key, the model you pick reads what search found and writes an answer that cites it.",
  },
  agentic: {
    h: "Agentic search needs your key",
    p: HOSTED
      ? "The search agent runs server-side on your OpenRouter key. Add one to try it. Regular chat also uses your key, straight from the browser."
      : "The search agent runs server-side on your OpenRouter key. Add one to try it. Regular chat works without it, on Ollama or your own OpenRouter models.",
  },
};
function KeyModal({ open, onSave, onClose }) {
  const [val, setVal] = useState("");
  // Reset on every open: a stale half-typed key from a previous open must not
  // be one Enter press away from becoming the active key.
  useEffect(() => { if (open) setVal(""); }, [open]);
  useEffect(() => {
    if (!open) return;
    const onEsc = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [open, onClose]);
  if (!open) return null;
  const copy = KEY_MODAL_COPY[open] || KEY_MODAL_COPY.agentic;
  const valid = val.trim().length > 0;
  const save = () => valid && onSave(val.trim());
  return (
    <div className="km-scrim" onClick={onClose}>
      <div className="km-card rise" role="dialog" aria-modal="true" aria-label="Add your OpenRouter key" onClick={(e) => e.stopPropagation()}>
        <h3>{copy.h}</h3>
        <p>{copy.p}</p>
        <input className="input" type="password" placeholder="sk-or-v1-…" value={val} autoFocus
          onChange={(e) => setVal(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") save(); }} />
        <p className="km-note">This server uses your key for the agent's request and does not keep it; chat sends it straight from this browser to OpenRouter. It lasts for this tab unless you tick "Remember on this device" in the key menu. Use a key with a credit limit. No key yet? <a href="https://openrouter.ai/settings/keys" target="_blank" rel="noopener">Creating one</a> takes about a minute.</p>
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
    // Deep-link support: /#inspection and the like (the legacy *.html pages
    // redirect here).
    const h = (location.hash || "").replace(/^#/, "");
    const valid = ["chat", "inspection", "papers", "figures", "why"];
    return (valid.includes(h) && h) || localStorage.getItem("sr-tab") || "chat";
  });
  const [navOpen, setNavOpen] = useState(false);
  // Provider + model survive reloads, with one remembered model per provider
  // so switching back doesn't clobber the other side's choice.
  const modelStoreKey = (p) => (p === "ollama" ? "sr-ollama-model" : "sr-or-model");
  const defaultModel = (p) => (p === "ollama" ? "" : "openai/gpt-4o-mini");
  // A persisted "ollama" pick is ignored on hosted pages (toggle is hidden there).
  const storedProvider = () => (HOSTED ? "openrouter" : localStorage.getItem("sr-provider") || "openrouter");
  const [provider, setProviderRaw] = useState(storedProvider);
  const [model, setModelRaw] = useState(() => {
    const p = storedProvider();
    return localStorage.getItem(modelStoreKey(p)) || defaultModel(p);
  });
  const setModel = (v) => {setModelRaw(v);localStorage.setItem(modelStoreKey(provider), v);};
  const setProvider = (p) => {
    setProviderRaw(p);
    localStorage.setItem("sr-provider", p);
    setModelRaw(localStorage.getItem(modelStoreKey(p)) || defaultModel(p));
  };
  // The OpenRouter key lasts for this tab (sessionStorage) unless the visitor
  // ticks "Remember on this device" (localStorage). ADR 0031 amendment.
  const [rememberKey, setRememberKeyRaw] = useState(() => !!localStorage.getItem("sr-key"));
  const [apiKey, setApiKeyRaw] = useState(() => sessionStorage.getItem("sr-key") || localStorage.getItem("sr-key") || "");
  const storeKey = (v, remember) => {
    localStorage.removeItem("sr-key");
    sessionStorage.removeItem("sr-key");
    if (v) (remember ? localStorage : sessionStorage).setItem("sr-key", v);
  };
  const setApiKey = (v) => {setApiKeyRaw(v);storeKey(v, rememberKey);};
  const setRememberKey = (r) => {setRememberKeyRaw(r);storeKey(apiKey, r);};
  const [settings, setSettings] = useState({ route: "auto", routingMode: "", topk: 5, paper: "" });
  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }));
  const [papers, setPapers] = useState([]);
  const [figures, setFigures] = useState(null);
  const [pagesAvailable, setPagesAvailable] = useState(false);
  const [routingAvailable, setRoutingAvailable] = useState(true);
  const [uploadAvailable, setUploadAvailable] = useState(false);
  const [keyModalOpen, setKeyModalOpen] = useState(false);
  // "probing" | "cold" | "ready" | "down". Drives the cold-start banner.
  const [backendWarm, setBackendWarm] = useState("probing");

  const setTheme = (th) => {setThemeRaw(th);localStorage.setItem("sr-theme", th);};
  useEffect(() => {document.documentElement.setAttribute("data-theme", theme);}, [theme]);
  useEffect(() => {
    localStorage.setItem("sr-tab", tab);
    // Keep the hash in sync. A stale deep-link hash would otherwise override
    // the saved tab on every reload.
    if ((location.hash || "").replace(/^#/, "") !== tab) history.replaceState(null, "", "#" + tab);
  }, [tab]);

  // Real corpus data: the paper list (feeds the paper filter) and whether page
  // PNGs are mounted (gates vision generation). Best-effort; on failure the
  // defaults (empty list, no images) keep the UI working.
  useEffect(() => {
    // The API scales to zero, and a cold instance answers /health only after
    // startup init (model loading), 2 to 5 minutes. Until then requests fail
    // or come back as 5xx from Cloud Run's front end. No reply within a few
    // seconds means a cold start is in progress, so show the banner instead of
    // silently empty tabs, and keep polling until the backend answers.
    const coldTimer = setTimeout(() => setBackendWarm((s) => (s === "probing" ? "cold" : s)), 3000);
    window.RAG.loadPapers().then(setPapers);
    window.RAG.loadFigures().then(setFigures);
    // routing_available must be POSITIVELY confirmed: an older server without
    // the field should not leave routing controls offered on a deployment that
    // can't honor them.
    window.RAG.waitForHealth().then((h) => {
      clearTimeout(coldTimer);
      if (!h) {
        setBackendWarm("down");
        return;
      }
      setBackendWarm("ready");
      setPagesAvailable(!!h.pages_available); setRoutingAvailable(h.routing_available === true); setUploadAvailable(!!h.upload_available);
      // /health resolving proves the backend is warm. The parallel corpus
      // fetches above can burn all their retries inside the cold-start window
      // (they give up in seconds; the cold hold lasts minutes) and would leave
      // the session with an empty Papers tab and id-only source labels until a
      // manual reload, so re-fetch whatever came back empty.
      window.RAG.loadPapers().then((p) => setPapers((prev) => (prev.length ? prev : p)));
      window.RAG.loadFigures().then((f) => setFigures((prev) => (prev && prev.length ? prev : f)));
    });
    return () => clearTimeout(coldTimer);
  }, []);

  const crumb = CRUMB[tab];
  const stats = { papers: papers.length, figures: figures ? figures.length : 0 };
  // Tapping a nav item also dismisses the mobile drawer.
  const selectTab = (id) => { setTab(id); setNavOpen(false); };
  // Re-fetch the corpus after an upload so the new paper + figures appear.
  const reloadCorpus = () => { window.RAG.loadPapers().then(setPapers); window.RAG.loadFigures().then(setFigures); };

  return (
    <div className="app">
      <Sidebar tab={tab} setTab={selectTab} theme={theme} setTheme={setTheme} stats={stats} open={navOpen} />
      {navOpen && <div className="nav-scrim" onClick={() => setNavOpen(false)}></div>}
      <main className="main">
        <div className="topbar">
          <button className="nav-burger" aria-label="Open menu" onClick={() => setNavOpen(true)}><Icon name="menu" size={18} /></button>
          <div>
            <div className="crumb"><b>{crumb.t}</b></div>
            <div className="topbar-sub">{crumb.s}</div>
          </div>
          <div className="topbar-right">
            {tab === "chat" &&
            <ConnectionControl apiKey={apiKey} setApiKey={setApiKey} rememberKey={rememberKey} setRememberKey={setRememberKey} provider={provider} setProvider={setProvider} model={model} setModel={setModel} />
            }
          </div>
        </div>

        {backendWarm === "cold" &&
        <div className="warmup" role="status">
          <span className="warmup-dot"></span>
          <span>The backend is waking up. Loading the retrieval models takes a few minutes. The page fills in on its own.</span>
        </div>}
        {backendWarm === "down" &&
        <div className="warmup" role="status">
          <span className="warmup-dot"></span>
          <span>The backend is not responding. Try reloading the page in a few minutes.</span>
        </div>}

        <div className="view">
          {/* ChatView stays mounted across tab switches. Unmounting would
              destroy the conversation while the chat itself points users at
              the Papers and Figures tabs. */}
          <div style={{ display: tab === "chat" ? "contents" : "none" }}>
            <ChatView settings={settings} set={set} apiKey={apiKey} provider={provider} model={model} papers={papers} figures={figures} pagesAvailable={pagesAvailable} routingAvailable={routingAvailable} onNeedKey={(kind) => setKeyModalOpen(kind || "agentic")} />
          </div>
          {tab === "inspection" && <InspectionView settings={settings} papers={papers} routingAvailable={routingAvailable} />}
          {tab === "papers" && <PapersView setTab={setTab} papers={papers} figures={figures} uploadAvailable={uploadAvailable} onUploaded={reloadCorpus} />}
          {tab === "figures" && <FiguresView figures={figures} />}
          {tab === "why" && <WhyView setTab={setTab} routingAvailable={routingAvailable} />}
        </div>
      </main>

      <KeyModal open={keyModalOpen} onClose={() => setKeyModalOpen(false)} onSave={(k) => { setApiKey(k); setKeyModalOpen(false); }} />
    </div>);

}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);