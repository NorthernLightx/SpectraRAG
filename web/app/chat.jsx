/* CHAT VIEW: conversation + live retrieval panel (the hero).
   Wired to the real backend via window.RAG: condense → /query → OpenRouter
   BYOK stream → renumbered citations. The retrieval panel reflects the actual
   chunks the server returned. */

// The sources shown under an answer are exactly what the answer CITED: text
// AND visual, nothing it didn't use. Deduped by page so one source is one tile.
function citedSources(citations) {
  const seen = new Set();
  const out = [];
  for (const c of citations || []) {
    if (c.page == null) continue;
    const key = `${c.paper}:${c.page}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(c);
  }
  return out;
}

// The advanced-settings search choices as the route/routingMode pair /query
// takes. "both" is the served default (ADR 0032); the old hybrid and cascade
// options returned the same ranking as the default.
const SEARCH_MODES = [
  { value: "both", route: "auto", routingMode: "" },
  { value: "text", route: "text", routingMode: "" },
  { value: "pages", route: "visual", routingMode: "", needsVisual: true },
  { value: "router", route: "auto", routingMode: "category", needsVisual: true },
  { value: "agentic", route: "agentic", routingMode: "" },
];
function searchMode(route, routingMode) {
  if (routingMode === "category") return "router";
  return { text: "text", visual: "pages", agentic: "agentic" }[route] || "both";
}

function AdvancedPanel({ settings, set, papers, routingAvailable }) {
  // When the server runs without the multimodal router (no visual leg),
  // force_route/routing_mode are no-ops, so grey them out rather than offering
  // switches that silently do nothing. "agentic" stays: DCI is its own path.
  const noRouter = routingAvailable === false;
  const offTitle = "Needs the visual leg, which is off on this server";
  return (
    <div className="adv-panel rise">
      <div className="field">
        <label>Search</label>
        <Segmented value={searchMode(settings.route, settings.routingMode)}
          onChange={(v) => { const m = SEARCH_MODES.find((x) => x.value === v); set("route", m.route); set("routingMode", m.routingMode); }}
          options={SEARCH_MODES.map((m) => ({ value: m.value, label: m.value, disabled: noRouter && m.needsVisual, disabledTitle: offTitle }))} />
        {noRouter
          ? <span className="field-note">visual retrieval is off: it needs the page index (scripts/build_visual_index.py) and RAG_ENABLE_MULTIMODAL=true. Retrieved pages still reach the model as images.</span>
          : <span className="field-note">both searches text and page images on every question; router lets a classifier pick one</span>}
      </div>
      <div className="field">
        <label>Context budget</label>
        <div className="range-row">
          <input type="range" min="3" max="16" value={settings.topk} onChange={(e) => set("topk", +e.target.value)} />
          <span className="range-val">{settings.topk}</span>
        </div>
      </div>
      <div className="field">
        <label>Force paper filter</label>
        <select className="select" value={settings.paper} onChange={(e) => set("paper", e.target.value)}>
          <option value="">All papers</option>
          {papers.map((p) => <option key={p.paper_id} value={p.paper_id}>{p.title ? `${p.title.slice(0, 32)}…` : p.paper_id}</option>)}
        </select>
      </div>
    </div>
  );
}

function EmptyState({ onAsk, routingAvailable, needsKey, onAddKey }) {
  // Without the multimodal router (RAG_ENABLE_MULTIMODAL off) every turn retrieves
  // text-side, so don't advertise per-question routes on the chips.
  const noRouter = routingAvailable === false;
  return (
    <div className="empty">
      <div className="empty-mark"><Icon name="layers" size={24} /></div>
      <h2>Ask across the corpus</h2>
      <p>{noRouter
        ? "20 research papers, indexed by text and figure. Watch retrieval rank the evidence live in the panel on the right."
        : "20 research papers, indexed by text and figure. Each question searches both the text and the page images, and the panel on the right shows what came back."}</p>
      {needsKey && (
        <p className="empty-key">Search is free; cited answers need an OpenRouter key.{" "}
          <button className="btn ghost sm" onClick={onAddKey}><Icon name="key" size={13} /> Add key</button>
        </p>
      )}
      <div className="suggest-grid">
        {window.RAG.SUGGESTIONS.map((s, i) => (
          <button key={i} className="suggest" onClick={() => onAsk(s.q)}>
            <span className="q">{s.q}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// A streaming answer cites raw chunk ids ("[paper::p2::fig1]"), renumbered to
// [n] only once it completes; hide them, and a half-streamed one, until then.
function streamingText(t) {
  return String(t || "").replace(/\s?\[[^\[\]]*::[^\[\]]*\]/g, "").replace(/\s?\[[^\]\s]*$/, "");
}

function AiMessage({ msg, onCite, onFig, onAddKey, onShowPanel, paperTitle, pendingLabel }) {
  const done = !msg.streaming;
  const cited = citedSources(msg.citations);
  const tokens = msg.usage ? (msg.usage.prompt_tokens || 0) + (msg.usage.completion_tokens || 0) : 0;
  // Mirror the retrieval panel's section counts so the footer and the panel
  // never disagree: ranked candidates, injected named-figure captions, and
  // page-image cites that never appeared as candidates.
  const addedN = (msg.injected || []).length;
  const pageCiteN = (msg.citations || []).filter(
    (c) => c.page_cite && !(msg.candidates || []).some((cd) => cd.chunk_id === c.id)
  ).length;
  // KaTeX over the finished answer only: a done message's props never change,
  // so React won't fight the DOM mutation (same pattern as the figure
  // captions in figures.jsx). During streaming the raw $...$ stays visible.
  const bodyRef = useRef(null);
  useEffect(() => {
    if (!done || msg.error || !bodyRef.current) return;
    if (typeof window.renderMathInElement !== "function") return;
    try {
      window.renderMathInElement(bodyRef.current, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
      });
    } catch (_) { /* leave the raw text on a KaTeX error */ }
  }, [done, msg.error, msg.answer]);
  return (
    <div className="msg msg-ai rise">
      <div className="ai-head">
        <div className="ai-avatar"><Icon name="spark" size={14} /></div>
        <span className="ai-name">SpectraRAG</span>
        {msg.route && <div className="ai-meta"><RoutePill route={msg.route} /></div>}
      </div>
      <div className={"ai-body" + (msg.error ? " err" : "")} ref={bodyRef}>
        {!done && !msg.answer ? (
          <span className="mono" style={{ fontSize: 12, color: "var(--text-faint)" }}>
            {pendingLabel || "routing query → re-retrieving"}<span className="caret"></span>
          </span>
        ) : (
          <React.Fragment>
            <Markdown text={done ? msg.answer : streamingText(msg.answer)} onCite={onCite} maxCite={(msg.citations || []).length} />
            {!done && <span className="caret"></span>}
          </React.Fragment>
        )}
      </div>
      {msg.needKey && onAddKey && (
        <div className="notice-cta"><button className="btn primary sm" onClick={onAddKey}><Icon name="key" size={13} /> Add a key</button></div>
      )}
      {done && !msg.error && cited.length > 0 && (
        <div className="answer-sources rise">
          <div className="src-head">Sources</div>
          {cited.map((c) => (
            <button key={c.id} className="src-row" title="View source"
              onClick={() => onFig({
                chunk_id: c.id, paper: c.paper, page: c.page, pages: [c.page],
                kind: c.kind, bbox: c.bbox || null, text: c.quote || "", page_cite: !!c.page_cite,
              })}>
              <span className="src-n">[{c.n}]</span>
              <span className="src-title">{clip(paperTitle ? paperTitle(c.paper) : c.paper, 80)}</span>
              <span className="src-page">{c.kind === "visual" && <Icon name="image" size={11} />} p.{c.page}</span>
            </button>
          ))}
        </div>
      )}
      {done && !msg.error && (msg.candidates || []).length > 0 && (
        <div className="ai-foot rise">
          <span className="metric"><Icon name="route" size={13} /> {msg.candidates.length} ranked{addedN ? ` · ${addedN} added` : ""}{pageCiteN ? ` · ${pageCiteN} page${pageCiteN > 1 ? "s" : ""}` : ""}</span>
          {typeof msg.latencyMs === "number" && <span className="metric"><b>{(msg.latencyMs / 1000).toFixed(2)}s</b></span>}
          {tokens > 0 && <span className="metric"><b>{tokens}</b> tok</span>}
          {onShowPanel && <button className="btn ghost sm phone-only" onClick={onShowPanel}>View results</button>}
        </div>
      )}
    </div>
  );
}

function Composer({ onAsk, busy }) {
  const [val, setVal] = useState("");
  const ref = useRef();
  const submit = () => { const v = val.trim(); if (!v || busy) return; onAsk(v); setVal(""); if (ref.current) ref.current.style.height = "auto"; };
  return (
    <div className="composer-wrap">
      <div className="composer">
        <div className="composer-box">
          <textarea ref={ref} rows={1} value={val} placeholder="Ask a question or follow up…"
            onChange={(e) => { setVal(e.target.value); e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; }}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }} />
          <button className="btn primary" onClick={submit} disabled={busy || !val.trim()}>
            <Icon name="send" size={15} /> Ask
          </button>
        </div>
        <div className="composer-hint">
          <span className="left"><span className="kbd">Enter</span> send · <span className="kbd">Shift+Enter</span> newline</span>
        </div>
      </div>
    </div>
  );
}

/* ---- retrieval panel ---- */
function RetrievalPanel({ turn, highlight, settings, paperTitle, routingAvailable, sheet, onCloseSheet }) {
  const [pageItem, setPageItem] = useState(null);
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem("sr-retr-collapsed") === "1");
  const toggleCollapsed = () =>
    setCollapsed((v) => { localStorage.setItem("sr-retr-collapsed", v ? "0" : "1"); return !v; });
  if (collapsed && !sheet) {
    return (
      <aside className="retr-panel collapsed">
        <button className="retr-rail" onClick={toggleCollapsed} title="Show retrieval panel" aria-label="Show retrieval panel">
          <Icon name="chevron" size={16} style={{ transform: "rotate(180deg)" }} />
          <span className="rail-label">Retrieval</span>
        </button>
      </aside>
    );
  }
  if (!turn || !turn.candidates) {
    return (
      <aside className="retr-panel">
        <div className="retr-head">
          <Icon name="route" size={15} /><h3>Retrieval</h3>
          <button className="retr-collapse" style={{ marginLeft: "auto" }} onClick={toggleCollapsed} title="Hide retrieval panel" aria-label="Hide retrieval panel"><Icon name="chevron" size={15} /></button>
        </div>
        <div className="retr-empty">No turn yet.<br />Ask a question and the live routing decision, ranked candidates, and retrieved figures appear here.</div>
      </aside>
    );
  }
  const cands = turn.candidates;
  // Text rerank scores (0 to 1, less any length penalty) and visual patch-sim
  // (~18+) are different scales, so bars and ranks compare within a leg, never
  // across. A shared min-max would flatten every text bar the moment one visual
  // candidate appears. Min-max per leg (rerank scores can bunch near 0 or 1, and
  // raw values would render near-empty or near-full bars), with a small floor so
  // the leg's worst row still shows a sliver instead of reading as zero relevance.
  const legScores = {
    text: cands.filter((c) => c.kind !== "visual").map((c) => c.score || 0),
    visual: cands.filter((c) => c.kind === "visual").map((c) => c.score || 0),
  };
  const relScore = (c) => {
    const arr = legScores[c.kind === "visual" ? "visual" : "text"];
    const mx = Math.max(...arr), mn = Math.min(...arr);
    return mx === mn ? 1 : 0.08 + 0.92 * (((c.score || 0) - mn) / (mx - mn));
  };
  const total = cands.length;
  // Evidence-mix bars reflect what the answer CITED, not just what was
  // retrieved. On a text route the model can still cite page images attached at
  // generation, so a retrieval-only split falsely reads 100% text. Count cited
  // modality (page_cite / fig_cite / visual candidates all carry kind "visual").
  // Rendered only once citations exist: a retrieved-mix placeholder would show
  // one basis during streaming and silently switch to another when the answer
  // lands, so the bars appear with their final value or not at all.
  const cites = turn.citations || [];
  const visCited = cites.filter((c) => c.kind === "visual").length;
  const visShare = cites.length ? Math.round((visCited / cites.length) * 100) : 0;
  const txtShare = 100 - visShare;
  const citedNum = (c) => {
    const ct = (turn.citations || []).find((x) => x.id === c.chunk_id);
    return ct ? ct.n : null;
  };
  // Page-image citations (`::page`) the model produced from the page images
  // attached at generation. They aren't retrieval candidates, so they get no
  // ranked row. Surface them here so a cited [n] always resolves to something
  // in the panel. Skip any whose id is already a candidate (hybrid route, where
  // the visual leg retrieved the same page).
  const citedPages = (turn.citations || []).filter(
    (c) => c.page_cite && !cands.some((cd) => cd.chunk_id === c.id)
  );
  // Surface the candidates the answer actually cited at the top, then the rest.
  // Each group keeps the retriever's existing rank order, which for a text-only
  // result set is score DESC. We deliberately don't re-sort by raw `score`: text
  // rerank scores and visual patch-sim are different scales, so a global sort
  // would interleave the legs wrongly. The [n] badge is the citation's own number.
  const orderedCands = [
    ...cands.filter((c) => citedNum(c) != null),
    ...cands.filter((c) => citedNum(c) == null),
  ];
  return (
    <aside className={"retr-panel" + (sheet ? " sheet" : "")}>
      <div className="retr-head">
        <Icon name="route" size={15} />
        <h3>Retrieval</h3>
        <span className="live"><span className="dot"></span>live</span>
        <button className="retr-collapse" onClick={sheet ? onCloseSheet : toggleCollapsed} title="Hide retrieval panel" aria-label="Hide retrieval panel"><Icon name="chevron" size={15} /></button>
      </div>
      <div className="retr-body">
        <div className="retr-section">
          <h4>Searched</h4>
          {routingAvailable === false ? (
            <div className="route-card">
              <span className="cand-src">visual retrieval off: every turn retrieves text-side, and the pages it finds reach the model as images. Turning it on takes the page index (scripts/build_visual_index.py) and RAG_ENABLE_MULTIMODAL=true. On MMDocIR, fusing the visual leg raised recall@10 from <b>0.46 to 0.78</b> (<a href="https://github.com/NorthernLightx/SpectraRAG/blob/main/docs/results.md#mmdocir-where-routing-stops-paying" target="_blank" rel="noopener">results</a>).</span>
            </div>
          ) : (
            <div className="route-card">
              <div className="gate"><RoutePill route={turn.route || "text"} />{turn.mode === "router" && <span className="cand-src" style={{ marginLeft: "auto" }}>picked by the router</span>}</div>
              {cites.length > 0 && (
                <div className="route-bars">
                  <span className="cand-src">evidence the answer cited</span>
                  <div className="rb"><span className="lbl">text</span><div className="scorebar"><i style={{ width: txtShare + "%" }}></i></div><span className="pct">{txtShare}%</span></div>
                  <div className="rb"><span className="lbl">visual</span><div className="scorebar visual"><i style={{ width: visShare + "%" }}></i></div><span className="pct">{visShare}%</span></div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Caption chunks injected for figures/tables the question names.
            Context the model saw and can cite, but not retrieval output, so
            they get their own labeled section instead of a ranked row. */}
        {turn.injected && turn.injected.length > 0 && (
          <div className="retr-section">
            <h4>Named-figure evidence <span className="n">{turn.injected.length}</span></h4>
            {turn.injected.map((f, i) => {
              const ct = (turn.citations || []).find((x) => x.id === f.chunkId);
              return (
                <div key={f.chunkId || i} className="cand row-clickable"
                  onClick={() => setPageItem({ chunk_id: f.chunkId, paper: f.paperId, page: f.page, pages: [f.page], kind: "visual", bbox: f.bbox || null, text: f.caption || "" })}
                  title="View source region on page">
                  <div className="cand-top">
                    <span className="cand-num visual">{ct ? ct.n : "·"}</span>
                    <span className="cand-src">{f.paperId} · p.{f.page}</span>
                    <span className="cand-score">added</span>
                  </div>
                  {f.caption && <div className="cand-quote">{clip(f.caption, 180)}</div>}
                  <div className="cand-meta">
                    <span className="pin-note">your question names this figure, so its caption joined the evidence directly</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div className="retr-section">
          <h4>Ranked candidates <span className="n">{total} chunks</span></h4>
          {orderedCands.map((c, i) => {
            const num = citedNum(c);
            const hl = highlight && num && String(num) === String(highlight);
            return (
              <div key={c.chunk_id || i} className={"cand row-clickable" + (hl ? " hl" : "")}
                onClick={() => setPageItem(c)} title="View source region on page">
                <div className="cand-top">
                  <span className={"cand-num" + (c.kind === "visual" ? " visual" : "")}>{num || (c.kind === "visual" ? "IMG" : "·")}</span>
                  <span className="cand-src">{c.paper} · p.{c.page}</span>
                  <span className="cand-score" title="Scores and bars compare within a leg only">{c.kind === "visual" ? "visual" : "text"} · {(c.score || 0).toFixed(2)}</span>
                </div>
                <ScoreBar score={relScore(c)} kind={c.kind} />
                {c.kind === "visual" ? (
                  <div className="answer-fig-img" style={{ height: 84, borderRadius: 6, margin: "6px 0 8px" }}>
                    <img src={window.RAG.pageImageUrl(c.paper, c.page)} alt={`page ${c.page}`} loading="lazy" />
                  </div>
                ) : (
                  c.text && <div className="cand-quote">{clip(c.text, 180)}</div>
                )}
                <div className="cand-meta">
                  <span className="tag">{clip(paperTitle(c.paper), 30)}</span>
                  <span className="view-region"><Icon name="search" size={11} /> region</span>
                </div>
              </div>
            );
          })}
        </div>

        {citedPages.length > 0 && (
          <div className="retr-section">
            <h4>Page images cited <span className="n">{citedPages.length}</span></h4>
            {citedPages.map((c, i) => (
              <div key={c.id || i} className="cand row-clickable"
                onClick={() => setPageItem({ chunk_id: c.id, paper: c.paper, page: c.page, pages: [c.page], kind: "visual", bbox: null, text: "", page_cite: true })}
                title="View cited page image">
                <div className="cand-top">
                  <span className="cand-num visual">{c.n}</span>
                  <span className="cand-src">{c.paper} · p.{c.page}</span>
                  <span className="cand-score">page</span>
                </div>
                <div className="cand-meta">
                  <span className="pin-note">the model read this whole page at generation; cited but not a ranked retrieval result</span>
                </div>
              </div>
            ))}
          </div>
        )}

      </div>
      <PageRegionModal item={pageItem} onClose={() => setPageItem(null)} paperTitle={paperTitle} />
    </aside>
  );
}

function ChatView({ settings, set, apiKey, provider, model, papers, figures, pagesAvailable, routingAvailable, onNeedKey }) {
  const [turns, setTurns] = useState([]);
  const [busy, setBusy] = useState(false);
  const [advOpen, setAdvOpen] = useState(false);
  const [highlight, setHighlight] = useState(null);
  const [status, setStatus] = useState("");
  const [pageItem, setPageItem] = useState(null);
  const scrollRef = useRef();
  const turnsRef = useRef(turns);
  useEffect(() => { turnsRef.current = turns; }, [turns]);

  const paperTitle = useCallback(
    (id) => { const p = papers.find((x) => x.paper_id === id); return (p && p.title) || id; },
    [papers]
  );

  const lastAssistant = useMemo(() => [...turns].reverse().find((t) => t.role === "assistant"), [turns]);

  const mounted = useRef(false);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (!mounted.current) { el.scrollTop = 0; mounted.current = true; return; }
    el.scrollTop = el.scrollHeight;
  }, [turns]);

  // Patch the most recent assistant turn (the live one).
  const updateLast = (patch) => setTurns((ts) => {
    const next = ts.slice();
    for (let i = next.length - 1; i >= 0; i--) {
      if (next[i].role === "assistant") {
        next[i] = { ...next[i], ...(typeof patch === "function" ? patch(next[i]) : patch) };
        break;
      }
    }
    return next;
  });

  // Incremented by New chat: an in-flight ask compares its captured value and
  // stops writing, so a zombie stream can't leak into the next conversation.
  const runSeq = useRef(0);

  const ask = useCallback(async (q) => {
    if (busy) return;
    const myRun = ++runSeq.current;
    const live = (fn) => { if (runSeq.current === myRun) fn(); };
    const upd = (patch) => live(() => updateLast(patch));
    setHighlight(null);
    setStatus("");
    setBusy(true);

    const priorTurns = turnsRef.current
      // Error and notice turns are UI copy ("add your key…"), not assistant
      // answers. Feeding them to condense/generation pollutes the history.
      .filter((t) => t.role === "user" || (t.role === "assistant" && t.answer && !t.error && !t.notice))
      .map((t) => ({ role: t.role, text: t.role === "user" ? t.text : t.answer }));

    setTurns((ts) => [
      ...ts,
      { role: "user", text: q },
      { role: "assistant", q, answer: "", streaming: true, candidates: null, citations: [], route: null, mode: searchMode(settings.route, settings.routingMode) },
    ]);

    let phase = "prepare";
    try {
      // Agentic search can't run keyless: the server-side agent spends the
      // caller's OpenRouter key. Stop before retrieval with a notice instead
      // of letting the thrown error render as a generic failure.
      if (settings.route === "agentic" && !(apiKey && apiKey.trim())) {
        upd({
          answer: "Agentic search runs a search agent server-side on your OpenRouter key, so it needs one. Add yours (top-right) to try it. The standard retrieval modes work without a key.",
          streaming: false,
          notice: true,
        });
        onNeedKey && onNeedKey("agentic");
        return;
      }

      // Whether this turn can generate at all: Ollama needs a picked model,
      // OpenRouter needs the visitor's key.
      const canGenerate = provider === "ollama" ? !!model : !!(apiKey && apiKey.trim());

      // Condense follow-ups into a standalone query through the chosen
      // provider. First turns retrieve as typed. A transient failure on this
      // 80-token call must not kill the turn: retrieval needs no generation,
      // so fall back to the raw message.
      let searchQuery = q;
      if (priorTurns.length && canGenerate) {
        try {
          searchQuery = await window.RAG.condense({ provider, model, apiKey }, priorTurns, q);
        } catch {
          searchQuery = q;
        }
      }
      upd({ searchedFor: searchQuery });

      // Retrieve.
      phase = "retrieve";
      const t0 = performance.now();
      const { results, routing, trace } = await window.RAG.retrieve(searchQuery, {
        topK: settings.topk,
        forceRoute: ["text", "visual", "hybrid"].includes(settings.route) ? settings.route : "",
        routingMode: settings.routingMode || "",
        paperId: settings.paper || "",
        dci: settings.route === "agentic",
        apiKey,
        onStatus: (s) => live(() => setStatus(s)),
      });
      live(() => setStatus(""));
      const tRetrieve = performance.now() - t0;
      phase = "generate";
      const candidates = results.map(toCand);
      // With no router on the deployment the route label is always "text",
      // which is noise, not information. Suppress the per-message pill there.
      // Agentic turns are their own path (DCI), not a router decision; the
      // pill should say so rather than defaulting to "text route".
      upd({
        candidates,
        route: settings.route === "agentic" ? "agentic" : routingAvailable === false ? null : window.RAG.routeLabel(routing),
        routing, trace,
      });

      if (results.length === 0) {
        upd({ answer: "No chunks retrieved. The corpus may not cover this query.", streaming: false, latencyMs: Math.round(tRetrieve) });
        return;
      }

      // Generation is browser-direct on the chosen provider. Without one
      // configured, stop at retrieval with a how-to notice.
      if (!canGenerate) {
        const notice = provider === "ollama"
          ? "Search found the chunks in the retrieval panel. Pick a vision model in the model menu to get a cited answer from them."
          : "Search found the chunks in the retrieval panel. Add an OpenRouter key to get a cited answer from them.";
        live(() => setStatus(""));
        upd({ answer: notice, streaming: false, notice: true, needKey: provider !== "ollama", latencyMs: Math.round(tRetrieve) });
        return;
      }

      // Both providers only offer vision-capable models; the server attaches
      // page images whenever it serves page renders.
      if (pagesAvailable) live(() => setStatus("Reading the retrieved page images…"));
      const { messages, injected } = await window.RAG.buildMessages(priorTurns, q, results);
      if (injected && injected.length) upd({ injected });
      const tGen = performance.now();
      const onDelta = (delta) => upd((prev) => ({ answer: prev.answer + delta }));
      const { text, usage } = await window.RAG.streamChat({ provider, model, apiKey }, messages, onDelta);

      // Renumber the model's chunk-id citations → [1][2] and build the list.
      const { newText, ids } = window.RAG.renumberCitations(text);
      const byId = new Map(results.map((c) => [c.chunk_id, c]));
      const citations = ids.map((id, i) => {
        // Page-image citations (`paper::pN::page`) point at an attached page,
        // not a retrieved chunk. The model cites them for figure claims.
        const pm = id.match(/^(.+)::p(\d+)::page$/);
        if (pm) return { n: i + 1, id, paper: pm[1], page: +pm[2], quote: null, kind: "visual", page_cite: true };
        const c = byId.get(id);
        if (!c) {
          // Injected figure/table caption (buildMessages adds it when the
          // question names the element). Resolve through the figure index.
          const fg = (figures || []).find((g) => g.chunk_id === id);
          if (fg) return { n: i + 1, id, paper: fg.paper_id, page: fg.page_number, quote: clip(fg.caption || "", 180), kind: "visual", fig_cite: true, bbox: fg.bbox || null };
        }
        const pages = c ? c.page_numbers || [] : [];
        return { n: i + 1, id, paper: c ? c.paper_id : id, page: pages[0], quote: c ? clip(c.text, 180) : null, kind: c && c.source === "visual" ? "visual" : "text" };
      });
      upd({
        answer: newText,
        streaming: false,
        citations,
        usage,
        latencyMs: Math.round(performance.now() - tGen + tRetrieve),
      });
    } catch (err) {
      if (err && (err.code === "ollama_down" || err.code === "stream_error")) {
        // Provider failure, not the visitor's fault. Say so without blaming
        // a key or model they configured fine, and keep any partial answer.
        upd((prev) => ({
          answer: prev.answer
            ? `${prev.answer}\n\nGeneration stopped early: ${(err && err.message) || "the model provider failed."}`
            : `${(err && err.message) || "The model provider failed."} Retrieval still worked. The chunks are on the right.`,
          streaming: false,
          notice: true,
        }));
      } else {
        // Keep whatever streamed before the failure. Wiping a half-answer is
        // worse than showing it with an interruption note.
        const why = String((err && err.message) || err).replace(/\.$/, "");
        upd((prev) => ({
          answer: prev.answer
            ? `${prev.answer}\n\nGeneration interrupted: ${(err && err.message) || err}`
            : phase === "retrieve"
            ? `Search failed: ${why}. The server may still be starting up. Try again in a minute.`
            : phase === "generate"
            ? `Answer failed: ${why}. The model provider rejected the request or could not be reached. The search results are in the retrieval panel.`
            : `Request failed: ${why}.`,
          streaming: false,
          error: !prev.answer,
          notice: !!prev.answer,
        }));
      }
      live(() => setStatus(""));
    } finally {
      live(() => setBusy(false));
    }
  }, [busy, apiKey, provider, model, settings, figures, pagesAvailable, routingAvailable, onNeedKey]);

  // Bumping runSeq orphans any in-flight ask: its guarded writes become no-ops.
  const newChat = () => { runSeq.current += 1; setTurns([]); setHighlight(null); setBusy(false); setStatus(""); };
  // The retrieval panel is hidden at phone width, so a
  // highlight there would be invisible, so open the source-page modal instead.
  // Page-image citations always open the modal: they have no panel row.
  const onCite = (tag, msg) => {
    if (tag[0] === "F") return;
    const cit = msg && msg.citations ? msg.citations.find((c) => String(c.n) === String(tag)) : null;
    if (cit && cit.page_cite) {
      setPageItem({ chunk_id: cit.id, paper: cit.paper, page: cit.page, pages: [cit.page], kind: "visual", bbox: null, text: "", page_cite: true });
      return;
    }
    if (cit && cit.fig_cite) {
      // Injected figure caption: open its page with the region box.
      setPageItem({ chunk_id: cit.id, paper: cit.paper, page: cit.page, pages: [cit.page], kind: "visual", bbox: cit.bbox || null, text: cit.quote || "" });
      return;
    }
    // The panel only ever shows the LAST turn's evidence, so a highlight is
    // wrong for citations in older messages, so open the modal for those too.
    const panelHidden = window.matchMedia && window.matchMedia("(max-width: 760px)").matches;
    const isLastTurn = msg === lastAssistant;
    if ((panelHidden || !isLastTurn) && cit && msg.candidates) {
      const cand = msg.candidates.find((cd) => cd.chunk_id === cit.id);
      if (cand) { setPageItem(cand); return; }
    }
    if (!isLastTurn) return;
    setHighlight(tag);
  };
  const openFig = (cand) => setPageItem(cand);
  const needsKey = provider !== "ollama" && !(apiKey && apiKey.trim());
  // At phone width the side panel is hidden; it opens as a bottom sheet instead.
  const [sheet, setSheet] = useState(false);
  const showPanel = () => { if (window.matchMedia("(max-width: 760px)").matches) setSheet(true); };
  useEffect(() => {
    if (!sheet) return;
    const onEsc = (e) => { if (e.key === "Escape") setSheet(false); };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [sheet]);

  return (
    <div className="chat-wrap">
      <div className="chat-col">
        <div className="adv-toggle-row">
          <button type="button" className={"adv-toggle" + (advOpen ? " open" : "")} aria-expanded={advOpen} onClick={() => setAdvOpen((o) => !o)}>
            <Icon name="chevron" size={14} /> <Icon name="sliders" size={13} /> Advanced retrieval settings
            <span className="mono" style={{ color: "var(--text-faint)", fontSize: 11, marginLeft: 4 }}>
              {searchMode(settings.route, settings.routingMode)} · ctx={settings.topk}
            </span>
          </button>
          <button className="btn ghost sm" onClick={newChat} title="Clear conversation"><Icon name="plus" size={14} /> New chat</button>
        </div>
        {advOpen && <AdvancedPanel settings={settings} set={set} papers={papers} routingAvailable={routingAvailable} />}

        <div className="chat-scroll" ref={scrollRef}>
          {turns.length === 0 ? (
            <EmptyState onAsk={ask} routingAvailable={routingAvailable} needsKey={needsKey} onAddKey={() => onNeedKey && onNeedKey("chat")} />
          ) : (
            <div className="chat-inner">
              {turns.map((t, i) =>
                t.role === "user" ? (
                  <div className="msg msg-user rise" key={i}><div className="bubble">{t.text}</div></div>
                ) : (
                  <AiMessage key={i} msg={t} onCite={(tag) => onCite(tag, t)} onFig={openFig} paperTitle={paperTitle}
                    onAddKey={onNeedKey ? () => onNeedKey("chat") : null} onShowPanel={t === lastAssistant ? showPanel : null}
                    pendingLabel={status || (routingAvailable === false ? "retrieving…" : undefined)} />
                )
              )}
            </div>
          )}
        </div>
        {status && turns.length === 0 && <div className="composer-hint" style={{ padding: "0 18px 8px", color: "var(--text-faint)" }}>{status}</div>}

        <Composer onAsk={ask} busy={busy} />
      </div>

      {sheet && <div className="sheet-scrim" onClick={() => setSheet(false)}></div>}
      <RetrievalPanel turn={lastAssistant} highlight={highlight} settings={settings} paperTitle={paperTitle} routingAvailable={routingAvailable}
        sheet={sheet} onCloseSheet={() => setSheet(false)} />
      <PageRegionModal item={pageItem} onClose={() => setPageItem(null)} paperTitle={paperTitle} />
    </div>
  );
}

window.ChatView = ChatView;
