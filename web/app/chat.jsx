/* CHAT VIEW — conversation + live retrieval panel (the hero).
   Wired to the real backend via window.RAG: condense → /query → OpenRouter
   BYOK stream → renumbered citations. The retrieval panel reflects the actual
   chunks the server returned. */

// A real RetrievalResult chunk → the flat shape the panel/cards render.
function toCand(c) {
  const pages = c.page_numbers || [];
  return {
    chunk_id: c.chunk_id,
    paper: c.paper_id,
    page: pages[0],
    pages,
    score: typeof c.score === "number" ? c.score : 0,
    kind: c.source === "visual" ? "visual" : "text",
    bbox: (c.metadata && c.metadata.bbox) || null,
    text: c.text || "",
  };
}

function previewQuote(raw, max = 180) {
  const t = String(raw || "").replace(/\s+/g, " ").trim();
  return t.length > max ? t.slice(0, max).trim() + "…" : t;
}

function AdvancedPanel({ settings, set, papers, routingAvailable }) {
  // When the server runs without the multimodal router (no GPU visual leg),
  // force_route/routing_mode are no-ops — grey them out rather than offering
  // switches that silently do nothing. "agentic" stays: DCI is its own path.
  const noRouter = routingAvailable === false;
  const offTitle = "Needs the multimodal router (GPU visual leg) — not available on this deployment";
  return (
    <div className="adv-panel rise">
      <div className="field">
        <label>Routing</label>
        <Segmented value={settings.route} onChange={(v) => set("route", v)}
          options={[{ value: "auto", label: "auto" }, { value: "text", label: "text" },
            { value: "visual", label: "visual", disabled: noRouter, disabledTitle: offTitle },
            { value: "agentic", label: "agentic" }]} />
        {noRouter &&
        <span className="field-note">visual routing needs the GPU leg — off on this CPU deployment (offline: +35% recall over text-only); figure questions still work via page images</span>
        }
      </div>
      <div className="field">
        <label>Routing mode</label>
        <Segmented value={settings.routingMode} onChange={(v) => set("routingMode", v)}
          options={[{ value: "", label: "default" },
            { value: "category", label: "category", disabled: noRouter, disabledTitle: offTitle },
            { value: "cascade", label: "cascade", disabled: noRouter, disabledTitle: offTitle }]} />
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

function EmptyState({ onAsk, routingAvailable }) {
  // Without the multimodal router (CPU-only deploys) every turn retrieves
  // text-side, so don't advertise per-question routes on the chips.
  const noRouter = routingAvailable === false;
  return (
    <div className="empty">
      <div className="empty-mark"><Icon name="layers" size={24} /></div>
      <h2>Ask across the corpus</h2>
      <p>{noRouter
        ? "20 research papers, indexed by text and figure. Watch retrieval rank the evidence live in the panel on the right."
        : "20 research papers, indexed by text and figure. Every turn re-retrieves against the right modality — watch it route in the panel on the right."}</p>
      <div className="suggest-grid">
        {window.RAG.SUGGESTIONS.map((s, i) => (
          <button key={i} className="suggest" onClick={() => onAsk(s.q)}>
            <span className="q">{s.q}</span>
            {!noRouter && <RoutePill route={s.route} />}
          </button>
        ))}
      </div>
    </div>
  );
}

function AiMessage({ msg, onCite, onFig, paperTitle, pendingLabel }) {
  const done = !msg.streaming;
  const figs = (msg.candidates || []).filter((c) => c.kind === "visual");
  const tokens = msg.usage ? (msg.usage.prompt_tokens || 0) + (msg.usage.completion_tokens || 0) : 0;
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
            <Markdown text={msg.answer} onCite={onCite} />
            {!done && <span className="caret"></span>}
          </React.Fragment>
        )}
      </div>
      {done && !msg.error && figs.length > 0 && (
        <div className="answer-figs rise">
          {figs.slice(0, 4).map((f, i) => (
            <button key={f.chunk_id || i} className="answer-fig" onClick={() => onFig(f)}>
              <div className="answer-fig-img" style={{ height: 92 }}>
                <img src={window.RAG.pageImageUrl(f.paper, f.page)} alt={`page ${f.page}`} loading="lazy" />
              </div>
              <div className="cap"><b>p.{f.page}</b> {previewQuote(f.text, 64)}</div>
            </button>
          ))}
        </div>
      )}
      {done && !msg.error && (msg.candidates || []).length > 0 && (
        <div className="ai-foot rise">
          <span className="metric"><Icon name="route" size={13} /> {msg.candidates.length} chunks</span>
          {typeof msg.latencyMs === "number" && <span className="metric"><b>{(msg.latencyMs / 1000).toFixed(2)}s</b></span>}
          {tokens > 0 && <span className="metric"><b>{tokens}</b> tok</span>}
          {msg.demo && <span className="metric" title="Generated with the shared free demo model. Add your own key for stronger models.">free demo model</span>}
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
function RetrievalPanel({ turn, highlight, settings, paperTitle, routingAvailable }) {
  const [pageItem, setPageItem] = useState(null);
  if (!turn || !turn.candidates) {
    return (
      <aside className="retr-panel">
        <div className="retr-head"><Icon name="route" size={15} /><h3>Retrieval</h3></div>
        <div className="retr-empty">No turn yet.<br />Ask a question and the live routing decision, ranked candidates, and retrieved figures appear here.</div>
      </aside>
    );
  }
  const cands = turn.candidates;
  // Rerank scores are logits (any sign, any magnitude) — min-max scale the
  // bars within the set so they stay comparative; all-negative sets would
  // otherwise render every bar empty.
  const scores = cands.map((c) => c.score || 0);
  const sMax = Math.max(...scores), sMin = Math.min(...scores);
  const relScore = (s) => (sMax === sMin ? 1 : (s - sMin) / (sMax - sMin));
  const vis = cands.filter((c) => c.kind === "visual");
  const total = cands.length;
  const visShare = total ? Math.round((vis.length / total) * 100) : 0;
  const txtShare = 100 - visShare;
  const citedNum = (c) => {
    const ct = (turn.citations || []).find((x) => x.id === c.chunk_id);
    return ct ? ct.n : null;
  };
  return (
    <aside className="retr-panel">
      <div className="retr-head">
        <Icon name="route" size={15} />
        <h3>Retrieval</h3>
        <span className="live"><span className="dot"></span>live</span>
      </div>
      <div className="retr-body">
        <div className="retr-section">
          <h4>Routing decision</h4>
          {routingAvailable === false ? (
            <div className="route-card">
              <span className="cand-src">router off on this CPU-only deployment — offline it measures <b>+35% recall</b> over text-only retrieval on MMLongBench (<a href="https://github.com/NorthernLightx/SpectraRAG/blob/main/docs/results.md" target="_blank" rel="noopener">results</a>). Here every turn retrieves text-side; figure questions read the page images at generation.</span>
            </div>
          ) : (
            <div className="route-card">
              <div className="gate"><RoutePill route={turn.route || "text"} /><span className="cand-src" style={{ marginLeft: "auto" }}>mode: {turn.mode || settings.route}</span></div>
              <div className="route-bars">
                <div className="rb"><span className="lbl">text</span><div className="scorebar"><i style={{ width: txtShare + "%" }}></i></div><span className="pct">{txtShare}%</span></div>
                <div className="rb"><span className="lbl">visual</span><div className="scorebar visual"><i style={{ width: visShare + "%" }}></i></div><span className="pct">{visShare}%</span></div>
              </div>
            </div>
          )}
        </div>

        {/* Caption chunks injected for figures/tables the question names —
            context the model saw and can cite, but not retrieval output, so
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
                  {f.caption && <div className="cand-quote">{previewQuote(f.caption)}</div>}
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
          {cands.map((c, i) => {
            const num = citedNum(c);
            const hl = highlight && num && String(num) === String(highlight);
            return (
              <div key={c.chunk_id || i} className={"cand row-clickable" + (hl ? " hl" : "")}
                onClick={() => setPageItem(c)} title="View source region on page">
                <div className="cand-top">
                  <span className={"cand-num" + (c.kind === "visual" ? " visual" : "")}>{num || (c.kind === "visual" ? "IMG" : "·")}</span>
                  <span className="cand-src">{c.paper} · p.{c.page}</span>
                  <span className="cand-score">{c.score.toFixed(3)}</span>
                </div>
                <ScoreBar score={relScore(c.score || 0)} kind={c.kind} />
                {c.text && <div className="cand-quote">{previewQuote(c.text)}</div>}
                <div className="cand-meta">
                  <span className="tag">{previewQuote(paperTitle(c.paper), 30)}</span>
                  <span className="view-region"><Icon name="search" size={11} /> region</span>
                </div>
              </div>
            );
          })}
        </div>

        {vis.length > 0 && (
          <div className="retr-section">
            <h4>Retrieved figures <span className="n">{vis.length}</span></h4>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              {vis.map((c, i) => (
                <div key={c.chunk_id || i} className="figthumb-click" onClick={() => setPageItem(c)} title="View source region on page">
                  <div className="figthumb">
                    <div className="answer-fig-img" style={{ height: 84 }}>
                      <img src={window.RAG.pageImageUrl(c.paper, c.page)} alt={`page ${c.page}`} loading="lazy" />
                    </div>
                    <div className="figthumb-meta"><span className="mono">{c.paper}</span> · p.{c.page}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      <PageRegionModal item={pageItem} onClose={() => setPageItem(null)} paperTitle={paperTitle} />
    </aside>
  );
}

function ChatView({ settings, set, layout, resetSignal, apiKey, model, papers, figures, pagesAvailable, demoAvailable, routingAvailable, onNeedKey }) {
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
      // answers — feeding them to condense/generation pollutes the history.
      .filter((t) => t.role === "user" || (t.role === "assistant" && t.answer && !t.error && !t.notice))
      .map((t) => ({ role: t.role, text: t.role === "user" ? t.text : t.answer }));

    setTurns((ts) => [
      ...ts,
      { role: "user", text: q },
      { role: "assistant", q, answer: "", streaming: true, candidates: null, citations: [], route: null, mode: settings.route },
    ]);

    try {
      // Agentic search can't run keyless: the server-side agent spends the
      // caller's OpenRouter key. Stop before retrieval with a notice instead
      // of letting the thrown error render as a generic failure.
      if (settings.route === "agentic" && !(apiKey && apiKey.trim())) {
        upd({
          answer: "Agentic search runs a search agent server-side on your OpenRouter key, so it needs one — add yours (top-right) to try it. The standard retrieval modes work without a key.",
          streaming: false,
          notice: true,
        });
        onNeedKey && onNeedKey("agentic");
        return;
      }

      // Condense follow-ups into a standalone query: BYOK browser-direct, or
      // through the demo path keyless (one extra demo call; falls back to the
      // raw message on failure). First turns retrieve as typed.
      const hasKeyForCondense = !!(apiKey && apiKey.trim());
      let searchQuery = q;
      if (priorTurns.length) {
        if (hasKeyForCondense) {
          // A transient 429/5xx on this 80-token call must not kill the turn:
          // retrieval needs no key — fall back to the raw message, like the
          // keyless condenseDemo does.
          try {
            searchQuery = await window.RAG.condense(apiKey, model, priorTurns, q);
          } catch {
            searchQuery = q;
          }
        } else if (demoAvailable) {
          setStatus("Condensing the follow-up into a search query…");
          searchQuery = await window.RAG.condenseDemo(priorTurns, q);
          live(() => setStatus(""));
        }
      }
      upd({ searchedFor: searchQuery });

      // Retrieve.
      const t0 = performance.now();
      const { results, routing, trace } = await window.RAG.retrieve(searchQuery, {
        topK: settings.topk,
        forceRoute: settings.route === "text" ? "text" : settings.route === "visual" ? "hybrid" : "",
        routingMode: settings.routingMode || "",
        paperId: settings.paper || "",
        dci: settings.route === "agentic",
        apiKey,
        onStatus: (s) => live(() => setStatus(s)),
      });
      live(() => setStatus(""));
      const tRetrieve = performance.now() - t0;
      const candidates = results.map(toCand);
      // With no router on the deployment the route label is always "text" —
      // noise, not information. Suppress the per-message pill there.
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

      // Generation: browser-direct with the visitor's key (BYOK) when one is
      // set, else the server's keyless demo path (free model, daily-capped).
      // Only when the server has no demo key either does this stop at
      // retrieval with the bring-a-key notice.
      const hasKey = !!(apiKey && apiKey.trim());
      if (!hasKey && !demoAvailable) {
        live(() => setStatus("Add your OpenRouter key (top-right) to generate a cited answer."));
        upd({
          answer: "Retrieved the chunks shown on the right. Add your OpenRouter key (top-right) to generate a cited answer from them.",
          streaming: false,
          notice: true,
          latencyMs: Math.round(tRetrieve),
        });
        return;
      }

      // The demo chain is all vision-capable models, so keyless turns always
      // attach page images when pages are served.
      const useImages = pagesAvailable && (hasKey ? window.RAG.supportsVision(model) : true);
      if (useImages) live(() => setStatus(hasKey ? "Reading the retrieved page images…" : "Free demo model is reading the retrieved pages…"));
      const { messages, injected } = await window.RAG.buildMessages(priorTurns, q, results, useImages, figures);
      if (injected && injected.length) upd({ injected });
      const tGen = performance.now();
      const onDelta = (delta) => upd((prev) => ({ answer: prev.answer + delta }));
      const { text, usage } = hasKey
        ? await window.RAG.streamChat(apiKey, model, messages, onDelta)
        : await window.RAG.streamDemoChat(messages, onDelta);

      // Renumber the model's chunk-id citations → [1][2] and build the list.
      const { newText, ids } = window.RAG.renumberCitations(text);
      const byId = new Map(results.map((c) => [c.chunk_id, c]));
      const citations = ids.map((id, i) => {
        // Page-image citations (`paper::pN::page`) point at an attached page,
        // not a retrieved chunk — the model cites them for figure claims.
        const pm = id.match(/^(.+)::p(\d+)::page$/);
        if (pm) return { n: i + 1, id, paper: pm[1], page: +pm[2], quote: null, kind: "visual", page_cite: true };
        const c = byId.get(id);
        if (!c) {
          // Injected figure/table caption (buildMessages adds it when the
          // question names the element) — resolve through the figure index.
          const fg = (figures || []).find((g) => g.chunk_id === id);
          if (fg) return { n: i + 1, id, paper: fg.paper_id, page: fg.page_number, quote: previewQuote(fg.caption || ""), kind: "visual", fig_cite: true, bbox: fg.bbox || null };
        }
        const pages = c ? c.page_numbers || [] : [];
        return { n: i + 1, id, paper: c ? c.paper_id : id, page: pages[0], quote: c ? previewQuote(c.text) : null, kind: c && c.source === "visual" ? "visual" : "text" };
      });
      upd({
        answer: newText,
        streaming: false,
        citations,
        usage,
        demo: !hasKey,
        latencyMs: Math.round(performance.now() - tGen + tRetrieve),
      });
    } catch (err) {
      if (err && err.code === "demo_quota") {
        // The shared free quota ran out for today: hand off to the key modal
        // instead of rendering it as a failure — retrieval still worked.
        onNeedKey && onNeedKey();
        upd({
          answer: "Today's free demo answers are used up, but retrieval still works — the chunks are on the right. Add your own OpenRouter key (top-right) to keep generating answers.",
          streaming: false,
          notice: true,
        });
      } else if (err && (err.code === "demo_down" || err.code === "stream_error")) {
        // Upstream model failure, not the visitor's fault — say so without
        // blaming a key they may not even have, and keep any partial answer.
        upd((prev) => ({
          answer: prev.answer
            ? `${prev.answer}\n\nGeneration stopped early: ${(err && err.message) || "the model provider failed."}`
            : `${(err && err.message) || "The model provider failed."} Retrieval still worked — the chunks are on the right.`,
          streaming: false,
          notice: true,
        }));
      } else {
        // Keep whatever streamed before the failure — wiping a half-answer is
        // worse than showing it with an honest interruption note.
        upd((prev) => ({
          answer: prev.answer
            ? `${prev.answer}\n\nGeneration interrupted: ${(err && err.message) || err}`
            : `Request failed: ${(err && err.message) || err}. Either the server isn't reachable, or your OpenRouter key is invalid.`,
          streaming: false,
          error: !prev.answer,
          notice: !!prev.answer,
        }));
      }
      live(() => setStatus(""));
    } finally {
      live(() => setBusy(false));
    }
  }, [busy, apiKey, model, settings, figures, pagesAvailable, demoAvailable, routingAvailable, onNeedKey]);

  // Bumping runSeq orphans any in-flight ask: its guarded writes become no-ops.
  const newChat = () => { runSeq.current += 1; setTurns([]); setHighlight(null); setBusy(false); setStatus(""); };
  // The retrieval panel is hidden in focus layout and at phone width, so a
  // highlight there would be invisible — open the source-page modal instead.
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
    // wrong for citations in older messages — open the modal for those too.
    const panelHidden = layout === "single" || (window.matchMedia && window.matchMedia("(max-width: 760px)").matches);
    const isLastTurn = msg === lastAssistant;
    if ((panelHidden || !isLastTurn) && cit && msg.candidates) {
      const cand = msg.candidates.find((cd) => cd.chunk_id === cit.id);
      if (cand) { setPageItem(cand); return; }
    }
    if (!isLastTurn) return;
    setHighlight(tag);
  };
  const openFig = (cand) => setPageItem(cand);

  const firstReset = useRef(true);
  useEffect(() => {
    if (firstReset.current) { firstReset.current = false; return; }
    newChat();
  }, [resetSignal]);

  return (
    <div className={"chat-wrap" + (layout === "single" ? " single" : "")}>
      <div className="chat-col">
        <div className="adv-toggle-row">
          <div className={"adv-toggle" + (advOpen ? " open" : "")} onClick={() => setAdvOpen((o) => !o)}>
            <Icon name="chevron" size={14} /> <Icon name="sliders" size={13} /> Advanced retrieval settings
            <span className="mono" style={{ color: "var(--text-faint)", fontSize: 11, marginLeft: 4 }}>
              {settings.route}{settings.routingMode ? " · " + settings.routingMode : ""} · ctx={settings.topk}
            </span>
          </div>
          <button className="btn ghost sm" onClick={newChat} title="Clear conversation"><Icon name="plus" size={14} /> New chat</button>
        </div>
        {advOpen && <AdvancedPanel settings={settings} set={set} papers={papers} routingAvailable={routingAvailable} />}

        <div className="chat-scroll" ref={scrollRef}>
          {turns.length === 0 ? (
            <EmptyState onAsk={ask} routingAvailable={routingAvailable} />
          ) : (
            <div className="chat-inner">
              {turns.map((t, i) =>
                t.role === "user" ? (
                  <div className="msg msg-user rise" key={i}><div className="bubble">{t.text}</div></div>
                ) : (
                  <AiMessage key={i} msg={t} onCite={(tag) => onCite(tag, t)} onFig={openFig} paperTitle={paperTitle}
                    pendingLabel={status || (routingAvailable === false ? "retrieving…" : undefined)} />
                )
              )}
            </div>
          )}
        </div>
        {status && turns.length === 0 && <div className="composer-hint" style={{ padding: "0 18px 8px", color: "var(--text-faint)" }}>{status}</div>}

        <Composer onAsk={ask} busy={busy} />
      </div>

      {layout !== "single" && <RetrievalPanel turn={lastAssistant} highlight={highlight} settings={settings} paperTitle={paperTitle} routingAvailable={routingAvailable} />}
      <PageRegionModal item={pageItem} onClose={() => setPageItem(null)} paperTitle={paperTitle} />
    </div>
  );
}

window.ChatView = ChatView;
