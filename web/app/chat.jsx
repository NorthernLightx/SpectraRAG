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

function AdvancedPanel({ settings, set, papers }) {
  return (
    <div className="adv-panel rise">
      <div className="field">
        <label>Routing</label>
        <Segmented value={settings.route} onChange={(v) => set("route", v)}
          options={[{ value: "auto", label: "auto" }, { value: "text", label: "text" }, { value: "visual", label: "visual" }, { value: "agentic", label: "agentic" }]} />
      </div>
      <div className="field">
        <label>Routing mode</label>
        <Segmented value={settings.routingMode} onChange={(v) => set("routingMode", v)}
          options={[{ value: "", label: "default" }, { value: "category", label: "category" }, { value: "cascade", label: "cascade" }]} />
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

function EmptyState({ onAsk }) {
  return (
    <div className="empty">
      <div className="empty-mark"><Icon name="layers" size={24} /></div>
      <h2>Ask across the corpus</h2>
      <p>20 research papers, indexed by text and figure. Every turn re-retrieves against the right modality — watch it route in the panel on the right.</p>
      <div className="suggest-grid">
        {window.RAG.SUGGESTIONS.map((s, i) => (
          <button key={i} className="suggest" onClick={() => onAsk(s.q)}>
            <span className="q">{s.q}</span>
            <RoutePill route={s.route} />
          </button>
        ))}
      </div>
    </div>
  );
}

function AiMessage({ msg, onCite, onFig, paperTitle }) {
  const done = !msg.streaming;
  const figs = (msg.candidates || []).filter((c) => c.kind === "visual");
  const tokens = msg.usage ? (msg.usage.prompt_tokens || 0) + (msg.usage.completion_tokens || 0) : 0;
  return (
    <div className="msg msg-ai rise">
      <div className="ai-head">
        <div className="ai-avatar"><Icon name="spark" size={14} /></div>
        <span className="ai-name">SpectraRAG</span>
        {msg.route && <div className="ai-meta"><RoutePill route={msg.route} /></div>}
      </div>
      <div className={"ai-body" + (msg.error ? " err" : "")}>
        <Markdown text={msg.answer} onCite={onCite} />
        {!done && <span className="caret"></span>}
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
function RetrievalPanel({ turn, highlight, settings, paperTitle }) {
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
          <div className="route-card">
            <div className="gate"><RoutePill route={turn.route} /><span className="cand-src" style={{ marginLeft: "auto" }}>mode: {settings.route}</span></div>
            <div className="route-bars">
              <div className="rb"><span className="lbl">text</span><div className="scorebar"><i style={{ width: txtShare + "%" }}></i></div><span className="pct">{txtShare}%</span></div>
              <div className="rb"><span className="lbl">visual</span><div className="scorebar visual"><i style={{ width: visShare + "%" }}></i></div><span className="pct">{visShare}%</span></div>
            </div>
          </div>
        </div>

        <div className="retr-section">
          <h4>Ranked candidates <span className="n">{total} chunks</span></h4>
          {cands.map((c, i) => {
            const num = citedNum(c);
            const hl = highlight && num && String(num) === String(highlight);
            return (
              <div key={c.chunk_id || i} className={"cand row-clickable" + (hl ? " hl" : "")}
                onClick={() => setPageItem(c)} title="View source region on page">
                <div className="cand-top">
                  <span className={"cand-num" + (c.kind === "visual" ? " visual" : "")}>{c.kind === "visual" ? "IMG" : (num || "·")}</span>
                  <span className="cand-src">{c.paper} · p.{c.page}</span>
                  <span className="cand-score">{c.score.toFixed(3)}</span>
                </div>
                <ScoreBar score={c.score} kind={c.kind} />
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

function ChatView({ settings, set, layout, resetSignal, apiKey, model, papers, pagesAvailable }) {
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

  const ask = useCallback(async (q) => {
    if (busy) return;
    setHighlight(null);
    setStatus("");
    setBusy(true);

    const priorTurns = turnsRef.current
      .filter((t) => t.role === "user" || (t.role === "assistant" && t.answer))
      .map((t) => ({ role: t.role, text: t.role === "user" ? t.text : t.answer }));

    setTurns((ts) => [
      ...ts,
      { role: "user", text: q },
      { role: "assistant", q, answer: "", streaming: true, candidates: null, citations: [], route: null },
    ]);

    try {
      // Condense (skip on first turn, or when there's no key to call with —
      // the raw message still retrieves fine).
      const searchQuery = (priorTurns.length && apiKey && apiKey.trim())
        ? await window.RAG.condense(apiKey, model, priorTurns, q)
        : q;
      updateLast({ searchedFor: searchQuery });

      // Retrieve.
      const t0 = performance.now();
      const { results, routing, trace } = await window.RAG.retrieve(searchQuery, {
        topK: settings.topk,
        forceRoute: settings.route === "text" ? "text" : settings.route === "visual" ? "hybrid" : "",
        routingMode: settings.routingMode || "",
        paperId: settings.paper || "",
        dci: settings.route === "agentic",
        apiKey,
        onStatus: setStatus,
      });
      setStatus("");
      const tRetrieve = performance.now() - t0;
      const candidates = results.map(toCand);
      updateLast({ candidates, route: window.RAG.routeLabel(routing), routing, trace });

      if (results.length === 0) {
        updateLast({ answer: "No chunks retrieved. The corpus may not cover this query.", streaming: false, latencyMs: Math.round(tRetrieve) });
        return;
      }

      // Generation needs the visitor's key (BYOK); retrieval above does not.
      // Without a key, stop here with the chunks shown on the right.
      if (!apiKey || !apiKey.trim()) {
        setStatus("Add your OpenRouter key (top-right) to generate a cited answer.");
        updateLast({
          answer: "Retrieved the chunks shown on the right. Add your OpenRouter key (top-right) to generate a cited answer from them.",
          streaming: false,
          notice: true,
          latencyMs: Math.round(tRetrieve),
        });
        return;
      }

      // Generate (client-side, BYOK). Stream tokens into the live turn.
      const useImages = pagesAvailable && window.RAG.supportsVision(model);
      if (useImages) setStatus("Reading the retrieved page images…");
      const messages = await window.RAG.buildMessages(priorTurns, q, results, useImages);
      const tGen = performance.now();
      const { text, usage } = await window.RAG.streamChat(apiKey, model, messages, (delta) => {
        updateLast((prev) => ({ answer: prev.answer + delta }));
      });

      // Renumber the model's chunk-id citations → [1][2] and build the list.
      const { newText, ids } = window.RAG.renumberCitations(text);
      const byId = new Map(results.map((c) => [c.chunk_id, c]));
      const citations = ids.map((id, i) => {
        const c = byId.get(id);
        const pages = c ? c.page_numbers || [] : [];
        return { n: i + 1, id, paper: c ? c.paper_id : id, page: pages[0], quote: c ? previewQuote(c.text) : null, kind: c && c.source === "visual" ? "visual" : "text" };
      });
      updateLast({
        answer: newText,
        streaming: false,
        citations,
        usage,
        latencyMs: Math.round(performance.now() - tGen + tRetrieve),
      });
    } catch (err) {
      updateLast({
        answer: `Request failed: ${(err && err.message) || err}. Either the server isn't reachable, or your OpenRouter key is invalid.`,
        streaming: false,
        error: true,
      });
      setStatus("");
    } finally {
      setBusy(false);
    }
  }, [busy, apiKey, model, settings, pagesAvailable]);

  const newChat = () => { setTurns([]); setHighlight(null); setBusy(false); setStatus(""); };
  const onCite = (tag) => { if (tag[0] !== "F") setHighlight(tag); };
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
        {advOpen && <AdvancedPanel settings={settings} set={set} papers={papers} />}

        <div className="chat-scroll" ref={scrollRef}>
          {turns.length === 0 ? (
            <EmptyState onAsk={ask} />
          ) : (
            <div className="chat-inner">
              {turns.map((t, i) =>
                t.role === "user" ? (
                  <div className="msg msg-user rise" key={i}><div className="bubble">{t.text}</div></div>
                ) : (
                  <AiMessage key={i} msg={t} onCite={onCite} onFig={openFig} paperTitle={paperTitle} />
                )
              )}
              {busy && lastAssistant && lastAssistant.streaming && !lastAssistant.answer && (
                <div className="msg msg-ai"><div className="ai-head"><div className="ai-avatar"><Icon name="spark" size={14} /></div><span className="ai-name">SpectraRAG</span></div>
                  <div className="ai-body" style={{ color: "var(--text-faint)" }}><span className="mono" style={{ fontSize: 12 }}>{status || "routing query → re-retrieving"}<span className="caret"></span></span></div></div>
              )}
            </div>
          )}
        </div>
        {status && turns.length === 0 && <div className="composer-hint" style={{ padding: "0 18px 8px", color: "var(--text-faint)" }}>{status}</div>}

        <Composer onAsk={ask} busy={busy} />
      </div>

      {layout !== "single" && <RetrievalPanel turn={lastAssistant} highlight={highlight} settings={settings} paperTitle={paperTitle} />}
      <PageRegionModal item={pageItem} onClose={() => setPageItem(null)} paperTitle={paperTitle} />
    </div>
  );
}

window.ChatView = ChatView;
