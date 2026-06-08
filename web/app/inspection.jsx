/* INSPECTION VIEW — trace a real query through the retrieval pipeline.
   Runs POST /query and shows the actual route decision, the ranked candidates
   with their real scores, and the assembled context. The pipeline strip labels
   each stage with real values; stages the backend doesn't expose internals for
   (embedding, the gate distribution) are described, not fabricated. */

function inspToCand(c) {
  const pages = c.page_numbers || [];
  return {
    chunk_id: c.chunk_id,
    paper: c.paper_id,
    page: pages[0],
    score: typeof c.score === "number" ? c.score : 0,
    kind: c.source === "visual" ? "visual" : "text",
    bbox: (c.metadata && c.metadata.bbox) || null,
    text: c.text || "",
  };
}

function Stage({ icon, label, value, sub, last, selected, onClick }) {
  return (
    <React.Fragment>
      <button className={"stage active" + (selected ? " selected" : "")} onClick={onClick}>
        <div className="stage-icon"><Icon name={icon} size={16} /></div>
        <div className="stage-label">{label}</div>
        <div className="stage-value mono">{value}</div>
        {sub && <div className="stage-sub">{sub}</div>}
      </button>
      {!last && <div className="stage-arrow"><Icon name="arrowRight" size={15} /></div>}
    </React.Fragment>
  );
}

function InspRow({ c, onOpen }) {
  return (
    <div className="insp-row row-clickable" onClick={() => onOpen(c)} title="View source region on page">
      <div className="insp-row-head">
        <span className={"cand-num" + (c.kind === "visual" ? " visual" : "")}>{c.kind === "visual" ? "IMG" : "TXT"}</span>
        <span className="cand-src">{c.paper} · p.{c.page}</span>
        <span className="cand-status kept">score {c.score.toFixed(3)}</span>
      </div>
      <div className="insp-scores">
        <div className="insp-score">
          <span className="isk">relevance</span>
          <ScoreBar score={Math.min(c.score, 1)} kind={c.kind} />
          <span className="isv mono">{c.score.toFixed(3)}</span>
        </div>
      </div>
      {c.text && <div className="cand-quote" style={{ marginTop: 6 }}>{clip(c.text, 150)}</div>}
    </div>
  );
}

function InspectionView({ settings, papers }) {
  const [draft, setDraft] = useState(window.RAG.SUGGESTIONS[0].q);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState(null); // { query, cands, routing }
  const [activeStage, setActiveStage] = useState(null);
  const [pageItem, setPageItem] = useState(null);

  const paperTitle = useCallback(
    (id) => { const p = (papers || []).find((x) => x.paper_id === id); return (p && p.title) || id; },
    [papers]
  );

  const trace = useCallback(async (text) => {
    const v = (text != null ? text : draft).trim();
    if (!v || busy) return;
    setBusy(true); setStatus(""); setActiveStage(null);
    try {
      const { results, routing } = await window.RAG.retrieve(v, {
        topK: settings.topk,
        forceRoute: settings.route === "text" ? "text" : settings.route === "visual" ? "hybrid" : "",
        routingMode: settings.routingMode || "",
        onStatus: setStatus,
      });
      setStatus("");
      setResult({ query: v, cands: results.map(inspToCand), routing });
    } catch (err) {
      setStatus(`Trace failed: ${(err && err.message) || err}`);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }, [draft, busy, settings]);

  const cands = result ? result.cands : [];
  const textCands = cands.filter((c) => c.kind === "text");
  const visCands = cands.filter((c) => c.kind === "visual");
  const routeLabel = result ? window.RAG.routeLabel(result.routing) : "text";

  const stageInfo = {
    query: { icon: "text", title: "Query", body: result ? `Normalized query: “${result.query}”` : "The turn, resolved against conversation history and classified by intent." },
    embed: { icon: "layers", title: "Embed", body: "Encoded with bge-m3 into a 1024-d dense vector (L2-normalized) for nearest-neighbour search. The raw vector isn't surfaced by the API." },
    route: { icon: "route", title: "Route gate", body: result ? `Routing mode: ${result.routing?.mode || settings.routingMode || "default"} · path: ${result.routing?.path || "text"}${result.routing?.forced ? " · forced" : ""}${result.routing?.category ? " · category: " + result.routing.category : ""}` : "A gate predicts which store(s) hold the answer." },
    retrieve: { icon: "search", title: "Retrieve", body: result ? `${cands.length} candidates returned (${textCands.length} text, ${visCands.length} visual) from the dense index.` : "Pulls top-k candidates from each enabled store." },
    rerank: { icon: "filter", title: "Rerank", body: "A MiniLM cross-encoder re-scores the candidate pool; the score shown on each row is the post-rerank relevance." },
    assemble: { icon: "check", title: "Assemble", body: result ? `Top ${cands.length} chunks packed into the generation context (budget ${settings.topk}).` : "Greedily packs the highest-scoring chunks under a token budget." },
  };

  return (
    <div className="scroll-view">
      <div className="list-toolbar" style={{ gap: 10 }}>
        <div className="search-box" style={{ flex: "1 1 440px", maxWidth: 620, height: 38 }}>
          <Icon name="search" size={15} />
          <input placeholder="Type a query to trace through routing, retrieval & rerank…"
            value={draft} onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") trace(); }} />
        </div>
        <button className="btn primary" onClick={() => trace()} disabled={!draft.trim() || busy}>
          <Icon name="route" size={15} /> {busy ? "Tracing…" : "Trace"}
        </button>
        {result && <RoutePill route={routeLabel} />}
      </div>
      <div className="insp-examples">
        <span className="mono insp-ex-label">EXAMPLES</span>
        <div className="tag-filters">
          {window.RAG.SUGGESTIONS.map((s, i) => (
            <button key={i} className="chip" onClick={() => { setDraft(s.q); trace(s.q); }}>{clip(s.q, 38)}</button>
          ))}
        </div>
      </div>

      <div className="content-pad">
        {status && <div className="retr-empty" style={{ marginBottom: 14 }}>{status}</div>}
        {!result && !status && <div className="retr-empty">Run a trace to see the real routing decision, ranked candidates, and assembled context.</div>}

        {result && (
          <React.Fragment>
            <div className="pipeline">
              <Stage icon="text" label="Query" value="1 turn" selected={activeStage === "query"} onClick={() => setActiveStage((a) => a === "query" ? null : "query")} />
              <Stage icon="layers" label="Embed" value="1024-d" sub="bge-m3" selected={activeStage === "embed"} onClick={() => setActiveStage((a) => a === "embed" ? null : "embed")} />
              <Stage icon="route" label="Route" value={result.routing?.path || "text"} sub={result.routing?.mode || "default"} selected={activeStage === "route"} onClick={() => setActiveStage((a) => a === "route" ? null : "route")} />
              <Stage icon="search" label="Retrieve" value={"k=" + settings.topk} sub={cands.length + " cands"} selected={activeStage === "retrieve"} onClick={() => setActiveStage((a) => a === "retrieve" ? null : "retrieve")} />
              <Stage icon="filter" label="Rerank" value="x-enc" sub="MiniLM" selected={activeStage === "rerank"} onClick={() => setActiveStage((a) => a === "rerank" ? null : "rerank")} />
              <Stage icon="check" label="Assemble" value={"top " + cands.length} sub={"budget " + settings.topk} last selected={activeStage === "assemble"} onClick={() => setActiveStage((a) => a === "assemble" ? null : "assemble")} />
            </div>

            {activeStage && stageInfo[activeStage] && (
              <div className="stage-detail rise">
                <div className="sd-head">
                  <div className="sd-icon"><Icon name={stageInfo[activeStage].icon} size={16} /></div>
                  <div className="sd-title"><h4>{stageInfo[activeStage].title}</h4><p>{stageInfo[activeStage].body}</p></div>
                  <button className="btn ghost sm" onClick={() => setActiveStage(null)}><Icon name="x" size={15} /></button>
                </div>
              </div>
            )}

            <div className="insp-stores">
              <div className="insp-store">
                <h4 className="section-h"><span className="dot-tag"><i style={{ background: "var(--accent)" }}></i>Text store</span><span className="result-count mono">{textCands.length}</span></h4>
                {textCands.length ? textCands.map((c, i) => <InspRow key={i} c={c} onOpen={setPageItem} />) : <div className="retr-empty">No text candidates.</div>}
              </div>
              <div className="insp-store">
                <h4 className="section-h"><span className="dot-tag"><i style={{ background: "var(--visual)" }}></i>Visual store</span><span className="result-count mono">{visCands.length}</span></h4>
                {visCands.length ? visCands.map((c, i) => <InspRow key={i} c={c} onOpen={setPageItem} />) : <div className="retr-empty">No visual candidates passed the gate for this query.</div>}
              </div>
            </div>

            <div className="insp-final">
              <h4 className="section-h"><Icon name="check" size={14} /> Assembled context <span className="result-count mono">{cands.length} chunks</span></h4>
              <div className="insp-final-list">
                {cands.map((c, i) => (
                  <div className="insp-final-item row-clickable" key={i} onClick={() => setPageItem(c)} title="View source region on page">
                    <span className="cand-num">{i + 1}</span>
                    <span className={"cand-num" + (c.kind === "visual" ? " visual" : "")} style={{ minWidth: 34, textAlign: "center" }}>{c.kind === "visual" ? "IMG" : "TXT"}</span>
                    <span className="cand-src">{c.paper} · p.{c.page}</span>
                    <div style={{ flex: 1 }}></div>
                    <span className="view-region"><Icon name="search" size={11} /> region</span>
                    <span className="cand-score">{c.score.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </React.Fragment>
        )}
      </div>

      <PageRegionModal item={pageItem} onClose={() => setPageItem(null)} paperTitle={paperTitle} />
    </div>
  );
}

window.InspectionView = InspectionView;
