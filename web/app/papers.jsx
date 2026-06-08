/* PAPERS VIEW — the real corpus from /papers, with per-paper figure counts
   from the figures index and a detail drawer. Only fields the API actually
   serves are shown (no fabricated authors/venue/citations). */

function PaperCard({ p, figCount, onOpen }) {
  return (
    <button className="paper-card" onClick={() => onOpen(p)}>
      <div className="paper-card-top">
        <span className="mono paper-id">{p.paper_id}</span>
        {p.is_arxiv && <span className="paper-venue">arXiv</span>}
      </div>
      <h3 className="serif paper-title">{p.title || p.paper_id}</h3>
      <div className="paper-stats">
        <span className="metric"><Icon name="papers" size={12} /> <b>{p.page_count}</b> pages</span>
        <span className="metric"><Icon name="image" size={12} /> <b>{figCount}</b> figs</span>
      </div>
    </button>
  );
}

function PaperDrawer({ p, figs, onClose }) {
  useEffect(() => {
    if (!p) return;
    const onEsc = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [p, onClose]);
  if (!p) return null;
  return (
    <div className="drawer-scrim" onClick={onClose}>
      <div className="drawer rise-r" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-head">
          <span className="mono paper-id">{p.paper_id}</span>
          <button className="btn ghost sm" onClick={onClose}><Icon name="x" size={15} /></button>
        </div>
        <div className="drawer-body">
          <h2 className="serif" style={{ margin: "0 0 10px", fontSize: 22, lineHeight: 1.25 }}>{p.title || p.paper_id}</h2>
          {p.is_arxiv && p.arxiv_url && (
            <div className="paper-venue" style={{ marginBottom: 16 }}>
              <a href={p.arxiv_url} target="_blank" rel="noopener" style={{ color: "var(--accent)" }}>{p.arxiv_url}</a>
            </div>
          )}

          <div className="drawer-stats">
            <div className="ds"><span className="dsv mono">{p.page_count}</span><span className="dsk">pages</span></div>
            <div className="ds"><span className="dsv mono">{figs.length}</span><span className="dsk">figures indexed</span></div>
            <div className="ds"><span className="dsv mono">1024-d</span><span className="dsk">embeddings</span></div>
          </div>

          {figs.length > 0 && (
            <div style={{ marginTop: 22 }}>
              <h4 className="section-h">Indexed figures</h4>
              <div className="fig-grid-2">
                {figs.slice(0, 8).map((f) => (
                  <div key={f.chunk_id} className="figthumb">
                    <FigCrop url={f.page_image_url} bbox={f.bbox} fallbackH={92} />
                    <div className="figthumb-meta"><span className="mono">p.{f.page_number}</span> · {clip(f.caption, 40)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PapersView({ setTab, papers, figures }) {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState("all");
  const [open, setOpen] = useState(null);

  const figByPaper = useMemo(() => {
    const m = {};
    (figures || []).forEach((f) => { m[f.paper_id] = (m[f.paper_id] || 0) + 1; });
    return m;
  }, [figures]);

  if (!papers || papers.length === 0) {
    return <div className="scroll-view"><div className="content-pad"><div className="retr-empty">Loading papers…</div></div></div>;
  }

  const filters = ["all", "arxiv", "other"];
  const filtered = papers.filter((p) => {
    const okF = filter === "all" || (filter === "arxiv" ? p.is_arxiv : !p.is_arxiv);
    const okQ = !q || ((p.title || "") + " " + p.paper_id).toLowerCase().includes(q.toLowerCase());
    return okF && okQ;
  });

  return (
    <div className="scroll-view">
      <div className="list-toolbar">
        <div className="search-box">
          <Icon name="search" size={15} />
          <input placeholder="Search titles, arXiv id…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <div className="tag-filters">
          {filters.map((t) => (
            <button key={t} className={"chip" + (filter === t ? " on" : "")} onClick={() => setFilter(t)}>{t}</button>
          ))}
        </div>
        <span className="result-count mono">{filtered.length} / {papers.length}</span>
      </div>
      <div className="content-pad">
        <div className="paper-grid">
          {filtered.map((p) => <PaperCard key={p.paper_id} p={p} figCount={figByPaper[p.paper_id] || 0} onOpen={setOpen} />)}
        </div>
      </div>
      <PaperDrawer p={open} figs={open ? (figures || []).filter((f) => f.paper_id === open.paper_id) : []} onClose={() => setOpen(null)} />
    </div>
  );
}

window.PapersView = PapersView;
