/* PAPERS VIEW: the real corpus from /papers, with per-paper figure counts
   from the figures index and a detail drawer. Only fields the API actually
   serves are shown (no fabricated authors/venue/citations). */

function PaperCard({ p, figCount, onOpen }) {
  // The id chip duplicates the heading whenever there's no real title: the
  // heading falls back to paper_id (data/paper_titles.json unpopulated). Show
  // the chip only when a distinct title exists, so the id appears once.
  const hasTitle = p.title && p.title.trim() && p.title.trim() !== p.paper_id;
  return (
    <button className="paper-card" onClick={() => onOpen(p)}>
      {(hasTitle || p.is_arxiv) && (
        <div className="paper-card-top">
          {hasTitle && <span className="mono paper-id">{p.paper_id}</span>}
          {p.is_arxiv && <span className="paper-venue">arXiv</span>}
        </div>
      )}
      <h3 className="serif paper-title">{p.title || p.paper_id}</h3>
      <div className="paper-stats">
        <span className="metric"><Icon name="papers" size={12} /> <b>{p.page_count}</b> pages</span>
        <span className="metric"><Icon name="image" size={12} /> <b>{figCount}</b> figs</span>
      </div>
    </button>
  );
}

/* First caption line, with internal [chunk_id] placeholders hidden. The
   Figures gallery cleans these the same way. */
function drawerCaption(raw) {
  const first = String(raw || "").split("\n")[0].trim();
  return !first || /^\[[^\]]+\]$/.test(first) ? "No caption captured" : first;
}

function PaperDrawer({ p, figs, onClose }) {
  const [pageItem, setPageItem] = useState(null);
  useEffect(() => {
    if (!p) return;
    const onEsc = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [p, onClose]);
  if (!p) return null;
  const hasTitle = p.title && p.title.trim() && p.title.trim() !== p.paper_id;
  return (
    <React.Fragment>
    <div className="drawer-scrim" onClick={onClose}>
      <div className="drawer rise-r" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-head">
          {hasTitle && <span className="mono paper-id">{p.paper_id}</span>}
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
                {figs.map((f) => (
                  <div key={f.chunk_id} className="figthumb figthumb-click"
                    onClick={() => setPageItem({ chunk_id: f.chunk_id, paper: f.paper_id, page: f.page_number, pages: [f.page_number], kind: "visual", bbox: f.bbox || null, text: f.caption || "" })}
                    title="View source region on page">
                    <FigCrop url={window.RAG.absPage(f.page_image_url)} bbox={f.bbox} fallbackH={92} eager thumb={window.RAG.figThumbUrl(f.paper_id, f.chunk_id)} />
                    <div className="figthumb-meta"><span className="mono">p.{f.page_number}</span> · {clip(drawerCaption(f.caption), 40)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
    <PageRegionModal item={pageItem} onClose={() => setPageItem(null)} paperTitle={() => p.title || p.paper_id} />
    </React.Fragment>
  );
}

function UploadControl({ onUploaded }) {
  const inputRef = useRef();
  const [status, setStatus] = useState(null);
  const onPick = async (e) => {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";
    if (!file) return;
    setStatus({ kind: "busy", msg: `Ingesting ${file.name}…` });
    try {
      const r = await window.RAG.ingestPdf(file);
      setStatus({ kind: "ok", msg: `Added ${r.paper_id} · ${r.chunks_added} chunks` });
      onUploaded && onUploaded();
    } catch (err) {
      setStatus({ kind: "err", msg: String((err && err.message) || err) });
    }
  };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <input ref={inputRef} type="file" accept="application/pdf,.pdf" style={{ display: "none" }} onChange={onPick} />
      <button className="btn primary sm" disabled={status && status.kind === "busy"}
        onClick={() => inputRef.current && inputRef.current.click()}>
        {status && status.kind === "busy" ? "Ingesting…" : "+ Add PDF"}
      </button>
      {status && status.kind !== "busy" &&
        <span className="mono" style={{ fontSize: 12, color: status.kind === "ok" ? "var(--accent)" : "#e5484d" }}>{status.msg}</span>}
    </div>
  );
}

function PapersView({ setTab, papers, figures, uploadAvailable, onUploaded }) {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState("all");
  const [open, setOpen] = useState(null);

  const figByPaper = useMemo(() => {
    const m = {};
    (figures || []).forEach((f) => { m[f.paper_id] = (m[f.paper_id] || 0) + 1; });
    return m;
  }, [figures]);

  if (!papers || papers.length === 0) {
    return (
      <div className="scroll-view"><div className="content-pad">
        <div className="retr-empty">Loading papers… If this doesn't resolve, the server may be unreachable or no corpus is indexed.</div>
        {uploadAvailable && <div style={{ marginTop: 16, display: "flex", justifyContent: "center" }}><UploadControl onUploaded={onUploaded} /></div>}
      </div></div>
    );
  }

  // Only offer source filters when the corpus actually mixes sources. A
  // permanently empty "other" chip reads as broken.
  const filters = papers.some((p) => !p.is_arxiv) ? ["all", "arxiv", "other"] : ["all"];
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
        {uploadAvailable && <UploadControl onUploaded={onUploaded} />}
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
