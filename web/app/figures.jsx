/* FIGURES VIEW — multimodal figure gallery over the real /figures index.
   Each card crops the source page image to the figure's bbox; the lightbox
   shows the full page with the bbox overlaid. */

// Render a caption with KaTeX. Captions carry relatex'd math in $...$ or \(...\)
// (the VLM emits either), so both delimiters are enabled. Caption-only by design.
function MathText({ text, className, style }) {
  const ref = useRef(null);
  // The relatex VLM writes LaTeX-correct `\%` for a literal percent, but KaTeX
  // only renders the math spans, so a prose `\%` shows its backslash. Unescape
  // `\%` -> `%` outside `$...$` (inside math, `%` is a comment so leave it).
  const cleaned = String(text || "").replace(/(\$[^$]*\$)|\\%/g, (_m, math) => math || "%");
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof window.renderMathInElement !== "function") return;
    try {
      window.renderMathInElement(el, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
      });
    } catch (_) { /* leave the raw text on a KaTeX error */ }
  }, [cleaned]);
  return <p ref={ref} className={className} style={style}>{cleaned}</p>;
}

// Drop repeated paragraphs. Table chunks store the caption twice — once as the
// extracted caption, once embedded in Docling's table markdown — and after the
// relatex pass one copy carries LaTeX ($\Delta V$) while the other is still flat
// (∆ V). Normalise to bare alphanumerics so the two collapse to one key, and
// keep the LaTeX copy. Plain duplicate paragraphs (identical) dedupe too.
function normCaption(p) {
  return String(p)
    .replace(/\$/g, "")
    .replace(/\\[a-zA-Z]+/g, "")
    .replace(/[{}^_\\]/g, "")
    .replace(/[^a-zA-Z0-9]/g, "")
    .toLowerCase();
}
function dedupeParagraphs(text) {
  const kept = [];
  const keys = [];
  for (const p of String(text || "").split(/\n{2,}/).map((s) => s.trim()).filter(Boolean)) {
    const k = normCaption(p);
    const idx = k ? keys.indexOf(k) : -1;
    if (idx === -1) { keys.push(k); kept.push(p); }
    else if (p.includes("$") && !kept[idx].includes("$")) { kept[idx] = p; }
  }
  return kept.join("\n\n");
}

// A table chunk's text is "caption\n\n<markdown>" (chunking.table_to_chunk).
// Split it into the descriptive name and the table markdown.
function splitCaptionData(text) {
  const lines = String(text || "").split("\n");
  const firstPipe = lines.findIndex((l) => l.trim().startsWith("|"));
  if (firstPipe === -1) return { name: dedupeParagraphs(text), data: "" };
  return {
    name: dedupeParagraphs(lines.slice(0, firstPipe).join("\n")),
    data: lines.slice(firstPipe).join("\n").trim(),
  };
}

// Crop a 150-DPI page image to a figure's PDF-point bbox. Computes the crop
// transform from the image's natural size on load; falls back to the full page
// width until then (and when a chunk has no bbox).
function FigCrop({ url, bbox, fallbackH = 150 }) {
  const [s, setS] = useState(null);
  const onLoad = (e) => {
    const img = e.target;
    const nW = img.naturalWidth, nH = img.naturalHeight;
    if (!nW || !nH || !Array.isArray(bbox) || bbox.length !== 4) return;
    const DPI = 150, pageW = (nW * 72) / DPI, pageH = (nH * 72) / DPI;
    const [x0, y0, x1, y1] = bbox;
    const fx = x0 / pageW, fy = y0 / pageH;
    const fw = Math.max((x1 - x0) / pageW, 0.02), fh = Math.max((y1 - y0) / pageH, 0.02);
    setS({
      widthPct: 100 / fw,
      leftPct: -(fx / fw) * 100,
      topPct: -(fy / fh) * 100,
      aspect: (fw * nW) / (fh * nH),
    });
  };
  return (
    <div className="fig-crop"
      style={s
        ? { position: "relative", width: "100%", aspectRatio: String(s.aspect), overflow: "hidden", background: "#fff" }
        : { position: "relative", width: "100%", height: fallbackH, overflow: "hidden", background: "#fff" }}>
      <img src={url} alt="" loading="lazy" onLoad={onLoad}
        style={s
          ? { position: "absolute", width: s.widthPct + "%", left: s.leftPct + "%", top: s.topPct + "%", maxWidth: "none" }
          : { width: "100%", display: "block" }} />
    </div>
  );
}

function FigureCard({ f, onOpen }) {
  const { name } = splitCaptionData(f.caption);
  const hasCap = name && !/^\[.+\]$/.test(name.trim());
  return (
    <button className="figure-card" onClick={() => onOpen(f)}>
      <FigCrop url={f.page_image_url} bbox={f.bbox} />
      <div className="figure-card-body">
        {hasCap
          ? (name.includes("$") || name.includes("\\("))
            ? <MathText text={name} className="figure-card-cap" />
            : <div className="figure-card-cap">{name}</div>
          : <div className="figure-card-cap" style={{ color: "var(--text-faint)", fontStyle: "italic" }}>No caption captured</div>}
        <div className="figure-card-meta">
          <span className="mono">{f.paper_id}</span>
          <span className="figure-card-page">p.{f.page_number}</span>
        </div>
        {f.docling_label && (
          <div className="figure-card-title serif">{(f.role || "figure")} · {f.docling_label.replace(/_/g, " ")}</div>
        )}
      </div>
    </button>
  );
}

function FigureLightbox({ f, onClose }) {
  const [ov, setOv] = useState(null);
  const [jsonView, setJsonView] = useState(false);
  const imgRef = useRef(null);

  // Place the bbox overlay in pixels relative to .lb-img, derived from the
  // image's own offset + rendered size. .lb-img has padding:22px and is a grid
  // cell that stretches to the (taller) side column, so a %-based overlay
  // measured the padded/stretched box, not the image — pixels off the image
  // geometry are robust to both.
  const place = useCallback(() => {
    const img = imgRef.current;
    if (!img || !img.naturalWidth || !Array.isArray(f?.bbox) || f.bbox.length !== 4) {
      setOv(null);
      return;
    }
    const DPI = 150, pW = (img.naturalWidth * 72) / DPI, pH = (img.naturalHeight * 72) / DPI;
    const [x0, y0, x1, y1] = f.bbox;
    setOv({
      top: img.offsetTop + (y0 / pH) * img.clientHeight,
      left: img.offsetLeft + (x0 / pW) * img.clientWidth,
      width: ((x1 - x0) / pW) * img.clientWidth,
      height: ((y1 - y0) / pH) * img.clientHeight,
    });
  }, [f]);

  useEffect(() => { setOv(null); setJsonView(false); }, [f]);
  useEffect(() => {
    if (!f) return;
    const onEsc = (e) => { if (e.key === "Escape") onClose(); };
    const onResize = () => place();
    document.addEventListener("keydown", onEsc);
    window.addEventListener("resize", onResize);
    return () => { document.removeEventListener("keydown", onEsc); window.removeEventListener("resize", onResize); };
  }, [f, onClose, place]);
  if (!f) return null;
  const { name, data } = splitCaptionData(f.caption);
  const hasCaption = name && !/^\[.+\]$/.test(name.trim());
  const noCap = <p className="lb-cap" style={{ fontSize: 13, margin: 0, color: "var(--text-faint)", fontStyle: "italic" }}>No caption captured.</p>;
  const jsonObj = {
    paper_id: f.paper_id,
    page: f.page_number,
    type: f.role || "figure",
    ...(f.docling_label ? { docling_label: f.docling_label } : {}),
    caption: hasCaption ? name : null,
    ...(data ? { data } : {}),
  };

  return (
    <div className="lb-scrim" onClick={onClose}>
      <div className="lb rise" onClick={(e) => e.stopPropagation()}>
        <div className="lb-img" style={{ position: "relative" }}>
          <img ref={imgRef} src={f.page_image_url} alt={`page ${f.page_number}`} onLoad={place} style={{ display: "block", width: "100%", height: "auto" }} />
          {ov && (
            <div className="pm-region visual" style={{ position: "absolute", top: ov.top + "px", left: ov.left + "px", width: ov.width + "px", height: ov.height + "px" }}>
              <span className="pm-region-tab">{f.role || "figure"} · selected</span>
            </div>
          )}
        </div>
        <div className="lb-side">
          <div className="lb-side-head">
            <span className="pill visual"><span className="dot"></span>{f.role || "figure"}</span>
            <button className="btn ghost sm" onClick={onClose}><Icon name="x" size={15} /></button>
          </div>
          <div className="lb-fignum mono">{f.paper_id} · page {f.page_number}{f.docling_label ? ` · ${f.docling_label.replace(/_/g, " ")}` : ""}</div>
          <div className="lb-viewtoggle">
            <button className={"vt-btn" + (!jsonView ? " on" : "")} onClick={() => setJsonView(false)}>Fields</button>
            <button className={"vt-btn" + (jsonView ? " on" : "")} onClick={() => setJsonView(true)}>JSON</button>
          </div>
          {jsonView ? (
            <pre className="md-pre mono">{JSON.stringify(jsonObj, null, 2)}</pre>
          ) : data ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <details open className="lb-drop">
                <summary>Caption</summary>
                {hasCaption ? <MathText text={name} className="lb-cap serif" style={{ fontSize: 14, margin: 0 }} /> : noCap}
              </details>
              <details open className="lb-drop">
                <summary>Data</summary>
                <pre className="md-pre mono">{data}</pre>
              </details>
            </div>
          ) : (
            hasCaption ? <MathText text={name} className="lb-cap serif" /> : noCap
          )}
          <hr className="divider" style={{ margin: "16px 0" }} />
          <div className="lb-note">
            <Icon name="route" size={13} />
            <span>Indexed as a {f.role || "figure"} chunk in the visual store; the box marks its region on the source page. {hasCaption ? (f.has_vlm_caption ? "Caption written by a VLM." : "Caption extracted from the document.") : "No caption was captured for this region."}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function FiguresView({ figures }) {
  const [q, setQ] = useState("");
  const [role, setRole] = useState("all");
  const [open, setOpen] = useState(null);

  if (!figures) {
    return <div className="scroll-view"><div className="content-pad"><div className="retr-empty">Loading figures…</div></div></div>;
  }
  const figs = figures;

  const roles = ["all", ...Array.from(new Set(figs.map((f) => f.role || "figure")))];
  const filtered = figs.filter((f) => {
    const okR = role === "all" || (f.role || "figure") === role;
    const okQ = !q || ((f.caption || "") + " " + f.paper_id).toLowerCase().includes(q.toLowerCase());
    return okR && okQ;
  });

  return (
    <div className="scroll-view">
      <div className="list-toolbar">
        <div className="search-box">
          <Icon name="search" size={15} />
          <input placeholder="Search captions…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <div className="tag-filters">
          {roles.map((k) => <button key={k} className={"chip" + (role === k ? " on" : "")} onClick={() => setRole(k)}>{k}</button>)}
        </div>
        <span className="result-count mono">{filtered.length} figures</span>
      </div>
      <div className="content-pad">
        <div className="figure-grid">
          {filtered.map((f) => <FigureCard key={f.chunk_id} f={f} onOpen={setOpen} />)}
        </div>
      </div>
      <FigureLightbox f={open} onClose={() => setOpen(null)} />
    </div>
  );
}

window.FiguresView = FiguresView;
