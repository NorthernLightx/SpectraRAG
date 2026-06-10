/* Shared primitives — exported to window for the other babel scripts. */
const { useState, useEffect, useRef, useMemo, useCallback } = React;

/* Collapse whitespace and truncate to n chars with an ellipsis. */
function clip(s, n) {
  const t = String(s || "").replace(/\s+/g, " ").trim();
  return t.length > n ? t.slice(0, n).trim() + "…" : t;
}

/* ---- icons (simple stroke set) ---- */
const PATHS = {
  chat: "M21 11.5a8.38 8.38 0 0 1-8.5 8.5 9 9 0 0 1-4-1L3 20l1-4.5a8.5 8.5 0 1 1 17-4Z",
  inspect: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM21 21l-4.3-4.3M11 8v6M8 11h6",
  papers: "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6",
  figures: "M3 5h18v14H3zM3 15l5-5 4 4 3-3 6 6",
  why: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM12 16v-4M12 8h.01",
  sun: "M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10ZM12 1v3M12 20v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M1 12h3M20 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1",
  moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z",
  github: "M9 19c-5 1.5-5-2.5-7-3m14 6v-3.9a3.4 3.4 0 0 0-.9-2.6c3-.3 6.2-1.5 6.2-6.7A5.2 5.2 0 0 0 20 4.8 4.8 4.8 0 0 0 19.9 1S18.7.6 16 2.5a13.4 13.4 0 0 0-7 0C6.3.6 5.1 1 5.1 1A4.8 4.8 0 0 0 5 4.8 5.2 5.2 0 0 0 3.7 8.3c0 5.2 3.2 6.4 6.2 6.7a3.4 3.4 0 0 0-.9 2.6V21",
  api: "M16 18l6-6-6-6M8 6l-6 6 6 6",
  server: "M3 4h18v6H3zM3 14h18v6H3zM7 7h.01M7 17h.01M11 7h6M11 17h6",
  arrowRight: "M5 12h14M13 6l6 6-6 6",
  search: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM21 21l-4.3-4.3",
  send: "M22 2 11 13M22 2l-7 20-4-9-9-4 20-7Z",
  plus: "M12 5v14M5 12h14",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM12 16v-4M12 8h.01",
  chevron: "M9 6l6 6-6 6",
  sliders: "M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6",
  check: "M20 6 9 17l-5-5",
  x: "M18 6 6 18M6 6l12 12",
  menu: "M4 6h16M4 12h16M4 18h16",
  route: "M6 3v12a4 4 0 0 0 4 4h8M18 3v0M6 3v0M18 16l3 3-3 3M3 6l3-3 3 3",
  layers: "M12 2 2 7l10 5 10-5-10-5ZM2 17l10 5 10-5M2 12l10 5 10-5",
  image: "M3 5h18v14H3zM8.5 11a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3ZM21 16l-5-5L5 21",
  text: "M4 6h16M4 12h16M4 18h10",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  spark: "M12 2v6M12 16v6M2 12h6M16 12h6M5 5l3 3M16 16l3 3M19 5l-3 3M8 16l-3 3",
  filter: "M3 4h18l-7 8v7l-4-2v-5L3 4Z",
  key: "M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4",
  external: "M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14 21 3",
};
function Icon({ name, size = 16, className = "", strokeWidth = 1.7, fill = false, style }) {
  const d = PATHS[name] || "";
  return (
    <svg className={"ico " + className} width={size} height={size} viewBox="0 0 24 24"
      fill={fill ? "currentColor" : "none"} stroke="currentColor" strokeWidth={strokeWidth}
      strokeLinecap="round" strokeLinejoin="round" style={style} aria-hidden="true">
      {d.split("M").filter(Boolean).map((seg, i) => <path key={i} d={"M" + seg} />)}
    </svg>
  );
}

function RoutePill({ route }) {
  const cls = route === "visual" ? "visual" : route.includes("+") ? "mixed" : "text";
  const label = route === "visual" ? "visual route" : route.includes("+") ? "text + visual" : "text route";
  return <span className={"pill " + cls}><span className="dot"></span>{label}</span>;
}

function ScoreBar({ score, kind = "text", dropped = false }) {
  const cls = ["scorebar", kind === "visual" ? "visual" : "", dropped ? "dropped" : ""].join(" ");
  return <div className={cls}><i style={{ width: Math.round(score * 100) + "%" }}></i></div>;
}

/* ---- tiny markdown -> react (bold, lists, citations) ---- */
function inlineNodes(text, onCite) {
  const out = [];
  const re = /\*\*(.+?)\*\*|\[(F?\d+)\]/g;
  let last = 0, m, k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1] !== undefined) {
      out.push(<strong key={"b" + k++}>{m[1]}</strong>);
    } else {
      const tag = m[2];
      const isFig = tag[0] === "F";
      out.push(
        <sup key={"c" + k++} className={"cite-ref " + (isFig ? "fig" : "")}
          onClick={() => onCite && onCite(tag)} title="Jump to evidence">{tag}</sup>
      );
    }
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
function Markdown({ text, onCite }) {
  const blocks = text.split("\n\n");
  return (
    <div className="md">
      {blocks.map((b, i) => {
        const lines = b.split("\n");
        const isOl = lines.every((l) => /^\d+\.\s/.test(l));
        if (isOl) {
          return (
            <ol key={i}>
              {lines.map((l, j) => <li key={j}>{inlineNodes(l.replace(/^\d+\.\s/, ""), onCite)}</li>)}
            </ol>
          );
        }
        return <p key={i}>{inlineNodes(b, onCite)}</p>;
      })}
    </div>
  );
}

/* shared small components */
function Segmented({ options, value, onChange }) {
  return (
    <div className="segmented">
      {options.map((o) => (
        <button key={o.value} className={"seg" + (value === o.value ? " on" : "")}
          onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

function Tooltip({ label, children }) {
  return <span className="tt" data-tt={label}>{children}</span>;
}

/* ---- source-page modal: real page image + cited-region bbox overlay ---- */
function PageRegionModal({ item, onClose, paperTitle }) {
  const [ov, setOv] = useState(null);     // pixel overlay box, measured off the image
  const [norm, setNorm] = useState(null); // normalized fractions for the side panel
  const imgRef = useRef(null);

  // Place the overlay in pixels off the image's own offset + rendered size. The
  // page PNG renders at 150 DPI (1 pt = 150/72 px). Pixels not % because
  // .pm-page carries CSS padding + a fixed aspect-ratio, so a %-based overlay
  // would measure that box, not the image (ADR 0009 figures/tables, 0021 text).
  const place = useCallback(() => {
    const img = imgRef.current;
    if (!img || !img.naturalWidth || !item || !Array.isArray(item.bbox) || item.bbox.length !== 4) {
      setOv(null); setNorm(null); return;
    }
    const DPI = 150, pW = (img.naturalWidth * 72) / DPI, pH = (img.naturalHeight * 72) / DPI;
    const [x0, y0, x1, y1] = item.bbox;
    const fl = x0 / pW, ft = y0 / pH, fw = (x1 - x0) / pW, fh = (y1 - y0) / pH;
    setNorm({ left: fl, top: ft, width: fw, height: fh });
    setOv({
      top: img.offsetTop + ft * img.clientHeight,
      left: img.offsetLeft + fl * img.clientWidth,
      width: fw * img.clientWidth,
      height: fh * img.clientHeight,
    });
  }, [item]);

  useEffect(() => { setOv(null); setNorm(null); }, [item]);
  useEffect(() => {
    if (!item) return;
    const onEsc = (e) => { if (e.key === "Escape") onClose(); };
    const onResize = () => place();
    document.addEventListener("keydown", onEsc);
    window.addEventListener("resize", onResize);
    return () => { document.removeEventListener("keydown", onEsc); window.removeEventListener("resize", onResize); };
  }, [item, onClose, place]);
  if (!item) return null;
  const isVis = item.kind === "visual";
  const title = paperTitle ? paperTitle(item.paper) : item.paper;
  const hasBbox = Array.isArray(item.bbox) && item.bbox.length === 4;
  const rawQuote = String(item.quote || item.text || "").replace(/\s+/g, " ").trim();
  const quote = rawQuote.length > 320 ? rawQuote.slice(0, 320).trim() + "…" : rawQuote;

  return ReactDOM.createPortal(
    <div className="pm-scrim" onClick={onClose}>
      <div className="pm rise" onClick={(e) => e.stopPropagation()}>
        <div className="pm-pagewrap">
          <div className="pm-page" style={{ position: "relative", padding: 0, overflow: "hidden", aspectRatio: "auto" }}>
            <img ref={imgRef} className="pm-page-img" src={window.RAG.pageImageUrl(item.paper, item.page)}
              alt={`page ${item.page}`} onLoad={place}
              style={{ display: "block", width: "100%", height: "auto" }} />
            {ov && (
              <div className={"pm-region " + (isVis ? "visual" : "text")}
                style={{ position: "absolute", top: ov.top + "px", left: ov.left + "px", width: ov.width + "px", height: ov.height + "px" }}>
                <span className="pm-region-tab">{isVis ? "figure" : "passage"} · selected</span>
              </div>
            )}
            <span className="pm-pagenum mono">p. {item.page}</span>
          </div>
        </div>
        <div className="pm-side">
          <div className="pm-side-head">
            <span className={"pill " + (isVis ? "visual" : "text")}><span className="dot"></span>{isVis ? "visual store" : "text store"}</span>
            <button className="btn ghost sm" onClick={onClose}><Icon name="x" size={15} /></button>
          </div>
          <div className="pm-src mono">{item.paper} · p.{item.page}</div>
          <h3 className="serif pm-paper">{title}</h3>

          {hasBbox && norm && (
            <div className="pm-bbox">
              <span className="section-h" style={{ margin: "0 0 9px" }}>Selected region</span>
              <div className="pm-bbox-grid">
                <div><span className="bk mono">x</span><span className="bv mono">{norm.left.toFixed(3)}</span></div>
                <div><span className="bk mono">y</span><span className="bv mono">{norm.top.toFixed(3)}</span></div>
                <div><span className="bk mono">w</span><span className="bv mono">{norm.width.toFixed(3)}</span></div>
                <div><span className="bk mono">h</span><span className="bv mono">{norm.height.toFixed(3)}</span></div>
              </div>
              <div className="pm-norm mono">normalized page coords · bbox overlay</div>
            </div>
          )}

          {typeof item.score === "number" && (
            <div className="pm-score">
              <div className="pm-score-row"><span className="isk">{isVis ? "patch sim" : "passage sim"}</span><span className="isv mono">{item.score.toFixed(3)}</span></div>
              <ScoreBar score={item.score} kind={item.kind} />
            </div>
          )}

          {quote && <p className="pm-quote serif">{"“" + quote + "”"}</p>}

          <div className="pm-note">
            <Icon name={isVis ? "image" : "text"} size={13} />
            <span>{isVis
              ? "Retrieved from the visual store over page images; the box marks the figure or table region on the source page."
              : "Retrieved from the text store by the dense retriever; the box marks the cited passage on the page."}</span>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}

Object.assign(window, {
  Icon, RoutePill, ScoreBar, Markdown, inlineNodes, Segmented, Tooltip,
  PageRegionModal,
  useState, useEffect, useRef, useMemo, useCallback,
});
