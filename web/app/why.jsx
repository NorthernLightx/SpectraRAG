/* WHY MULTIMODAL — the pitch, driven by real MMLongBench results in
   /why-multimodal.json. Each card is a question whose answer lives in a figure:
   text_pages = the text retriever's top hits (gold page absent); router_pages =
   the router's top hits (gold page present). */

function PageChips({ pages, gold }) {
  return (
    <span style={{ display: "inline-flex", flexWrap: "wrap", gap: 4 }}>
      {pages.length === 0
        ? <span className="mono" style={{ color: "var(--text-faint)" }}>nothing on-topic</span>
        : pages.map((p, i) => (
          <span key={i} className="mono" style={{
            padding: "1px 6px", borderRadius: 5, fontSize: 11,
            background: gold.includes(p) ? "var(--good, #16a34a)" : "var(--canvas, rgba(127,127,127,.15))",
            color: gold.includes(p) ? "#fff" : "var(--text-dim)",
            fontWeight: gold.includes(p) ? 700 : 400,
          }}>p{p}</span>
        ))}
    </span>
  );
}

function WhyView({ setTab }) {
  const [data, setData] = useState(null);
  const [active, setActive] = useState(0);

  useEffect(() => {
    fetch("/why-multimodal.json")
      .then((r) => (r.ok ? r.json() : { cards: [] }))
      .then(setData)
      .catch(() => setData({ cards: [] }));
  }, []);

  const cards = (data && data.cards) || [];
  const card = cards[active];
  const inList = (g, list) => g.some((x) => list.includes(x));
  const textHit = cards.filter((c) => inList(c.gold_pages, c.text_pages)).length;
  const routerHit = cards.filter((c) => inList(c.gold_pages, c.router_pages)).length;
  const pct = (n, d) => (d ? Math.round((n / d) * 100) : 0);

  return (
    <div className="scroll-view">
      <div className="content-pad why">
        <section className="why-hero">
          <span className="why-eyebrow mono">THE PROBLEM</span>
          <h1 className="serif">Most RAG can't read the figure.</h1>
          <p>A large share of a document's answers live where a text chunker never looks — leaderboard <b>tables</b>, architecture <b>diagrams</b>, values printed inside <b>charts</b>. Embed only the body text and those answers are simply not in the index. SpectraRAG indexes the page images too and routes each question to the store that actually holds the answer. Every example below is a real MMLongBench question whose answer sits in a figure.</p>
        </section>

        {card && (
          <section className="why-example">
            <div className="why-q">
              <span className="mono why-q-label">QUESTION</span>
              <span className="serif">“{card.question}”</span>
            </div>
            {cards.length > 1 && (
              <div className="tag-filters" style={{ margin: "12px 0 4px" }}>
                {cards.map((c, i) => (
                  <button key={c.id} className={"chip" + (i === active ? " on" : "")} onClick={() => setActive(i)}>
                    {c.figure_label || c.id}
                  </button>
                ))}
              </div>
            )}
            <div className="vs-grid">
              <div className="vs-card bad">
                <div className="vs-head"><span className="vs-tag bad">Text-only retrieval</span><Icon name="x" size={16} /></div>
                <div style={{ margin: "6px 0 12px" }}>top-10 pages: <PageChips pages={card.text_pages} gold={card.gold_pages} /></div>
                <p>Gold page <b>p{card.gold_pages[0]}</b> ({card.figure_label}) is <b>not</b> in the text retriever's top hits — the answer is printed in the figure, which never enters the text index. The model has no grounding for it.</p>
                <div className="vs-verdict bad"><Icon name="x" size={13} /> gold page missed</div>
              </div>
              <div className="vs-card good">
                <div className="vs-head"><span className="vs-tag good">SpectraRAG router</span><Icon name="check" size={16} /></div>
                <img src={"/" + card.image} alt={card.figure_label}
                  style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border, rgba(127,127,127,.2))", marginBottom: 12 }} loading="lazy" />
                <div style={{ margin: "0 0 10px" }}>top-10 pages: <PageChips pages={card.router_pages} gold={card.gold_pages} /></div>
                <p>The router flags a figure-bound query, searches the visual store, and pulls gold page <b>p{card.gold_pages[0]}</b>. The model reads the answer off the {card.figure_label}.</p>
                <div className="vs-verdict good"><Icon name="check" size={13} /> grounded · answer: <b>{card.answer}</b></div>
              </div>
            </div>
          </section>
        )}

        <section className="why-chart">
          <div className="why-chart-text">
            <span className="why-eyebrow mono">MMLONGBENCH · FIGURE-BOUND ITEMS</span>
            <h2 className="serif">Routing recovers the gold page that text-only retrieval drops.</h2>
            <p>Across these {cards.length} figure-bound examples, the gold page — the one where the answer actually appears — lands in the text retriever's top-10 <b>{textHit}/{cards.length}</b> times. The router recovers it <b>{routerHit}/{cards.length}</b>. Same query, same corpus, different store.</p>
          </div>
          <div className="bar-chart">
            <div className="bar-row">
              <span className="bar-label">Text-only · {textHit}/{cards.length}</span>
              <div className="bar-track"><div className="bar-fill muted" style={{ width: pct(textHit, cards.length) + "%" }}></div></div>
            </div>
            <div className="bar-row">
              <span className="bar-label">+ router · {routerHit}/{cards.length}</span>
              <div className="bar-track"><div className="bar-fill accent" style={{ width: pct(routerHit, cards.length) + "%" }}></div></div>
            </div>
            <div className="bar-foot mono">gold page present in top-10 · {cards.length} figure-bound MMLongBench items</div>
          </div>
        </section>

        <section className="why-how">
          <span className="why-eyebrow mono">HOW ROUTING WORKS</span>
          <h2 className="serif">One gate, two stores, per turn.</h2>
          <div className="how-steps">
            <div className="how-step">
              <span className="how-n mono">01</span>
              <h3>Classify the query</h3>
              <p>A lightweight router reads the turn (plus prior context) and predicts whether the answer is likely text-bound, figure-bound, or both.</p>
            </div>
            <div className="how-step">
              <span className="how-n mono">02</span>
              <h3>Retrieve from the right store</h3>
              <p>Text routes to a dense bge-m3 passage index; figure-bound queries also pull the page images so the answer's page is in context.</p>
            </div>
            <div className="how-step">
              <span className="how-n mono">03</span>
              <h3>Rerank &amp; cite</h3>
              <p>A cross-encoder reranks the candidates, and the answer cites the exact chunk or page each claim came from.</p>
            </div>
          </div>
        </section>

        <section className="why-cta">
          <div>
            <h2 className="serif">See it route in real time.</h2>
            <p>Ask a figure-bound question and watch the retrieval panel pick the page.</p>
          </div>
          <div className="why-cta-btns">
            <button className="btn primary" onClick={() => setTab("chat")}><Icon name="chat" size={15} /> Open chat</button>
            <button className="btn" onClick={() => setTab("inspection")}><Icon name="inspect" size={15} /> Inspect a trace</button>
          </div>
        </section>
      </div>
    </div>
  );
}

window.WhyView = WhyView;
