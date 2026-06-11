/* Real backend wiring for the SpectraRAG SPA.

   Ported from the prior vanilla chat (web/index.html) so the React views talk
   to the same endpoints with the same battle-tested behaviour: same-origin
   POST /query (with the post-cold-start 503 warm-up retry), POST /query/dci for
   the agentic tier, and client-side OpenRouter generation with the visitor's
   own key (BYOK). Without a key, generation falls back to the server's
   keyless demo path (POST /demo/chat — free model, daily-capped, ADR 0027).
   These helpers return data; the components own the rendering. */
(function () {
  const ORIGIN = window.location.origin;

  // The real, supported model slate (mirrors the prior chat's <select>).
  // The ":free" entries are the demo chain's models, selectable here so a
  // keyed visitor can also run at zero cost (free-tier rate limits apply).
  const MODELS = [
    { id: "openai/gpt-4o-mini", note: "vision · cheapest" },
    { id: "anthropic/claude-sonnet-4.6", note: "vision" },
    { id: "openai/gpt-4o", note: "vision" },
    { id: "qwen/qwen3-vl-32b-instruct", note: "vision · open" },
    { id: "meta-llama/llama-3.1-70b-instruct", note: "text-only" },
    { id: "google/gemma-4-26b-a4b-it:free", note: "vision · free" },
    { id: "nvidia/nemotron-nano-12b-v2-vl:free", note: "vision · free" },
  ];

  // Suggestion chips. Carried over from the prior chat, where each was checked
  // to retrieve its target paper as the top hit against the live corpus. One
  // per modality bucket so the chips always advertise text + figure + table.
  const SUGGESTIONS = [
    { q: "What is exploration hacking?", route: "text" },
    { q: "What does Figure 1 in HERMES++ illustrate about the proposed framework?", route: "visual" },
    { q: "Which surrogate losses are compared by convexity, smoothness, and consistency?", route: "text + visual" },
  ];

  function supportsVision(model) {
    // Mirrors the prior chat: the text-only Llama can't read page images.
    return !model.includes("llama-3.1-70b");
  }

  function pageImageUrl(paperId, page) {
    return `${ORIGIN}/pages/${encodeURIComponent(paperId)}/${encodeURIComponent(paperId)}_p${page}.png`;
  }

  async function loadPapers() {
    try {
      const r = await fetch("/papers");
      return r.ok ? await r.json() : [];
    } catch {
      return [];
    }
  }

  async function loadHealth() {
    try {
      const r = await fetch("/health");
      return await r.json();
    } catch {
      return {};
    }
  }

  // Every figure/table chunk in the index: caption, bbox, page image URL,
  // docling role/label. Used by the Figures gallery and the corpus counts.
  async function loadFigures(limit = 1000) {
    try {
      const r = await fetch(`/figures?limit=${limit}`);
      return r.ok ? await r.json() : [];
    } catch {
      return [];
    }
  }

  // Fresh retrieval for a turn. Returns { results, routing, trace }. `results`
  // are the server's RetrievalResult chunks; `routing` is the route metadata;
  // `trace` is the agent tool-loop (DCI only, else null).
  async function retrieve(query, opts) {
    const {
      topK = 5,
      forceRoute = "",
      routingMode = "",
      paperId = "",
      dci = false,
      apiKey = "",
      onStatus,
    } = opts || {};

    const body = { text: query, top_k: Math.min(Math.max(topK, 1), 20) };
    if (forceRoute) body.force_route = forceRoute;
    if (routingMode) body.routing_mode = routingMode;
    if (paperId) body.filters = { paper_id: paperId };

    // Agentic search (DCI) runs the agent server-side: the key goes in a header
    // (not the body — bodies are logged). No warm-up retry; a 503 here means
    // "no key", not "warming up".
    if (dci) {
      if (!apiKey) {
        throw new Error("Agentic search runs server-side and needs your OpenRouter key.");
      }
      const res = await fetch("/query/dci", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-OpenRouter-Key": apiKey },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
      }
      const data = await res.json();
      return { results: data.results || [], routing: data.routing || null, trace: data.trace || null };
    }

    // The server wires the retriever during lifespan startup, so a /query 503
    // is either Cloud Run still routing to a starting instance (transient) or
    // "Retriever not configured" — a corpus that failed to load, which no
    // amount of waiting fixes. Retry both briefly (transient 503s are real),
    // but say which one is happening; give the permanent case a short budget.
    const start = performance.now();
    while (true) {
      const res = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        const data = await res.json();
        return { results: data.results || [], routing: data.routing || null, trace: null };
      }
      const detail = await res.text();
      const noCorpus = detail.includes("Retriever not configured");
      const budget = noCorpus ? 20000 : 120000;
      if (res.status === 503 && performance.now() - start < budget) {
        onStatus &&
          onStatus(noCorpus
            ? "The server reports no corpus is loaded. Retrying briefly in case it is still starting…"
            : "Server is warming up after a cold start. The first query can take a minute or two. Retrying automatically…");
        await new Promise((r) => setTimeout(r, 3000));
        continue;
      }
      throw new Error(`${res.status} ${res.statusText}: ${detail}`);
    }
  }

  // Condense prior turns + the latest message into one standalone search query.
  function condenseMessages(priorTurns, latest) {
    const transcript = priorTurns
      .map((t) => `${t.role === "user" ? "User" : "Assistant"}: ${t.text || t.answer || ""}`)
      .join("\n");
    return [
      {
        role: "system",
        content:
          "Rewrite the user's latest message into a single standalone search query " +
          "for a corpus of research papers. Resolve pronouns and references using " +
          "the conversation history. Output only the query — no quotes, no preamble.",
      },
      {
        role: "user",
        content:
          `Conversation so far:\n${transcript}\n\n` +
          `Latest user message: ${latest}\n\nStandalone search query:`,
      },
    ];
  }

  // BYOK condense: non-streaming, low max_tokens, the user's chosen model.
  async function condense(apiKey, model, priorTurns, latest) {
    const messages = condenseMessages(priorTurns, latest);
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
        "HTTP-Referer": ORIGIN,
        "X-Title": "SpectraRAG (condense)",
      },
      body: JSON.stringify({ model, messages, temperature: 0, max_tokens: 80 }),
    });
    if (!res.ok) {
      throw new Error(`Condense failed (${res.status}): ${await res.text()}`);
    }
    const data = await res.json();
    return (data.choices?.[0]?.message?.content || "").trim() || latest;
  }

  // Keyless condense through the server's demo path, so follow-up turns
  // retrieve with a resolved query instead of raw text like "what is that".
  // Costs one extra demo-quota call per follow-up; falls back to the raw
  // message on any failure (quota, flaky free endpoint).
  async function condenseDemo(priorTurns, latest) {
    try {
      const { text } = await streamDemoChat(condenseMessages(priorTurns, latest), () => {});
      const q = (text || "").trim().split("\n")[0].trim();
      return q && q.length <= 300 ? q : latest;
    } catch {
      return latest;
    }
  }

  // Build the OpenRouter chat messages. Based on src/prompts/library/answer.yaml
  // v5's refusal contract, relaxed for chat: meta-conversation and follow-ups
  // about the previous answer are answered conversationally rather than
  // refused, and attached page images carry citable ids for figure claims.
  // Fetch a page image (same-origin) and inline it as a base64 data URL.
  // Passing a link (localhost or even the public domain) makes the model's
  // provider fetch it server-side, which fails for localhost and is flaky for
  // public URLs — so we send the bytes inline instead. Returns null on failure.
  async function imageToDataUrl(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) return null;
      const blob = await res.blob();
      return await new Promise((resolve) => {
        const fr = new FileReader();
        fr.onloadend = () => resolve(typeof fr.result === "string" ? fr.result : null);
        fr.onerror = () => resolve(null);
        fr.readAsDataURL(blob);
      });
    } catch {
      return null;
    }
  }

  // When the question names "Figure N" / "Table N", find that element's real
  // page via the figure index (/figures). Retrieval often returns body text
  // that only references the figure from another page — without this, the
  // page that actually shows it never reaches the model. Scoped to papers in
  // the top retrieved chunks; at most two extra pages.
  function referencedFigurePages(question, chunks, figureIndex) {
    if (!Array.isArray(figureIndex) || figureIndex.length === 0) return [];
    // Plural-aware: "figures 2 and 3" / "tables 1, 2" carry one keyword for a
    // list of numbers, so capture the whole number list and split it.
    const refs = [];
    for (const m of question.matchAll(/\b(figs?\.?|figures?|tables?)\s*(\d+(?:\s*(?:,|and|&|–|-)\s*\d+)*)\b/gi)) {
      const isTable = /^t/i.test(m[1]);
      for (const num of m[2].match(/\d+/g) || []) refs.push({ isTable, num });
    }
    if (refs.length === 0) return [];
    const papers = [...new Set(chunks.slice(0, 3).map((c) => c.paper_id))];
    const out = [];
    for (const { isTable, num } of refs) {
      const re = isTable
        ? new RegExp(`^table\\.?\\s*${num}\\b`, "i")
        : new RegExp(`^fig(?:ure)?\\.?\\s*${num}\\b`, "i");
      for (const paperId of papers) {
        const f = figureIndex.find((g) => g.paper_id === paperId && re.test(String(g.caption || "").trim()));
        if (f && typeof f.page_number === "number") {
          out.push({ paperId, page: f.page_number });
          // One page per reference — matching the same "Figure 2" in a second
          // paper would crowd out the question's other references.
          break;
        }
      }
    }
    return out.slice(0, 2);
  }

  async function buildMessages(priorTurns, latestUserText, chunks, useImages, figureIndex) {
    const system = [
      "You are a careful research assistant answering questions from the supplied documents.",
      '- Use only the provided context for factual claims. If the user asks a factual question the context cannot answer, say exactly "Not stated in the provided context." — do not speculate.',
      "- Out-of-domain questions (e.g. about a topic completely unrelated to the chunks) must be refused with the exact phrase above. Do not produce a generic summary of the chunks instead.",
      "- Not every message is a question. If the user is reacting to your previous answer — challenging it, asking what you checked, or making conversation — reply naturally using the prior turns: explain what evidence you used (the chunks and page images you cited) and offer to check something specific. Never use the refusal phrase for such messages.",
      "- Questions about the corpus as a whole (how many papers, which papers cover a topic) cannot be answered from the few retrieved excerpts you see. Say that plainly and point to the Papers and Figures tabs for corpus-wide browsing — do not use the refusal phrase and do not guess a count.",
      "- Analysis questions (implications, comparisons, how an idea could transfer or be useful elsewhere) call for reasoning, not lookup. Reason from the provided context, label that reasoning as your own interpretation, and cite the chunks that anchor it. The refusal phrase is for missing facts only — never for questions that ask you to think.",
      "- When one answer mentions figures from more than one paper, say which paper each figure belongs to.",
      "- If the user asks to see a figure, plot, or graph and the retrieved chunks don't contain one matching the question, say so plainly — do not claim you cannot display images.",
      "- Cite specific chunk IDs when making factual claims by wrapping the literal id in square brackets. Example: if a chunk header is [2604.22753v1::p5::c24], cite it as [2604.22753v1::p5::c24] — NOT [chunk_id 2604.22753v1::p5::c24] and NOT [chunk 24]. Use only ids that appear in the provided context.",
      "- Attached page images are labeled with their own id, e.g. [page image 2604.22753v1::p5::page]. When you describe what a figure, plot, table, or diagram shows based on looking at a page image, cite that page id (e.g. [2604.22753v1::p5::page]) — not a text chunk. Cite text chunk ids only for claims supported by the chunk text itself.",
      "- Several pages may be attached. Cite the id of the page that actually contains the figure you are describing — check the label immediately before the image you read; a page that merely mentions the figure in its text is the wrong citation.",
      "- Prior turns are included for reference. If the user follows up about something from your own previous answer (a term you used, a claim you made) and the current chunks don't cover it, explain it from the previous turn's evidence — without bracket citations — instead of refusing.",
      "- Keep answers concise (3-6 sentences unless the question demands more).",
    ].join("\n");

    const messages = [{ role: "system", content: system }];
    for (const t of priorTurns) {
      messages.push({ role: t.role, content: t.text || t.answer || "" });
    }

    const content = [];
    const seenPages = new Set();
    // Page images average ~0.5 MB as base64; with the ctx slider at 16 an
    // uncapped loop could inline 15+ MB and blow provider payload limits or
    // the demo relay's read timeout. Six chunk pages plus the two referenced-
    // figure extras keeps the body well under that.
    let chunkImages = 0;
    for (const c of chunks) {
      content.push({
        type: "text",
        text: `[chunk ${c.chunk_id}] paper=${c.paper_id} pages=${(c.page_numbers || []).join(",")}\n${c.text || ""}`,
      });
      if (useImages && Array.isArray(c.page_numbers)) {
        for (const page of c.page_numbers) {
          if (chunkImages >= 6) break;
          const key = `${c.paper_id}:${page}`;
          if (seenPages.has(key)) continue;
          seenPages.add(key);
          const dataUrl = await imageToDataUrl(pageImageUrl(c.paper_id, page));
          if (dataUrl) {
            chunkImages += 1;
            // Label the image so visual claims have a citable id (the system
            // prompt directs figure descriptions at these, not text chunks).
            content.push({ type: "text", text: `[page image ${c.paper_id}::p${page}::page]` });
            content.push({ type: "image_url", image_url: { url: dataUrl } });
          }
        }
      }
    }
    // Attach the page that actually shows a figure/table the question names,
    // when retrieval didn't already bring it in.
    if (useImages) {
      for (const ref of referencedFigurePages(latestUserText, chunks, figureIndex)) {
        const key = `${ref.paperId}:${ref.page}`;
        if (seenPages.has(key)) continue;
        seenPages.add(key);
        const dataUrl = await imageToDataUrl(pageImageUrl(ref.paperId, ref.page));
        if (dataUrl) {
          content.push({ type: "text", text: `[page image ${ref.paperId}::p${ref.page}::page]` });
          content.push({ type: "image_url", image_url: { url: dataUrl } });
        }
      }
    }
    content.push({ type: "text", text: `\nQuestion: ${latestUserText}` });
    messages.push({ role: "user", content });
    return messages;
  }

  // Read an OpenRouter-style SSE stream, invoking onDelta(text) per token.
  // Returns { text, usage }. Shared by the BYOK and demo streaming paths.
  async function readSse(res, onDelta) {
    let acc = "";
    let usage = { prompt_tokens: 0, completion_tokens: 0 };
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        let obj = null;
        try {
          obj = JSON.parse(payload);
        } catch {
          continue; // heartbeat / partial — skip
        }
        // OpenRouter delivers mid-stream failures as an error frame on an
        // HTTP-200 stream (common on free-tier endpoints under load).
        // Swallowing it would render a finished, silently empty answer.
        if (obj.error) {
          const err = new Error(obj.error.message || "The model provider failed mid-answer.");
          err.code = "stream_error";
          throw err;
        }
        const delta = obj.choices?.[0]?.delta?.content || "";
        if (delta) {
          acc += delta;
          onDelta(delta);
        }
        if (obj.usage) usage = obj.usage;
      }
    }
    // Flush a final line that arrived without a trailing newline — the
    // usage-bearing frame is often the last thing in the stream.
    const tail = buf.trim();
    if (tail.startsWith("data:")) {
      const payload = tail.slice(5).trim();
      if (payload && payload !== "[DONE]") {
        try {
          const obj = JSON.parse(payload);
          if (obj.usage) usage = obj.usage;
          const delta = obj.choices?.[0]?.delta?.content || "";
          if (delta) {
            acc += delta;
            onDelta(delta);
          }
        } catch { /* partial frame — drop */ }
      }
    }
    return { text: acc, usage };
  }

  // Stream a completion from OpenRouter, invoking onDelta(text) per token.
  // Returns { text, usage }.
  async function streamChat(apiKey, model, messages, onDelta) {
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
        "HTTP-Referer": ORIGIN,
        "X-Title": "SpectraRAG",
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0.2,
        max_tokens: 800,
        stream: true,
        usage: { include: true },
      }),
    });
    if (!res.ok || !res.body) {
      throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
    }
    return readSse(res, onDelta);
  }

  // Keyless path: the server generates with its own caged key on a free
  // model. Model choice, price pinning, and the daily quota all live
  // server-side — the browser only sends messages. A 429 means the shared
  // demo quota ran out; callers surface the bring-your-own-key prompt.
  async function streamDemoChat(messages, onDelta) {
    const res = await fetch("/demo/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    });
    if (res.status === 429) {
      const err = new Error("The free demo hit its daily limit.");
      err.code = "demo_quota";
      throw err;
    }
    if (res.status === 502) {
      // The server tried the whole free-model chain and every endpoint was
      // down (free-pool churn). Keyless visitors have no key to blame.
      const err = new Error("The free demo models are all busy right now. Retry in a minute, or add your own OpenRouter key (top-right).");
      err.code = "demo_down";
      throw err;
    }
    if (!res.ok || !res.body) {
      throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
    }
    return readSse(res, onDelta);
  }

  // Rewrite the model's inline chunk-id citations (`[<paper>::p<N>::c<N>]`, or
  // several comma-separated in one bracket) into compact numeric refs `[1][2]`,
  // preserving first-cited order. Returns { newText, ids } so the citation
  // panel can render a matching ordered list.
  function renumberCitations(text) {
    const idMap = new Map();
    const orderedIds = [];
    const chunkIdRe = /[A-Za-z0-9_.\-]+::p\d+::(?:(?:c|tab|fig)\d+|page)/g;
    const newText = text.replace(/\[([^\]]+)\]/g, (match, inner) => {
      const ids = inner.match(chunkIdRe);
      if (!ids || ids.length === 0) return match;
      return ids
        .map((id) => {
          if (!idMap.has(id)) {
            idMap.set(id, orderedIds.length + 1);
            orderedIds.push(id);
          }
          return `[${idMap.get(id)}]`;
        })
        .join("");
    });
    return { newText, ids: orderedIds };
  }

  // Normalize the server's routing metadata into the { label, path } the
  // RoutePill/route-bars want. path ∈ "text" | "hybrid"/"visual".
  function routeLabel(routing) {
    if (!routing) return "text";
    const path = routing.path || "text";
    if (path === "visual") return "visual";
    if (path === "hybrid" || path.includes("+")) return "text + visual";
    return "text";
  }

  window.RAG = {
    MODELS,
    SUGGESTIONS,
    supportsVision,
    pageImageUrl,
    loadPapers,
    loadHealth,
    loadFigures,
    retrieve,
    condense,
    condenseDemo,
    buildMessages,
    streamChat,
    streamDemoChat,
    renumberCitations,
    routeLabel,
  };
})();
