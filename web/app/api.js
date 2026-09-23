/* Real backend wiring for the SpectraRAG SPA.

   Ported from the prior vanilla chat (web/index.html) so the React views talk
   to the same endpoints with the same battle-tested behaviour: same-origin
   POST /query (with the post-cold-start 503 warm-up retry), POST /query/dci
   for the agentic tier, and client-side generation against the visitor's own
   provider (ADR 0031): OpenRouter with their key, or a local Ollama through
   its OpenAI-compatible endpoint. These helpers return data; the components
   own the rendering. */
(function () {
  const ORIGIN = window.location.origin;
  // API base: same-origin by default (local dev + the monolith deploy). When the
  // frontend is hosted separately, the page sets window.SPECTRARAG_API_BASE to the
  // API service URL (via a local-only config.js) and every call routes there.
  const API = window.SPECTRARAG_API_BASE || ORIGIN;
  const OPENROUTER_URL = "https://openrouter.ai/api/v1";
  // The UI is local-first (served by `spectrarag serve`), so the browser can
  // reach a sibling Ollama directly. Its default CORS allows localhost origins.
  const OLLAMA_URL = "http://localhost:11434";

  // Curated OpenRouter shortlist, pinned above the fetched list in the model
  // menu (and the whole menu when the /models fetch fails). Vision-capable
  // only: the corpus is text+figures and generation attaches page images.
  const PINNED = [
    { id: "openai/gpt-4o-mini", note: "vision · cheapest" },
    { id: "anthropic/claude-sonnet-4.6", note: "vision" },
    { id: "openai/gpt-4o", note: "vision" },
    { id: "qwen/qwen3-vl-32b-instruct", note: "vision · open" },
    { id: "google/gemma-4-26b-a4b-it:free", note: "vision · free" },
    { id: "nvidia/nemotron-nano-12b-v2-vl:free", note: "vision · free" },
  ];

  // Full OpenRouter catalog, vision-capable only. Public endpoint, no key
  // needed. One in-flight/settled promise per page load: the list is large
  // and model churn within a session doesn't matter. Resolves null on failure
  // (the menu then shows just the pins).
  let _orModels = null;
  function loadOpenRouterModels() {
    if (_orModels) return _orModels;
    _orModels = fetch(`${OPENROUTER_URL}/models`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!data || !Array.isArray(data.data)) return null;
        return data.data
          .filter((m) => (m.architecture?.input_modalities || []).includes("image"))
          .map((m) => ({
            id: m.id,
            name: m.name || m.id,
            ctx: m.context_length || 0,
            free: m.id.endsWith(":free"),
          }))
          .sort((a, b) => a.id.localeCompare(b.id));
      })
      .catch(() => null);
    return _orModels;
  }

  // Vision models worth pulling, shown under the installed list so a fresh
  // Ollama gets a one-click start. Tags and download sizes verified against
  // registry.ollama.ai manifests.
  const OLLAMA_SUGGESTED = [
    { id: "qwen2.5vl:3b", note: "3.2 GB" },
    { id: "qwen2.5vl:7b", note: "6.0 GB" },
    { id: "granite3.2-vision:2b", note: "2.4 GB · document-focused" },
    { id: "minicpm-v:8b", note: "5.5 GB" },
    { id: "llama3.2-vision:11b", note: "7.8 GB" },
  ];

  // Stream a model download through Ollama's /api/pull. onProgress gets
  // { status, pct } per frame (pct only while a layer reports total bytes).
  // Resolves when the final "success" frame arrives; rejects on an error
  // frame or an unreachable Ollama.
  async function pullOllamaModel(name, onProgress) {
    const res = await fetch(`${OLLAMA_URL}/api/pull`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: name, stream: true }),
    });
    if (!res.ok || !res.body) {
      throw new Error(ollamaErrorText(await res.text()));
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let ok = false;
    const eat = (line) => {
      if (!line) return;
      let obj = null;
      try {
        obj = JSON.parse(line);
      } catch {
        return;
      }
      if (obj.error) throw new Error(obj.error);
      if (obj.status === "success") ok = true;
      const pct = obj.total ? Math.round(((obj.completed || 0) / obj.total) * 100) : null;
      onProgress({ status: obj.status || "", pct });
    };
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        eat(buf.slice(0, nl).trim());
        buf = buf.slice(nl + 1);
      }
    }
    eat(buf.trim());
    if (!ok) throw new Error("Pull ended without success.");
  }

  // Local Ollama vision models. /api/tags reports per-model `capabilities`
  // (Ollama ≥0.4), so one request tells us which models can read page images.
  // Models with a `remote_host` are ollama.com cloud passthroughs. They can
  // be retired upstream while still listed locally, so each gets an /api/show
  // probe (returns the retirement error without spending cloud quota) and
  // retired ones are dropped from the list. Truly local models sort first.
  // { ok: false } when Ollama isn't reachable; `force` re-probes.
  let _ollamaModels = null;
  function loadOllamaModels(force) {
    if (_ollamaModels && !force) return _ollamaModels;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 2000);
    _ollamaModels = fetch(`${OLLAMA_URL}/api/tags`, { signal: ctrl.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then(async (data) => {
        if (!data || !Array.isArray(data.models)) return { ok: false };
        const models = data.models
          .filter((m) => (m.capabilities || []).includes("vision"))
          .map((m) => {
            const d = m.details || {};
            const cloud = !!m.remote_host;
            const bits = [];
            if (cloud) bits.push("cloud");
            if (d.parameter_size) bits.push(d.parameter_size);
            if (d.context_length) bits.push(`${Math.round(d.context_length / 1000)}k ctx`);
            return { id: m.name, note: bits.join(" · "), cloud, dead: false };
          })
          .sort((a, b) => (a.cloud === b.cloud ? a.id.localeCompare(b.id) : a.cloud ? 1 : -1));
        await Promise.all(models.filter((m) => m.cloud).map(async (m) => {
          try {
            const r = await fetch(`${OLLAMA_URL}/api/show`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ model: m.id }),
            });
            const d = await r.json();
            if (d && d.error) m.dead = true;
          } catch { /* probe failure is not proof of retirement, leave usable */ }
        }));
        return { ok: true, models: models.filter((m) => !m.dead) };
      })
      .catch(() => ({ ok: false }))
      .finally(() => clearTimeout(timer));
    return _ollamaModels;
  }

  // Suggestion chips. Carried over from the prior chat, where each was checked
  // to retrieve its target paper as the top hit against the live corpus. One
  // per modality bucket so the chips always advertise text + figure + table.
  const SUGGESTIONS = [
    { q: "What is exploration hacking?", route: "text" },
    { q: "What does Figure 1 in HERMES++ illustrate about the proposed framework?", route: "visual" },
    { q: "Which surrogate losses are compared by convexity, smoothness, and consistency?", route: "text + visual" },
  ];

  function pageImageUrl(paperId, page) {
    return `${API}/pages/${encodeURIComponent(paperId)}/${encodeURIComponent(paperId)}_p${page}.png`;
  }

  // Pre-rendered figure/table thumbnail (scripts/render_figure_thumbs.py). The
  // file is keyed by chunk_id with ":" → "_" (mirrors the Docling crop name).
  // Small WebP, served same-origin: bundled with the frontend on the split
  // deploy (Firebase), or from the backend /pages mount on the combined deploy.
  // Callers fall back to a full-page CSS-crop when a thumb is absent.
  function figThumbUrl(paperId, chunkId) {
    const safe = chunkId.replace(/:/g, "_");
    return `/pages/${encodeURIComponent(paperId)}/thumbs/${encodeURIComponent(safe)}.webp`;
  }

  // /figures + /papers return page_image_url root-relative (/pages/...). On the
  // split frontend (different origin than the API) it must resolve against the
  // API host, not window.location.
  function absPage(u) {
    return u && u.startsWith("/") ? `${API}${u}` : u;
  }

  // Cloud Run scales to zero, so the first request after idle (or during a
  // redeploy) can transiently fail or 5xx while the container spins up. Retry a
  // few times so a cold start doesn't leave the tab empty with no recovery.
  async function fetchRetry(url, tries = 3) {
    for (let i = 0; ; i++) {
      try {
        const r = await fetch(url);
        if (r.ok || i >= tries - 1) return r;
      } catch (e) {
        if (i >= tries - 1) throw e;
      }
      await new Promise((f) => setTimeout(f, 1200 * (i + 1)));
    }
  }

  async function loadPapers() {
    try {
      const r = await fetchRetry(`${API}/papers`);
      return r.ok ? await r.json() : [];
    } catch {
      return [];
    }
  }

  const WARMING_STATUS = "Server is warming up after a cold start. The first query can take a minute or two. Retrying automatically…";

  // Poll /health until the backend answers. A cold start takes about two
  // minutes, and meanwhile Cloud Run's front end rejects requests with a 5xx or
  // the connection fails; neither says what the backend serves once it is up.
  // Resolves null only after `deadlineMs`. One poll per page: queries await the
  // same promise.
  let _health = null;
  function waitForHealth(deadlineMs = 300000) {
    if (!_health) _health = pollHealth(deadlineMs);
    return _health;
  }
  async function pollHealth(deadlineMs) {
    const start = performance.now();
    for (let wait = 2000; ; wait = Math.min(wait * 1.5, 10000)) {
      try {
        const r = await fetch(`${API}/health`);
        if (r.ok) return await r.json();
      } catch { /* not reachable yet */ }
      if (performance.now() - start + wait > deadlineMs) return null;
      await new Promise((f) => setTimeout(f, wait));
    }
  }

  // Every figure/table chunk in the index: caption, bbox, page image URL,
  // docling role/label. Used by the Figures gallery and the corpus counts.
  async function loadFigures(limit = 1000) {
    try {
      const r = await fetchRetry(`${API}/figures?limit=${limit}`);
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

    // Hold the query until /health has answered. During a cold start Cloud Run
    // holds requests and drops them at the service's 120 s request timeout,
    // which a cold start also takes.
    if (_health) {
      const notice = setTimeout(() => onStatus && onStatus(WARMING_STATUS), 300);
      await _health;
      clearTimeout(notice);
    }

    // Agentic search (DCI) runs the agent server-side: the key goes in a header,
    // not the body (bodies are logged). No warm-up retry; a 503 here means
    // "no key", not "warming up".
    if (dci) {
      if (!apiKey) {
        throw new Error("Agentic search runs server-side and needs your OpenRouter key.");
      }
      const res = await fetch(`${API}/query/dci`, {
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
    // "Retriever not configured", a corpus that failed to load, which no
    // amount of waiting fixes. Retry both briefly (transient 503s are real),
    // but say which one is happening; give the permanent case a short budget.
    // A cold start also shows up as a failed connection or as Cloud Run's front
    // end answering with a non-JSON 5xx; the app's own errors are JSON.
    const start = performance.now();
    while (true) {
      let res;
      try {
        res = await fetch(`${API}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          // Past the service's 120 s request timeout: a hung request surfaces.
          signal: AbortSignal.timeout(150000),
        });
      } catch (err) {
        if (err && err.name === "TimeoutError") throw new Error("The server did not answer within 150 s.");
        if (performance.now() - start >= 120000) throw err;
        onStatus && onStatus(WARMING_STATUS);
        await new Promise((r) => setTimeout(r, 3000));
        continue;
      }
      if (res.ok) {
        const data = await res.json();
        return { results: data.results || [], routing: data.routing || null, trace: null };
      }
      const detail = await res.text();
      const noCorpus = detail.includes("Retriever not configured");
      const budget = noCorpus ? 20000 : 120000;
      const frontEnd5xx = res.status >= 500 && !(res.headers.get("content-type") || "").includes("application/json");
      if ((res.status === 503 || frontEnd5xx) && performance.now() - start < budget) {
        onStatus &&
          onStatus(noCorpus
            ? "The server reports no corpus is loaded. Retrying briefly in case it is still starting…"
            : WARMING_STATUS);
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
          "for a corpus of research papers. If the latest message already makes sense " +
          "on its own, return it unchanged, even when it changes the subject. Only when " +
          "it refers back to the conversation (a pronoun such as \"it\" or \"they\", a " +
          "phrase such as \"that figure\" or \"the paper\", or a follow-up such as \"what " +
          "about X?\"), replace each reference with the subject it points to. Never add " +
          "a paper, method or term from earlier turns that the latest message does not " +
          "refer to. Do not answer the message. Output only the query, with no quotes " +
          "and no preamble.\n\n" +
          "Examples, after a conversation about ResNet:\n" +
          "\"How deep is it?\" becomes: How deep is ResNet?\n" +
          "\"What about its Table 4?\" becomes: What does Table 4 of the ResNet paper show?\n" +
          "\"Which datasets does the BLEU paper use?\" stays: Which datasets does the BLEU paper use?",
      },
      {
        role: "user",
        content:
          `Conversation so far:\n${transcript}\n\n` +
          `Latest user message: ${latest}\n\nStandalone search query:`,
      },
    ];
  }

  // gen = { provider: "openrouter" | "ollama", model, apiKey } is the one
  // object the chat flow threads into every generation call.
  //
  // Ollama goes through its NATIVE /api/chat, not the OpenAI-compat /v1
  // endpoint: /v1 cannot set num_ctx and reloads the model at the runtime
  // default (4096), which rejects any page-image prompt (~16k tokens). The
  // native route takes options.num_ctx per request.
  const OLLAMA_NUM_CTX = 32768;

  // OpenAI-style multimodal content arrays → Ollama-native messages: text
  // parts joined, image data URLs stripped to bare base64 in `images`. The
  // in-text [page image …] labels keep their order, matching the image order.
  function toOllamaMessages(messages) {
    return messages.map((m) => {
      if (typeof m.content === "string") return { role: m.role, content: m.content };
      const texts = [];
      const images = [];
      for (const part of m.content) {
        if (part.type === "text") texts.push(part.text);
        else if (part.type === "image_url") {
          const url = (part.image_url && part.image_url.url) || "";
          const i = url.indexOf("base64,");
          if (i >= 0) images.push(url.slice(i + 7));
        }
      }
      const out = { role: m.role, content: texts.join("\n") };
      if (images.length) out.images = images;
      return out;
    });
  }

  function ollamaBody(gen, messages, { stream, maxTokens, temperature }) {
    return {
      model: gen.model,
      messages: toOllamaMessages(messages),
      stream,
      keep_alive: "10m",
      options: { num_ctx: OLLAMA_NUM_CTX, num_predict: maxTokens, temperature },
    };
  }

  function openrouterRequest(gen, messages, { stream, maxTokens, temperature }) {
    const body = { model: gen.model, messages, temperature, max_tokens: maxTokens, stream };
    if (stream) body.usage = { include: true };
    return {
      url: `${OPENROUTER_URL}/chat/completions`,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${gen.apiKey}`,
        "HTTP-Referer": ORIGIN,
        "X-Title": "SpectraRAG",
      },
      body,
    };
  }

  // Native Ollama error bodies are {"error": "..."} (a string) or
  // {"error": {"message": "..."}} depending on the path. Unwrap either.
  function ollamaErrorText(raw) {
    try {
      const e = JSON.parse(raw).error;
      return (e && (e.message || (typeof e === "string" ? e : ""))) || raw;
    } catch {
      return raw;
    }
  }

  // Condense: non-streaming, low max_tokens, the user's chosen model.
  async function condense(gen, priorTurns, latest) {
    const messages = condenseMessages(priorTurns, latest);
    const opts = { stream: false, maxTokens: 80, temperature: 0 };
    if (gen.provider === "ollama") {
      const res = await fetch(`${OLLAMA_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ollamaBody(gen, messages, opts)),
      });
      if (!res.ok) throw new Error(`Condense failed (${res.status}): ${await res.text()}`);
      const data = await res.json();
      return ((data.message && data.message.content) || "").trim() || latest;
    }
    const req = openrouterRequest(gen, messages, opts);
    const res = await fetch(req.url, {
      method: "POST",
      headers: req.headers,
      body: JSON.stringify(req.body),
    });
    if (!res.ok) {
      throw new Error(`Condense failed (${res.status}): ${await res.text()}`);
    }
    const data = await res.json();
    return (data.choices?.[0]?.message?.content || "").trim() || latest;
  }

  // Fetch a page image (same-origin) and inline it as a base64 data URL.
  // Passing a link (localhost or even the public domain) makes the model's
  // provider fetch it server-side, which fails for localhost and is flaky for
  // public URLs, so we send the bytes inline instead. Returns null on failure.
  async function imageToDataUrl(url) {
    try {
      // cache: "no-store". The retrieval panel's <img> tags fetch these same
      // URLs without an Origin header, and the server only emits
      // Access-Control-Allow-Origin (and Vary: Origin) when Origin is present.
      // Chrome then serves that headerless cached response to this cors-mode
      // fetch and blocks it, so the visually-retrieved page would silently
      // never reach the model on the split-origin deploy.
      const res = await fetch(url, { cache: "no-store" });
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

  // The reader's messages come from the server (POST /context, ADR 0033), built
  // by the same code /answer and the eval use. The browser only swaps each page
  // image ref for the image bytes (fetched in parallel, kept in order) and sends
  // the result to the provider on the visitor's key, which never reaches the
  // server. A page whose image fails to load drops out with its label. The
  // server decides whether pages are attached (it knows what it serves), so a
  // question asked before /health returns still gets its images.
  async function buildMessages(priorTurns, latestUserText, chunks) {
    const res = await fetch(`${API}/context`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: latestUserText,
        results: chunks,
        prior_turns: priorTurns,
      }),
    });
    if (!res.ok) throw new Error(`Context failed (${res.status}): ${await res.text()}`);
    const ctx = await res.json();
    const messages = await Promise.all(
      ctx.messages.map(async (m) => {
        if (typeof m.content === "string") return { role: m.role, content: m.content };
        const resolved = await Promise.all(
          m.content.map(async (part) => {
            if (part.type !== "page_image") return [{ type: "text", text: part.text }];
            const dataUrl = await imageToDataUrl(pageImageUrl(part.paper_id, part.page));
            return dataUrl
              ? [{ type: "text", text: part.label }, { type: "image_url", image_url: { url: dataUrl } }]
              : [];
          }),
        );
        return { role: m.role, content: resolved.flat() };
      }),
    );
    // `injected` rides along so the UI can show this evidence in the panel.
    // It is context the model saw, but it is not a retrieval result.
    const injected = (ctx.injected || []).map((f) => ({
      paperId: f.paper_id,
      page: f.page,
      chunkId: f.chunk_id,
      caption: f.caption,
      bbox: f.bbox || null,
    }));
    return { messages, injected };
  }

  // Read an OpenRouter SSE stream, invoking onDelta(text) per token.
  // Returns { text, usage }.
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
          continue; // heartbeat / partial, skip
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
    // Flush a final line that arrived without a trailing newline. The
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
        } catch { /* partial frame, drop */ }
      }
    }
    return { text: acc, usage };
  }

  // Read Ollama's native NDJSON stream (one JSON object per line), invoking
  // onDelta(text) per token. The final done frame carries the token counts,
  // mapped onto the OpenAI usage shape the UI already renders.
  async function readNdjson(res, onDelta) {
    let acc = "";
    let usage = { prompt_tokens: 0, completion_tokens: 0 };
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    const eat = (line) => {
      if (!line) return;
      let obj = null;
      try {
        obj = JSON.parse(line);
      } catch {
        return; // partial line, skip
      }
      if (obj.error) {
        const err = new Error(ollamaErrorText(line));
        err.code = "stream_error";
        throw err;
      }
      const delta = (obj.message && obj.message.content) || "";
      if (delta) {
        acc += delta;
        onDelta(delta);
      }
      if (obj.done) {
        usage = { prompt_tokens: obj.prompt_eval_count || 0, completion_tokens: obj.eval_count || 0 };
      }
    };
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        eat(buf.slice(0, nl).trim());
        buf = buf.slice(nl + 1);
      }
    }
    eat(buf.trim());
    return { text: acc, usage };
  }

  // Stream a completion from the chosen provider, invoking onDelta(text) per
  // token. Returns { text, usage }.
  async function streamChat(gen, messages, onDelta) {
    const opts = { stream: true, maxTokens: 800, temperature: 0.2 };
    if (gen.provider === "ollama") {
      let res;
      try {
        res = await fetch(`${OLLAMA_URL}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(ollamaBody(gen, messages, opts)),
        });
      } catch {
        const err = new Error("Can't reach Ollama at localhost:11434. Is it running?");
        err.code = "ollama_down";
        throw err;
      }
      if (!res.ok || !res.body) {
        throw new Error(`Ollama: ${ollamaErrorText(await res.text())}`);
      }
      return readNdjson(res, onDelta);
    }
    const req = openrouterRequest(gen, messages, opts);
    const res = await fetch(req.url, {
      method: "POST",
      headers: req.headers,
      body: JSON.stringify(req.body),
    });
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

  // Upload a PDF into the local corpus (POST /ingest, ADR 0029). Multipart
  // form-data. 403 when RAG_ENABLE_UPLOAD is off; the Papers
  // tab hides the control via the upload_available health flag.
  async function ingestPdf(file) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API}/ingest`, { method: "POST", body: form });
    if (!res.ok) {
      let detail = await res.text();
      try { detail = JSON.parse(detail).detail || detail; } catch { /* keep raw text */ }
      throw new Error(detail);
    }
    return res.json(); // { paper_id, chunks_added, corpus_chunks }
  }

  window.RAG = {
    PINNED,
    SUGGESTIONS,
    loadOpenRouterModels,
    loadOllamaModels,
    OLLAMA_SUGGESTED,
    pullOllamaModel,
    pageImageUrl,
    figThumbUrl,
    absPage,
    loadPapers,
    waitForHealth,
    loadFigures,
    ingestPdf,
    retrieve,
    condense,
    buildMessages,
    streamChat,
    renumberCitations,
    routeLabel,
  };
})();
