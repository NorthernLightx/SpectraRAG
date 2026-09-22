# ADR 0033: One reader context for the chat, /answer and the eval

Status: accepted

## Context

ADR 0032 names the reader as the binding constraint: handed the right page it
answers about a third of queries. Every change aimed at that constraint is
judged by the eval, and the eval did not measure what users get.

Three code paths built the reader's input:

- **The chat UI** (`web/app/api.js`, generation in the browser per ADR 0031)
  sent its own system prompt, attached the page image of every retrieved chunk
  up to six, and injected the caption and page of any figure the question named
  by number.
- **`/answer` and `eval_run`** used `Generator` with `answer.yaml`. It attached
  images only for results whose `source` was `visual`, up to four, appended at
  the end of the message with no labels, and injected nothing.
- **The reading experiments** in ADR 0032's amendments ran from scripts in
  `scripts/experiments/`, a third builder.

The two maintained paths also combined badly with page-level fusion. When both
legs find a page, `_fuse_page_level` returns the text chunk for it, so on the
Python path the pages both legs agreed on (the likeliest to be right) reached
the reader without their image.

## Decision

`src/rag/context.py:build_reader_context` builds the reader's messages, and
everything calls it:

- The prompt is `src/prompts/library/chat.yaml`, the chat UI's prompt moved
  verbatim out of `api.js`. A reader prompt's `user_template` takes only
  `{query}`; `Generator` rejects one with `{context}` at construction.
- Each retrieved chunk is followed by the page image of every page it cites,
  up to six images, each preceded by its citable label
  (`[page image <paper>::p<N>::page]`). Text and visual results are treated
  alike, so an agreed page keeps its pixels.
- A figure or table the question names by number, found in the papers of the
  top three chunks, joins the context as its caption plus its page (two at
  most).
- `POST /context` returns the messages with page images as refs. The browser
  swaps each ref for the image bytes and sends the result to the provider on
  the visitor's key, which still never reaches the server (ADR 0031).
- `Generator` resolves the refs to files under `pages_dir` and sends the same
  messages through an `LLMClient`. `eval_run` measures that `Generator`.

`Message.content` accepts a list of text and image parts so a label stays next
to its image in both OpenRouter's content blocks and Ollama's native format.
Image data URLs carry the file's real type; JPEG pages used to go out labelled
`image/png`.

## Consequences

- Generation baselines produced with `answer.yaml` (for example
  `data/eval/baseline.json`) are not comparable with new runs: the prompt, the
  image policy and the default cap (4 to 6) all changed. Re-run both arms
  rather than gating a new run against them. `eval_run` records
  `generator_prompt` in the run config.
- Citation grounding counts a citation as grounded when it names anything the
  reader was shown (`Answer.context_ids`), which now includes page images and
  injected captions.
- A new frontend needs an API that serves `/context`. Deploy the API first.
- `answer.yaml` stays for the legacy and experiment scripts that load it.
  `scripts/experiments/study_context.py` built a `Generator` on it and now
  stops at the template check; like the other experiment scripts, it
  reproduces at the commit it was written for.
