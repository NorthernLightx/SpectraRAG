# ADR 0034: Recordings are documents whose pages are time segments

Status: accepted

## Context

Talks, meetings and calls hold answers the way PDFs do: in what is said and,
for a talk, on its slides. The stack is keyed on pages throughout. Chunk ids
are `doc::pN::cM`, page images are `<doc>_p<N>.png`, fusion is per page, and
the reader labels each page image it is shown. A recording needed a unit that
fits those keys.

## Decision

A recording is a document whose pages are its time segments
([`src/ingestion/media.py`](../../src/ingestion/media.py)).

- A talk video is cut where the slide changes. Grey 64 by 36 thumbnails are
  sampled once a second, and a segment ends when a frame differs from the
  segment's last stable frame by more than 4 grey levels, within 5 to 90
  seconds. The keyframe is the segment's last stable frame, so a cross-fade
  never becomes the page image. The ColQwen2 page index reads it like a
  rendered PDF page.
- An audio file has no slides. Its pages are windows that close at the first
  sentence end after 45 seconds, or at 90 seconds. Audio documents have no
  page images and use the text leg only. Cover art in an audio file is a
  one-picture video stream flagged as attached, and does not make it a video.
- faster-whisper large-v3-turbo transcribes on CPU in int8, with word
  timestamps. A word belongs to the segment holding its midpoint and chunks
  never cross a segment, so each chunk is evidence for one page and carries
  `start_s` and `end_s`.
- A manifest beside the pages records the segment times, the transcriber and
  the segmentation parameters. Gold evidence is labelled in seconds and mapped
  onto the current segments at scoring time
  ([`src/eval/spans.py`](../../src/eval/spans.py)), so a re-segmentation never
  invalidates a label.

Rejected: a separate media retriever with its own fusion, a second
construction path; and Docling's ASR pipeline, which wraps openai-whisper,
reads no video frames and attaches time as a track source the Docling chunker
drops.

On five MCIF talks (28 minutes), large-v3-turbo transcribed at 2.4 times real
time with 6.2 % word error against MCIF's gold transcripts. Whisper small ran
at 4 times real time with 7.4 %, and Parakeet TDT 0.6B v2 at 9 times real time
with 9.1 %, including acronyms such as GPT heard as GPD. Receipt:
[`data/eval/asr-bakeoff-mcif5.json`](../../data/eval/asr-bakeoff-mcif5.json).

## Gold labels without new labelling

Both eval sets reuse human labels; only their unit changes.

### Talks: MCIF

MCIF ([arXiv 2507.19634](https://arxiv.org/abs/2507.19634), CC BY 4.0) has 21
ACL 2023 talks with human-written questions and answers. The
annotators marked where each answer starts and ends in the talk, and the
benchmark build
([`dataset_build/testset_generator.py`](https://github.com/hlt-mt/mcif/blob/main/dataset_build/testset_generator.py))
publishes only the short clip that overlaps that range.
[`scripts/locate_mcif_spans.py`](../../scripts/locate_mcif_spans.py) finds each
clip in its talk's audio by normalised cross-correlation of its first and last
three seconds, which returns the marked range widened to the clip's edges.

- 192 of 195 answerable questions were located, median span 15 seconds. The
  three that did not match are left out.
- All 60 title-slide questions (authors, affiliations, speaker) start within
  15 seconds of their talk.
- The reference answer's words occur in the speech at the span 57 % of the
  time, against 14 % thirty seconds later; the span wins on 124 of 140
  questions.

### Meetings: QMSum on AMI

QMSum ([arXiv 2104.05938](https://arxiv.org/abs/2104.05938), MIT) runs on the
AMI Meeting Corpus (CC BY 4.0). QMSum marks evidence as transcript turns;
AMI times every word by forced alignment.
[`scripts/build_qmsum_golden.py`](../../scripts/build_qmsum_golden.py) matches
each turn's words, in order, against its speaker's timed words.

- Across the 137 product meetings, 131 match at 95 % or better (mean 99.3 %),
  and 38 of 65,280 timed turns run backward by more than five seconds.
- Meetings below the threshold are dropped, and so are two that AMI lists with
  bad timings or an audio offset.
- The test split keeps 18 meetings and 117 specific queries. General queries,
  whose evidence is the whole meeting, are left out.

The machine does not author these labels. It locates a published clip in its
source and converts a turn range to seconds.

## What was measured

Retrieval within each recording on the served `cpu` stack (fingerprint
`c898a7415fec`), against a random ranking of the same recording's pages at the
same depth.

Talks, 192 questions; arms derived from the recorded legs with
`derive_arms --weight 1 --top-k 10`:

| arm | recall@1 | recall@3 | MRR |
|---|---|---|---|
| text and visual, fused | 0.53 | 0.77 | 0.71 |
| text only | 0.50 | 0.71 | 0.69 |
| visual only | 0.48 | 0.76 | 0.68 |
| random | 0.08 | 0.24 | 0.26 |

- Fused beats text only at recall@3 by 0.06 (95 % bootstrap interval 0.01 to
  0.12). Fused and visual only do not differ.
- The MiniLM reranker adds 0.14 recall@1 on transcripts (0.08 to 0.20).
- The visual leg finds the title-slide questions (recall@3 0.92); the text leg
  is ahead on questions about the paper's content. Title-slide questions are
  31 % of the set.

Meetings, 117 queries, text leg. Evidence spans a median two minutes, several
pages, so a hit counts any evidence page in the top k:

| | hit@1 | hit@3 | MRR |
|---|---|---|---|
| served | 0.47 | 0.75 | 0.65 |
| random | 0.17 | 0.39 | 0.32 |

- On evidence of two or three pages, hit@1 is 0.40 against 0.08 at random; on
  four or more pages, 0.54 against 0.26.
- Without the reranker, hit@1 is 0.06 higher, inside the interval -0.03 to
  0.15. It stays on.

Receipts: [`data/eval/mcif-talks-legs.json.gz`](../../data/eval/mcif-talks-legs.json.gz),
[`data/eval/mcif-talks-legs-norerank.json.gz`](../../data/eval/mcif-talks-legs-norerank.json.gz),
[`data/eval/ami-meetings-text.json.gz`](../../data/eval/ami-meetings-text.json.gz),
[`data/eval/ami-meetings-text-norerank.json.gz`](../../data/eval/ami-meetings-text-norerank.json.gz).
Golden sets: [`data/golden/mcif-v1.yaml`](../../data/golden/mcif-v1.yaml),
[`data/golden/qmsum-ami-test-v1.yaml`](../../data/golden/qmsum-ami-test-v1.yaml).

## Consequences

- Talks and audio are searchable through the same API; each result carries
  `start_s` and `end_s`, and `/papers` lists audio recordings from their
  manifests.
- The reader does not see timestamps yet, so answers cite segment ids.
- No speaker labels.
- Transcription takes about 25 minutes of CPU per hour of audio, so recordings
  are indexed by [`scripts/ingest_media.py`](../../scripts/ingest_media.py);
  `POST /ingest` stays PDF only.
- The eval corpora are downloaded, not committed:
  [`scripts/fetch_mcif.py`](../../scripts/fetch_mcif.py) and
  [`scripts/fetch_ami.py`](../../scripts/fetch_ami.py).
- The golden sets carry third-party text. MCIF questions and answers are CC BY
  4.0, with the prompt preamble removed and time spans added. QMSum queries and
  answers are MIT, Copyright (c) 2021 Yale-LILY, with only specific queries kept
  and turn spans converted to seconds using AMI word timings (CC BY 4.0).

## Amendment (2026-09-27): meeting labels v2, and checks that changed nothing

**Stacks.** The talk runs above are not the served stack. They add the
ColQwen2 leg, hybrid routing and a fusion depth of 50 to it, set by flags
because `--profile cpu` refuses `--fusion-depth`; their fingerprint is
`a3ccfc9428b3`. The meeting runs are the served stack, `c898a7415fec`.

**Meeting labels v2.** The 131 of 137 above was an aligner fault. An unmatched
turn leaves its speaker's position where it was, and the aligner searched only
200 words past it, so after one long unmatched turn that speaker could stop
matching for the rest of the meeting. With a 2,000-word window all 135 meetings AMI does not flag match at
98.8 % or better (mean 99.9 %), 835 specific queries across the splits. The
test split is now
[`data/golden/qmsum-ami-test-v2.yaml`](../../data/golden/qmsum-ami-test-v2.yaml):
19 meetings (ES2004b is back) and 123 queries. v1 stays for the runs above.

| v2, served text leg | hit@1 | hit@3 | MRR |
|---|---|---|---|
| served | 0.46 | 0.75 | 0.64 |
| random | 0.17 | 0.39 | 0.32 |

On evidence of two or three pages, hit@1 is 0.39 against 0.08 at random; on
four or more, 0.52 against 0.26. Receipt:
[`data/eval/ami-meetings-text-v2.json.gz`](../../data/eval/ami-meetings-text-v2.json.gz).

**Title slides.** The scorer also reports a ranking that lists a recording's
pages in time order. On talks it gets hit@1 0.33 overall, against 0.57 fused.
On the 62 questions MCIF files as General (authors, affiliations, speaker), it
gets 0.98, against 0.66 for the visual leg and 0.55 fused. Slide text does not
close that gap. Docling OCR of every keyframe, added as one text chunk per
page (299 chunks), moved fused hit@1 by -0.02 (-0.06 to +0.02) and General
hit@1 by -0.05 (-0.15 to +0.05). The names are in the OCR text, but "How many
authors are involved in the paper?" shares no words with a list of names: the
title slide's chunk ranked first in the text leg on 3 of the 62. Slide text is
not indexed. Receipt:
[`data/eval/mcif-talks-legs-slidetext.json.gz`](../../data/eval/mcif-talks-legs-slidetext.json.gz).

**Checks that changed nothing.**

- A visual fusion weight of 1.5 or 2 moves talk hit@3 by +0.02 (-0.02 to
  +0.05) and hit@1 by 0.00. The weight stays 1.
- Turning the reranker's length norm off moves fused talk hit@1 by -0.005
  (-0.03 to +0.02). Receipt:
  [`data/eval/mcif-talks-legs-nolengthnorm.json.gz`](../../data/eval/mcif-talks-legs-nolengthnorm.json.gz).
- Against AMI's manual words, meeting transcripts have a median 27 % word
  error, mostly deletions. A third of the deleted words are fillers, and 5 %
  fall in 19 skips of 20 words or more. Receipt:
  [`data/eval/ami-meetings-asr-wer.json`](../../data/eval/ami-meetings-asr-wer.json).
- Slide-cut segmentation on camera footage cuts on movement. An AMI close-up
  camera cuts 3.9 times a minute and a slide talk 5.1; two AMI cameras are
  still 57 % and 80 % of the time, two slide talks 88 % and 95 %. Neither
  measure leaves a margin to detect camera footage by, so a filmed meeting is
  indexed from its audio track. Receipt:
  [`data/eval/segmentation-camera-vs-slides.json`](../../data/eval/segmentation-camera-vs-slides.json).

The reader now sees each transcript chunk's time range as `time=m:ss-m:ss`
([`src/rag/context.py`](../../src/rag/context.py)).
