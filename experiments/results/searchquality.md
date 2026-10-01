# Search quality: what we can measure, and what we got wrong

Run 2026-10-01 on an Apple M5, `-Dllama -Doptimize=ReleaseFast`.

Until now every quality number dve produced was scored against its own output. `wikitest
--verify` compares the index to an exhaustive exact-cosine scan of our own store, which says
whether stage one surfaced what stage two would have ranked -- not whether that answer is any
good. `src/benchmark.zig` scored the embedder on corpora of 2 to 14 documents.

## 1. BEIR / SciFact: dve is at parity with its embedder

`experiments/beirbench` runs dve against BEIR, which has human relevance judgments and
published per-model scores. SciFact: 5,183 documents, 300 judged test queries, binary
relevance. Ingested as `title\nbody`, scored at the document level by best chunk
(`uniqueSearch`), threshold 0 so the ranking is not truncated.

| metric | llama / nomic-embed-text-v1.5 |
|---|---:|
| nDCG@10 | **0.6939** |
| Recall@100 | 0.9367 |
| MRR@10 | 0.6577 |
| search | 11.2 ms/query at candidates=2000 |

nomic-embed-text-v1.5's own published SciFact nDCG@10 is in roughly this range, so **dve is
not damaging the embedder** on clean prose. Recall@100 of 0.94 says the gold document is
almost always within reach; the loss is ordering inside the top 100, which is a reranking
problem rather than a retrieval one.

This is the measurement that separates "the embedder is mediocre" from "we are wrecking a good
embedder", and it was the one thing no other benchmark here could do.

## 2. A correction: the Wikipedia results were not a ranking failure

The previous write-up pointed at this output from the 1.18M-vector index as evidence that
search was broken:

```
"capital city of france"   recall 1.000
    0.8230  Northern_League_(baseball,_1993-2010).md
    0.8088  Villy,_Yonne.md
```

**Paris was never in the corpus.** That run indexed 15,000 of 283,547 articles chosen by a
seeded shuffle; checking the manifest of the 22,000-article index afterwards finds no
`Paris.md` and no `Photosynthesis.md` either. The system was asked an unanswerable question
and returned the nearest available text, and `recall 1.000` was telling us so -- the index
returned exactly the best-scoring vectors in the store.

`wikitest.md` had already warned about this ("a random 2,000-article sample often has no good
answer to a given question"). It was written down and still read as a quality failure.

## 3. The real defect: short chunks are universal attractors

Asking the 2.7M-vector index four deliberately unanswerable questions:

```
"quarterly revenue by region"   -> 10 results, all above llama's 0.55 threshold
    0.7420  UNRWA.md [1975..1982]
    0.7419  MediaFire.md [466..473]
    0.7419  HSBC_Bank_USA.md [583..590]
    0.7419  Gillet.md [263..270]
    0.7419  Valve_Corporation.md [799..806]
```

Nine documents, **bit-identical** scores, and the offsets are the tell: `[466..473]` is a
seven-byte chunk. These are the two-word fragments `.!?\n` makes out of markdown lists and
infobox fields -- 82% of the corpus is ≤5 words (`src/chunking.md`) -- and a two-word string
sits in a region of the embedding space that is mildly close to everything. They score ~0.74
against an arbitrary question.

So the chunking is the defect, but not by the mechanism previously claimed. Ranking is fine.
The problem is that these chunks are indexed at all, which means:

- an unanswerable query returns a full page of confident-looking garbage, and
- an absolute `threshold` cannot fix it, because the floor is set by chunk *length*, not by
  relevance.

## 4. New cases in src/benchmark.zig

Three groups added, scored into the same totals.

| group | mpnet | nlembedding | llama |
|---|---:|---:|---:|
| markdown structure | 100% | 93.3% | 100% |
| distractor density | 100% | 50.0% | 100% |
| absent answer precision | **62.5%** | **62.5%** | **62.5%** |
| all | 88.5% | 67.3% | 89.6% |

Two of the three pass, and that is a result in itself:

- **markdown structure** -- list items, table rows, headings, front matter, and one real
  sentence buried in stubs. Passes, so markdown structure alone does not break retrieval when
  the gold answer is a real sentence.
- **distractor density** -- 30 near-identical French commune stubs plus one Paris, and 25
  nutrient-deficiency facts plus one scurvy. Passes for mpnet and llama. A few dozen adjacent
  distractors is **not** enough to reproduce the Wikipedia behaviour; the estimate that it
  would be was wrong. It does separate nlembedding at 50%, so it earns its place as an
  embedder discriminator.
- **absent answer precision** -- unanswerable queries against a corpus that *contains stub
  chunks*. 62.5% for every model, identically, which is the signature of a chunking problem
  rather than an embedding one.

The first version of that last test scored 100% and measured nothing: its corpus had no stubs
in it to find. The failure needs the attractors present, not just the answer absent.

## What this changes

Search quality is not broken. It is at embedder parity on judged data. The confirmed defect is
narrower and more fixable than "the results are wrong": **don't index chunks too short to
carry meaning.** `src/chunking.md` already costed that -- 36.9M vectors down to 6.7M -- and it
now has two benchmarks that will show whether it helps: `absent answer precision` should rise,
and SciFact's nDCG@10 should not fall.

## Not measured

- **Scale.** SciFact is 5,183 documents. Neither benchmark reaches the millions of chunks where
  the attractor effect actually bites; the 2.7M evidence above is anecdotal, not scored.
  MS MARCO (8.8M passages) has judgments at that scale and is the obvious next step.
- **Markdown with judgments.** SciFact is prose abstracts, so it does not exercise the chunker
  at all. CQADupStack (StackExchange) is the closest BEIR dataset in form.
- **Lexical queries.** Proper nouns, identifiers, error codes, acronyms -- the things a notes
  user actually types, and where dense retrieval is weakest. No coverage.
