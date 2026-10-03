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

---

# Appendix: why the CoreML backend cannot use the GPU [2026-10-02]

The mpnet tests aborted about one process in twelve inside
MetalPerformanceShadersGraph with `shape.count = 0 != strides.count = 3`. The fix shipped
was to stop offering CoreML the GPU (`compute_units = .cpu_and_neural_engine`). This is
the follow-up asking whether the conversion was at fault, since the model is converted by
hand from sentence-transformers via coremltools.

Two conversion settings were suspects: `compute_precision=FLOAT32` and
`minimum_deployment_target=macOS13`. Both were tested against the shipped model.

| conversion | GPU aborts | vectors | speed |
|---|---:|---|---:|
| fp32 / macOS13 (shipped) | 9/124 | reference | 51.1 chunks/s |
| fp32 / macOS15 | 1/40 | cosine **1.000000** vs reference | 55.6 chunks/s |
| fp16 / macOS15 | 0/66 | **broken** | 225.5 chunks/s |
| fp32 / either, CPU+ANE | 0/40 | reference | 30.7 chunks/s |

**The opset is not the cause.** macOS15 produces bit-identical vectors and still aborts.

**fp16 is not usable for this model.** It never aborted and ran 4.4x faster, and it is
also wrong:

| | dog~puppy | dog~airplane | cat-on-mat ~ feline-on-rug | cat-on-mat ~ QCD |
|---|---:|---:|---:|---:|
| fp32 | 0.778 | 0.301 | 0.708 | 0.007 |
| fp16 | 1.000 | 1.000 | 0.315 | 0.811 |

Short texts collapse onto one vector and unrelated pairs outscore related ones, which is
saturation somewhere in the network rather than quantization noise. BERT-family models are
known to need op-level exclusions to survive fp16. Had this been judged on abort rate and
throughput alone it would have looked like a 4.4x win.

That also explains the throughput table: the Neural Engine is fp16-only, so an fp32 model
cannot run on it. `cpu_and_neural_engine` on the shipped model is really *CPU*, which is why
it is slower than the GPU rather than faster, and why fp16 scored the same (225.5 / 221.8) on
`all` and `cpu_and_neural_engine` -- it was on the ANE both times.

## So, for CoreML as a general backend

The crash is Apple's, on the fp32 GPU path, and reachable by an ordinary graph: 720 ops, 16 op
types, and 86 rank-0 constants that are just layer-norm epsilons and the attention scale. We
did not do anything unusual to provoke it.

What that leaves for a second CoreML model:

- Convert **fp16** if the model survives it, and verify semantically rather than by cosine
  against an fp32 build -- an fp16 model that is broken still self-consistently embeds.
  `models/gen-coreml.py --precision` now exposes this, defaulting to fp16 with the hazard
  written down next to the flag.
- fp16 is also the only way onto the Neural Engine, which is worth 4.4x.
- A model that needs fp32 is CPU-only in practice, because the GPU aborts and the ANE will
  not take it.

One trap was removed along the way: the output `MLMultiArray` was read as `[*]f32` without
checking `dataType`. Any fp16-output model would have returned silent nonsense -- measured
0.13 cosine when that was first suspected here. `embed` now switches on the reported dataType.
The Float16 branch is **unexercised**: every model to hand declares an fp32 output even when
its weights are fp16.

---

# Appendix: fp16, and the Neural Engine [2026-10-03]

The previous appendix concluded that mpnet could not be converted to fp16. That was wrong --
or rather, it was right about the naive conversion and wrong to stop there.

## The cause was one constant

Scanning the fp32 MIL program for values outside fp16's ±65504 range finds exactly one:

```
ops with immediate constants exceeding fp16 max (65504):
  const            max |value| = 3.403e+38  (1 values)
```

That is `torch.finfo(float32).min`, the value transformers uses for masked-out positions.
The mask is applied as `(1 - mask) * value`, so in fp16 that constant becomes `-inf` and a
*real* token computes `0 * -inf = NaN`. The NaN spreads through attention and flattens every
embedding, which is why the first attempt scored `dog~airplane` at 1.000.

`models/gen-coreml.py` now overrides `get_extended_attention_mask` to use -1e4, which is the
value original BERT used. `exp(-1e4)` underflows to zero in fp32 as well as fp16, so softmax
sees the same thing either way. After the change the program contains no constant outside
fp16 range.

## Which makes the Neural Engine reachable

| config | chunks/s | aborts | agreement |
|---|---:|---:|---|
| fp32, GPU | 51.1 | 9/124 | reference |
| fp32, CPU+ANE (really CPU) | 30.7 | 0/40 | reference |
| fp16-safe, `all` | 226.5 | 0/40 | cosine 0.999982 |
| **fp16-safe, CPU+ANE** | **248.8** | **0/40** | cosine 0.999982 |

**8.1x the shipped configuration**, no aborts, and correctness preserved:

| | dog~puppy | dog~airplane | cat-on-mat ~ feline-on-rug | cat-on-mat ~ QCD |
|---|---:|---:|---:|---:|
| fp32 | 0.778 | 0.301 | 0.708 | 0.007 |
| fp16-safe | 0.779 | 0.302 | 0.708 | 0.007 |

Every group in `src/benchmark.zig` scores identically, total 88.5% either way.

Note the ordering of those first two rows: the Neural Engine is the fastest engine on this
hardware and the GPU is the middle one, 4.4x behind it. For CoreML the ANE is the target, and
fp16 is the price of entry -- an fp32 model cannot use it at all.

## Cost: fidelity to the reference, slightly

Against the Python reference implementation, fp16 drifts about 1e-3 per component:

```
c0: ref 0.02624974 got 0.02701905 delta 7.69e-4
c1: ref 0.01339556 got 0.01353736 delta 1.42e-4
c2: ref -0.00453320 got -0.00417337 delta 3.60e-4
sum: ref -0.21557690 got -0.21755816 delta 1.98e-3
```

`embed - mpnetembed solo` caught this, which is the test doing its job. Its bounds were
widened to 2e-3 per component and 5e-3 on the sum -- sized to catch a model that is wrong
rather than one that is imprecise, and the broken conversion would not have squeezed through.
It also gained a guard that does not care about precision at all: `dog~puppy` must beat
`dog~airplane` by 0.2. That is the assertion that would have caught the first attempt, and
no numeric tolerance would have.

## Releasing it

The model ships as a GitHub release tarball, so this cannot be finished from here.
`models/coreml_models_v5.tar.gz` is built and its package hash computed:

```
coreml_models-5.0.0-AAAAAPi8Cg30SeBVYopp2lv7QG-GGvb099KEn_PG_s2l
```

Until it is published, `InitOptions.compute_units` stays at `cpu_and_neural_engine`: the
released model is still fp32, and `all` would hand it the GPU and bring the abort back. The
default and the dependency have to move together.
