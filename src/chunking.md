# What we embed

A note on chunking, not on storage. Nothing here changes `vstore.zig`; it changes how many
vectors ever reach it. Written down because it was found while sizing the in-memory codes
cache and is the largest single lever on that cache's size -- but it is a separate decision,
about *what* we embed rather than *how* we store it, and it is not scheduled.

## The finding

`SENTENCE_SPLIT_DELIMITERS` is `".!?\n"` (`src/embed.zig:811`), so **every newline starts a new
unit**. In prose that is roughly right. In the Simple Wikipedia corpus it is not: mediawiki
markup, link lists, category lines, infobox fields and table cells are all one line each, and
each becomes its own embedded vector.

Measured over 400 random articles from `wikitest/wikidata/md`, applying the real splitter's
delimiters, punctuation strip and `wordlike()` filter:

| | |
|---|---:|
| embeddable chunks per article | 130.0 |
| median chunk length | **2 words** |
| mean chunk length | 3.5 words |

That 130 is not an artifact of the approximation -- the existing `wikitest-db` holds 37,689
vectors from 300 articles, which is 125.6 per article.

Cumulative, and what dropping each tail would do to the full 283,547-article corpus:

| cutoff | share of chunks | vectors over the corpus | 1-bit codes in RAM |
|---|---:|---:|---:|
| (none, as indexed today) | -- | 36.9M | 3.4 GB |
| drop <= 1 word | 34.6% | 24.1M | 2.2 GB |
| drop <= 2 words | 51.3% | 18.0M | 1.6 GB |
| drop <= 3 words | 66.1% | 12.5M | 1.1 GB |
| **drop <= 5 words** | **81.9%** | **6.7M** | **610 MB** |
| drop <= 8 words | 91.9% | 3.0M | 270 MB |

**Four out of five vectors we store are five words or shorter.** A two-word fragment's
embedding carries very little of what semantic search is for, so this is likely costing result
quality as well as memory -- a short chunk is a nearly-free match for a wide range of queries,
which is the shape of a false positive.

## Why it is not being acted on yet

It is a product decision about corpus preparation, and it is entangled with things this note
has not measured:

- **Does dropping them improve results, or just shrink the index?** Plausible, unmeasured.
  `src/benchmark.zig` is the harness that could answer it.
- **What is the right cutoff?** A word count is the crudest possible filter. Dropping markup
  specifically, or merging short adjacent lines into their surrounding paragraph, are both
  better and both more work.
- **Some short chunks are real.** Headings and one-line definitions are short and meaningful.
  A blanket cutoff throws those away too.

The storage work is sized against the corpus as it exists today (~35M vectors, 3.4 GB of 1-bit
codes), so nothing downstream depends on this landing. If it does land, the codes cache gets
5x smaller and easier, which is a reason to keep it in view rather than a reason to rush it.
