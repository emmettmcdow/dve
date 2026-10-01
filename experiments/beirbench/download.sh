#!/bin/sh
# Fetches a BEIR retrieval dataset into data/<name>/.
#
# BEIR is the standard zero-shot retrieval benchmark: 18 datasets, nDCG@10 as the
# headline metric, and published per-dataset scores for most public embedding
# models -- which is the point. Running dve on the same data as a model's own
# published number separates "the embedder is mediocre" from "we are damaging a
# good embedder", and nothing else we measure can tell those apart.
#
# Defaults to scifact: 5k documents and ~300 judged queries, small enough to run
# in minutes and real enough to have human relevance judgments.
#
#   ./download.sh            # scifact
#   ./download.sh nfcorpus   # 3.6k docs, also small
#   ./download.sh cqadupstack  # StackExchange; closest in form to notes
set -eu

NAME="${1:-scifact}"
BASE="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DIR="$(dirname "$0")/data"

mkdir -p "$DIR"
if [ -d "$DIR/$NAME" ]; then
    echo "$DIR/$NAME already exists; delete it to re-download"
    exit 0
fi

echo "fetching $NAME..."
curl -fL --progress-bar -o "$DIR/$NAME.zip" "$BASE/$NAME.zip"
unzip -oq "$DIR/$NAME.zip" -d "$DIR"
rm "$DIR/$NAME.zip"

echo
echo "$NAME:"
wc -l "$DIR/$NAME/corpus.jsonl" "$DIR/$NAME/queries.jsonl" "$DIR/$NAME"/qrels/*.tsv
