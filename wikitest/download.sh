#!/usr/bin/env bash
set -ex

mkdir wikidata
cd wikidata/
curl -LO https://dumps.wikimedia.org/other/mediawiki_content_current/simplewiki/2026-08-01/xml/bzip2/simplewiki-2026-08-01-p1p1277745.xml.bz2
bzip2 -d simplewiki-2026-08-01-p1p1277745.xml.bz2

mkdir rawmw/
../split_dump.py simplewiki-2026-08-01-p1p1277745.xml -o rawmw/ 

mkdir md/
# pandoc's mediawiki reader rejects some real-world wikitext (unbalanced
# templates, parser functions inside <ref>, odd table attributes) and writes
# NO output at all when it does -- roughly 1.3% of pages. So: retry with
# <ref>...</ref> stripped, then fall back to the raw wikitext, and log the
# page instead of aborting the whole run.
find rawmw -name '*.wiki' -print0 |
  xargs -0 -P "$(getconf _NPROCESSORS_ONLN)" -n1 sh -c '
    o=${1##*/}; o="md/${o%.wiki}.md"
    pandoc -f mediawiki -t markdown "$1" -o "$o" 2>/dev/null && exit 0
    perl -0777 -pe "s{<ref\b[^>]*/>}{}g; s{<ref\b[^>]*>.*?</ref>}{}gs" "$1" |
      pandoc -f mediawiki -t markdown -o "$o" 2>/dev/null && exit 0
    cp "$1" "$o"
    echo "$1" >> pandoc-failures.log
  ' _
