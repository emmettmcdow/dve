#!/usr/bin/env python3
"""Split a MediaWiki XML dump into one wikitext file per page.

Streams the dump with iterparse so memory stays flat regardless of dump size.

    ./split_dump.py simplewiki-2026-08-01-p1p1277745.xml -o pages
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path
from urllib.parse import quote
from xml.etree.ElementTree import iterparse

# Characters that are illegal or troublesome in filenames on macOS/Linux/Windows.
ILLEGAL = set('/\\:*?"<>|')
# Leave room under the common 255-byte filename limit for the extension and a
# collision suffix.
MAX_STEM_BYTES = 180


def localname(tag):
    """Strip the '{namespace}' prefix ElementTree puts on every tag."""
    return tag.rpartition('}')[2]


def child(elem, name):
    """Last direct child with the given local name, or None."""
    found = None
    for sub in elem:
        if localname(sub.tag) == name:
            found = sub
    return found


def safe_stem(title):
    """Turn a page title into a filename stem: readable, unique, portable."""
    stem = title.replace(' ', '_')
    stem = ''.join(
        quote(ch, safe='') if ch in ILLEGAL or ord(ch) < 32 else ch
        for ch in stem
    )
    encoded = stem.encode('utf-8')
    if len(encoded) > MAX_STEM_BYTES:
        digest = hashlib.sha1(title.encode('utf-8')).hexdigest()[:8]
        stem = encoded[:MAX_STEM_BYTES].decode('utf-8', 'ignore') + '-' + digest
    if stem.startswith('.') or stem in ('', '_'):
        stem = '_' + stem
    return stem


def iter_pages(path):
    """Yield each <page> element, freeing it before moving on."""
    context = iterparse(path, events=('start', 'end'))
    _, root = next(context)  # start of <mediawiki>
    for event, elem in context:
        if event == 'end' and localname(elem.tag) == 'page':
            yield elem
            root.clear()  # drop the page we just handed out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dump', type=Path, help='MediaWiki XML dump (uncompressed)')
    ap.add_argument('-o', '--out', type=Path, default=Path('pages'),
                    help='output directory (default: pages)')
    ap.add_argument('--ext', default='.wiki', help='file extension (default: .wiki)')
    ap.add_argument('--ns', default='0',
                    help='comma-separated namespace keys to keep, or "all" '
                         '(default: 0, i.e. articles only)')
    ap.add_argument('--redirects', action='store_true',
                    help='also write redirect stubs (skipped by default)')
    ap.add_argument('--shard', action='store_true',
                    help='bucket output into 256 subdirs by title hash; keeps '
                         'directory listings fast for large dumps')
    ap.add_argument('--limit', type=int, help='stop after N written pages')
    args = ap.parse_args()

    keep_ns = None if args.ns == 'all' else {n.strip() for n in args.ns.split(',')}

    args.out.mkdir(parents=True, exist_ok=True)
    if args.shard:
        for i in range(256):
            (args.out / f'{i:02x}').mkdir(exist_ok=True)

    seen = set()
    written = skipped = 0
    start = time.monotonic()

    for page in iter_pages(args.dump):
        title_el = child(page, 'title')
        ns_el = child(page, 'ns')
        if title_el is None or not title_el.text:
            skipped += 1
            continue
        if keep_ns is not None and (ns_el is None or ns_el.text not in keep_ns):
            skipped += 1
            continue
        if not args.redirects and child(page, 'redirect') is not None:
            skipped += 1
            continue

        revision = child(page, 'revision')
        text_el = child(revision, 'text') if revision is not None else None
        if text_el is None or not text_el.text:
            skipped += 1  # deleted/suppressed or genuinely empty
            continue

        stem = safe_stem(title_el.text)
        # casefold: APFS and NTFS are case-insensitive, so "Apple"/"APPLE" collide.
        key = stem.casefold()
        if key in seen:
            page_id = child(page, 'id')
            suffix = page_id.text if page_id is not None and page_id.text else str(written)
            stem = f'{stem}-{suffix}'
            key = stem.casefold()
        seen.add(key)

        dest = args.out
        if args.shard:
            dest = dest / hashlib.sha1(key.encode('utf-8')).hexdigest()[:2]
        (dest / (stem + args.ext)).write_text(text_el.text, encoding='utf-8')

        written += 1
        if written % 10000 == 0:
            rate = written / (time.monotonic() - start)
            print(f'{written:>8} written, {skipped:>8} skipped '
                  f'({rate:,.0f} pages/s)', file=sys.stderr)
        if args.limit and written >= args.limit:
            break

    elapsed = time.monotonic() - start
    print(f'Done: {written} pages written to {args.out}/, {skipped} skipped, '
          f'in {elapsed:.1f}s', file=sys.stderr)


if __name__ == '__main__':
    main()
