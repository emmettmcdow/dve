# Wikitest
This is a benchmark which performs searches against the simple wikipedia corpus.

## Usage
Make sure [pandoc](https://pandoc.org/index.html) is installed. You can do that on a Mac by running
`brew install pandoc`. You will also need python3 and bzip2.

From this directory simply run `download.sh`, and it should output:
- `simplewiki-[DATE]-[VARIOUS META].xml` - this is an XML file with every page from simple
wikipedia.
- `rawmw/*.wiki` - every page contained in `simplewiki*.xml` split into individual files per-page. 
- `md/*.md` - every page from `rawmw/*.wiki` converted to markdown (best effort). This is the main
thing we will be using in our tests.

## ETC
If you are profiling this application, make sure to run the following command to generate a
`.dSYM`:
```bash
dsymutil zig-out/bin/wikitest
```
