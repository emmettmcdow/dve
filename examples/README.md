# Examples

Each example embeds a small fixed set of documents, prints them, then drops into an
interactive query loop. Type a query to find the most similar documents, or `quit` to exit.

> These examples use the default Apple NaturalLanguage embedding model — no model files needed.
> To use the higher-quality mpnet model, see [Model selection](../USAGE.md#model-selection) in USAGE.md.

## Zig

```sh
cd zig
zig build run
```

## Swift (Experimental)

> **Note:** The Swift example works but the bindings are experimental. First-class Swift support
> is planned for a future release.

Requires building the XCFramework first (from the repo root):
```sh
zig build xcframework
```

Then:
```sh
cd swift
swift run dve-repl
```
