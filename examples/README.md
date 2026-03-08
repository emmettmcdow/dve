# Examples

Each example embeds a small fixed set of documents, prints them, then drops into an
interactive query loop. Type a query to find the most similar documents, or `quit` to exit.

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

### Model selection

By default, the example uses Apple's NaturalLanguage framework — no model files needed.

To use the higher-quality mpnet model, download it first:
```sh
./scripts/download_model.sh mpnet
```

Then set environment variables when running:
```sh
DVE_MODEL_PATH=/path/to/all_mpnet_base_v2.mlpackage \
DVE_TOKENIZER_PATH=/path/to/tokenizer.json \
swift run dve-repl
```

> **Note:** The database format differs between models. Use the same model consistently for a given database directory.
