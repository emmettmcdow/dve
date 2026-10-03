# Examples

Each example embeds a small fixed set of documents, prints them, then drops into an
interactive query loop. Type a query to find the most similar documents, or `quit` to exit.

> The Zig example uses the default mpnet embedding model, so the first build downloads the
> model files. To use the lighter Apple NaturalLanguage model instead, see
> [Model selection](../USAGE.md#model-selection) in USAGE.md.

## Zig

```sh
cd zig
zig build run
```

Pick a different embedding model with `-Dmodel`:

```sh
zig build run -Dmodel=apple_nlembedding
zig build run -Dmodel=llama_nomic_embed_text_v1_5_f32
```
