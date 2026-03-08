# dve - dve vector engine
dve is a library for creating and searching vector embeddings locally on Apple devices.
Built with Zig. Experimental Swift bindings are also available.

dve is early-stage and actively developed. Bug reports, issues, and pull requests are welcome on
GitHub.

If you're trying dve and run into trouble, feel free to reach out directly:
[@emmettmcdow](https://github.com/emmettmcdow).

## Usage
```zig
const dve = @import("dve");

// Open a directory to store the vector database.
const dir = try std.fs.cwd().makeOpenPath("my_vectors", .{});

// VectorEngine is generic over the embedding model, which is set at build time.
const VectorEngine = dve.VectorEngine(dve.embedding_model);
var embedder = try dve.embed.NLEmbedder.init();
const vectors = try VectorEngine.init(allocator, dir, embedder.embedder());
defer vectors.deinit();

// Embed text. The key identifies the entry (typically a file path).
try vectors.embedText("doc1", "Machine learning enables computers to learn from data");
try vectors.embedText("doc2", "The solar system has eight planets");

// Search returns results ordered by similarity.
var results: [10]dve.SearchResult = undefined;
const n = try vectors.search("artificial intelligence", &results);
// results[0].path == "doc1"
```

See the [examples](./examples) directory for complete working demos in Zig and Swift.
See [USAGE.md](./USAGE.md) for installation and full usage details.

## Why
Vector search is a powerful tool for apps, but on-device implementations are surprisingly hard to
come by. Developers typically choose between calling a cloud API (adding cost and a third-party
dependency) or reaching for heavy all-in-one frameworks like Hugging Face Transformers.

This problem is especially acute on Apple platforms. ML frameworks overwhelmingly prioritize
Linux servers, CoreML is poorly documented, and cross-platform libraries rarely integrate cleanly
into macOS or iOS apps. dve was built to fill that gap. It starts with Apple with portability as a
core design goal.

## Core Principles
- Portable - dve should use few libraries, such that it can be easily used on any platform.
- Simple - dve should have a simple but configurable interface, with sane defaults.
- Local - dve should run performantly, and should never make network calls.

## Roadmap
- Add Linux support.
- Make Swift bindings more stable.
- Make C/C++ bindings more stable.
- iOS support.
- Multi-modal embedding support.
- Download links within text documents and embed them.
- Support for multiple model types (beyond mpnet and Apple NL).
- Support multiple database instances within a single process.
