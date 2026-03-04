# dve - dve vector engine
dve is a library for creating and searching vector embeddings locally on Apple devices.

## Usage
### Zig
```zig
const dve = @import("dve");

// Open a directory to store the vector database.
const dir = try std.fs.cwd().makeOpenPath("my_db", .{});

// VectorDB is generic over the embedding model, which is set at build time.
const VectorDB = dve.VectorDB(dve.embedding_model);
// Initialize the embedder. NLEmbedder uses Apple's NaturalLanguage framework.
var embedder = try dve.embed.NLEmbedder.init();
const db = try VectorDB.init(allocator, dir, embedder.embedder());
defer db.deinit();

// Embed text. Each entry is matches with a key. The key can be a path to a file...
try db.embedText("path/to/some/file.md", "The quick brown fox jumps over the lazy dog");
// ... or an arbitrary key.
try db.embedText("machine learning", "Machine learning enables computers to learn from data");
// embedText is blocking, but asyncEmbedText runs on a background thread.
try db.asyncEmbedText("running on a thread", "This runs quickly!");

// Search returns results ordered by similarity.
var results: [10]dve.SearchResult = undefined;
const n = try db.search("computing", &results);
for (results[0..n]) |result| {
    std.debug.print("{s} (similarity: {d:.2})\n", .{ result.path, result.similarity });
}
```

### Swift
TODO

## Install
### Zig
Run the following to add dve to your project:
```sh
zig fetch --save git+https://github.com/emmettmcdow/dve
```

Then in your `build.zig`, fetch the dependency and add it to your compile target:
```zig
const dve_dep = b.dependency("dve", .{
    .target = target,
    .optimize = optimize,
});
const dve_module = dve_dep.module("dve");

// Add to your executable or library
exe.root_module.addImport("dve", dve_module);

// Link required Apple frameworks
exe.root_module.linkFramework("NaturalLanguage", .{});
exe.root_module.linkFramework("CoreML", .{});
exe.root_module.linkFramework("Foundation", .{});
```

### Swift
TODO

## Why
Vector search is a powerful tool for apps, but on-device implementations are surprisingly hard to
come by. Developers typically choose between calling a cloud API (adding cost and a third-party
dependency) or reaching for all-in-one frameworks like Hugging Face Transformers - which are heavy,
slow, and difficult to embed in native applications.

This problem is especially acute on Apple platforms. ML frameworks overwhelmingly prioritize Linux,
CoreML is poorly documented, and cross-platform libraries rarely integrate cleanly into macOS or
iOS apps. dve was built to fill that gap. It starts with Apple - where the need is greatest - with
portability as a core design goal.

## Core Principles
- Portable - dve should use few libraries, such that it can be easily used on any platform.
- Simple - dve should have a simple but configurable interface, with sane defaults. 
- Local - dve should run performantly, and should never make network calls.

## Roadmap
- Add Linux support.
- Add Windows support.
- Multi-modal embedding support.
- Download links within text documents and embed them.
