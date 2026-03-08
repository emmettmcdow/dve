# Usage & Installation

## Model selection

By default, dve uses Apple's NaturalLanguage framework — no model files required, good for
development but less accurate.

For higher-quality embeddings, use the mpnet model. Download it first:
```sh
./scripts/download_model.sh mpnet
```

Then initialize with explicit model paths (Zig example):
```zig
var embedder = try dve.embed.MPNetEmbedder.init(
    allocator,
    "/path/to/all_mpnet_base_v2.mlpackage",
    "/path/to/tokenizer.json",
);
const vectors = try VectorEngine.init(allocator, dir, embedder.embedder());
```

> **Note:** The database format differs between models. Use the same model consistently
> for a given database directory.

## Zig

### Requirements
- Zig 0.15.1

### Install

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

exe.root_module.addImport("dve", dve_module);

// Link required Apple frameworks
exe.root_module.linkFramework("NaturalLanguage", .{});
exe.root_module.linkFramework("CoreML", .{});
exe.root_module.linkFramework("Foundation", .{});
```

### Usage

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
// embedTextAsync returns immediately and embeds on a background thread.
try vectors.embedTextAsync("doc2", "The solar system has eight planets");

// Search returns results ordered by similarity.
var results: [10]dve.SearchResult = undefined;
const n = try vectors.search("artificial intelligence", &results);
for (results[0..n]) |result| {
    std.debug.print("{s} (similarity: {d:.2})\n", .{ result.path, result.similarity });
}
```

## Swift
> **Experimental:** Swift bindings work but are not yet polished or well-documented. They are
> intended for developers comfortable reading source code and debugging FFI issues themselves.
> First-class Swift support is planned for a future release.

### Requirements
- Zig 0.15.1 (to build the XCFramework)
- Xcode 15+

### Install
First, build the XCFramework from the repo root:
```sh
zig build xcframework
```

If you are using XCode, you can add the dependency using
`File -> Add Package Dependencies -> Add Local`, and selecting `dve/bindings/swift/`.

Or add it to your `Package.swift`:
```swift
dependencies: [
    .package(path: "/path/to/dve/bindings/swift"),
],
targets: [
    .target(
        name: "MyTarget",
        dependencies: [
            .product(name: "DVEKit", package: "swift"),
        ]
    ),
]
```

### Usage

```swift
import DVEKit

// Open (or create) a directory to store the vector database.
let vectors = try VectorEngine(directory: myURL, model: .appleNL)

// Embed text. The key identifies the entry (typically a file path).
try vectors.embed(key: "doc1", content: "Machine learning enables computers to learn from data")
// embedAsync returns immediately and embeds on a background thread.
try vectors.embedAsync(key: "doc2", content: "The solar system has eight planets")

// Search returns results ordered by similarity.
let results = try vectors.search("artificial intelligence", maxResults: 10)
for result in results {
    print("\(result.key) (similarity: \(result.similarity))")
}
```
