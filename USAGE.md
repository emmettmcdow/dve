# Usage & Installation

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
const dir = try std.fs.cwd().makeOpenPath("my_db", .{});

// VectorDB is generic over the embedding model, which is set at build time.
const VectorDB = dve.VectorDB(dve.embedding_model);
var embedder = try dve.embed.NLEmbedder.init();
const db = try VectorDB.init(allocator, dir, embedder.embedder());
defer db.deinit();

// Embed text. The key identifies the entry (typically a file path).
try db.embedText("doc1", "Machine learning enables computers to learn from data");
// embedTextAsync returns immediately and embeds on a background thread.
try db.embedTextAsync("doc2", "The solar system has eight planets");

// Search returns results ordered by similarity.
var results: [10]dve.SearchResult = undefined;
const n = try db.search("artificial intelligence", &results);
for (results[0..n]) |result| {
    std.debug.print("{s} (similarity: {d:.2})\n", .{ result.path, result.similarity });
}
```

## Swift

### Requirements
- Zig 0.15.1 (to build the XCFramework)
- Xcode 15+

### Install

First, build the XCFramework from the repo root:
```sh
zig build xcframework
```

Then add the package to your project in Xcode via **File → Add Package Dependencies**,
or add it to your `Package.swift`:
```swift
dependencies: [
    .package(path: "/path/to/dve/bindings/swift"),
],
targets: [
    .target(
        name: "MyTarget",
        dependencies: [
            // "swift" is the directory name of the local package, which SPM
            // uses as the package identity for local path dependencies.
            .product(name: "DveKit", package: "swift"),
        ]
    ),
]
```

> **Note:** DveKit will be available via a remote SPM URL once a release is published. Until then, reference the local path above.

### Usage

```swift
import DveKit

// Open (or create) a directory to store the vector database.
let db = try DveDatabase(directory: myURL, model: .appleNL)

// Embed text. The key identifies the entry (typically a file path).
try db.embed(key: "doc1", content: "Machine learning enables computers to learn from data")
// embedAsync returns immediately and embeds on a background thread.
try db.embedAsync(key: "doc2", content: "The solar system has eight planets")

// Search returns results ordered by similarity.
let results = try db.search("artificial intelligence", maxResults: 10)
for result in results {
    print("\(result.key) (similarity: \(result.similarity))")
}
```

### Model selection

By default, dve uses Apple's NaturalLanguage framework — no model files required, good for
development and lower-stakes use cases.

For higher-quality embeddings, use the mpnet model. Download it first:
```sh
./scripts/download_model.sh mpnet
```

Then initialize with explicit model paths:
```swift
let db = try DveDatabase(
    directory: myURL,
    model: .mpnet(
        modelURL: URL(fileURLWithPath: "/path/to/all_mpnet_base_v2.mlpackage"),
        tokenizerURL: URL(fileURLWithPath: "/path/to/tokenizer.json")
    )
)
```

> **Note:** The database format differs between models. Use the same model consistently
> for a given database directory.
