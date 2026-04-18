# Usage & Installation
The main interface for this library is defined in [vector.zig](src/vector.zig).

## Model selection

dve supports two embedding backends:

- **Apple NaturalLanguage** (default) — no model files required, works out of the box, good for
  development and prototyping. Scores 66% on our benchmarks. 
- **mpnet** (`sentence-transformers/all-mpnet-base-v2`) — higher quality embeddings. Scores 88%
  on our benchmarks.

> **Note:** The database format differs between models. Use the same model consistently
> for a given database directory.

### Using the mpnet model

To use mpnet, pass the `embedding-model` option when declaring the dve dependency in your
`build.zig`:

```zig
const dve_dep = b.dependency("dve", .{
    .target = target,
    .optimize = optimize,
    .@"embedding-model" = .mpnet_embedding,
});
b.getInstallStep().dependOn(dve_dep.builder.getInstallStep());
```

On the first build, Zig will automatically download the pre-converted CoreML model (~400MB)
from GitHub Releases and cache it in your local Zig package cache. Subsequent builds skip the
download.

Then initialize with the model:
```zig
var embedder = try dve.embed.MpnetEmbedder.init(.{});
const vectors = try VectorEngine.init(allocator, dir, embedder.embedder());
```

## Zig

### Requirements
- Zig 0.15.1

### Install

Run the following to add dve to your project:
```sh
zig fetch --save git+https://github.com/emmettmcdow/dve
```

Then in your `build.zig`, fetch the dependency and add it to your compile target. Use
`.apple_nlembedding` (default, no setup required) or `.mpnet_embedding` (see [Model
selection](#model-selection) above):
```zig
const dve_dep = b.dependency("dve", .{
    .target = target,
    .optimize = optimize,
    // .@"embedding-model" = .mpnet_embedding, // uncomment for higher-quality embeddings
});
const dve_module = dve_dep.module("dve");

exe.root_module.addImport("dve", dve_module);

// Link required Apple frameworks
exe.root_module.linkFramework("NaturalLanguage", .{});
exe.root_module.linkFramework("CoreML", .{});
exe.root_module.linkFramework("Foundation", .{});

b.installArtifact(exe);

// If using mpnet, install model files into your project's zig-out/share/ so the
// exe can find them at their default paths.
//const coreml_models = dve_dep.builder.dependency("coreml_models", .{});
//const install_model = b.addInstallDirectory(.{
//    .source_dir = coreml_models.path("all_mpnet_base_v2/all_mpnet_base_v2.mlpackage"),
//    .install_dir = .{ .custom = "share" },
//    .install_subdir = "all_mpnet_base_v2.mlpackage",
//});
//const install_tokenizer = b.addInstallFile(
//    coreml_models.path("all_mpnet_base_v2/tokenizer.json"),
//    "share/tokenizer.json",
//);
//b.getInstallStep().dependOn(&install_model.step);
//b.getInstallStep().dependOn(&install_tokenizer.step);
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
