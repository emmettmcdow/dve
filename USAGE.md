# Usage & Installation
The main interface for this library is defined in [vector.zig](src/vector.zig).

## Model selection

dve supports two embedding backends:

- **mpnet** (`sentence-transformers/all-mpnet-base-v2`) — default. Scores 88% on our benchmarks.
- **Apple NaturalLanguage** — no model files required, good for quick prototyping. Scores 66%
  on our benchmarks. Opt in with `.@"embedding-model" = .apple_nlembedding`.

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
    // .@"embedding-model" = .apple_nlembedding, // lighter, no model download
});
const dve_module = dve_dep.module("dve");

exe.root_module.addImport("dve", dve_module);

// Link required Apple frameworks
exe.root_module.linkFramework("NaturalLanguage", .{});
exe.root_module.linkFramework("CoreML", .{});
exe.root_module.linkFramework("Foundation", .{});

b.installArtifact(exe);

// Install model files into your project's zig-out/share/ so the exe can find them.
b.getInstallStep().dependOn(dve_dep.builder.getInstallStep());
```

### Usage

```zig
const dve = @import("dve");

// Open a directory to store the vector database.
const dir = try std.fs.cwd().makeOpenPath("my_vectors", .{});

const VectorEngine = dve.VectorEngine(dve.embedding_model);
var embedder = try dve.embed.MpnetEmbedder.init(.{});
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
- Xcode 15+

### Install

**From a release tag** (recommended):
```swift
dependencies: [
    .package(url: "https://github.com/emmettmcdow/dve", from: "0.0.1"),
],
targets: [
    .target(
        name: "MyTarget",
        dependencies: [
            .product(name: "DVEKit", package: "DVEKit"),
        ]
    ),
]
```

**From source** (requires Zig 0.15.1):

Build the XCFramework first:
```sh
zig build xcframework
```

Then add DVEKit as a local package. In Xcode: `File → Add Package Dependencies → Add Local`,
select `dve/bindings/swift/`. Or in your `Package.swift`:
```swift
dependencies: [
    .package(path: "/path/to/dve/bindings/swift"),
],
```

### Usage

```swift
import DVEKit

// Open (or create) a directory to store the vector database.
let vectors = try VectorEngine(directory: myURL)

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

### Release process (for maintainers)

See [DEV.md](DEV.md).
