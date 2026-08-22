# Usage & Installation
The main interface for this library is defined in [vector.zig](src/vector.zig).

## Model selection

Every embedding backend is compiled into every build of dve. You pick one at the call site by
naming it when you instantiate the engine:

```zig
const VectorEngine = dve.VectorEngine(.mpnet_embedding);
```

- **`.mpnet_embedding`** (`sentence-transformers/all-mpnet-base-v2`) — 768 dimensions, scores 88%
  on our benchmarks. Needs model files; see [Install](#install) below.
- **`.apple_nlembedding`** (Apple NaturalLanguage) — 512 dimensions, scores 66% on our benchmarks.
  Served by the OS, so it needs no model files. Good for quick prototyping.

You own the model files. `.mpnet_embedding` looks for them next to your executable
(`<exe>/../share/`) or in the app bundle's `Resources/`.
`@import("dve").installModels(b, dve_dep)` puts them there for you. Paths can be overridden
per-instance with `.{ .model_path = "...", .tokenizer_path = "..." }`.

> **Note:** The database format differs between models — 768-dim vectors are not readable as
> 512-dim ones. Use the same model consistently for a given database directory.

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
// Install the mpnet model files into your project's zig-out/share/ so the exe can find
// them. Skip this if you only use .apple_nlembedding, which needs no model files.
@import("dve").installModels(b, dve_dep);
```

### Usage

```zig
const dve = @import("dve");

// Open a directory to store the vector database.
const dir = try std.fs.cwd().makeOpenPath("my_vectors", .{});

// Name the model you want; see "Model selection" above.
const VectorEngine = dve.VectorEngine(.mpnet_embedding);
// model files can be changed from their defaults using .{ .model_path = "...", .tokenizer_path = "..." }.
const vectors = try VectorEngine.init(allocator, dir, .{});
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
    .package(url: "https://github.com/emmettmcdow/dve", from: "0.1.2"),
],
targets: [
    .target(
        name: "MyTarget",
        dependencies: [
            .product(name: "DVEKit", package: "dve"),
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
select the `dve/` repo root. Or in your `Package.swift`:
```swift
dependencies: [
    .package(path: "/path/to/dve"),
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
