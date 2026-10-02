# Usage & Installation
The main interface for this library is defined in [vector.zig](src/vector.zig).

## Requirements
- Zig 0.15.1

## Install
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

## Short Example

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

## Model selection
dve can be instantiated with one of a variety of embedding models.
```zig
const VectorEngine = dve.VectorEngine(.mpnet_embedding);
```

- **`.mpnet_embedding`** (`sentence-transformers/all-mpnet-base-v2`) — 768 dimensions, scores 88%
  on our benchmarks. Needs model files; see [Install](#install) below.
- **`.apple_nlembedding`** (Apple NaturalLanguage) — 512 dimensions, scores 66% on our benchmarks.
  Served by the OS, so it needs no model files. Good for quick prototyping.
- **`.llama_nomic_embed_text_v1_5_f32`** (llama.cpp / nomic-embed-text-v1.5) — 768 dimensions,
  scores 90% on our benchmarks, with an 8192-token context. The one backend that is *not*
  compiled in by default: it needs a prebuilt llama.cpp and `-Dllama`. See
  [llama.cpp backend](#llamacpp-backend).

You own the model files. `.mpnet_embedding` looks for them next to your executable
(`<exe>/../share/`) or in the app bundle's `Resources/`.
`@import("dve").installModels(b, dve_dep)` puts them there for you. Paths can be overridden
per-instance with `.{ .model_path = "...", .tokenizer_path = "..." }`.

> **Note:** The database format differs between models — 768-dim vectors are not readable as
> 512-dim ones. Use the same model consistently for a given database directory.

### llama.cpp backend
This backend is the only one with a dependency outside the Zig package graph, so unlike the other
two it is not compiled in by default: you opt in with `-Dllama`, and without that flag dve builds
with no llama.cpp present. Selecting `.llama_nomic_embed_text_v1_5_f32` in a build that was not
given `-Dllama` fails with `error.LlamaNotLinked`; `dve.llama.enabled` reports which you have.

You need a llama.cpp checkout built as shared libraries, and the nomic-embed-text-v1.5 GGUF:

```sh
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -B ~/llama.cpp/build -S ~/llama.cpp -DBUILD_SHARED_LIBS=ON
cmake --build ~/llama.cpp/build --config Release
# then place nomic-embed-text-v1.5.f32.gguf in ~/llama.cpp/build/bin/
```

dve looks for `include/`, `ggml/include/` and the shared libraries in `build/bin/` under that
checkout. Point it elsewhere with `-Dllama-path`, and override the GGUF it picks with
`-Dllama-model`:

```sh
zig build -Dllama -Dllama-path=/opt/llama.cpp
```

A consumer forwards the same flags to the dependency:

```zig
const dve_dep = b.dependency("dve", .{
    .target = target,
    .optimize = optimize,
    .llama = true,
    .@"llama-path" = "/opt/llama.cpp",
});
```

The build bakes the model path into the binary, and `DVE_LLAMA_MODEL` overrides it at run time.
Set `DVE_LLAMA_VERBOSE` to let llama.cpp's own logging through; it is suppressed by default.

The libllama link and its rpath travel with the `dve` module, so beyond the flags above a
consumer needs nothing in its own `build.zig`.
