# Usage & Installation
The main interface for this library is defined in [vector.zig](src/vector.zig).

## Requirements
- Zig 0.15.1
- `curl`, only if `installModels` fetches the llama model

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
// Fetch the files for the models you use and install them into your project's zig-out/share/,
// where the exe looks for them. Fields are named after the models and default to false. Skip
// this if you only use .apple_nlembedding, which needs no model files.
@import("dve").installModels(b, dve_dep, .{ .mpnet_embedding = true });
```

## Short Example

```zig
const dve = @import("dve");

// Open a directory to store the vector database.
const dir = try std.fs.cwd().makeOpenPath("my_vectors", .{});

// Name the model you want; see "Model selection" above.
const VectorEngine = dve.VectorEngine(.mpnet_embedding);
// Model files can be changed from their defaults using:
//    VectorEngine.init(..., ..., .{ .model_path = "...", .tokenizer_path = "..." });
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

> **Note:** The database format differs between models — 768-dim vectors are not readable as
> 512-dim ones. Use the same model consistently for a given database directory.

You own the model files. If you consume dve from Zig, the build system can get them for you:
`installModels` from [Install](#install) fetches the files for the models you name and installs
them to `zig-out/share/`, which is where each model looks by default. There is nothing else to do.

```zig
@import("dve").installModels(b, dve_dep, .{
    .mpnet_embedding = true,
    .llama_nomic_embed_text_v1_5_f32 = true,
});
```

If you are not building with Zig, or want the files somewhere else, each model below lists a
`curl` command as the fallback. Every model then takes its path the same way, through the options
passed to `init`:

```zig
const vectors = try VectorEngine.init(allocator, dir, .{ .model_path = "path/to/model" });
```

If a model file is not found, `init` fails with `error.ModelNotFound` and logs where it looked
and the command to fetch it.

### `.mpnet_embedding`
`sentence-transformers/all-mpnet-base-v2` — 768 dimensions, scores 88% on our benchmarks.

- Files: `all_mpnet_base_v2.mlpackage` and `tokenizer.json`, about 200 MB together.
- With Zig: `installModels(b, dve_dep, .{ .mpnet_embedding = true })`.
- Default location: the app bundle's `Resources/`, then `<exe>/../share/`.
- Options: `.model_path` and `.tokenizer_path`. mpnet is the one model with two files.

Fallback download, which unpacks both files into `./all_mpnet_base_v2/`:

```sh
URL=https://github.com/emmettmcdow/dve/releases/download/coreml-models-v5/coreml_models_v5.tar.gz
curl -L "$URL" | tar -xz all_mpnet_base_v2
```

### `.apple_nlembedding`
Apple NaturalLanguage — 512 dimensions, scores 66% on our benchmarks. Served by the OS, so it
needs no model files. Good for quick prototyping.

### `.llama_nomic_embed_text_v1_5_f32`
llama.cpp / nomic-embed-text-v1.5 — 768 dimensions, scores 90% on our benchmarks, with an
8192-token context.

This backend is the only one with a dependency outside the Zig package graph, so unlike the other
two it is not compiled in by default: you opt in with `-Dllama`, and without that flag dve builds
with no llama.cpp present. Selecting `.llama_nomic_embed_text_v1_5_f32` in a build that was not
given `-Dllama` fails with `error.LlamaNotLinked`; `dve.llama.enabled` reports which you have.

#### Building llama.cpp
You need a llama.cpp checkout built as shared libraries:

```sh
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -B ~/llama.cpp/build -S ~/llama.cpp -DBUILD_SHARED_LIBS=ON
cmake --build ~/llama.cpp/build --config Release
```

dve looks for `include/`, `ggml/include/` and the shared libraries in `build/bin/` under that
checkout. Point it elsewhere with `-Dllama-path`:

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

The libllama link and its rpath travel with the `dve` module, so beyond the flags above a
consumer needs nothing in its own `build.zig`.

#### Model file
- Files: `nomic-embed-text-v1.5.f32.gguf`, 547 MB.
- With Zig: `installModels(b, dve_dep, .{ .llama_nomic_embed_text_v1_5_f32 = true })`. This
  downloads the file with `curl`, which must be on your `PATH`; the other models have no such
  requirement.
- Default location: the `DVE_LLAMA_MODEL` environment variable, then the app bundle's
  `Resources/`, then `<exe>/../share/`.
- Options: `.model_path`.

Fallback download, into the current directory:

```sh
URL=https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f32.gguf
curl -LO "$URL"
```

The bridge holds one model per process, so a second engine asking for a different GGUF fails with
`error.ModelAlreadyLoaded`. Set `DVE_LLAMA_VERBOSE` to let llama.cpp's own logging through; it is
suppressed by default.
