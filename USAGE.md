# Usage & Installation
The main interface for this library is defined in [vector.zig](src/vector.zig).

## Requirements
- Zig 0.17.0
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

// `io` is your program's `std.Io`, e.g. `init.io` in `pub fn main(init: std.process.Init)`.
// The engine keeps it for every file operation and lock, on both of its threads.
// Open a directory to store the vector database.
const dir = try std.Io.Dir.cwd().createDirPathOpen(io, "my_vectors", .{});

// Name the model you want; see "Model selection" below.
const VectorEngine = dve.VectorEngine(.mpnet_embedding);
// Model files can be changed from their defaults using:
//    VectorEngine.init(..., ..., ..., .{ .model_path = "...", .tokenizer_path = "..." });
const vectors = try VectorEngine.init(allocator, io, dir, .{});
defer vectors.deinit();

// Embed text. The key identifies the entry (typically a file path).
try vectors.embedText("doc1", "Machine learning enables computers to learn from data");
// embedTextAsync returns immediately and embeds on a background thread.
try vectors.embedTextAsync("doc2", "The solar system has eight planets");

// Search returns results ordered by similarity.
var results: [10]dve.SearchResult = undefined;
const n = try vectors.search("artificial intelligence", &results);
for (results[0..n]) |result| {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    // A result carries the document's id, not its path; `pathOfResult` copies the path into a
    // buffer you own.
    const path = try vectors.pathOfResult(result, &path_buf) orelse continue;
    std.debug.print("{s} (similarity: {d:.2})\n", .{ path, result.similarity });
}

// To check whether a known document came back, IDs can be compared.
const top_is_doc1 = n > 0 and results[0].id == vectors.idOf("doc1");
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
const vectors = try VectorEngine.init(allocator, io, dir, .{ .model_path = "path/to/model" });
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
URL=https://github.com/emmettmcdow/dve/releases/download/coreml-models-v7/coreml_models_v7.tar.gz
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

## Logging
dve's log lines go to `std.log` under the `.dve` scope by default, so your program's
`std_options` controls them like any other library's: `log_level` sets what is compiled in, and
`log_scope_levels` can quiet dve alone.

To handle them yourself instead, set a function. It applies process-wide and is not
synchronized against logging, so set it before creating an engine:

```zig
fn onLog(level: std.log.Level, msg: [:0]const u8, ctx: ?*anyopaque) void {
    _ = ctx;
    std.debug.print("dve {t}: {s}\n", .{ level, msg });
}

dve.log.setLogFn(onLog, null, .debug);
```

The function runs on whichever thread logged, including the engine's work queue, sometimes
while the engine holds its locks: it must be thread-safe and must not call back into dve.
`std_options.log_level` still decides which levels are compiled in; the last argument filters
below that at runtime.

From C, `dve_set_log_callback` in `dve.h` is the same thing. Without a callback, the C bindings
write warnings and errors to stderr.

Only debug lines contain search queries or note text. Note paths and the database directory can
appear at any level.
