# dve - dve vector engine
dve is a vector search library for Apple devices.

dve is early-stage and actively developed. Bug reports, issues, and pull requests are welcome on
GitHub.

If you're trying dve and run into trouble, feel free to reach out directly:
[@emmettmcdow](https://github.com/emmettmcdow).


## Why
dve aims to be the SQLite of search. It should run quickly on hardware large and small. Also:

- search is hard (let dve do it for you)
- it's free, it doesn't call out to inference APIs (let dve do it for you)
- portable and lightweight inference is tricky (let dve do it for you)
- ... especially on Apple devices (let dve do it for you)


## Usage
```zig
const dve = @import("dve");

// `io` is your program's `std.Io`, e.g. `init.io` in `pub fn main(init: std.process.Init)`.
// Open a directory to store the vector database.
const dir = try std.Io.Dir.cwd().createDirPathOpen(io, "my_vectors", .{});

// Select the model you want. See USAGE.md for available model options and tradeoffs.
const VectorEngine = dve.VectorEngine(.mpnet_embedding);
const vectors = try VectorEngine.init(allocator, io, dir, .{});
defer vectors.deinit();

// Embed text. The key identifies the entry (typically a file path).
try vectors.embedText("doc1", "Machine learning enables computers to learn from data");
try vectors.embedText("doc2", "The solar system has eight planets");

// Search returns results ordered by similarity.
var results: [10]dve.SearchResult = undefined;
const n = try vectors.search("artificial intelligence", &results);
// results[0].path == "doc1"
```

See the [examples](./examples) directory for complete working demos in Zig.
See [USAGE.md](./USAGE.md) for installation and full usage details.


## Core Principles
- Fast - dve ought to be the fastest local search library.
- Simple - dve should have a simple but configurable interface, with sane defaults.
- Local - dve should run on a single machine without making API calls.


## Roadmap
- Add Linux support.
- Make C/C++ bindings more stable.
- iOS support.
- Multi-modal embedding support.
- Download links within text documents and embed them.
- Generalize llama.cpp and CoreML engines to run any supported model.
- Support multiple database instances within a single process.
