# dve - dve vector engine
dve is a library for creating and searching vector embeddings locally on Apple devices.
Built with zig, but bindings are available for Swift.

## Usage
```swift
import DveKit

let db = try DveDatabase(directory: myURL, model: .appleNL)

try db.embed(key: "doc1", content: "Machine learning enables computers to learn from data")
try db.embed(key: "path/to/doc2", content: "The solar system has eight planets")

let results = try db.search("artificial intelligence")
// results[0].key == "doc1"
```

See the [examples](./examples) directory for complete working demos in Zig and Swift.
See [USAGE.md](./USAGE.md) for installation and full usage details.

## Why
Vector search is a powerful tool for apps, but on-device implementations are surprisingly hard to
come by. Developers typically choose between calling a cloud API (adding cost and a third-party
dependency) or reaching for heavy all-in-one frameworks like Hugging Face Transformers.

This problem is especially acute on Apple platforms. ML frameworks overwhelmingly prioritize
Linux servers, CoreML is poorly documented, and cross-platform libraries rarely integrate cleanly
into macOS or iOS apps. dve was built to fill that gap. It starts with Apple with portability as a
core design goal.

## Core Principles
- Portable - dve should use few libraries, such that it can be easily used on any platform.
- Simple - dve should have a simple but configurable interface, with sane defaults.
- Local - dve should run performantly, and should never make network calls.

## Roadmap
- Add Linux support.
- Add Windows support.
- Multi-modal embedding support.
- Download links within text documents and embed them.
- Support for multiple model types (beyond mpnet and Apple NL).
- Support multiple database instances within a single process.
- iOS support.
- C/C++ bindings.
