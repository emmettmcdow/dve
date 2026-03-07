import DVEKit
import Foundation

let documents: [(key: String, text: String)] = [
    ("solar-system",     "The solar system consists of the Sun and the objects that orbit it, including eight planets, their moons, and countless asteroids and comets."),
    ("photosynthesis",   "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of glucose."),
    ("machine-learning", "Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."),
    ("ancient-rome",     "Ancient Rome was a civilization that grew from a small agricultural community on the Italian Peninsula into a vast empire spanning much of Europe, the Middle East, and North Africa."),
    ("quantum-mechanics","Quantum mechanics is a fundamental theory in physics that describes the behavior of nature at the smallest scales, where particles can exist in multiple states simultaneously."),
]

// Use a temp directory for the database, cleaned up on exit.
let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory())
    .appendingPathComponent("dve-demo-\(ProcessInfo.processInfo.processIdentifier)", isDirectory: true)
try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: tmpDir) }

let vectors = try VectorEngine(directory: tmpDir, model: .appleNL)

// Embed all documents.
print("Embedding documents...\n")
for doc in documents {
    try vectors.embed(key: doc.key, content: doc.text)
    print("  [\(doc.key)]\n  \(doc.text)\n")
}

// Query loop.
while true {
    print("Query (or 'quit'): ", terminator: "")
    guard let line = readLine(strippingNewline: true) else { break }
    if line == "quit" { break }
    if line.isEmpty { continue }

    let results = try vectors.search(line, maxResults: 5)
    if results.isEmpty {
        print("No results.\n")
        continue
    }
    for result in results {
        print("  [\(result.key)] (similarity: \(String(format: "%.2f", result.similarity)))")
    }
    print("")
}
