import DVECore
import Darwin
import Foundation

// MARK: - Public types

public struct SearchResult {
    public let key: String
    public let startIndex: Int
    public let endIndex: Int
    public let similarity: Float
}

public enum VectorEngineError: Error {
    case initFailed(code: Int32)
    case alreadyInitialized
    case notInitialized
    case embedFailed(code: Int32)
    case searchFailed(code: Int32)
    case removeFailed(code: Int32)
    case renameFailed(code: Int32)
    /// The mpnet model files could not be found in DVECore.framework/Resources/.
    /// Ensure DVECore.framework is properly embedded in your app.
    case modelNotFound
}

// MARK: - VectorEngine

/// A local vector embedding database. One instance per process (singleton).
public final class VectorEngine {

    /// Initialize the database.
    ///
    /// - Parameters:
    ///   - directory: Directory where the database files will be stored.
    public init(directory: URL) throws {
        let basedir = directory.path
        let (modelURL, tokenizerURL) = try VectorEngine.resolveMpnetURLs()
        let code = dve_init(basedir, modelURL.path, tokenizerURL.path)
        guard code == DVE_SUCCESS.rawValue else {
            throw VectorEngineError.initFailed(code: code)
        }
    }

    deinit {
        dve_deinit()
    }

    // MARK: - Embedding

    /// Embed text synchronously. Blocks until complete.
    public func embed(key: String, content: String) throws {
        let result = dve_embed(key, content)
        guard result == DVE_SUCCESS.rawValue else {
            throw VectorEngineError.embedFailed(code: result)
        }
    }

    /// Embed text on a background thread. Returns immediately.
    public func embedAsync(key: String, content: String) throws {
        let result = dve_embed_async(key, content)
        guard result == DVE_SUCCESS.rawValue else {
            throw VectorEngineError.embedFailed(code: result)
        }
    }

    // MARK: - Search

    /// Search for text semantically similar to the query.
    ///
    /// - Parameters:
    ///   - query: The search query.
    ///   - maxResults: Maximum number of results to return (default 20).
    public func search(_ query: String, maxResults: Int = 20) throws -> [SearchResult] {
        var buf = [DVESearchResult](repeating: DVESearchResult(), count: maxResults)
        let n = dve_search(query, &buf, UInt32(maxResults))
        guard n >= 0 else {
            throw VectorEngineError.searchFailed(code: n)
        }
        return buf[0..<Int(n)].map { r in
            SearchResult(
                key: withUnsafeBytes(of: r.key) { bytes in
                    String(cString: bytes.baseAddress!.assumingMemoryBound(to: CChar.self))
                },
                startIndex: Int(r.start_i),
                endIndex: Int(r.end_i),
                similarity: r.similarity
            )
        }
    }

    // MARK: - Management

    /// Remove all embeddings associated with key.
    public func remove(key: String) throws {
        let result = dve_remove(key)
        guard result == DVE_SUCCESS.rawValue else {
            throw VectorEngineError.removeFailed(code: result)
        }
    }

    /// Rename a key, preserving its embeddings.
    public func rename(from oldKey: String, to newKey: String) throws {
        let result = dve_rename(oldKey, newKey)
        guard result == DVE_SUCCESS.rawValue else {
            throw VectorEngineError.renameFailed(code: result)
        }
    }

    // MARK: - Private

    /// Resolves the mpnet model and tokenizer paths from DVECore.framework's Resources/.
    private static func resolveMpnetURLs() throws -> (model: URL, tokenizer: URL) {
        // Find the path of the loaded DVECore dylib via its exported symbol.
        guard let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "dve_init") else {
            throw VectorEngineError.modelNotFound
        }
        var info = Dl_info()
        guard dladdr(sym, &info) != 0, let fname = info.dli_fname else {
            throw VectorEngineError.modelNotFound
        }
        // fname is e.g. ".../DVECore.framework/DVECore"
        // One level up is the framework directory.
        let frameworkURL = URL(fileURLWithPath: String(cString: fname))
            .deletingLastPathComponent()
        let resources = frameworkURL.appendingPathComponent("Resources")
        let modelURL = resources.appendingPathComponent("all_mpnet_base_v2.mlpackage")
        let tokenizerURL = resources.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: modelURL.path),
            FileManager.default.fileExists(atPath: tokenizerURL.path)
        else {
            throw VectorEngineError.modelNotFound
        }
        return (modelURL, tokenizerURL)
    }
}
