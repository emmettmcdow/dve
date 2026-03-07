// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DVERepl",
    platforms: [
        .macOS(.v13),
    ],
    targets: [
        // Pre-built XCFramework (run `zig build xcframework` from repo root first).
        .binaryTarget(
            name: "DVECore",
            path: "../../zig-out/DVECore.xcframework"
        ),
        // Pull DVEKit sources via symlink (Sources/DVEKit -> bindings/swift/Sources/DVEKit).
        // This avoids a cross-package dependency that would confuse SPM's identity resolution.
        // NOTE: In a real project, add DVEKit via its Package.swift in bindings/swift.
        // The linker settings below are only needed here because this example pulls
        // DVEKit sources via a symlink rather than a proper package dependency.
        // Real consumers get these settings automatically from bindings/swift/Package.swift.
        .target(
            name: "DVEKit",
            dependencies: ["DVECore"],
            linkerSettings: [
                .linkedFramework("NaturalLanguage"),
                .linkedFramework("CoreML"),
                .linkedLibrary("c++"),
            ]
        ),
        .executableTarget(
            name: "dve-repl",
            dependencies: ["DVEKit"]
        ),
    ]
)
