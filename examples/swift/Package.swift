// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DveRepl",
    platforms: [
        .macOS(.v13),
    ],
    targets: [
        // Pre-built XCFramework (run `zig build xcframework` from repo root first).
        .binaryTarget(
            name: "DveCore",
            path: "../../zig-out/DveCore.xcframework"
        ),
        // Pull DveKit sources via symlink (Sources/DveKit -> bindings/swift/Sources/DveKit).
        // This avoids a cross-package dependency that would confuse SPM's identity resolution.
        // NOTE: In a real project, add DveKit via its Package.swift in bindings/swift.
        // The linker settings below are only needed here because this example pulls
        // DveKit sources via a symlink rather than a proper package dependency.
        // Real consumers get these settings automatically from bindings/swift/Package.swift.
        .target(
            name: "DveKit",
            dependencies: ["DveCore"],
            linkerSettings: [
                .linkedFramework("NaturalLanguage"),
                .linkedFramework("CoreML"),
                .linkedLibrary("c++"),
            ]
        ),
        .executableTarget(
            name: "dve-repl",
            dependencies: ["DveKit"]
        ),
    ]
)
