// swift-tools-version: 5.9
import PackageDescription

// Development vs release usage:
//
// DEVELOPMENT (Zig installed):
//   1. From the dve repo root: `zig build xcframework`
//   2. Add this package as a local dependency in your project:
//        .package(path: "/path/to/dve/bindings/swift")
//   The binaryTarget below points to the locally built XCFramework.
//
// RELEASE (no Zig required):
//   Add dve as a remote Swift package at a release tag. The release tag's
//   Package.swift replaces the binaryTarget below with a remote URL+checksum
//   pointing to a pre-built DVECore.xcframework.zip on GitHub Releases.
//   SPM handles the download automatically.

let package = Package(
    name: "DVEKit",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(name: "DVEKit", targets: ["DVEKit"])
    ],
    targets: [
        // Release: pre-built XCFramework downloaded by SPM from GitHub Releases.
        // For local development, replace with:
        // .binaryTarget(name: "DVECore", path: "../../zig-out/DVECore.xcframework"),
        .binaryTarget(
            name: "DVECore",
            url:
                "https://github.com/emmettmcdow/dve/releases/download/v0.1.0/DVECore-0.1.0.xcframework.zip",
            checksum: "bde44fa984e6e1dde3c6ddae7c40207e89e0ef44d1793c0466785390e22b9d7a"
        ),
        .target(
            name: "DVEKit",
            dependencies: ["DVECore"],
            linkerSettings: [
                // DVECore links into these Apple frameworks at runtime.
                // Declaring them here propagates the requirement to any binary
                // that depends on DVEKit, so consumers don't need to add them manually.
                .linkedFramework("NaturalLanguage"),
                .linkedFramework("CoreML"),
                .linkedLibrary("c++"),
            ]
        ),
    ]
)
