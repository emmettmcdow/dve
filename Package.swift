// swift-tools-version: 5.9
import PackageDescription

// This is the root Package.swift, enabling:
//   .package(url: "https://github.com/emmettmcdow/dve", from: "0.1.1")
//
// Development (Zig installed):
//   Run `zig build xcframework` first. The binaryTarget below points to the
//   locally built XCFramework at zig-out/DVECore.xcframework.
//
// Release:
//   At tag time, replace the binaryTarget below with:
//     .binaryTarget(name: "DVECore", url: "<github-release-url>", checksum: "<sha256>")
//   See DEV.md for the full release process.

let package = Package(
    name: "dve",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(name: "DVEKit", targets: ["DVEKit"])
    ],
    targets: [
        .binaryTarget(
            name: "DVECore",
            url: "https://github.com/emmettmcdow/dve/releases/download/v0.1.1/DVECore-0.1.1.xcframework.zip",
            checksum: "d6db9be1f14205b26b70dfcd3c163236476315d8637c38d2bbd6f6f55090a41e"
        ),
        .target(
            name: "DVEKit",
            dependencies: ["DVECore"],
            path: "bindings/swift/Sources/DveKit",
            linkerSettings: [
                .linkedFramework("NaturalLanguage"),
                .linkedFramework("CoreML"),
                .linkedLibrary("c++"),
            ]
        ),
    ]
)
