// swift-tools-version: 5.9
import PackageDescription

// This is the root Package.swift, enabling:
//   .package(url: "https://github.com/emmettmcdow/dve", from: "0.1.3")
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
            url:
                "https://github.com/emmettmcdow/dve/releases/download/v0.1.3/DVECore-0.1.3.xcframework.zip",
            checksum: "f1abcfcdf471549f06bc5e7778d587f861fc1a6a7d1723e15a3ac32bc81d2361"
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
