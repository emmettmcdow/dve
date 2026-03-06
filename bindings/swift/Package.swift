// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DveKit",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .library(name: "DveKit", targets: ["DveKit"]),
    ],
    targets: [
        // Pre-built XCFramework containing the Zig/C core.
        // For local development: run `zig build xcframework` from the dve repo root,
        // then reference the output at ../../zig-out/DveCore.xcframework.
        // For released versions: replace with a remote binaryTarget(url:checksum:).
        .binaryTarget(
            name: "DveCore",
            path: "../../zig-out/DveCore.xcframework"
        ),
        .target(
            name: "DveKit",
            dependencies: ["DveCore"],
            linkerSettings: [
                // DveCore links into these Apple frameworks at runtime.
                // Declaring them here propagates the requirement to any binary
                // that depends on DveKit, so consumers don't need to add them manually.
                .linkedFramework("NaturalLanguage"),
                .linkedFramework("CoreML"),
                .linkedLibrary("c++"),
            ]
        ),
    ]
)
