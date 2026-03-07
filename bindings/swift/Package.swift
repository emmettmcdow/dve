// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DVEKit",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .library(name: "DVEKit", targets: ["DVEKit"]),
    ],
    targets: [
        // Pre-built XCFramework containing the Zig/C core.
        // For local development: run `zig build xcframework` from the dve repo root,
        // then reference the output at ../../zig-out/DVECore.xcframework.
        // For released versions: replace with a remote binaryTarget(url:checksum:).
        .binaryTarget(
            name: "DVECore",
            path: "../../zig-out/DVECore.xcframework"
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
