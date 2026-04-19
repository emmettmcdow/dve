// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DVERepl",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .binaryTarget(
            name: "DVECore",
            path: "../../zig-out/DVECore.xcframework"
        ),
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
