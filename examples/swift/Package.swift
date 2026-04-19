// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DVERepl",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        // .binaryTarget(
        //     name: "DVECore",
        //     path: "../../zig-out/DVECore.xcframework"
        // ),
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
