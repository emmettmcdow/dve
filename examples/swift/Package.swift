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
                "https://github.com/emmettmcdow/dve/releases/download/v0.1.1/DVECore-0.1.1.xcframework.zip",
            checksum: "d6db9be1f14205b26b70dfcd3c163236476315d8637c38d2bbd6f6f55090a41e"
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
