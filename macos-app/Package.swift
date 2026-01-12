// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AlphaArena",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "AlphaArena", targets: ["AlphaArena"])
    ],
    targets: [
        .executableTarget(
            name: "AlphaArena",
            path: "AlphaArena",
            resources: [
                .process("Resources/Assets.xcassets")
            ]
        )
    ]
)
