// swift-tools-version: 6.1
// A small static library that exposes FluidAudio's Parakeet speech-to-text (Core ML, Apple Neural
// Engine) to Rust through a C ABI. Built by ../build.rs; see Sources/ParakeetBridge.
import PackageDescription

let package = Package(
    name: "ParakeetBridge",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "ParakeetBridge", type: .static, targets: ["ParakeetBridge"])
    ],
    dependencies: [
        // Pinned to the commit that was benchmarked (see BENCHMARKS.md). `traits: []` builds
        // FluidAudio's ASR only: no TTS, diarizer, bundled resources or prebuilt binaries.
        .package(
            url: "https://github.com/FluidInference/FluidAudio.git",
            revision: "e2ea498dafd6db9b3f18f0f644bc35312b3a7679",
            traits: []
        )
    ],
    targets: [
        .target(
            name: "ParakeetBridge",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")]
        )
    ],
    swiftLanguageModes: [.v5]
)
