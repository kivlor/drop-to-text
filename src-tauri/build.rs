use std::path::PathBuf;
use std::process::Command;

fn main() {
    build_parakeet_bridge();
    tauri_build::build()
}

/// Build swift/ (FluidAudio's Parakeet behind a C ABI) as a static library and link it, together
/// with the Swift runtime and the system frameworks it uses.
fn build_parakeet_bridge() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let package = manifest.join("swift");
    for p in ["Package.swift", "Package.resolved", "Sources"] {
        println!("cargo:rerun-if-changed={}", package.join(p).display());
    }

    // Always an optimised Swift build: FluidAudio's decoder loop is far too slow in debug.
    let scratch = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("swift");
    let status = Command::new("xcrun")
        .args([
            "swift",
            "build",
            "-c",
            "release",
            "--product",
            "ParakeetBridge",
        ])
        .arg("--package-path")
        .arg(&package)
        .arg("--scratch-path")
        .arg(&scratch)
        .status()
        .expect("running `swift build` (install Xcode or the Command Line Tools)");
    assert!(
        status.success(),
        "building the Swift Parakeet bridge failed"
    );

    println!(
        "cargo:rustc-link-search=native={}",
        scratch.join("release").display()
    );
    println!("cargo:rustc-link-lib=static=ParakeetBridge");

    // Swift runtime: the OS copy at run time; the toolchain's back-deployment shims at link time.
    let swift = xcrun(&["--find", "swift"]);
    let toolchain_lib = PathBuf::from(swift.trim())
        .parent()
        .and_then(|bin| bin.parent())
        .map(|usr| usr.join("lib/swift/macosx"))
        .unwrap();
    let sdk = xcrun(&["--sdk", "macosx", "--show-sdk-path"]);
    println!("cargo:rustc-link-search=native={}", toolchain_lib.display());
    println!(
        "cargo:rustc-link-search=native={}/usr/lib/swift",
        sdk.trim()
    );
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    for fw in [
        "Foundation",
        "CoreML",
        "Accelerate",
        "AVFoundation",
        "CoreAudio",
    ] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }
}

fn xcrun(args: &[&str]) -> String {
    let out = Command::new("xcrun")
        .args(args)
        .output()
        .expect("running xcrun");
    assert!(out.status.success(), "xcrun {args:?} failed");
    String::from_utf8(out.stdout).unwrap()
}
