use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir")).join("../..");
    let generator = root.join("scripts/generate_il_rust.py");
    let schema = root.join("babeldoc/format/pdf/document_il/il_version_1.py");
    let output = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("il_types.rs");
    println!("cargo:rerun-if-changed={}", generator.display());
    println!("cargo:rerun-if-changed={}", schema.display());
    let status = Command::new("python3")
        .arg(&generator)
        .arg("--schema")
        .arg(&schema)
        .arg("--output")
        .arg(&output)
        .arg("--mode")
        .arg("types")
        .status()
        .expect("run IL type generator");
    assert!(status.success(), "IL type generation failed");
}
