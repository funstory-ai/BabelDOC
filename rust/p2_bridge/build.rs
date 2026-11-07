use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir")).join("../..");
    let generator = root.join("scripts/generate_il_rust.py");
    let schema = root.join("babeldoc/format/pdf/document_il/il_version_1.py");
    println!("cargo:rerun-if-changed={}", generator.display());
    println!("cargo:rerun-if-changed={}", schema.display());
    println!(
        "cargo:rerun-if-changed={}",
        root.join("scripts/il_rust_m.py").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("scripts/il_rust_m_declarations.py").display()
    );
    let status = Command::new("python3")
        .arg(&generator)
        .arg("--schema")
        .arg(&schema)
        .arg("--mode")
        .arg("m-reader")
        .arg("--output")
        .arg(PathBuf::from(env::var("OUT_DIR").unwrap()).join("m_reader.rs"))
        .status()
        .expect("run M reader generator");
    assert!(status.success(), "M reader generation failed");
}
