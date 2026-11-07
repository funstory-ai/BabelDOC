use std::{env, path::PathBuf, process::Command};

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let generator = root.join("scripts/generate_il_rust.py");
    let schema = root.join("babeldoc/format/pdf/document_il/il_version_1.py");
    for path in [
        &generator,
        &schema,
        &root.join("scripts/il_rust_m.py"),
        &root.join("scripts/il_rust_m_declarations.py"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let status = Command::new("python3")
        .arg(generator)
        .arg("--schema")
        .arg(schema)
        .arg("--mode")
        .arg("m-core")
        .arg("--output")
        .arg(PathBuf::from(env::var("OUT_DIR").unwrap()).join("schema.rs"))
        .status()
        .expect("execute schema generator");
    assert!(status.success(), "M schema generation failed");
}
