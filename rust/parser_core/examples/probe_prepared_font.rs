//! Local numeric-reading witness for parser differential diagnosis.
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let doc = lopdf::Document::load(&args[1]).unwrap();
    let xref: u32 = args[2].parse().unwrap();
    let cid: u32 = args[3].parse().unwrap();
    let font = babeldoc_parser_core::fonts::FontBackend::from_document(&doc, (xref, 0)).unwrap();
    println!(
        "font={xref} cid={cid} width_em={:.17}",
        font.width(cid).unwrap()
    );
    println!(
        "decimal_f64={:.17} lopdf_real_f64={:.17}",
        333.008_f64, 333.008_f32 as f64
    );
}
