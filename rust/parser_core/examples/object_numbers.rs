use std::io::{self, BufRead};
fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let x = if let Some(bits) = line.strip_prefix('@') {
            f32::from_bits(u32::from_str_radix(bits, 16).unwrap())
        } else {
            babeldoc_parser_core::object_numbers::lex_real(&line)
        };
        println!(
            "{:08x}\t{:016x}",
            x.to_bits(),
            babeldoc_parser_core::object_numbers::printed_real(x).to_bits()
        );
    }
}
