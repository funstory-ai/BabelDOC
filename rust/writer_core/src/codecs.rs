use crate::{Result, WriterError};
use std::io::Read;

pub fn base85(s: &str) -> Result<Vec<u8>> {
    const ALPHABET: &[u8] =
        b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";
    let pad = (5 - s.len() % 5) % 5;
    let mut input = s.as_bytes().to_vec();
    input.extend(std::iter::repeat_n(b'~', pad));
    let mut out = vec![];
    for chunk in input.chunks(5) {
        let mut n = 0u64;
        for c in chunk {
            let d = ALPHABET
                .iter()
                .position(|a| a == c)
                .ok_or_else(|| WriterError::Invalid("invalid base85 character".into()))?;
            n = n * 85 + d as u64;
        }
        let n = u32::try_from(n).map_err(|_| WriterError::Invalid("base85 overflow".into()))?;
        out.extend(n.to_be_bytes());
    }
    out.truncate(out.len().saturating_sub(pad));
    Ok(out)
}

pub fn operations(s: &str) -> Result<Vec<u8>> {
    let encoded = base85(s)?;
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(encoded.as_slice())
        .map_err(|e| WriterError::Invalid(format!("zstd: {e}")))?;
    let mut bytes = vec![];
    decoder.read_to_end(&mut bytes)?;
    String::from_utf8(bytes.clone())
        .map_err(|e| WriterError::Invalid(format!("base_operations UTF-8: {e}")))?;
    Ok(bytes)
}

/// CPython's binascii.a2b_base64 with strict_mode=False, including early padding termination.
pub fn base64(s: &str) -> Option<Vec<u8>> {
    if !s.is_ascii() {
        return None;
    }
    let mut out = vec![];
    let (mut quad, mut pads, mut bits) = (0usize, 0usize, 0u32);
    for c in s.bytes() {
        if c == b'=' {
            if quad >= 2 {
                pads += 1;
                if quad + pads >= 4 {
                    return Some(out);
                }
            }
            continue;
        }
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => continue,
        };
        pads = 0;
        bits = (bits << 6) | v as u32;
        quad += 1;
        match quad {
            2 => out.push((bits >> 4) as u8),
            3 => out.push((bits >> 2) as u8),
            4 => {
                out.push(bits as u8);
                quad = 0;
                bits = 0;
            }
            _ => (),
        }
    }
    (quad == 0).then_some(out)
}

#[cfg(test)]
mod codecs_tests;
