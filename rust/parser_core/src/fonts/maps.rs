//! Local interval compilation. All endpoints use u64 events so MAX+1 cannot wrap.
use super::*;
use lopdf::resource::{Limit, Policy};
use std::collections::BTreeSet;
use std::sync::Arc;

/// Immutable mapping bytes; splitting a winner interval only clones this handle.
/// The original origin remains unchanged, including for UTF16 wrapping increments.
#[derive(Clone, Debug)]
pub(super) struct UnicodeTarget {
    pub bytes: Arc<[u8]>,
    pub origin: u32,
}
impl UnicodeTarget {
    pub fn new(bytes: Vec<u8>, origin: u32) -> Self {
        Self {
            bytes: bytes.into(),
            origin,
        }
    }
}

pub(super) fn check_mapping(policy: Option<&Policy>, bytes: usize) -> Result<(), FontError> {
    if let Some(policy) = policy {
        policy.check(Limit::UnicodeMappingBytes, bytes)?;
    }
    Ok(())
}

/// Compile overlapping declarations into sorted, disjoint winner segments in O(R log R).
/// The result refers to original declarations; values/offsets are interpreted by callers.
pub(super) fn overlay(bounds: &[(u32, u32)], last_wins: bool) -> Vec<(u32, u32, usize)> {
    let mut events = Vec::new();
    for (index, &(lo, hi)) in bounds.iter().enumerate() {
        if lo <= hi {
            events.push((u64::from(lo), true, index));
            events.push((u64::from(hi) + 1, false, index));
        }
    }
    events.sort_unstable();
    let mut active = BTreeSet::new();
    let mut result: Vec<(u32, u32, usize)> = Vec::new();
    let mut i = 0;
    while i < events.len() {
        let position = events[i].0;
        while i < events.len() && events[i].0 == position {
            let (_, insert, index) = events[i];
            if insert {
                active.insert(index);
            } else {
                active.remove(&index);
            }
            i += 1;
        }
        let Some(next) = events.get(i).map(|event| event.0) else {
            break;
        };
        let winner = if last_wins {
            active.last()
        } else {
            active.first()
        };
        if let Some(&winner) = winner {
            if let Some(previous) = result.last_mut() {
                if previous.2 == winner && u64::from(previous.1) + 1 == position {
                    previous.1 = (next - 1) as u32;
                    continue;
                }
            }
            result.push((position as u32, (next - 1) as u32, winner));
        }
    }
    result
}

pub(super) fn lookup<T>(ranges: &[T], code: u32, bounds: impl Fn(&T) -> (u32, u32)) -> Option<&T> {
    let index = ranges
        .partition_point(|range| bounds(range).0 <= code)
        .checked_sub(1)?;
    let range = &ranges[index];
    (code <= bounds(range).1).then_some(range)
}

pub(super) fn incremented_bytes(bytes: &[u8], increment: u32) -> Vec<u8> {
    let n = bytes.len().min(4);
    let split = bytes.len() - n;
    let base = bytes[split..]
        .iter()
        .fold(0u32, |n, b| (n << 8) | u32::from(*b));
    let value = base.wrapping_add(increment).to_be_bytes();
    let mut result = bytes[..split].to_vec();
    result.extend_from_slice(&value[4 - n..]);
    result
}

pub(super) fn decode_mapping(bytes: &[u8], policy: Option<&Policy>) -> Result<String, FontError> {
    check_mapping(policy, bytes.len())?;
    let mut text = String::new();
    for character in char::decode_utf16(
        bytes
            .chunks_exact(2)
            .map(|b| u16::from_be_bytes([b[0], b[1]])),
    )
    .filter_map(Result::ok)
    {
        // Both parser admission and lazy lookup measure raw mapping bytes.
        // UTF-8 output can be 3/2 of valid UTF-16 input; that representation
        // conversion does not change the per-mapping resource's unit.
        text.push(character);
    }
    Ok(text)
}

/// Validate before allocating the per-code raw target. Token storage belongs to the input parser.
pub(super) fn mapping_bytes(
    token: &str,
    policy: Option<&Policy>,
) -> Result<Option<Vec<u8>>, FontError> {
    let Some(body) = token.strip_prefix('<').and_then(|s| s.strip_suffix('>')) else {
        return Ok(None);
    };
    let mut digits = 0usize;
    for character in body
        .chars()
        .filter(|c| !matches!(c, '\0' | '\t' | '\n' | '\x0c' | '\r' | ' '))
    {
        if !character.is_ascii_hexdigit() {
            return Ok(None);
        }
        digits += 1;
    }
    check_mapping(policy, digits / 2 + digits % 2)?;
    Ok(hex_string_bytes(token))
}

pub(super) fn mapping_pair(
    src: &str,
    dst: &str,
    policy: Option<&Policy>,
) -> Result<Option<(u32, String)>, FontError> {
    let Some((code, _)) = parse_unicode_code(src) else {
        return Ok(None);
    };
    let Some(raw) = mapping_bytes(dst, policy)? else {
        return Ok(None);
    };
    Ok(Some((code, decode_mapping(&raw, policy)?)))
}

/// Packed [lo, hi] inclusive endpoints, in first-declaration encounter order.
/// Empty and nonempty-but-outside-bbox-domain remain distinct.
pub(crate) fn projection_codes(
    packed: &[u32],
    bbox_count: usize,
) -> impl Iterator<Item = u32> + '_ {
    let end = (bbox_count as u64).min(u64::from(u32::MAX) + 1);
    packed
        .chunks_exact(2)
        .flat_map(move |range| u64::from(range[0])..(u64::from(range[1]) + 1).min(end))
        .map(|code| code as u32)
}

pub(super) fn compile_unicode(ranges: &mut Vec<(u32, u32, UnicodeTarget, usize)>) {
    let segments = overlay(&ranges.iter().map(|r| (r.0, r.1)).collect::<Vec<_>>(), true);
    let compiled = segments
        .into_iter()
        .map(|(lo, hi, index)| {
            let original = &ranges[index];
            (lo, hi, original.2.clone(), original.3)
        })
        .collect();
    *ranges = compiled;
}
