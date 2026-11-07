use super::unicode_data;

pub fn category_is(character: char, category: &str) -> bool {
    let ranges = match category {
        "Mn" => unicode_data::CATEGORY_Mn,
        "Sk" => unicode_data::CATEGORY_Sk,
        "Sm" => unicode_data::CATEGORY_Sm,
        "Zl" => unicode_data::CATEGORY_Zl,
        "Zp" => unicode_data::CATEGORY_Zp,
        "Zs" => unicode_data::CATEGORY_Zs,
        "Co" => unicode_data::CATEGORY_Co,
        _ => return false,
    };
    in_ranges(character as u32, ranges)
}

pub fn is_whitespace(character: char) -> bool {
    in_ranges(character as u32, unicode_data::WHITESPACE)
}

pub fn is_python_strip_whitespace(character: char) -> bool {
    in_ranges(character as u32, unicode_data::PYTHON_STRIP_WHITESPACE)
}

fn in_ranges(value: u32, ranges: &[(u32, u32)]) -> bool {
    ranges
        .binary_search_by(|&(start, end)| {
            if value < start {
                std::cmp::Ordering::Greater
            } else if value > end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

pub fn nfkc(input: &str) -> String {
    let mut decomposed = Vec::with_capacity(input.chars().count());
    for character in input.chars() {
        decompose(character as u32, &mut decomposed);
    }
    canonical_order(&mut decomposed);
    String::from_iter(compose(decomposed).into_iter().filter_map(char::from_u32))
}

fn decompose(codepoint: u32, output: &mut Vec<u32>) {
    // Unicode 15.1 algorithmic Hangul decomposition.
    const S_BASE: u32 = 0xAC00;
    const L_BASE: u32 = 0x1100;
    const V_BASE: u32 = 0x1161;
    const T_BASE: u32 = 0x11A7;
    const L_COUNT: u32 = 19;
    const V_COUNT: u32 = 21;
    const T_COUNT: u32 = 28;
    const N_COUNT: u32 = V_COUNT * T_COUNT;
    const S_COUNT: u32 = L_COUNT * N_COUNT;
    if (S_BASE..S_BASE + S_COUNT).contains(&codepoint) {
        let index = codepoint - S_BASE;
        output.push(L_BASE + index / N_COUNT);
        output.push(V_BASE + (index % N_COUNT) / T_COUNT);
        let trailing = index % T_COUNT;
        if trailing != 0 {
            output.push(T_BASE + trailing);
        }
        return;
    }
    if let Ok(index) = unicode_data::DECOMPOSITIONS.binary_search_by_key(&codepoint, |x| x.0) {
        for child in unicode_data::DECOMPOSITIONS[index].1 {
            decompose(*child, output);
        }
    } else {
        output.push(codepoint);
    }
}

fn combining_class(codepoint: u32) -> u8 {
    unicode_data::COMBINING_CLASSES
        .binary_search_by_key(&codepoint, |x| x.0)
        .ok()
        .map_or(0, |index| unicode_data::COMBINING_CLASSES[index].1)
}

fn canonical_order(codepoints: &mut [u32]) {
    for index in 1..codepoints.len() {
        let class = combining_class(codepoints[index]);
        if class == 0 {
            continue;
        }
        let mut cursor = index;
        while cursor > 0 {
            let previous_class = combining_class(codepoints[cursor - 1]);
            if previous_class == 0 || previous_class <= class {
                break;
            }
            codepoints.swap(cursor - 1, cursor);
            cursor -= 1;
        }
    }
}

fn compose(codepoints: Vec<u32>) -> Vec<u32> {
    if codepoints.is_empty() {
        return codepoints;
    }
    let mut output = Vec::with_capacity(codepoints.len());
    output.push(codepoints[0]);
    let mut starter_pos = 0usize;
    let mut starter = codepoints[0];
    let mut previous_class = 0u8;
    for codepoint in codepoints.into_iter().skip(1) {
        let class = combining_class(codepoint);
        let composite = if previous_class == 0 || previous_class < class {
            compose_pair(starter, codepoint)
        } else {
            None
        };
        if let Some(composite) = composite {
            output[starter_pos] = composite;
            starter = composite;
        } else {
            if class == 0 {
                starter_pos = output.len();
                starter = codepoint;
            }
            output.push(codepoint);
            previous_class = class;
        }
    }
    output
}

fn compose_pair(first: u32, second: u32) -> Option<u32> {
    // Unicode 15.1 algorithmic Hangul composition.
    const L_BASE: u32 = 0x1100;
    const V_BASE: u32 = 0x1161;
    const T_BASE: u32 = 0x11A7;
    const L_COUNT: u32 = 19;
    const V_COUNT: u32 = 21;
    const T_COUNT: u32 = 28;
    const N_COUNT: u32 = V_COUNT * T_COUNT;
    const S_BASE: u32 = 0xAC00;
    if (L_BASE..L_BASE + L_COUNT).contains(&first) && (V_BASE..V_BASE + V_COUNT).contains(&second) {
        return Some(S_BASE + (first - L_BASE) * N_COUNT + (second - V_BASE) * T_COUNT);
    }
    if (S_BASE..S_BASE + L_COUNT * N_COUNT).contains(&first)
        && (first - S_BASE) % T_COUNT == 0
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&second)
    {
        return Some(first + second - T_BASE);
    }
    let key = (first, second);
    unicode_data::COMPOSITIONS
        .binary_search_by_key(&key, |entry| (entry.0, entry.1))
        .ok()
        .map(|index| unicode_data::COMPOSITIONS[index].2)
}
