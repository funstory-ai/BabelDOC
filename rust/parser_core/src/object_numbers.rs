//! MuPDF 1.27.2 object-number reading and printing, ported to Rust.
//! Object dictionaries use this path; decoded content/CMap bytes do not.
// strtof portions Copyright (C) 2004-2021 Artifex Software, Inc., AGPL-3.0-or-later.
/*
    Copyright (c) 2009 Florian Loitsch

    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use,
    copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following
    conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.
*/

const PRINT_POWERS: [u64; 84] = [
    0x881cea14545c7575,
    0xaa242499697392d3,
    0xd4ad2dbfc3d07788,
    0x84ec3c97da624ab5,
    0xa6274bbdd0fadd62,
    0xcfb11ead453994ba,
    0x81ceb32c4b43fcf5,
    0xa2425ff75e14fc32,
    0xcad2f7f5359a3b3e,
    0xfd87b5f28300ca0e,
    0x9e74d1b791e07e48,
    0xc612062576589ddb,
    0xf79687aed3eec551,
    0x9abe14cd44753b53,
    0xc16d9a0095928a27,
    0xf1c90080baf72cb1,
    0x971da05074da7bef,
    0xbce5086492111aeb,
    0xec1e4a7db69561a5,
    0x9392ee8e921d5d07,
    0xb877aa3236a4b449,
    0xe69594bec44de15b,
    0x901d7cf73ab0acd9,
    0xb424dc35095cd80f,
    0xe12e13424bb40e13,
    0x8cbccc096f5088cc,
    0xafebff0bcb24aaff,
    0xdbe6fecebdedd5bf,
    0x89705f4136b4a597,
    0xabcc77118461cefd,
    0xd6bf94d5e57a42bc,
    0x8637bd05af6c69b6,
    0xa7c5ac471b478423,
    0xd1b71758e219652c,
    0x83126e978d4fdf3b,
    0xa3d70a3d70a3d70a,
    0xcccccccccccccccd,
    0x8000000000000000,
    0xa000000000000000,
    0xc800000000000000,
    0xfa00000000000000,
    0x9c40000000000000,
    0xc350000000000000,
    0xf424000000000000,
    0x9896800000000000,
    0xbebc200000000000,
    0xee6b280000000000,
    0x9502f90000000000,
    0xba43b74000000000,
    0xe8d4a51000000000,
    0x9184e72a00000000,
    0xb5e620f480000000,
    0xe35fa931a0000000,
    0x8e1bc9bf04000000,
    0xb1a2bc2ec5000000,
    0xde0b6b3a76400000,
    0x8ac7230489e80000,
    0xad78ebc5ac620000,
    0xd8d726b7177a8000,
    0x878678326eac9000,
    0xa968163f0a57b400,
    0xd3c21bcecceda100,
    0x84595161401484a0,
    0xa56fa5b99019a5c8,
    0xcecb8f27f4200f3a,
    0x813f3978f8940984,
    0xa18f07d736b90be5,
    0xc9f2c9cd04674edf,
    0xfc6f7c4045812296,
    0x9dc5ada82b70b59e,
    0xc5371912364ce305,
    0xf684df56c3e01bc7,
    0x9a130b963a6c115c,
    0xc097ce7bc90715b3,
    0xf0bdc21abb48db20,
    0x96769950b50d88f4,
    0xbc143fa4e250eb31,
    0xeb194f8e1ae525fd,
    0x92efd1b8d0cf37be,
    0xb7abc627050305ae,
    0xe596b7b0c643c719,
    0x8f7e32ce7bea5c70,
    0xb35dbf821ae4f38c,
    0xe0352f62a19e306f,
];
const PRINT_EXPONENTS: [i32; 84] = [
    -186, -183, -180, -176, -173, -170, -166, -163, -160, -157, -153, -150, -147, -143, -140, -137,
    -133, -130, -127, -123, -120, -117, -113, -110, -107, -103, -100, -97, -93, -90, -87, -83, -80,
    -77, -73, -70, -67, -63, -60, -57, -54, -50, -47, -44, -40, -37, -34, -30, -27, -24, -20, -17,
    -14, -10, -7, -4, 0, 3, 6, 10, 13, 16, 20, 23, 26, 30, 33, 36, 39, 43, 46, 49, 53, 56, 59, 63,
    66, 69, 73, 76, 79, 83, 86, 89,
];
const READ_POWERS: [u32; 55] = [
    0x80000000, 0xa0000000, 0xc8000000, 0xfa000000, 0x9c400000, 0xc3500000, 0xf4240000, 0x98968000,
    0xbebc2000, 0xee6b2800, 0x9502f900, 0xba43b740, 0xe8d4a510, 0x9184e72a, 0xb5e620f4, 0xe35fa932,
    0x8e1bc9bf, 0xb1a2bc2f, 0xde0b6b3a, 0x8ac72305, 0xad78ebc6, 0xd8d726b7, 0x87867832, 0xa968163f,
    0xd3c21bcf, 0x84595161, 0xa56fa5ba, 0xcecb8f28, 0x813f3979, 0xa18f07d7, 0xc9f2c9cd, 0xfc6f7c40,
    0x9dc5ada8, 0xc5371912, 0xf684df57, 0x9a130b96, 0xc097ce7c, 0xf0bdc21b, 0x96769951, 0xbc143fa5,
    0xeb194f8e, 0x92efd1b9, 0xb7abc627, 0xe596b7b1, 0x8f7e32ce, 0xb35dbf82, 0xe0352f63, 0x8c213d9e,
    0xaf298d05, 0xdaf3f046, 0x88d8762c, 0xab0e93b7, 0xd5d238a5, 0x85a36367, 0xa70c3c41,
];
const READ_EXPONENTS: [i32; 55] = [
    -31, -28, -25, -22, -18, -15, -12, -8, -5, -2, 2, 5, 8, 12, 15, 18, 22, 25, 28, 32, 35, 38, 42,
    45, 48, 52, 55, 58, 62, 65, 68, 71, 75, 78, 81, 85, 88, 91, 95, 98, 101, 105, 108, 111, 115,
    118, 121, 125, 128, 131, 135, 138, 141, 145, 148,
];
#[derive(Clone, Copy)]
struct Diy {
    f: u64,
    e: i32,
}
fn multiply(a: Diy, b: Diy) -> Diy {
    Diy {
        f: (((a.f as u128) * (b.f as u128) + (1u128 << 63)) >> 64) as u64,
        e: a.e + b.e + 64,
    }
}
fn grisu(v: f32) -> (String, i32) {
    let bits = v.to_bits();
    let exponent = ((bits & 0x7f800000) >> 23) as i32;
    let significand = (bits & 0x7fffff) as u64;
    let x = Diy {
        f: if exponent == 0 {
            significand
        } else {
            significand + 0x800000
        },
        e: if exponent == 0 { -149 } else { exponent - 150 },
    };
    let mut upper = Diy {
        f: (x.f << 1) + 1,
        e: x.e - 1,
    };
    while upper.f & 0x1000000 == 0 {
        upper.f <<= 1;
        upper.e -= 1;
    }
    upper.f <<= 39;
    upper.e -= 39;
    let mut lower = if x.f == 0x800000 {
        Diy {
            f: (x.f << 2) - 1,
            e: x.e - 2,
        }
    } else {
        Diy {
            f: (x.f << 1) - 1,
            e: x.e - 1,
        }
    };
    lower.f <<= lower.e - upper.e;
    lower.e = upper.e;
    upper.f -= 1 << 10;
    lower.f += 1 << 10;
    let tmp = -60 - upper.e - 1;
    let mk = tmp * 1233 / 4096 + if tmp > 0 { 1 } else { 0 };
    let power = Diy {
        f: PRINT_POWERS[(mk + 37) as usize],
        e: PRINT_EXPONENTS[(mk + 37) as usize],
    };
    upper = multiply(upper, power);
    lower = multiply(lower, power);
    upper.f -= 1;
    lower.f += 1;
    let mut delta = upper.f - lower.f;
    let shift = -upper.e;
    let one = 1u64 << shift;
    let mut p1 = (upper.f >> shift) as u8;
    let mut p2 = upper.f & (one - 1);
    let mut digits = String::new();
    let mut div = 10u8;
    for kappa in (1..=2).rev() {
        let digit = p1 / div;
        if digit != 0 || !digits.is_empty() {
            digits.push((b'0' + digit) as char);
        }
        p1 %= div;
        div /= 10;
        if ((p1 as u64) << shift) + p2 <= delta {
            return (digits, -mk + kappa - 1);
        }
    }
    let mut kappa = 0;
    loop {
        p2 *= 10;
        digits.push((b'0' + (p2 >> shift) as u8) as char);
        p2 &= one - 1;
        kappa -= 1;
        delta *= 10;
        if p2 <= delta {
            return (digits, -mk + kappa);
        }
    }
}

/// MuPDF pdf-object.c's real-object print followed by Python float(text).
pub fn printed_real(x: f32) -> f64 {
    printed_real_text(x).parse().expect("MuPDF real text")
}

/// The exact spelling also matters to reference helpers that inspect object text.
pub fn printed_real_text(mut x: f32) -> String {
    if x.is_nan() {
        x = 0.;
    }
    if x.is_infinite() {
        x = if x.is_sign_negative() {
            -f32::MAX
        } else {
            f32::MAX
        };
    }
    // pdf_print_obj uses integer printing first, including negative zero.
    if x == 0. {
        return "0".into();
    }
    if x >= -2147483648. && x < 2147483648. && x == x.trunc() {
        return (x as i32).to_string();
    }
    let (digits, exponent) = grisu(x.abs());
    let point = digits.len() as i32 + exponent;
    let mut out = if x.is_sign_negative() {
        "-".to_owned()
    } else {
        String::new()
    };
    if point <= 0 {
        out.push('.');
        out.extend(std::iter::repeat_n('0', (-point) as usize));
        out.push_str(&digits);
    } else if point >= digits.len() as i32 {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n(
            '0',
            (point - digits.len() as i32) as usize,
        ));
    } else {
        out.push_str(&digits[..point as usize]);
        out.push('.');
        out.push_str(&digits[point as usize..]);
    }
    out
}

fn scaled(mut m: u32, mut n: i32, negative: bool) -> f32 {
    if m == 0 {
        return if negative { -0. } else { 0. };
    }
    // fz_atof replaces a nonzero underflow with 1 and clamps overflow.
    if n > 38 {
        return if negative { -f32::MAX } else { f32::MAX };
    }
    if n < -54 {
        return 1.;
    }
    while n > 13 && m <= u32::MAX / 10 {
        m *= 10;
        n -= 1;
    }
    while n < -13 && m % 10 == 0 {
        m /= 10;
        n += 1;
    }
    let shift = m.leading_zeros();
    let f = m << shift;
    let e = -(shift as i32);
    let power = READ_POWERS[n.unsigned_abs() as usize];
    let power_e = READ_EXPONENTS[n.unsigned_abs() as usize];
    let (f, e) = if n >= 0 {
        let mut product = (f as u64) * (power as u64);
        let mut exponent = e + power_e + 32;
        if product < (1u64 << 63) {
            product <<= 1;
            exponent -= 1;
        }
        (
            ((product >> 32) as u32) | u32::from(product & 0xffffffff != 0),
            exponent,
        )
    } else {
        let product = (f as u64) << 32;
        let quotient = product / power as u64;
        let remainder = product % power as u64;
        let extra = quotient > u32::MAX as u64;
        (
            ((quotient >> u32::from(extra)) as u32) | u32::from(remainder != 0),
            e - power_e - 32 + i32::from(extra),
        )
    };
    let mut result;
    if e > 96 || (e == 96 && f >= 0xffffff80) {
        result = 0x7f800000;
    } else if e > -158 {
        result = (((e + 158) as u32) << 23) | ((f >> 8) & 0x7fffff);
        if f & 0x80 != 0 && (f & 0x7f != 0 || f & 0x100 != 0) {
            result += 1;
        }
    } else if e == -158 && f >= 0xffffff00 {
        result = 1 << 23;
    } else if e > -181 {
        let shift = -149 - e;
        result = f >> shift;
        if f & (1 << (shift - 1)) != 0
            && (f & ((1 << (shift - 1)) - 1) != 0 || f & (1 << shift) != 0)
        {
            result += 1;
        }
    } else if e == -181 && f > 0x80000000 {
        result = 1;
    } else {
        return 1.;
    }
    if result == 0x7f800000 {
        return if negative { -f32::MAX } else { f32::MAX };
    }
    if negative {
        result |= 0x80000000;
    }
    f32::from_bits(result)
}

/// MuPDF pdf-lex.c's real path: fz_atof/fz_strtof, or Adobe overflow handling.
pub fn lex_real(text: &str) -> f32 {
    let raw = text.as_bytes();
    if raw.iter().position(|b| *b == b'.').unwrap_or(raw.len()) >= 10 {
        let mut i = 0;
        let mut negative = false;
        while raw.get(i) == Some(&b'-') {
            negative = true;
            i += 1;
        }
        while raw.get(i) == Some(&b'+') {
            i += 1;
        }
        let mut whole = 0i32;
        while let Some(b'0'..=b'9') = raw.get(i) {
            whole = whole.wrapping_mul(10).wrapping_add((raw[i] - b'0') as i32);
            i += 1;
        }
        let mut value = whole as f32;
        if raw.get(i) == Some(&b'.') {
            i += 1;
            let (mut numerator, mut denominator) = (0f32, 1f32);
            while let Some(b'0'..=b'9') = raw.get(i) {
                numerator = 10. * numerator + (raw[i] - b'0') as f32;
                denominator *= 10.;
                i += 1;
            }
            value += numerator / denominator;
        }
        return if negative { -value } else { value };
    }
    let mut i = 0;
    let mut negative = false;
    if raw.get(i) == Some(&b'+') {
        i += 1;
    }
    if raw.get(i) == Some(&b'-') {
        negative = true;
        i += 1;
    }
    let (mut m, mut n, mut digits) = (0u32, 0i32, 0);
    while let Some(b'0'..=b'9') = raw.get(i) {
        let digit = (raw[i] - b'0') as u32;
        if digits != 0 {
            if digits < 9 {
                digits += 1;
                m = m * 10 + digit;
            } else if n < 1000 {
                n += 1;
            }
        } else if digit != 0 {
            m = digit;
            digits += 1;
        }
        i += 1;
    }
    if raw.get(i) == Some(&b'.') {
        i += 1;
    }
    while let Some(b'0'..=b'9') = raw.get(i) {
        let digit = (raw[i] - b'0') as u32;
        if digits < 9 {
            if digits != 0 || digit != 0 {
                digits += 1;
                m = m * 10 + digit;
            }
            n -= 1;
        }
        i += 1;
    }
    scaled(m, n, negative)
}

#[cfg(test)]
mod object_numbers_tests;
