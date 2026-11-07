use super::{Dictionary, Object, ObjectId, Reader, Stream, StringFormat};
use crate::Error;
use crate::content::*;
use crate::error;
use crate::xref::*;
use std::collections::HashSet;
use std::str::{self, FromStr};

use nom::branch::alt;
use nom::bytes::complete::{tag, take, take_while, take_while_m_n, take_while1};
use nom::character::complete::multispace1;
use nom::character::complete::{digit0, digit1, one_of};
use nom::character::complete::{space0, space1};
use nom::combinator::cut;
use nom::combinator::{map, map_opt, map_res, opt, verify};
use nom::error::{ErrorKind, ParseError};
use nom::multi::{fold_many0, fold_many1, many0, many0_count};
use nom::sequence::{delimited, pair, preceded, separated_pair, terminated};
use nom::{AsBytes, AsChar, IResult, Input, Parser};

thread_local! {
    static REAL_PARSER: std::cell::Cell<Option<fn(&str) -> f32>> = const { std::cell::Cell::new(None) };
}
pub(crate) fn real_parser() -> Option<fn(&str) -> f32> { REAL_PARSER.get() }
pub(crate) fn with_real_parser<T>(parser: Option<fn(&str) -> f32>, f: impl FnOnce() -> T) -> T {
    struct Reset(Option<fn(&str) -> f32>);
    impl Drop for Reset { fn drop(&mut self) { REAL_PARSER.set(self.0); } }
    let _reset = Reset(REAL_PARSER.replace(parser));
    f()
}

pub(crate) mod cmap_parser;

pub(crate) type ParserInput<'a> = &'a [u8];
// Change this to something else that implements ParseError to get a
// different error type out of nom.
#[derive(Debug, PartialEq)]
pub(crate) struct NomError<'a> {
    syntax: nom::error::Error<ParserInput<'a>>,
    resource: Option<Box<crate::resource::ResourceLimitExceeded>>,
}
impl<'a> ParseError<ParserInput<'a>> for NomError<'a> {
    fn from_error_kind(input: ParserInput<'a>, kind: ErrorKind) -> Self {
        Self { syntax: nom::error::Error::new(input, kind), resource: None }
    }
    fn append(_: ParserInput<'a>, _: ErrorKind, other: Self) -> Self { other }
    fn or(self, other: Self) -> Self { if self.resource.is_some() { self } else { other } }
}
impl<'a, E> nom::error::FromExternalError<ParserInput<'a>, E> for NomError<'a> {
    fn from_external_error(input: ParserInput<'a>, kind: ErrorKind, _: E) -> Self { Self::from_error_kind(input, kind) }
}
fn fatal_resource<'a>(input: ParserInput<'a>, e: &Error) -> nom::Err<NomError<'a>> {
    nom::Err::Failure(NomError { syntax: nom::error::Error::new(input, ErrorKind::Fail),
        resource: e.resource_limit().cloned().map(Box::new) })
}
fn indirect_nom_error(e: nom::Err<NomError<'_>>, offset: usize) -> Error {
    match e {
        nom::Err::Error(e) | nom::Err::Failure(e) if e.resource.is_some() => Error::ResourceLimitExceeded(*e.resource.unwrap()),
        _ => Error::IndirectObject { offset },
    }
}

pub(crate) type NomResult<'a, O, E = NomError<'a>> = IResult<ParserInput<'a>, O, E>;

#[inline]
fn strip_nom<O>(r: NomResult<O>) -> Option<O> {
    r.ok().map(|(_, o)| o)
}

#[inline]
fn convert_result<O, E>(result: Result<O, E>, input: ParserInput, error_kind: ErrorKind) -> NomResult<O> {
    result.map(|o| (input, o)).map_err(|_| {
        // this is a unit bind if NomError = ()
        let err: NomError = NomError::from_error_kind(input, error_kind);
        nom::Err::Error(err)
    })
}

#[inline]
fn offset_stream(object: &mut Object, offset: usize) {
    if let Object::Stream(stream) = object {
        stream.start_position = stream.start_position.and_then(|sp| sp.checked_add(offset));
    }
}

pub(crate) fn eol(input: ParserInput) -> NomResult<ParserInput> {
    alt((tag(&b"\r\n"[..]), tag(&b"\n"[..]), tag(&b"\r"[..]))).parse(input)
}

pub(crate) fn comment(input: ParserInput) -> NomResult<()> {
    map((tag(&b"%"[..]), take_while(|c: u8| !b"\r\n".contains(&c)), eol), |_| ()).parse(input)
}

#[inline]
fn is_whitespace(c: u8) -> bool {
    b" \t\n\r\0\x0C".contains(&c)
}

#[inline]
fn is_delimiter(c: u8) -> bool {
    b"()<>[]{}/%".contains(&c)
}

#[inline]
fn is_regular(c: u8) -> bool {
    !is_whitespace(c) && !is_delimiter(c)
}

#[inline]
fn is_direct_literal_string(c: u8) -> bool {
    !b"()\\\r\n".contains(&c)
}

fn white_space(input: ParserInput) -> NomResult<()> {
    map(take_while(is_whitespace), |_| ()).parse(input)
}

fn space(input: ParserInput) -> NomResult<()> {
    fold_many0(
        alt((map(take_while1(is_whitespace), |_| ()), comment)),
        || {},
        |_, _| (),
    )
    .parse(input)
}

fn integer(input: ParserInput) -> NomResult<i64> {
    let (i, _) = pair(opt(one_of("+-")), digit1).parse(input)?;

    let int_input = &input[..input.len() - i.len()];
    convert_result(i64::from_str(str::from_utf8(int_input).unwrap()), i, ErrorKind::Digit)
}

fn real(input: ParserInput) -> NomResult<f32> {
    let (i, _) = pair(
        opt(one_of("+-")),
        alt((
            map((digit1, tag(&b"."[..]), digit0), |_| ()),
            map(pair(tag(&b"."[..]), digit1), |_| ()),
        )),
    )
    .parse(input)?;

    let float_input = &input[..input.len() - i.len()];
    let text = str::from_utf8(float_input).unwrap();
    if let Some(parse) = real_parser() { Ok((i, parse(text))) }
    else { convert_result(f32::from_str(text), i, ErrorKind::Digit) }
}

pub(crate) fn hex_char(input: ParserInput) -> NomResult<u8> {
    map_res(
        verify(take(2usize), |h: ParserInput| {
            h.as_bytes().iter().copied().all(AsChar::is_hex_digit)
        }),
        |x: ParserInput| u8::from_str_radix(str::from_utf8(x).unwrap(), 16),
    )
    .parse(input)
}

fn oct_char(input: ParserInput) -> NomResult<u8> {
    map_res(
        take_while_m_n(1, 3, AsChar::is_oct_digit),
        // Spec requires us to ignore any overflow.
        |x: ParserInput| u16::from_str_radix(str::from_utf8(x).unwrap(), 8).map(|o| o as u8),
    )
    .parse(input)
}

pub(crate) fn name(input: ParserInput) -> NomResult<Vec<u8>> {
    preceded(
        tag(&b"/"[..]),
        many0(alt((
            preceded(tag(&b"#"[..]), hex_char),
            map_opt(take(1usize), |c: ParserInput| {
                if c[0] != b'#' && is_regular(c[0]) {
                    Some(c[0])
                } else {
                    None
                }
            }),
        ))),
    )
    .parse(input)
}

fn escape_sequence(input: ParserInput) -> NomResult<Option<u8>> {
    preceded(
        tag(&b"\\"[..]),
        alt((
            map(oct_char, Some),
            map(eol, |_| None),
            map(tag(&b"n"[..]), |_| Some(b'\n')),
            map(tag(&b"r"[..]), |_| Some(b'\r')),
            map(tag(&b"t"[..]), |_| Some(b'\t')),
            map(tag(&b"b"[..]), |_| Some(b'\x08')),
            map(tag(&b"f"[..]), |_| Some(b'\x0C')),
            map(take(1usize), |c: ParserInput| Some(c[0])),
        )),
    )
    .parse(input)
}

enum InnerLiteralString<'a> {
    Direct(ParserInput<'a>),
    Escape(Option<u8>),
    Eol(ParserInput<'a>),
    Nested(Vec<u8>),
}

impl InnerLiteralString<'_> {
    fn push(&self, output: &mut Vec<u8>) {
        match self {
            InnerLiteralString::Direct(s) | InnerLiteralString::Eol(s) => output.extend_from_slice(s),
            InnerLiteralString::Escape(e) => output.extend(e),
            InnerLiteralString::Nested(n) => output.extend_from_slice(n),
        }
    }
}

fn inner_literal_string(depth: usize) -> impl Fn(ParserInput) -> NomResult<Vec<u8>> {
    move |input| {
        fold_many0(
            alt((
                map(take_while1(is_direct_literal_string), InnerLiteralString::Direct),
                map(escape_sequence, InnerLiteralString::Escape),
                map(eol, InnerLiteralString::Eol),
                map(nested_literal_string(depth), InnerLiteralString::Nested),
            )),
            Vec::new,
            |mut out: Vec<u8>, value| {
                value.push(&mut out);
                out
            },
        )
        .parse(input)
    }
}

fn nested_literal_string(depth: usize) -> impl Fn(ParserInput) -> NomResult<Vec<u8>> {
    move |input| {
        if depth == 0 {
            map(verify(tag(&b"too deep"[..]), |_: &[u8]| false), |_| vec![]).parse(input)
        } else {
            map(
                delimited(tag(&b"("[..]), inner_literal_string(depth - 1), tag(&b")"[..])),
                |mut content| {
                    content.insert(0, b'(');
                    content.push(b')');
                    content
                },
            )
            .parse(input)
        }
    }
}

fn literal_string(input: ParserInput) -> NomResult<Vec<u8>> {
    delimited(
        tag(&b"("[..]),
        inner_literal_string(crate::reader::MAX_BRACKET),
        tag(&b")"[..]),
    )
    .parse(input)
}

#[inline]
fn hex_digit(input: ParserInput) -> NomResult<u8> {
    map_opt(take(1usize), |c: ParserInput| {
        str::from_utf8(c).ok().and_then(|c| u8::from_str_radix(c, 16).ok())
    })
    .parse(input)
}

fn hexadecimal_string(input: ParserInput) -> NomResult<Object> {
    map(
        delimited(
            tag(&b"<"[..]),
            terminated(
                fold_many0(
                    preceded(white_space, hex_digit),
                    || -> (Vec<u8>, bool) { (Vec::new(), false) },
                    |state, c| match state {
                        (mut out, false) => {
                            out.push(c << 4);
                            (out, true)
                        }
                        (mut out, true) => {
                            *out.last_mut().unwrap() |= c;
                            (out, false)
                        }
                    },
                ),
                white_space,
            ),
            tag(&b">"[..]),
        ),
        |(bytes, _)| Object::String(bytes, StringFormat::Hexadecimal),
    )
    .parse(input)
}

fn boolean(input: ParserInput) -> NomResult<Object> {
    alt((
        map(tag(&b"true"[..]), |_| Object::Boolean(true)),
        map(tag(&b"false"[..]), |_| Object::Boolean(false)),
    ))
    .parse(input)
}

fn null(input: ParserInput) -> NomResult<Object> {
    map(tag(&b"null"[..]), |_| Object::Null).parse(input)
}

fn array(depth: usize) -> impl Fn(ParserInput) -> NomResult<Vec<Object>> {
    move |input| {
        let (remaining, mut values) = delimited(
            pair(tag(&b"["[..]), space),
            many0(_direct_object(depth)),
            tag(&b"]"[..]),
        )
        .parse(input)?;
        values.shrink_to_fit();
        Ok((remaining, values))
    }
}

pub(crate) fn dictionary(input: ParserInput) -> NomResult<Dictionary> {
    _dictionary(crate::reader::MAX_NESTING_DEPTH)(input)
}

fn _dictionary(depth: usize) -> impl Fn(ParserInput) -> NomResult<Dictionary> {
    move |input| delimited(pair(tag(&b"<<"[..]), space), inner_dictionary(depth), tag(&b">>"[..])).parse(input)
}

fn inner_dictionary(depth: usize) -> impl Fn(ParserInput) -> NomResult<Dictionary> {
    move |input| {
        let (remaining, mut dictionary) = fold_many0(
            pair(terminated(name, space), _direct_object(depth)),
            Dictionary::new,
            |mut dict, (key, value)| {
                dict.set(key, value);
                dict
            },
        )
        .parse(input)?;
        dictionary.as_hashmap_mut().shrink_to_fit();
        Ok((remaining, dictionary))
    }
}

pub(crate) fn dict_dup(input: ParserInput) -> NomResult<Dictionary> {
    delimited(
        (
            digit1,
            space1,
            tag(&b"dict"[..]),
            space1,
            tag(&b"dup"[..]),
            space1,
            tag(&b"begin"[..]),
            multispace1,
        ),
        fold_many0(
            terminated(
                pair(
                    terminated(name, space),
                    _direct_object(crate::reader::MAX_NESTING_DEPTH),
                ),
                pair(tag(&b"def"[..]), multispace1),
            ),
            Dictionary::new,
            |mut dict, (key, value)| {
                dict.set(key, value);
                dict
            },
        ),
        tag(&b"end"[..]),
    )
    .parse(input)
}

/// Recover the sole EOL-framed `endstream` immediately followed by `endobj`.
/// The caller must pass only bytes up to the current indirect-object boundary
/// so the scan cannot cross into a neighboring object.
fn recover_stream_length(input: ParserInput) -> Option<(ParserInput, ParserInput)> {
    const ENDSTREAM: &[u8] = b"endstream";
    let mut recovered = None;

    for (position, candidate) in input.windows(ENDSTREAM.len()).enumerate() {
        if candidate != ENDSTREAM {
            continue;
        }

        let data_end = if input[..position].ends_with(b"\r\n") {
            position - 2
        } else if input[..position].ends_with(b"\n") || input[..position].ends_with(b"\r") {
            position - 1
        } else {
            continue;
        };
        let after_endstream = &input[position + ENDSTREAM.len()..];
        let Ok((after_endobj, _)) = preceded(space, tag(&b"endobj"[..])).parse(after_endstream) else {
            continue;
        };
        if after_endobj.first().is_some_and(|&byte| !is_whitespace(byte)) {
            continue;
        }
        if recovered.is_some() {
            return None;
        }
        recovered = Some((after_endstream, &input[..data_end]));
    }

    recovered
}

fn stream<'a>(
    input: ParserInput<'a>, reader: &Reader, already_seen: &mut HashSet<ObjectId>, recover_length: bool,
    recovery_bound: Option<usize>,
) -> NomResult<'a, Object> {
    let (i, dict) = terminated(dictionary, (space, tag(&b"stream"[..]), space0, eol)).parse(input)?;

    let length = dict.get(b"Length").and_then(|value| {
        if let Ok(id) = value.as_reference() {
            reader.get_object(id, already_seen).and_then(|value| value.as_i64())
        } else {
            value.as_i64()
        }
    });
    if let Err(e) = &length { if e.is_resource_limit() { return Err(fatal_resource(i, e)); } }
    if let Ok(length) = length {
        if length < 0 {
            // artificial error kind is created to allow descriptive nom errors
            return Err(nom::Err::Failure(NomError::from_error_kind(i, ErrorKind::LengthValue)));
        }
        let Ok(length) = usize::try_from(length) else {
            return Err(nom::Err::Failure(NomError::from_error_kind(i, ErrorKind::LengthValue)));
        };
        match terminated(take(length), pair(opt(eol), tag(&b"endstream"[..]))).parse(i) {
            Ok((remaining, data)) => Ok((remaining, Object::from(Stream::with_content(dict, reader.stream_content(data))))),
            Err(_) if recover_length && !reader.strict => {
                // The scan must not cross into a neighbouring indirect object,
                // so it stops at the xref-derived bound; parsing itself stays
                // unbounded. The bound arrived here as `input.len() - end`, and
                // `i` starts `input.len() - i.len()` bytes later, so the scan
                // covers exactly the first `i.len() - bound` bytes of `i`.
                let scan_end = recovery_bound.map_or(i.len(), |bound| i.len().saturating_sub(bound));
                let Some((remaining, data)) = recover_stream_length(&i[..scan_end]) else {
                    return Err(nom::Err::Failure(NomError::from_error_kind(i, ErrorKind::LengthValue)));
                };
                log::warn!(
                    "Stream Length is {length}, but the unambiguous object boundary gives {} bytes; using the recovered length.",
                    data.len()
                );
                Ok((remaining, Object::from(Stream::new(dict, data.to_vec()))))
            }
            Err(_) => Err(nom::Err::Failure(NomError::from_error_kind(i, ErrorKind::LengthValue))),
        }
    } else {
        // Return position relative to the start of the stream dictionary.
        Ok((i, Object::from(Stream::with_position(dict, input.len() - i.len()))))
    }
}

fn unsigned_int<I: FromStr>(input: ParserInput) -> NomResult<I> {
    map_res(digit1, |digits: ParserInput| {
        I::from_str(str::from_utf8(digits).unwrap())
    })
    .parse(input)
}

fn object_id(input: ParserInput) -> NomResult<ObjectId> {
    pair(terminated(unsigned_int, space), terminated(unsigned_int, space)).parse(input)
}

fn reference(input: ParserInput) -> NomResult<Object> {
    map(terminated(object_id, tag(&b"R"[..])), Object::Reference).parse(input)
}

fn _direct_objects(depth: usize) -> impl Fn(ParserInput) -> NomResult<Object> {
    move |input| {
        alt((
            null,
            boolean,
            reference,
            map(real, Object::Real),
            map(integer, Object::Integer),
            map(name, Object::Name),
            map(literal_string, Object::string_literal),
            hexadecimal_string,
            map(array(depth), Object::Array),
            map(_dictionary(depth), Object::Dictionary),
        ))
        .parse(input)
    }
}

fn _direct_object(depth: usize) -> impl Fn(ParserInput) -> NomResult<Object> {
    move |input| {
        if depth == 0 {
            return Err(nom::Err::Failure(NomError::from_error_kind(input, ErrorKind::TooLarge)));
        }
        terminated(_direct_objects(depth - 1), space).parse(input)
    }
}

pub fn direct_object(input: ParserInput) -> Option<Object> {
    strip_nom(_direct_object(crate::reader::MAX_NESTING_DEPTH)(input))
}

fn object<'a>(
    input: ParserInput<'a>, reader: &Reader, already_seen: &mut HashSet<ObjectId>, recover_stream_length: bool,
    recovery_bound: Option<usize>,
) -> NomResult<'a, Object> {
    terminated(
        alt((
            |input| stream(input, reader, already_seen, recover_stream_length, recovery_bound),
            _direct_objects(crate::reader::MAX_NESTING_DEPTH),
        )),
        space,
    )
    .parse(input)
}

pub fn indirect_object(
    input: ParserInput, offset: usize, expected_id: Option<ObjectId>, reader: &Reader,
    already_seen: &mut HashSet<ObjectId>, recovery_bound: Option<usize>,
) -> crate::Result<(ObjectId, Object)> {
    // Every downstream slice is a suffix of `input`, so express the absolute
    // end offset as the maximum length a suffix may have.
    let recovery_bound = recovery_bound.map(|end| input.len().saturating_sub(end));
    let (id, mut object) = _indirect_object(
        input.take_from(offset),
        offset,
        expected_id,
        reader,
        already_seen,
        true,
        recovery_bound,
    )?;

    offset_stream(&mut object, offset);

    Ok((id, object))
}

fn _indirect_object<'a>(
    input: ParserInput<'a>, offset: usize, expected_id: Option<ObjectId>, reader: &Reader,
    already_seen: &mut HashSet<ObjectId>, recover_stream_length: bool, recovery_bound: Option<usize>,
) -> crate::Result<(ObjectId, Object)> {
    let (i, (_, object_id)) = terminated((space, object_id), pair(tag(&b"obj"[..]), space))
        .parse(input)
        .map_err(|e| indirect_nom_error(e, offset))?;
    if let Some(expected_id) = expected_id
        && object_id != expected_id
    {
        return Err(crate::error::Error::ObjectIdMismatch);
    }

    let object_offset = input.len() - i.len();
    let (_, mut object) = terminated(
        |i: ParserInput<'a>| object(i, reader, already_seen, recover_stream_length, recovery_bound),
        (space, opt(tag(&b"endobj"[..])), space),
    )
    .parse(i)
    .map_err(|e| indirect_nom_error(e, offset))?;

    offset_stream(&mut object, object_offset);

    Ok((object_id, object))
}

pub fn header(input: ParserInput, strict: bool) -> Option<String> {
    // Parse version digits (e.g. "1.7") separately from any trailing bytes
    // before the newline.  Some PDF generators (e.g. ImageMill) place binary
    // marker bytes on the header line which would fail UTF-8 validation.
    // In strict mode we reject such trailing bytes; in lenient mode we skip them.
    let (_, (version_raw, trailing)) = delimited(
        tag(&b"%PDF-"[..]),
        pair(
            take_while(|c: u8| c.is_ascii_digit() || c == b'.'),
            take_while(|c: u8| !b"\r\n".contains(&c)),
        ),
        pair(eol, many0_count(comment)),
    )
    .parse(input)
    .ok()?;

    if strict && !trailing.is_empty() {
        return None;
    }

    let version = str::from_utf8(version_raw).ok()?.to_string();
    Some(version)
}

pub fn binary_mark(input: ParserInput) -> Option<Vec<u8>> {
    strip_nom(
        map_res(
            delimited(
                tag(&b"%"[..]),
                take_while(|c: u8| !b"\r\n".contains(&c)),
                pair(eol, many0_count(comment)),
            ),
            |v: ParserInput| Ok::<Vec<u8>, ()>(v.to_vec()),
        )
        .parse(input),
    )
}

/// Decode CrossReferenceTable
fn xref(input: ParserInput, strict: bool) -> NomResult<Xref> {
    // ISO 32000-1 s7.5.4 requires every entry to be exactly 20 bytes, ending in one of the
    // 2-byte terminators SP CR, SP LF or CR LF. Many generators instead emit 19-byte entries
    // ending in a bare LF, which qpdf, pikepdf, PDFium and PDF.js all accept. Accept those
    // too when parsing leniently, but keep the conforming 2-byte forms first in the `alt` so
    // that a bare CR never matches the CR of a conforming CR LF and strands its LF.
    let xref_eol = move |i| {
        let conforming = alt((tag(&b" \r"[..]), tag(&b" \n"[..]), tag(&b"\r\n"[..])));
        if strict {
            map(conforming, |_| ()).parse(i)
        } else {
            map(alt((conforming, tag(&b"\n"[..]), tag(&b"\r"[..]))), |_| ()).parse(i)
        }
    };
    let xref_entry = pair(
        separated_pair(unsigned_int, tag(&b" "[..]), unsigned_int::<u32>),
        delimited(tag(&b" "[..]), map(one_of("nf"), |k| k == 'n'), xref_eol),
    );

    let xref_section = pair(
        separated_pair(unsigned_int::<usize>, tag(&b" "[..]), unsigned_int::<u32>),
        preceded(pair(opt(tag(&b" "[..])), eol), many0(xref_entry)),
    );

    delimited(
        pair(tag(&b"xref"[..]), preceded(opt(tag(&b" "[..])), eol)),
        fold_many1(
            xref_section,
            || -> Xref { Xref::new(0, XrefType::CrossReferenceTable) },
            |mut xref, ((start, _count), entries)| {
                let mut skipped = 0usize;
                for (index, ((offset, generation), is_normal)) in entries.into_iter().enumerate() {
                    if is_normal && let Ok(generation) = generation.try_into() {
                        // `start` is read from the subsection header, so it is untrusted:
                        // `start + index` can overflow, and object numbers are u32, so a
                        // `start` above u32::MAX would truncate into a valid-looking number
                        // that silently displaces a legitimate entry. Skip whatever cannot
                        // be represented, as an out-of-range generation is skipped above.
                        match start.checked_add(index).and_then(|id| u32::try_from(id).ok()) {
                            Some(id) => xref.insert(id, XrefEntry::Normal { offset, generation }),
                            None => skipped += 1,
                        }
                    }
                }
                if skipped > 0 {
                    log::warn!(
                        "cross-reference subsection starting at {start} has {skipped} entr{} with an object number beyond u32; skipping",
                        if skipped == 1 { "y" } else { "ies" }
                    );
                }
                xref
            },
        ),
        space,
    )
    .parse(input)
}

fn trailer(input: ParserInput) -> NomResult<Dictionary> {
    delimited(pair(tag(&b"trailer"[..]), space), dictionary, space).parse(input)
}

pub fn xref_and_trailer(input: ParserInput, reader: &Reader) -> crate::Result<(Xref, Dictionary)> {
    let xref_trailer = map(pair(|i| xref(i, reader.strict), trailer), |(mut xref, trailer)| {
        xref.size = trailer
            .get(b"Size")
            .and_then(Object::as_i64)
.map_err(|_| error::ParseError::InvalidTrailer)? as u32;
        Ok((xref, trailer))
    });
    alt((
        xref_trailer,
        (|input| {
            _indirect_object(input, 0, None, reader, &mut HashSet::new(), false, None)
                .map(|(_, obj)| {
                    let res = match obj {
                        Object::Stream(stream) => crate::parser_aux::decode_xref_stream_with_policy(*stream, reader.max_decompressed_size, reader.parser_policy.as_ref()),
                        _ => Err(crate::error::ParseError::InvalidXref.into()),
                    };
                    (input, res)
                })
                .map_err(|e| {
                    if e.is_resource_limit() { return fatal_resource(input, &e); }
                    // artificial error kind is created to allow descriptive nom errors
                    nom::Err::Error(NomError::from_error_kind(input, ErrorKind::Fail))
                })
        }),
    ))
    .parse(input)
    .map(|(_, o)| o)
    .map_err(|e| {
        match e {
            nom::Err::Error(e) | nom::Err::Failure(e) if e.resource.is_some() => Error::ResourceLimitExceeded(*e.resource.unwrap()),
            _ => error::ParseError::InvalidTrailer.into(),
        }
    })?
}

pub fn xref_start(input: ParserInput) -> Option<i64> {
    strip_nom(
        delimited(
            pair(tag(&b"startxref"[..]), preceded(opt(tag(&b" "[..])), eol)),
            trim_spaces(integer),
            (eol, tag(&b"%%EOF"[..]), space),
        )
        .parse(input),
    )
}

fn trim_spaces<'a, O>(
    p: impl Parser<ParserInput<'a>, Output = O, Error = NomError<'a>>,
) -> impl Parser<ParserInput<'a>, Output = O, Error = NomError<'a>> {
    delimited(many0(tag(" ")), p, many0(tag(" ")))
}

// The following code create parser to parse content stream.

fn content_space(input: ParserInput) -> NomResult<()> {
    map(take_while(|c| b" \t\r\n".contains(&c)), |_| ()).parse(input)
}

fn operator(input: ParserInput) -> NomResult<String> {
    map_res(
        take_while1(|c: u8| c.is_ascii_alphabetic() || b"*'\"".contains(&c)),
        |op: ParserInput| str::from_utf8(op).map(Into::into),
    )
    .parse(input)
}

fn operand(input: ParserInput) -> NomResult<Object> {
    terminated(
        alt((
            null,
            boolean,
            map(real, Object::Real),
            map(integer, Object::Integer),
            map(name, Object::Name),
            map(literal_string, Object::string_literal),
            hexadecimal_string,
            map(array(crate::reader::MAX_NESTING_DEPTH), Object::Array),
            map(dictionary, Object::Dictionary),
        )),
        content_space,
    )
    .parse(input)
}

fn operation(input: ParserInput) -> NomResult<Operation> {
    map(
        preceded(
            // A comment consumes its end-of-line marker, so also skip any
            // white space that follows it (e.g. an indented next line).
            many0(terminated(comment, content_space)),
            alt((inline_image, terminated(pair(many0(operand), operator), content_space))),
        ),
        |(operands, operator)| Operation { operator, operands },
    )
    .parse(input)
}

fn inline_image(input: ParserInput) -> NomResult<(Vec<Object>, String)> {
    preceded(pair(tag(&b"BI"[..]), content_space), cut(inline_image_impl)).parse(input)
}

fn inline_image_impl(input: ParserInput) -> NomResult<(Vec<Object>, String)> {
    let (input, stream_dict) = inner_dictionary(crate::reader::MAX_NESTING_DEPTH).parse(input)?;
    let (input, _) = pair(tag(&b"ID"[..]), content_space).parse(input)?;
    match image_data_stream(input, stream_dict) {
        Ok((input, stream)) => {
            let (input, _) = (content_space, tag(&b"EI"[..]), content_space).parse(input)?;
            Ok((input, (vec![Object::from(stream)], String::from("BI"))))
        }
        Err(e) => {
            // Skip to EI marker so the rest of the content stream can still be parsed.
            log::warn!("Skipping unparseable inline image: {e}");
            let bytes = input;
            // EI must appear after whitespace to distinguish from data bytes.
            let ei_pos = bytes
                .windows(4)
                .position(|w| {
                    (w[0] == b' ' || w[0] == b'\n' || w[0] == b'\r')
                        && w[1] == b'E'
                        && w[2] == b'I'
                        && (w[3] == b' ' || w[3] == b'\n' || w[3] == b'\r')
                })
                .ok_or_else(|| {
                    let err: NomError = NomError::from_error_kind(input, ErrorKind::Fail);
                    nom::Err::Failure(err)
                })?;
            let (input, _) = take(ei_pos + 3).parse(input).map_err(|_: nom::Err<()>| {
                let err: NomError = NomError::from_error_kind(input, ErrorKind::Fail);
                nom::Err::Failure(err)
            })?;
            let (input, _) = content_space(input)?;
            Ok((input, (vec![], String::from("BI"))))
        }
    }
}

fn image_data_stream(input: ParserInput, stream_dict: Dictionary) -> crate::Result<(ParserInput, Stream)> {
    let get_abbr = |key_abbr: &[u8], key: &[u8]| stream_dict.get(key_abbr).or_else(|_| stream_dict.get(key));
    let width = get_abbr(b"W", b"Width")?.as_i64()? as usize;
    let height = get_abbr(b"H", b"Height")?.as_i64()? as usize;
    let bpc = get_abbr(b"BPC", b"BitsPerComponent")?.as_i64()? as usize;
    let im = get_abbr(b"IM", b"ImageMask").and_then(|x| x.as_bool());
    let num_colors = match im {
        // If we have an image mask then we don't have a colorspace
        Ok(true) => 1,
        _ => {
            let colorspace = get_abbr(b"CS", b"ColorSpace")?.as_name()?;
            match colorspace {
                b"DeviceGray" | b"Gray" => 1,
                b"DeviceRGB" | b"RGB" => 3,
                b"DeviceRGBA" | b"RGBA" => 4,
                b"DeviceCMYK" | b"CMYK" => 4,
                b"Pattern" => {
                    log::warn!("Pattern colorspace is not allowed in inline images");
                    return Err(Error::InvalidInlineImage(String::from(
                        "Pattern colorspace is not allowed in inline images",
                    )));
                }
                _ => {
                    log::warn!("Colorspace of inline image not recognized / not yet implemented");
                    return Err(Error::Unimplemented("inline image colorspaces"));
                }
            }
        }
    };

    let stride = (width * (num_colors * bpc)).div_ceil(8);
    let length = height * stride;

    let (input, content) = match get_abbr(b"F", b"Filter") {
        Err(_) => {
            // no decompression needed as no filter was applied
            take(length)
                .parse(input)
                .map_err(|_: nom::Err<()>| crate::error::ParseError::EndOfInput)?
        }
        Ok(Object::Name(_filter)) => {
            log::warn!("Filters for inline images are not yet implemented");
            return Err(Error::Unimplemented("filters for inline images"));
        }
        Ok(Object::Array(_filters)) => {
            log::warn!("Filters for inline images are not yet implemented");
            return Err(Error::Unimplemented("filters for inline images"));
        }
        Ok(obj) => {
            log::warn!("Filter must be either a Name or and Array.");
            return Err(Error::ObjectType {
                expected: "Name or Array",
                found: obj.enum_variant(),
            });
        }
    };
    Ok((input, Stream::new(stream_dict, content.to_vec())))
}

fn _content(input: ParserInput) -> NomResult<Content<Vec<Operation>>> {
    delimited(
        content_space,
        map(many0(operation), |operations| Content { operations }),
        many0(terminated(comment, content_space)),
    )
    .parse(input)
}

pub fn content(input: ParserInput) -> Option<Content<Vec<Operation>>> {
    strip_nom(_content.parse(input))
}

pub fn content_strict(input: ParserInput) -> Result<Content<Vec<Operation>>, error::ParseError> {
    let (rest, content) = _content
        .parse(input)
        .map_err(|_| error::ParseError::InvalidContentStream)?;
    if !rest.is_empty() {
        return Err(error::ParseError::InvalidContentStream);
    }
    Ok(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_span(s: &'_ [u8]) -> ParserInput<'_> {
        s
    }

    fn tstrip<O>(r: NomResult<O>) -> Option<O> {
        r.ok().and_then(|(i, o)| if !i.is_empty() { None } else { Some(o) })
    }

    #[test]
    fn parse_real_number() {
        let real = |i| tstrip(real(i));

        assert_eq!(real(test_span(b"0.12")), Some(0.12));
        assert_eq!(real(test_span(b"-.12")), Some(-0.12));
        assert_eq!(real(test_span(b"10.")), Some(10.0));
    }

    #[test]
    fn parse_string() {
        let literal_string = |i| tstrip(literal_string(i));

        let data = vec![
            ("()", ""),
            ("(text())", "text()"),
            ("(text\r\n\\\\(nested\\t\\b\\f))", "text\r\n\\(nested\t\x08\x0C)"),
            ("(text\\0\\53\\053\\0053)", "text\0++\x053"),
            ("(text line\\\n())", "text line()"),
        ];

        for (input, expected) in data {
            assert_eq!(
                literal_string(test_span(input.as_bytes())),
                Some(expected.as_bytes().to_vec()),
                "input: {:?} output: {:?}",
                input,
                expected,
            );
        }
    }

    #[test]
    fn parse_name() {
        let (text, expected) = (b"/ABC#5f", b"ABC\x5F");
        let result = tstrip(name(test_span(text)));
        assert_eq!(result, Some(expected.to_vec()));

        let (text, expected) = (b"/#cb#ce#cc#e5", b"\xcb\xce\xcc\xe5");
        let result = tstrip(name(test_span(text)));
        assert_eq!(result, Some(expected.to_vec()));
    }

    #[test]
    /// Run `cargo test -- --nocapture` to see output
    fn parse_content() {
        let stream = b"
2 J
BT
/F1 12 Tf
0 Tc
0 Tw
72.5 712 TD
[(Unencoded streams can be read easily) 65 (,) ] TJ
0 -14 TD
[(b) 20 (ut generally tak) 10 (e more space than \\311)] TJ
T* (encoded streams.) Tj
		";
        let content = tstrip(_content(test_span(stream)));
        println!("{:?}", content);
        assert!(content.is_some());
    }

    #[test]
    fn hex_partial() {
        // Example from PDF specification.
        let out = tstrip(hexadecimal_string(test_span(b"<901FA>")));

        match out {
            Some(Object::String(s, _)) => assert_eq!(s, b"\x90\x1F\xA0".to_vec()),
            _ => panic!("unexpected {:?}", out),
        }
    }

    #[test]
    fn hex_separated() {
        let out = tstrip(hexadecimal_string(test_span(b"<9 01F A>")));

        match out {
            Some(Object::String(s, _)) => assert_eq!(s, b"\x90\x1F\xA0".to_vec()),
            _ => panic!("unexpected {:?}", out),
        }
    }

    #[test]
    fn big_generation_value() {
        let input = b"xref
0 1
0000000000 65536 f\x20
0 16
0000000000 65535 f\x20
0000153238 00000 n\x20
0000000019 00000 n\x20
0000000313 00000 n\x20
0000000333 00000 n\x20
0000145531 00000 n\x20
0000153407 00000 n\x20
0000145554 00000 n\x20
0000152303 00000 n\x20
0000152324 00000 n\x20
0000152514 00000 n\x20
0000152880 00000 n\x20
0000153106 00000 n\x20
0000153139 00000 n\x20
0000153532 00000 n\x20
0000153629 00000 n\x20
trailer
<</Size 16/Root 14 0 R
/Info 15 0 R
/ID [ <9DDC4B621B3F485FF5ED0F57D00A028F>
<9DDC4B621B3F485FF5ED0F57D00A028F> ]
/DocChecksum /2BCC3C7DE26E6BF3573E4A6E8362221F
>>
startxref
153804\x20
%%EOF
";
        match xref(test_span(input), false) {
            Ok((_, re)) => assert_eq!(re.entries.len(), 15),
            Err(err) => panic!("unexpected {:?}", err),
        }
    }

    #[test]
    fn space_in_startxref_number() {
        let input = b"startxref
153804\x20
%%EOF
";
        match xref_start(test_span(input)) {
            Some(num) => assert_eq!(num, 153804),
            None => panic!("could not parse number in startxref"),
        }
    }

    #[test]
    fn header_standard() {
        // Standard header with proper EOL
        let input = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n";
        assert_eq!(header(test_span(input), false), Some("1.7".to_string()));
    }

    #[test]
    fn header_with_binary_bytes_on_same_line() {
        // Some generators (e.g. ImageMill) place binary marker bytes on the
        // header line without a separating newline or '%' prefix.
        let input = b"%PDF-1.3 \xb0\x9f\x92\x9c\x9f\xd4\xe0\xce\xd0\xd0\xd0\r1 0 obj\r";
        assert_eq!(header(test_span(input), false), Some("1.3".to_string()));
    }

    #[test]
    fn header_with_binary_bytes_strict_rejects() {
        // In strict mode, binary bytes on the header line should cause a
        // parse failure (the raw bytes are not valid UTF-8).
        let input = b"%PDF-1.3 \xb0\x9f\x92\x9c\x9f\xd4\xe0\xce\xd0\xd0\xd0\r1 0 obj\r";
        assert_eq!(header(test_span(input), true), None);
    }

    #[test]
    fn header_cr_line_ending() {
        // CR-only line ending (common in older PDFs)
        let input = b"%PDF-1.3\r%\xe2\xe3\xcf\xd3\r";
        assert_eq!(header(test_span(input), false), Some("1.3".to_string()));
    }

    #[test]
    fn header_crlf_line_ending() {
        // CRLF line ending (common on Windows-generated PDFs)
        let input = b"%PDF-1.7\r\n%\xe2\xe3\xcf\xd3\r\n";
        assert_eq!(header(test_span(input), false), Some("1.7".to_string()));
    }

    #[test]
    fn header_pdf_2_0() {
        let input = b"%PDF-2.0\n%\xe2\xe3\xcf\xd3\n";
        assert_eq!(header(test_span(input), false), Some("2.0".to_string()));
    }

    #[test]
    fn content_with_comments() {
        // It should be processed as usual but ignoring the comments
        let input = b"0.5 0.5 0.5 setrgbcolor
% This is a comment
100 100 moveto
(Hello, world!) show
% Another comment
";
        let out = content(test_span(input)).unwrap();
        let out_strict = content_strict(test_span(input)).unwrap();
        assert_eq!(out.operations.len(), out_strict.operations.len());
        assert_eq!(out.operations.len(), 3);
    }

    #[test]
    fn content_with_comment_followed_by_indented_line() {
        // A comment is equivalent to a single white-space character
        // (ISO 32000-2, 7.2.4), so a line starting with white space right
        // after a comment must not stop the parser (issue #535).
        let input = b"BT /F1 24 Tf
% comment
  100 100 Td (Hello World) Tj ET";
        let out = content(test_span(input)).unwrap();
        let out_strict = content_strict(test_span(input)).unwrap();
        assert_eq!(out.operations.len(), out_strict.operations.len());
        let ops: Vec<&str> = out.operations.iter().map(|o| o.operator.as_str()).collect();
        assert_eq!(ops, vec!["BT", "Tf", "Td", "Tj", "ET"]);
    }

    #[test]
    fn inline_image_unknown_colorspace_skipped() {
        // Inline image with an unrecognized colorspace ("ICCBased" is not handled).
        // The parser should skip it and still parse the surrounding operations.
        let input = b"q 100 100 moveto
BI /W 2 /H 2 /CS /ICCBased /BPC 8
ID
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
EI
(Hello) Tj Q";
        let out = content(test_span(input)).unwrap();
        // Should have: q, moveto, BI (skipped), Tj, Q = 5 operations
        let ops: Vec<&str> = out.operations.iter().map(|o| o.operator.as_str()).collect();
        assert!(ops.contains(&"q"), "missing q, got: {:?}", ops);
        assert!(ops.contains(&"Tj"), "missing Tj, got: {:?}", ops);
        assert!(ops.contains(&"Q"), "missing Q, got: {:?}", ops);
    }

    #[test]
    fn inline_image() {
        let _ = env_logger::try_init();
        let input = b"BI /W 4 /H 4 /CS /RGB /BPC 8
ID
00000z0z00zzz00z0zzz0zzzEI aazazaazzzaazazzzazzz
EI";
        let out = super::inline_image(test_span(input)).unwrap().1;
        assert_eq!(&out.1, "BI");
        assert_eq!(
            &out.0[0].as_stream().unwrap().content,
            b"00000z0z00zzz00z0zzz0zzzEI aazazaazzzaazazzzazzz"
        )
    }

    #[test]
    fn xref_trailing_space_after_keyword() {
        // Some PDF generators emit "xref \n" with a trailing space.
        let input = b"xref \n0 3\n0000000000 65535 f \n0000000017 00000 n \n0000000081 00000 n \ntrailer\n<</Size 3/Root 1 0 R>>\nstartxref\n175\n%%EOF\n";
        match xref(test_span(input), false) {
            Ok((_, re)) => assert_eq!(re.entries.len(), 2),
            Err(err) => panic!("xref with trailing space should parse: {:?}", err),
        }
    }

    #[test]
    fn xref_entries_with_bare_eol_terminator() {
        // ISO 32000-1 s7.5.4 requires 20-byte entries, so the terminator is two bytes
        // (" \r", " \n" or "\r\n"). Many generators drop the padding space and emit
        // 19-byte entries ending in a bare "\n"; qpdf, pikepdf, PDFium and PDF.js all
        // accept these, so lenient parsing accepts them too.
        for (name, input) in [
            (
                "bare LF",
                &b"xref\n0 3\n0000000000 65535 f\n0000000017 00000 n\n0000000081 00000 n\ntrailer\n<</Size 3/Root 1 0 R>>\n"[..],
            ),
            (
                "bare CR",
                &b"xref\n0 3\n0000000000 65535 f\r0000000017 00000 n\r0000000081 00000 n\rtrailer\n<</Size 3/Root 1 0 R>>\n"[..],
            ),
        ] {
            match xref(test_span(input), false) {
                Ok((_, re)) => assert_eq!(re.entries.len(), 2, "{name} should yield both normal entries"),
                Err(err) => panic!("19-byte entries ({name}) should parse when lenient: {err:?}"),
            }
        }
    }

    #[test]
    fn xref_entries_with_bare_eol_rejected_when_strict() {
        // The 19-byte form is non-conforming, so strict mode must keep rejecting it.
        // At this level the rejection is indirect: the entries simply are not recognised,
        // leaving an empty section whose unconsumed lines then displace `trailer`. Assert
        // both halves -- the empty table here, and the document-level failure below.
        let input =
            b"xref\n0 3\n0000000000 65535 f\n0000000017 00000 n\n0000000081 00000 n\ntrailer\n<</Size 3/Root 1 0 R>>\n";
        if let Ok((_, re)) = xref(test_span(input), true) {
            assert_eq!(re.entries.len(), 0, "strict must not accept 19-byte entries");
        }

        // The contract that matters to callers: a document with 19-byte entries loads
        // leniently and is refused under `LoadOptions::strict`. The 20-byte build of the
        // very same document is the control -- it must load in *both* modes, so that the
        // strict rejection below is attributable to the terminator and nothing else.
        let load = |bytes: &[u8], strict: bool| {
            crate::Document::load_mem_with_options(
                bytes,
                crate::LoadOptions {
                    strict,
                    ..Default::default()
                },
            )
        };

        for (name, term, strict_should_load) in [("19-byte", "\n", false), ("20-byte", " \n", true)] {
            let header = "%PDF-1.7\n";
            let obj1 = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n";
            let obj2 = "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n";
            let obj3 = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>\nendobj\n";
            let o1 = header.len();
            let o2 = o1 + obj1.len();
            let o3 = o2 + obj2.len();
            let body = format!("{header}{obj1}{obj2}{obj3}");
            let xref_pos = body.len();
            let doc = format!(
                "{body}xref\n0 4\n0000000000 65535 f{term}{o1:010} 00000 n{term}{o2:010} 00000 n{term}\
                 {o3:010} 00000 n{term}trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
            );

            assert!(
                load(doc.as_bytes(), false).is_ok(),
                "{name} entries should load when lenient"
            );
            assert_eq!(
                load(doc.as_bytes(), true).is_ok(),
                strict_should_load,
                "{name} entries under strict parsing"
            );
        }
    }

    #[test]
    fn xref_entries_with_conforming_terminators() {
        // The three 20-byte terminators of s7.5.4 must keep parsing in both modes. In
        // particular the bare-CR alternative must not match the CR of a "\r\n" pair and
        // strand its LF, which would break the following entry.
        for (name, input) in [
            (
                "SP LF",
                &b"xref\n0 3\n0000000000 65535 f \n0000000017 00000 n \n0000000081 00000 n \ntrailer\n<</Size 3/Root 1 0 R>>\n"[..],
            ),
            (
                "SP CR",
                &b"xref\n0 3\n0000000000 65535 f \r0000000017 00000 n \r0000000081 00000 n \rtrailer\n<</Size 3/Root 1 0 R>>\n"[..],
            ),
            (
                "CR LF",
                &b"xref\n0 3\n0000000000 65535 f\r\n0000000017 00000 n\r\n0000000081 00000 n\r\ntrailer\n<</Size 3/Root 1 0 R>>\n"[..],
            ),
        ] {
            for strict in [false, true] {
                match xref(test_span(input), strict) {
                    Ok((_, re)) => assert_eq!(re.entries.len(), 2, "{name} (strict={strict}) lost an entry"),
                    Err(err) => panic!("conforming {name} entries should parse (strict={strict}): {err:?}"),
                }
            }
        }
    }

    #[test]
    fn xref_subsection_start_near_usize_max_is_skipped() {
        // A subsection header's start is read straight from the file, so it is untrusted.
        // A start of usize::MAX made `start + index` overflow: a panic wherever overflow
        // checks are on (a denial of service for any consumer parsing untrusted PDFs), and
        // a silent wrap to object 0 where they are not.
        let input = &b"xref\n18446744073709551615 2\n0000000000 65535 f \n0000000009 00000 n \ntrailer\n<</Size 2/Root 1 0 R>>\n"[..];
        for strict in [false, true] {
            match xref(test_span(input), strict) {
                Ok((_, re)) => assert!(
                    re.entries.is_empty(),
                    "unrepresentable object number was inserted (strict={strict}): {:?}",
                    re.entries
                ),
                Err(err) => panic!("xref should still parse (strict={strict}): {err:?}"),
            }
        }
    }

    #[test]
    fn xref_subsection_start_beyond_u32_does_not_displace_entries() {
        // Object numbers are u32, but the subsection start is parsed as usize and was cast
        // with `as u32`. A start above u32::MAX truncated into a valid-looking number --
        // 4294967297 becomes 1 -- silently overwriting a legitimate entry with an arbitrary
        // offset. No overflow occurs here, so a checked add alone would not catch it.
        let input = &b"xref\n0 2\n0000000000 65535 f \n0000000009 00000 n \n4294967297 1\n0000000999 00000 n \ntrailer\n<</Size 2/Root 1 0 R>>\n"[..];
        for strict in [false, true] {
            match xref(test_span(input), strict) {
                Ok((_, re)) => {
                    assert_eq!(
                        re.entries.len(),
                        1,
                        "(strict={strict}) unexpected entries: {:?}",
                        re.entries
                    );
                    assert!(
                        matches!(
                            re.get(1),
                            Some(XrefEntry::Normal {
                                offset: 9,
                                generation: 0
                            })
                        ),
                        "entry for object 1 was displaced (strict={strict}): {:?}",
                        re.get(1)
                    );
                }
                Err(err) => panic!("xref should still parse (strict={strict}): {err:?}"),
            }
        }
    }

    #[test]
    fn startxref_trailing_space_after_keyword() {
        // Some PDF generators emit "startxref \n" with a trailing space.
        let input = b"startxref \n135738\n%%EOF\n";
        match xref_start(test_span(input)) {
            Some(num) => assert_eq!(num, 135738),
            None => panic!("startxref with trailing space should parse"),
        }
    }

    #[test]
    fn content_silently_truncates_corrupted_data() {
        // Corrupted data with unterminated string literal
        let data = b"q 1 0 0 1 10 10 cm (corrupted Q";

        let content = content(data).unwrap();

        // Operations before the corruption returned without an error.
        // Trailing Q was silently dropped.
        assert_eq!(content.operations.len(), 2);
        assert_eq!(content.operations[0].operator, "q");
        assert_eq!(content.operations[1].operator, "cm");
    }

    #[test]
    fn content_strict_rejects_corrupted_data() {
        let data = b"q 1 0 0 1 10 10 cm (corrupted Q";
        assert!(content_strict(data).is_err());
    }

    fn on_big_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn deeply_nested_array_is_rejected() {
        on_big_stack(|| {
            let depth = 200_000;
            let mut input = Vec::with_capacity(depth * 2);
            input.extend(std::iter::repeat_n(b'[', depth));
            input.extend(std::iter::repeat_n(b']', depth));
            let result = _direct_object(crate::reader::MAX_NESTING_DEPTH)(input.as_slice());
            assert!(result.is_err());
        });
    }

    #[test]
    fn deeply_nested_dictionary_is_rejected() {
        on_big_stack(|| {
            let depth = 200_000;
            let mut input = Vec::with_capacity(depth * 6);
            for _ in 0..depth {
                input.extend_from_slice(b"<</K ");
            }
            for _ in 0..depth {
                input.extend_from_slice(b">>");
            }
            let result = _direct_object(crate::reader::MAX_NESTING_DEPTH)(input.as_slice());
            assert!(result.is_err());
        });
    }

    #[test]
    fn modestly_nested_array_still_parses() {
        let input = b"[[[[[1]]]]]";
        let obj = _direct_object(crate::reader::MAX_NESTING_DEPTH)(test_span(input));
        assert!(obj.is_ok());
    }
}
