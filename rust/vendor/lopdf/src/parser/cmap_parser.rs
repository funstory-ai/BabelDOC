use crate::cmap_section::{ArrayOfTargetStrings, CMapParseError, CMapSection, CodeLen, SourceCode, SourceRangeMapping};
use crate::parser::{NomResult, ParserInput, comment, dict_dup, dictionary, eol, hex_char, name};
use nom::Parser;
use nom::branch::alt;
pub use nom::bytes::complete::tag;
use nom::combinator::{map, opt};
use nom::error::ParseError;
use nom::multi::{fold_many_m_n, fold_many0, fold_many1, many_m_n, many0, many1, separated_list1};
use nom::sequence::{pair, preceded, separated_pair, terminated};
use nom::{character::complete::digit1, sequence::delimited};

impl<E> From<nom::Err<E>> for CMapParseError {
    fn from(err: nom::Err<E>) -> Self {
        match err {
            nom::Err::Incomplete(_) => CMapParseError::Incomplete,
            // normally nom::Err::Error is a recoverable error, but CMapParseError is the return type
            // so we assume that there are no more parsing branches to check
            nom::Err::Failure(_) | nom::Err::Error(_) => CMapParseError::Error,
        }
    }
}

pub(crate) fn parse(stream_content: ParserInput) -> Result<Vec<CMapSection>, CMapParseError> {
    let result = cmap_stream(stream_content);
    let result = result.map_err(CMapParseError::from)?;
    Ok(result.1)
}

fn cmap_stream(input: ParserInput) -> NomResult<Vec<CMapSection>> {
    delimited(
        cidinit_procset,
        cmap_resource_dictionary,
        (tag(&b"end"[..]), multispace0),
    )
    .parse(input)
}

fn space0(input: ParserInput) -> NomResult<()> {
    fold_many0(alt((tag(&b" "[..]), tag("\t"))), || {}, |_, _| ()).parse(input)
}

fn space1(input: ParserInput) -> NomResult<()> {
    fold_many1(alt((tag(&b" "[..]), tag("\t"))), || {}, |_, _| ()).parse(input)
}

fn multispace0(input: ParserInput) -> NomResult<()> {
    let space = tag(&b" "[..]).map(|_| ());
    let tab = tag("\t").map(|_| ());
    let eol = eol.map(|_| ());
    fold_many0(alt((space, tab, eol, comment)), || {}, |_, _| ()).parse(input)
}

fn multispace1(input: ParserInput) -> NomResult<()> {
    let space = tag(&b" "[..]).map(|_| ());
    let tab = tag("\t").map(|_| ());
    let eol = eol.map(|_| ());
    fold_many1(alt((space, tab, eol, comment)), || {}, |_, _| ()).parse(input)
}

fn cidinit_procset(input: ParserInput) -> NomResult<()> {
    (
        opt(tag("\u{FEFF}".as_bytes())),
        multispace0,
        tag(&b"/CIDInit"[..]),
        space0,
        alt((tag(&b"/ProcSet"[..]), tag(&b"/Procset"[..]))),
        space1,
        tag(&b"findresource"[..]),
        space1,
        tag(&b"begin"[..]),
        multispace1,
    )
        .parse(input)
        .map(|(i, _)| (i, ()))
}

fn cmap_resource_dictionary(input: ParserInput) -> NomResult<Vec<CMapSection>> {
    let begin_parser = (
        digit1,
        space1,
        tag(&b"dict"[..]),
        space1,
        tag(&b"begin"[..]),
        multispace1,
    );
    let end_parser = (tag(&b"end"[..]), multispace1);
    delimited(begin_parser, cmap_data, end_parser).parse(input)
}

fn cmap_data(input: ParserInput) -> NomResult<Vec<CMapSection>> {
    let cmap_end = (
        tag(&b"endcmap"[..]),
        multispace1,
        tag(&b"CMapName"[..]),
        space1,
        tag(&b"currentdict"[..]),
        space1,
        tag(&b"/CMap"[..]),
        space1,
        tag(&b"defineresource"[..]),
        space1,
        tag(&b"pop"[..]),
        multispace1,
    );
    delimited(
        (tag(&b"begincmap"[..]), multispace1),
        preceded(cmap_metadata, cmap_codespace_and_mappings),
        cmap_end,
    )
    .parse(input)
}

fn cmap_metadata(input: ParserInput) -> NomResult<()> {
    let metadata_parser = alt((
        cid_system_info,
        cmap_name,
        cmap_type,
        cmap_version,
        uidoffset,
        wmode,
        xuid,
    ));
    fold_many_m_n(1, 7, metadata_parser, || (), |_, _| ()).parse(input)
}

fn cid_system_info(input: ParserInput) -> NomResult<()> {
    // Note: Can array of CIDSystemInfo occur here?
    // Normally in cmap this can be an array, but can it be also if this is a ToUnicode cmap?
    (
        tag(&b"/CIDSystemInfo"[..]),
        multispace0,
        alt((dictionary, dict_dup)),
        multispace1,
        tag(&b"def"[..]),
        multispace1,
    )
        .parse(input)
        .map(|(i, _)| (i, ()))
}

fn cmap_name(input: ParserInput) -> NomResult<()> {
    (
        tag(&b"/CMapName"[..]),
        space0,
        name,
        space1,
        tag(&b"def"[..]),
        multispace1,
    )
        .parse(input)
        .map(|(i, _)| (i, ()))
}

fn cmap_type(input: ParserInput) -> NomResult<()> {
    (
        tag(&b"/CMapType"[..]),
        space1,
        digit1,
        space1,
        tag(&b"def"[..]),
        multispace1,
    )
        .parse(input)
        .map(|(i, _)| (i, ()))
}

fn cmap_version(input: ParserInput) -> NomResult<()> {
    let version = (digit1, opt((tag(&b"."[..]), digit1)));
    (
        tag(&b"/CMapVersion"[..]),
        space1,
        version,
        space1,
        tag(&b"def"[..]),
        multispace1,
    )
        .parse(input)
        .map(|(i, _)| (i, ()))
}

fn wmode(input: ParserInput) -> NomResult<()> {
    (
        tag(&b"/WMode"[..]),
        space1,
        digit1,
        space1,
        tag(&b"def"[..]),
        multispace1,
    )
        .parse(input)
        .map(|(i, _)| (i, ()))
}

fn uidoffset(input: ParserInput) -> NomResult<()> {
    (
        tag(&b"/UIDOffset"[..]),
        space1,
        digit1,
        space1,
        tag(&b"def"[..]),
        multispace1,
    )
        .parse(input)
        .map(|(i, _)| (i, ()))
}

fn xuid(input: ParserInput) -> NomResult<()> {
    let array = (tag(&b"["[..]), separated_list1(space1, digit1), tag(&b"]"[..]));
    (tag(&b"/XUID"[..]), space1, array, space1, tag(&b"def"[..]), multispace1)
        .parse(input)
        .map(|(i, _)| (i, ()))
}

fn cmap_codespace_and_mappings(input: ParserInput) -> NomResult<Vec<CMapSection>> {
    many1(alt((codespace_range_section, bf_char_section, bf_range_section))).parse(input)
}

fn codespace_range_section(input: ParserInput) -> NomResult<CMapSection> {
    let begin_section = (digit1, space1, tag(&b"begincodespacerange"[..]), multispace1);
    let end_section = (tag(&b"endcodespacerange"[..]), multispace1);
    let parse_range = delimited(space0, code_range_pair, multispace1);
    let (rest_of_input, ranges_result) = delimited(begin_section, many1(parse_range), end_section).parse(input)?;
    Ok((rest_of_input, CMapSection::CsRange(ranges_result)))
}

fn code_range_pair(input: ParserInput) -> NomResult<(SourceCode, SourceCode, CodeLen)> {
    let (rest_of_input, ((code_begin, code_len_beg), (code_end, code_len_end))) =
        separated_pair(source_code, space0, source_code).parse(input)?;
    if code_len_beg != code_len_end {
        create_code_len_err(rest_of_input)
    } else {
        Ok((rest_of_input, (code_begin, code_end, code_len_beg)))
    }
}

fn create_code_len_err<'a, T, E: ParseError<ParserInput<'a>>>(input: ParserInput<'a>) -> Result<T, nom::Err<E>> {
    Err(nom::Err::Failure(nom::error::make_error(
        input,
        nom::error::ErrorKind::LengthValue,
    )))
}

fn source_code(input: ParserInput) -> NomResult<(SourceCode, CodeLen)> {
    let (rest_of_input, bytes) = delimited(tag(&b"<"[..]), many_m_n(1, 4, hex_char), tag(&b">"[..])).parse(input)?;
    let code_len = bytes.len();
    let source_code = bytes
        .into_iter()
        .rev()
        .zip(0..4)
        .map(|(byte, i)| 256u32.pow(i) * byte as u32)
        .sum();
    Ok((rest_of_input, (source_code, code_len as u8)))
}

fn hex_u16(input: ParserInput) -> NomResult<u16> {
    map(pair(hex_char, hex_char), |(h1, h2)| h1 as u16 * 256 + h2 as u16).parse(input)
}

fn bf_char_section(input: ParserInput) -> NomResult<CMapSection> {
    let begin_section = (digit1, space1, tag(&b"beginbfchar"[..]), multispace1);
    let end_section = (tag(&b"endbfchar"[..]), multispace1);
    let bf_char_line = delimited(space0, separated_pair(source_code, space0, target_string), multispace1);
    // Some real-world ToUnicode CMaps contain sections like `0 beginbfchar ... endbfchar`.
    // Accept empty sections to avoid failing extraction (specifically calling extract_text)
    let (rest_of_input, bf_char_mappings) = delimited(begin_section, many0(bf_char_line), end_section).parse(input)?;
    Ok((rest_of_input, CMapSection::BfChar(bf_char_mappings)))
}

fn target_string(input: ParserInput) -> NomResult<Vec<u16>> {
    // according to specification dstString can be up to 512 bytes
    // in ToUnicode cmap these should be 2-byte big endian Unicode values
    delimited(
        tag(&b"<"[..]),
        many_m_n(1, 256, terminated(hex_u16, multispace0)),
        tag(&b">"[..]),
    )
    .parse(input)
}

fn bf_range_section(input: ParserInput) -> NomResult<CMapSection> {
    let begin_section = (digit1, space1, tag(&b"beginbfrange"[..]), multispace1);
    let end_section = (tag(&b"endbfrange"[..]), multispace1);
    // Some real-world ToUnicode CMaps contain sections like `0 beginbfrange ... endbfrange`.
    // Accept empty sections to avoid failing extraction (specifically calling extract_text)
    let (rest_of_input, bf_range_mappings) =
        delimited(begin_section, many0(bf_range_line), end_section).parse(input)?;
    Ok((rest_of_input, CMapSection::BfRange(bf_range_mappings)))
}

fn bf_range_line(input: ParserInput) -> NomResult<SourceRangeMapping> {
    let bf_range_parser = separated_pair(
        code_range_pair,
        space0,
        alt((target_string.map(|res| vec![res]), range_target_array)),
    );
    delimited(space0, bf_range_parser, multispace1).parse(input)
}

fn range_target_array(input: ParserInput) -> NomResult<ArrayOfTargetStrings> {
    delimited(
        (tag(&b"["[..]), space0),
        separated_list1(space1, target_string),
        (space0, tag(&b"]"[..])),
    )
    .parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_span(s: &'_ [u8]) -> ParserInput<'_> {
        s
    }
    #[test]
    fn parse_1byte_source_code() {
        let data = b"<0A>";
        let (rem, res) = source_code(test_span(data)).unwrap();
        assert_eq!(rem, b"");
        assert_eq!(res, (0x0a, 1));
    }

    #[test]
    fn parse_source_code() {
        let data = b"<080F>";
        let (rem, res) = source_code(test_span(data)).unwrap();
        assert_eq!(rem, b"");
        assert_eq!(res, (0x080f, 2));
    }

    #[test]
    fn parse_invalid_source_code() {
        let data = b"<080g01>";
        assert!(source_code(test_span(data)).is_err())
    }

    #[test]
    fn parse_too_long_source_code() {
        let data = b"<080g01030a>";
        assert!(source_code(test_span(data)).is_err())
    }

    #[test]
    fn parse_code_range_pair() {
        let data = b"<080F> <08FF> ";
        let (rem, res) = code_range_pair(test_span(data)).unwrap();
        assert_eq!(rem, b" ");
        assert_eq!(res, (0x080f, 0x08ff, 2));
    }

    #[test]
    fn parse_code_range_pair_without_spaces() {
        let data = b"<080F><08FF>";

        let (rem, res) = code_range_pair(test_span(data)).unwrap();
        assert_eq!(rem, b"");
        assert_eq!(res, (0x080f, 0x08ff, 2));
    }

    #[test]
    fn parse_code_range_pair_with_not_matching_len() {
        let data = b"<080F> <08>";
        assert!(code_range_pair(test_span(data)).is_err())
    }

    #[test]
    fn parse_bfrange_line() {
        let data = b"<080f> <08ff> <09000110>\n";
        let (rem, res) = bf_range_line(test_span(data)).unwrap();
        assert_eq!(rem, b"");
        assert_eq!(res, ((0x080f, 0x08ff, 2), vec![vec![0x0900, 0x0110]]));
    }
    #[test]
    fn parse_bfrange_line_without_spaces() {
        let data = b"<080f><08ff><09000110>\n";
        let (rem, res) = bf_range_line(test_span(data)).unwrap();
        assert_eq!(rem, b"");
        assert_eq!(res, ((0x080f, 0x08ff, 2), vec![vec![0x0900, 0x0110]]));
    }

    #[test]
    fn parse_bfrange_line_array() {
        let data = b"<080f> <08ff> [ <09000110> <08fe> ] \n";
        let (rem, res) = bf_range_line(test_span(data)).unwrap();
        assert_eq!(rem, b"");
        assert_eq!(res, ((0x080f, 0x08ff, 2), vec![vec![0x0900, 0x0110], vec![0x08fe]]));
    }
    #[test]
    fn parse_invalid_bfrange_line() {
        let data = b"<080f> <08ff> [ <09000110> <08FF> <09fe80> ]\n";
        assert!(bf_range_line(test_span(data)).is_err())
    }

    #[test]
    fn parse_codespace_range_section() {
        let data = b"1 begincodespacerange\n\
            <0000> <FFFF> \n\
        endcodespacerange\n";
        let (rem, res) = codespace_range_section(test_span(data)).unwrap();
        assert_eq!(rem, b"");
        assert_eq!(res, CMapSection::CsRange(vec![(0x0000, 0xffff, 2)]));
    }

    #[test]
    fn parse_bf_range_section() {
        let data = b"3 beginbfrange \n\
            <0000> <000f> <0000>\n\
            <0010> <001f> <00000010> \n\
            <0020>  <002f> [<0000> <00000010> ]\n\
        endbfrange\n";

        let (rem, res) = bf_range_section(test_span(data)).unwrap();
        assert_eq!(rem, b"");
        assert_eq!(
            res,
            CMapSection::BfRange(vec![
                ((0x0000, 0x000f, 2), vec![vec![0x0000]]),
                ((0x0010, 0x001f, 2), vec![vec![0x0000, 0x0010]]),
                ((0x0020, 0x002f, 2), vec![vec![0x0000], vec![0x0000, 0x0010]]),
            ])
        );
    }

    #[test]
    fn parse_bf_char_section() {
        let data = b"4 beginbfchar \n\
            <1d> <0066 0069>\n\
            <1e> <00A0>\n\
            <1f> <0066 0066>
            <20> <0020>\n\
        endbfchar\n";
        let (rem, res) = bf_char_section(test_span(data)).unwrap();
        assert_eq!(rem, b"");
        assert_eq!(
            res,
            CMapSection::BfChar(vec![
                ((0x1d, 1), vec![0x0066, 0x0069]),
                ((0x1e, 1), vec![0x00a0]),
                ((0x1f, 1), vec![0x0066, 0x0066]),
                ((0x20, 1), vec![0x0020]),
            ])
        );
    }

    #[test]
    fn parse_cid_system_info() {
        let data = b"/CIDSystemInfo <<
/Registry (Adobe)
/Ordering (UCS)
/Supplement 0
>> def
";
        assert!(cid_system_info(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cid_system_info_dict_dup() {
        let data = b"/CIDSystemInfo 3 dict dup begin
  /Registry (callas) def
  /Ordering (MyriadPro-Regular14-UCMap) def
  /Supplement 0 def
end def
";
        assert!(cid_system_info(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cid_system_info_with_spaces() {
        let data = b"/CIDSystemInfo
<< /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n";
        assert!(cid_system_info(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cmap_name() {
        let data = b"/CMapName /Adobe-Identity-UCS def\n";
        assert!(cmap_name(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cmap_name2() {
        let data = b"/CMapName /Adobe-UCS-0 def\n";
        assert!(cmap_name(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cmap_type() {
        let data = b"/CMapType 2 def\n";
        assert!(cmap_type(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cmap_version() {
        let data = b"/CMapVersion 0 def\n";
        assert!(cmap_version(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cmap_version2() {
        let data = b"/CMapVersion 10.001 def\n";
        assert!(cmap_version(test_span(data)).is_ok())
    }

    #[test]
    fn parse_uidoffset() {
        let data = b"/UIDOffset 950 def\n";
        assert!(uidoffset(test_span(data)).is_ok())
    }

    #[test]
    fn parse_xuid() {
        let data = b"/XUID [1 10 25343] def\n";
        assert!(xuid(test_span(data)).is_ok())
    }

    #[test]
    fn parse_wmode() {
        let data = b"/WMode 0 def\n";
        assert!(wmode(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cmap_section_1() {
        let data = b"/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo <<
/Registry (Adobe)
/Ordering (UCS)
/Supplement 0
>> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
96 beginbfrange
<0000> <00FF> <0000>
<0100> <01FF> <0100>
<0200> <02FF> <0200>
<0300> <03FF> <0300>
<0400> <04FF> <0400>
<0500> <05FF> <0500>
<0600> <06FF> <0600>
<0700> <07FF> <0700>
<0800> <08FF> <0800>
<0900> <09FF> <0900>
<0A00> <0AFF> <0A00>
<0B00> <0BFF> <0B00>
<0C00> <0CFF> <0C00>
<0D00> <0DFF> <0D00>
<0E00> <0EFF> <0E00>
<0F00> <0FFF> <0F00>
<1000> <10FF> <1000>
<1100> <11FF> <1100>
<1200> <12FF> <1200>
<1300> <13FF> <1300>
<1400> <14FF> <1400>
<1500> <15FF> <1500>
<1600> <16FF> <1600>
<1700> <17FF> <1700>
<1800> <18FF> <1800>
<1900> <19FF> <1900>
<1A00> <1AFF> <1A00>
<1B00> <1BFF> <1B00>
<1C00> <1CFF> <1C00>
<1D00> <1DFF> <1D00>
<1E00> <1EFF> <1E00>
<1F00> <1FFF> <1F00>
<2000> <20FF> <2000>
<2100> <21FF> <2100>
<2200> <22FF> <2200>
<2300> <23FF> <2300>
<2400> <24FF> <2400>
<2500> <25FF> <2500>
<2600> <26FF> <2600>
<2700> <27FF> <2700>
<2800> <28FF> <2800>
<2900> <29FF> <2900>
<2A00> <2AFF> <2A00>
<2B00> <2BFF> <2B00>
<2C00> <2CFF> <2C00>
<2D00> <2DFF> <2D00>
<2E00> <2EFF> <2E00>
<2F00> <2FFF> <2F00>
<3000> <30FF> <3000>
<3100> <31FF> <3100>
<3200> <32FF> <3200>
<3300> <33FF> <3300>
<3400> <34FF> <3400>
<3500> <35FF> <3500>
<3600> <36FF> <3600>
<3700> <37FF> <3700>
<3800> <38FF> <3800>
<3900> <39FF> <3900>
<3A00> <3AFF> <3A00>
<3B00> <3BFF> <3B00>
<3C00> <3CFF> <3C00>
<3D00> <3DFF> <3D00>
<3E00> <3EFF> <3E00>
<3F00> <3FFF> <3F00>
<4000> <40FF> <4000>
<4100> <41FF> <4100>
<4200> <42FF> <4200>
<4300> <43FF> <4300>
<4400> <44FF> <4400>
<4500> <45FF> <4500>
<4600> <46FF> <4600>
<4700> <47FF> <4700>
<4800> <48FF> <4800>
<4900> <49FF> <4900>
<4A00> <4AFF> <4A00>
<4B00> <4BFF> <4B00>
<4C00> <4CFF> <4C00>
<4D00> <4DFF> <4D00>
<4E00> <4EFF> <4E00>
<4F00> <4FFF> <4F00>
<5000> <50FF> <5000>
<5100> <51FF> <5100>
<5200> <52FF> <5200>
<5300> <53FF> <5300>
<5400> <54FF> <5400>
<5500> <55FF> <5500>
<5600> <56FF> <5600>
<5700> <57FF> <5700>
<5800> <58FF> <5800>
<5900> <59FF> <5900>
<5A00> <5AFF> <5A00>
<5B00> <5BFF> <5B00>
<5C00> <5CFF> <5C00>
<5D00> <5DFF> <5D00>
<5E00> <5EFF> <5E00>
<5F00> <5FFF> <5F00>
endbfrange
96 beginbfrange
<6000> <60FF> <6000>
<6100> <61FF> <6100>
<6200> <62FF> <6200>
<6300> <63FF> <6300>
<6400> <64FF> <6400>
<6500> <65FF> <6500>
<6600> <66FF> <6600>
<6700> <67FF> <6700>
<6800> <68FF> <6800>
<6900> <69FF> <6900>
<6A00> <6AFF> <6A00>
<6B00> <6BFF> <6B00>
<6C00> <6CFF> <6C00>
<6D00> <6DFF> <6D00>
<6E00> <6EFF> <6E00>
<6F00> <6FFF> <6F00>
<7000> <70FF> <7000>
<7100> <71FF> <7100>
<7200> <72FF> <7200>
<7300> <73FF> <7300>
<7400> <74FF> <7400>
<7500> <75FF> <7500>
<7600> <76FF> <7600>
<7700> <77FF> <7700>
<7800> <78FF> <7800>
<7900> <79FF> <7900>
<7A00> <7AFF> <7A00>
<7B00> <7BFF> <7B00>
<7C00> <7CFF> <7C00>
<7D00> <7DFF> <7D00>
<7E00> <7EFF> <7E00>
<7F00> <7FFF> <7F00>
<8000> <80FF> <8000>
<8100> <81FF> <8100>
<8200> <82FF> <8200>
<8300> <83FF> <8300>
<8400> <84FF> <8400>
<8500> <85FF> <8500>
<8600> <86FF> <8600>
<8700> <87FF> <8700>
<8800> <88FF> <8800>
<8900> <89FF> <8900>
<8A00> <8AFF> <8A00>
<8B00> <8BFF> <8B00>
<8C00> <8CFF> <8C00>
<8D00> <8DFF> <8D00>
<8E00> <8EFF> <8E00>
<8F00> <8FFF> <8F00>
<9000> <90FF> <9000>
<9100> <91FF> <9100>
<9200> <92FF> <9200>
<9300> <93FF> <9300>
<9400> <94FF> <9400>
<9500> <95FF> <9500>
<9600> <96FF> <9600>
<9700> <97FF> <9700>
<9800> <98FF> <9800>
<9900> <99FF> <9900>
<9A00> <9AFF> <9A00>
<9B00> <9BFF> <9B00>
<9C00> <9CFF> <9C00>
<9D00> <9DFF> <9D00>
<9E00> <9EFF> <9E00>
<9F00> <9FFF> <9F00>
<A000> <A0FF> <A000>
<A100> <A1FF> <A100>
<A200> <A2FF> <A200>
<A300> <A3FF> <A300>
<A400> <A4FF> <A400>
<A500> <A5FF> <A500>
<A600> <A6FF> <A600>
<A700> <A7FF> <A700>
<A800> <A8FF> <A800>
<A900> <A9FF> <A900>
<AA00> <AAFF> <AA00>
<AB00> <ABFF> <AB00>
<AC00> <ACFF> <AC00>
<AD00> <ADFF> <AD00>
<AE00> <AEFF> <AE00>
<AF00> <AFFF> <AF00>
<B000> <B0FF> <B000>
<B100> <B1FF> <B100>
<B200> <B2FF> <B200>
<B300> <B3FF> <B300>
<B400> <B4FF> <B400>
<B500> <B5FF> <B500>
<B600> <B6FF> <B600>
<B700> <B7FF> <B700>
<B800> <B8FF> <B800>
<B900> <B9FF> <B900>
<BA00> <BAFF> <BA00>
<BB00> <BBFF> <BB00>
<BC00> <BCFF> <BC00>
<BD00> <BDFF> <BD00>
<BE00> <BEFF> <BE00>
<BF00> <BFFF> <BF00>
endbfrange
64 beginbfrange
<C000> <C0FF> <C000>
<C100> <C1FF> <C100>
<C200> <C2FF> <C200>
<C300> <C3FF> <C300>
<C400> <C4FF> <C400>
<C500> <C5FF> <C500>
<C600> <C6FF> <C600>
<C700> <C7FF> <C700>
<C800> <C8FF> <C800>
<C900> <C9FF> <C900>
<CA00> <CAFF> <CA00>
<CB00> <CBFF> <CB00>
<CC00> <CCFF> <CC00>
<CD00> <CDFF> <CD00>
<CE00> <CEFF> <CE00>
<CF00> <CFFF> <CF00>
<D000> <D0FF> <D000>
<D100> <D1FF> <D100>
<D200> <D2FF> <D200>
<D300> <D3FF> <D300>
<D400> <D4FF> <D400>
<D500> <D5FF> <D500>
<D600> <D6FF> <D600>
<D700> <D7FF> <D700>
<D800> <D8FF> <D800>
<D900> <D9FF> <D900>
<DA00> <DAFF> <DA00>
<DB00> <DBFF> <DB00>
<DC00> <DCFF> <DC00>
<DD00> <DDFF> <DD00>
<DE00> <DEFF> <DE00>
<DF00> <DFFF> <DF00>
<E000> <E0FF> <E000>
<E100> <E1FF> <E100>
<E200> <E2FF> <E200>
<E300> <E3FF> <E300>
<E400> <E4FF> <E400>
<E500> <E5FF> <E500>
<E600> <E6FF> <E600>
<E700> <E7FF> <E700>
<E800> <E8FF> <E800>
<E900> <E9FF> <E900>
<EA00> <EAFF> <EA00>
<EB00> <EBFF> <EB00>
<EC00> <ECFF> <EC00>
<ED00> <EDFF> <ED00>
<EE00> <EEFF> <EE00>
<EF00> <EFFF> <EF00>
<F000> <F0FF> <F000>
<F100> <F1FF> <F100>
<F200> <F2FF> <F200>
<F300> <F3FF> <F300>
<F400> <F4FF> <F400>
<F500> <F5FF> <F500>
<F600> <F6FF> <F600>
<F700> <F7FF> <F700>
<F800> <F8FF> <F800>
<F900> <F9FF> <F900>
<FA00> <FAFF> <FA00>
<FB00> <FBFF> <FB00>\r
<FC00> <FCFF> <FC00>
<FD00> <FDFF> <FD00>
<FE00> <FEFF> <FE00>
<FF00> <FFFF> <FF00>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end";
        assert!(cmap_stream(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cmap_section_2() {
        let data = b"/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CMapType 2 def
/CMapName/R27 def
1 begincodespacerange
<0000><ffff>
endcodespacerange
78 beginbfrange
<0020><0020><0020>
<0028><0028><0028>
<0029><0029><0029>
<002b><002b><002b>
<002c><002c><002c>
<002d><002d><002d>
<002e><002e><002e>
<002f><002f><002f>
<0030><0030><0030>
<0031><0031><0031>
<0032><0032><0032>
<0033><0033><0033>
<0034><0034><0034>
<0035><0035><0035>
<0036><0036><0036>
<0037><0037><0037>
<0038><0038><0038>
<0039><0039><0039>
<003a><003a><003a>
<003d><003d><003d>
<0041><0041><0041>
<0042><0042><0042>
<0043><0043><0043>
<0044><0044><0044>
<0045><0045><0045>
<0046><0046><0046>
<0047><0047><0047>
<0048><0048><0048>
<0049><0049><0049>
<004a><004a><004a>
<004b><004b><004b>
<004c><004c><004c>
<004d><004d><004d>
<004e><004e><004e>
<004f><004f><004f>
<0050><0050><0050>
<0052><0052><0052>
<0053><0053><0053>
<0054><0054><0054>
<0055><0055><0055>
<0056><0056><0056>
<0057><0057><0057>
<0058><0058><0058>
<005a><005a><005a>
<005c><005c><005c>
<0061><0061><0061>
<0062><0062><0062>
<0063><0063><0063>
<0064><0064><0064>
<0065><0065><0065>
<0066><0066><0066>
<0067><0067><0067>
<0068><0068><0068>
<0069><0069><0069>
<006a><006a><006a>
<006b><006b><006b>
<006c><006c><006c>
<006d><006d><006d>
<006e><006e><006e>
<006f><006f><006f>
<0070><0070><0070>
<0072><0072><0072>
<0073><0073><0073>
<0074><0074><0074>
<0075><0075><0075>
<0076><0076><0076>
<0077><0077><0077>
<0078><0078><0078>
<0079><0079><0079>
<007a><007a><007a>
<007c><007c><007c>
<00a3><00a3><00a3>
<00d6><00d6><00d6>
<00df><00df><00df>
<00e4><00e4><00e4>
<00f6><00f6><00f6>
<00fc><00fc><00fc>
<2019><2019><2019>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end end
";
        assert!(cmap_stream(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cmap_section_bfchar() {
        let data = b"/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo
<< /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-UCS-0 def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
36 beginbfchar
<0000> <0000>
<0001> <004C>
<0002> <0069>
<0003> <0073>
<0004> <0074>
<0005> <0061>
<0006> <006F>
<0007> <0070>
<0008> <0065>
<0009> <0072>
<000A> <0063>
<000B> <006A>
<000C> <007A>
<000D> <006B>
<000E> <0064>
<000F> <0032>
<0010> <0030>
<0011> <0033>
<0012> <002D>
<0013> <0035>
<0014> <0031>
<0015> <0039>
<0016> <0034>
<0017> <0057>
<0018> <006C>
<0019> <0075>
<001A> <0142>
<001B> <0079>
<001C> <0077>
<001D> <004F>
<001E> <0044>
<001F> <0052>
<0020> <0068>
<0021> <006E>
<0022> <004B>
<0023> <0067>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end";
        assert!(cmap_stream(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cmap_section_with_discontigous_range() {
        let data = b"/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo
<< /Registry (Adobe)
/Ordering (UCS)
/Supplement 0
>> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
2 beginbfrange
<0000> <005E> <0020>
<005F> <0061> [<D83DDE00> <D83DDD27> <D83DDD28>]
endbfrange
1 beginbfchar
<3A51> <D840DC3E>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end";
        assert!(cmap_stream(test_span(data)).is_ok())
    }

    #[test]
    fn parse_truetype_cmap_section_with_1byte_bfchars() {
        let data = b"/CIDInit/ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo<<
/Registry (Adobe)
/Ordering (UCS)
/Supplement 0
>> def
/CMapName/Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<00> <FF>
endcodespacerange
29 beginbfchar
<01> <0054>
<02> <0068>
<03> <0065>
<04> <0020>
<05> <0044>
<06> <0061>
<07> <006E>
<08> <0067>
<09> <0072>
<0A> <006F>
<0B> <0066>
<0C> <0045>
<0D> <0062>
<0E> <006B>
<0F> <0073>
<10> <004A>
<11> <0069>
<12> <0075>
<13> <0063>
<14> <003A>
<15> <0070>
<16> <0074>
<17> <002F>
<18> <0076>
<19> <0042>
<1A> <0079>
<1B> <002E>
<1C> <006D>
<1D> <006C>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end\n";
        assert!(cmap_stream(test_span(data)).is_ok())
    }

    #[test]
    fn parse_cmap_section_with_lowercase_pracset_and_nospace_target_string() {
        let data = b"/CIDInit /Procset findresource begin
12 dict begin
begincmap
/CMapType 2 def
1 begincodespacerange
<0000><ffff>
endcodespacerange
4 beginbfchar
<1D50><AC1C>
<1E29><ACF5>
<43ED><D2B9>
<46FC><D5C8>
endbfchar
1 beginbfrange
<067B><0692><0020>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end\n";
        assert!(cmap_stream(test_span(data)).is_ok())
    }
    #[test]
    fn parse_cmap_section_error() {
        let data = b"%!PS-Adobe-3.0 Resource-CMap
%%DocumentNeededResources: ProcSet (CIDInit)
%%IncludeResource: ProcSet (CIDInit)
%%BeginResource: CMap (MyriadPro-Regular14-UCMap)
%%Title: (MyriadPro-Regular14-UCMap callas MyriadPro-Regular14-UCMap 0)
%%EndComments

/CIDInit /ProcSet findresource begin

12 dict begin

begincmap

/CIDSystemInfo 3 dict dup begin
  /Registry (callas) def
  /Ordering (MyriadPro-Regular14-UCMap) def
  /Supplement 0 def
end def

/CMapName /MyriadPro-Regular14-UCMap def
/CMapType 2 def

1 begincodespacerange
<1e> <a9>
endcodespacerange
7 beginbfchar
<1e> <00A0>
<1f> <0066 0066>
<20> <0020>
<56> <0056>
<5f> <005F>
<96> <2013>
<a9> <00A9>
endbfchar
8 beginbfrange
<28> <29> <0028>
<2c> <3a> <002C>
<41> <4b> <0041>
<4d> <54> <004D>
<61> <69> <0061>
<6c> <70> <006C>
<72> <77> <0072>
<79> <7a> <0079>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end

%%EndResource
%%EOF
";
        let res = cmap_stream(test_span(data));
        println!("{:#?}", res);
        assert!(res.is_ok())
    }

    #[test]
    fn parse_cmap_byte_order_mark() {
        let data = b"\xEF\xBB\xBF/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe)/Ordering (UCS)/Supplement 0>> def
/CMapName /Adobe-Identity-UCS def /CMapType 2 def
1 begincodespacerange
<0003><0081>
endcodespacerange
23 beginbfrange
<0025><0025><0042>
<004F><004F><006C>
<0052><0052><006F>
<0046><0046><0063>
<004E><004E><006B>
<0048><0048><0065>
<0047><0047><0064>
<0003><0003><0020>
<0049><0049><0066>
<0055><0055><0072>
<0051><0051><006E>
<005A><005A><0077>
<0044><0044><0061>
<0053><0053><0070>
<004C><004C><0069>
<0057><0057><0074>
<0012><0012><002F>
<0029><0029><0046>
<0081><0081><00FC>
<0031><0031><004E>
<0058><0058><0075>
<004A><004A><0067>
<0056><0056><0073>
endbfrange
endcmap CMapName currentdict /CMap defineresource pop end end";
        assert!(cmap_stream(test_span(data)).is_ok())
    }
}
