/*
ToUnicode CMaps are special CMaps and thus can be parsed simpler. Assumptions:
- /CMapType is always 2
- only bfchar and bfrange sections allowed
- no glyph names as target allowed, only hex strings
- target encoded in UTF16-BE
 */

pub(crate) type ArrayOfTargetStrings = Vec<Vec<u16>>;

// According to pdf documentation source codes can be of various byte length but they
// should be smaller than integer
pub(crate) type SourceCode = u32;
pub(crate) type CodeLen = u8;
pub(crate) type SourceRangeMapping = ((SourceCode, SourceCode, CodeLen), ArrayOfTargetStrings);
pub(crate) type SourceCharMapping = ((SourceCode, CodeLen), Vec<u16>);
#[derive(Debug, PartialEq)]
pub enum CMapSection {
    CsRange(Vec<(SourceCode, SourceCode, CodeLen)>),
    BfChar(Vec<SourceCharMapping>),
    BfRange(Vec<SourceRangeMapping>),
}

#[derive(Debug)]
pub enum CMapParseError {
    Incomplete,
    Error,
}
