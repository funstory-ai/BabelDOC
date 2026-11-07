//! b20d3984 read functions are the independent oracle, not Policy::disabled().
//! This vendored dependency is not a workspace member. Compile this test file
//! with rustc --test and the workspace's already-built lopdf/flate2/brotli rlibs.
use lopdf::{Stream, Dictionary, Error, DecompressError, Result, dictionary, resource::Policy};
macro_rules! warn { ($($arg:tt)*) => {{ let _ = format_args!($($arg)*); }}; }
struct B20;
impl B20 {
    fn read_capped<R: std::io::Read>(reader: R, output: &mut Vec<u8>, limit: Option<usize>) -> std::io::Result<()> {
        use std::io::Read;
        match limit {
            Some(max) => Read::take(reader, (max as u64).saturating_add(1))
                .read_to_end(output)
                .map(|_| ()),
            None => {
                let mut reader = reader;
                reader.read_to_end(output).map(|_| ())
            }
        }
    }

    fn decompress_brotli(input: &[u8], limit: Option<usize>) -> Result<Vec<u8>> {
        use brotli_decompressor::Decompressor;

        // Unlike the Flate path there is no fallback (e.g. raw deflate), so a
        // truncated or corrupt stream fails hard instead of yielding partial
        // output — intentional for a new filter, not an oversight.
        let initial_capacity = match limit {
            Some(max) => input.len().saturating_mul(2).min(max.saturating_add(1)),
            None => input.len().saturating_mul(2),
        };
        let mut output = Vec::with_capacity(initial_capacity);
        Self::read_capped(Decompressor::new(input, 4096), &mut output, limit)
            .map_err(|err| Error::InvalidStream(format!("Brotli decompression failed: {err}")))?;
        if let Some(max) = limit
            && output.len() > max
        {
            return Err(DecompressError::MemoryLimitExceeded { limit: max }.into());
        }
        // /DecodeParms predictors are deliberately NOT applied here. The PDF
        // Association's prototype files never pair /BrotliDecode with a
        // /Predictor (their Brotli xref streams store raw W entries), pdf.js
        // and pypdf ignore the parameters too; only MuPDF applies them.
        // Applying them would also corrupt [/BrotliDecode /FlateDecode]
        // chains, since the shared dict-form /DecodeParms is passed to every
        // filter and a predictor there belongs to the Flate layer.
        Ok(output)
    }

    fn decompress_zlib(input: &[u8], params: Option<&Dictionary>, limit: Option<usize>) -> Result<Vec<u8>> {
        use flate2::read::ZlibDecoder;

        // Reserve a starting capacity, but never pre-allocate beyond the limit so
        // a bomb cannot force a huge allocation up front.
        let initial_capacity = match limit {
            Some(max) => input.len().saturating_mul(2).min(max.saturating_add(1)),
            None => input.len().saturating_mul(2),
        };
        let mut output = Vec::with_capacity(initial_capacity);

        if !input.is_empty()
            && let Err(err) = Self::read_capped(ZlibDecoder::new(input), &mut output, limit)
        {
            warn!("{err}");
            // Zlib decompression failed (e.g. corrupt adler32 checksum in
            // encrypted PDFs). Retry with raw deflate, skipping the 2-byte
            // zlib header and ignoring the checksum.
            if output.is_empty() && input.len() > 2 {
                use flate2::read::DeflateDecoder;
                if let Err(raw_err) = Self::read_capped(DeflateDecoder::new(&input[2..]), &mut output, limit) {
                    warn!("raw deflate fallback also failed: {raw_err}");
                }
            }
        }
        if let Some(max) = limit
            && output.len() > max
        {
            return Err(DecompressError::MemoryLimitExceeded { limit: max }.into());
        }
        Self::decompress_predictor(output, params)
    }


    fn decompress_predictor(data: Vec<u8>, params: Option<&Dictionary>) -> Result<Vec<u8>> {
        assert!(params.is_none()); Ok(data)
    }
}

fn compressed(bytes:&[u8])->Vec<u8> {
    use std::io::Write;
    let mut out=flate2::write::ZlibEncoder::new(Vec::new(),flate2::Compression::fast());
    out.write_all(bytes).unwrap();out.finish().unwrap()
}
fn plain()->Vec<u8> {
    let mut seed=123456789u32;
    (0..200000).map(|i| {seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;if i%5<3 {seed as u8} else {0}}).collect()
}
#[test]
fn flate_checksum_and_middle_bit_preserve_b20_partial_output() {
    let bytes=plain();
    for middle in [false,true] {
        let mut input=compressed(&bytes);
        let index=if middle {input.len()/2} else {input.len()-1}; input[index]^=1;
        let expected=B20::decompress_zlib(&input,None,None).unwrap();
        assert!(expected.len()>8192);
        let stream=Stream::new(dictionary!{"Filter"=>"FlateDecode"},input);
        for p in [Policy::default(),Policy::disabled(),Policy::for_test(bytes.len(),16,16*1024*1024,16*1024*1024)] {
            let actual=stream.get_plain_content_with_policy(Some(&p),None).unwrap();
            assert!(actual==expected,"middle={middle}, got {} bytes, b20 {} bytes",actual.len(),expected.len());
        }
    }
}
#[test]
fn flate_raw_fallback_and_valid_stream_share_b20_read_semantics() {
    let plain=plain();
    let valid=compressed(&plain);
    let mut raw=valid.clone();raw[..2].copy_from_slice(b"XX");
    for input in [valid,raw] {
        let expected=B20::decompress_zlib(&input,None,None).unwrap();
        let stream=Stream::new(dictionary!{"Filter"=>"FlateDecode"},input);
        for p in [Policy::default(),Policy::disabled()] {
            let actual=stream.get_plain_content_with_policy(Some(&p),None).unwrap();
            assert!(actual==expected,"got {} bytes, b20 {} bytes",actual.len(),expected.len());
        }
    }
}
#[test]
fn flate_small_output_cap_reports_before_accepting_a_read() {
    let stream=Stream::new(dictionary!{"Filter"=>"FlateDecode"},compressed(b"abcdef"));
    for cap in [7,6] {
        let p=Policy::for_test(cap,16,16,1<<24);
        assert_eq!(stream.get_plain_content_with_policy(Some(&p),None).unwrap(),b"abcdef");
    }
    let p=Policy::for_test(5,16,16,1<<24);
    let error=stream.get_plain_content_with_policy(Some(&p),None).unwrap_err();
    let resource=error.resource_limit().unwrap();
    assert_eq!(resource.limit_name,"max_decoded_stream_bytes");
    assert_eq!(resource.observed_or_requested,6);
}
#[test]
fn brotli_extension_headers_keep_b20_ordinary_error() {
    for bits in 10..=30 {
        let input=[0x11,bits];
        let expected=B20::decompress_brotli(&input,None).unwrap_err().to_string();
        let stream=Stream::new(dictionary!{"Filter"=>"BrotliDecode"},input.to_vec());
        for p in [Policy::default(),Policy::disabled()] {
            let actual=stream.get_plain_content_with_policy(Some(&p),None).unwrap_err();
            assert!(!actual.is_resource_limit());
            assert_eq!(actual.to_string(),expected);
        }
    }
}

#[test]
fn large_flate_output_boundary_and_legacy_take_match_b20() {
    let bytes=plain();let encoded=compressed(&bytes);
    let stream=Stream::new(dictionary!{"Filter"=>"FlateDecode"},encoded.clone());
    for cap in [bytes.len()+1,bytes.len()] {
        let p=Policy::for_test(cap,16,16,1<<24);
        assert_eq!(stream.get_plain_content_with_policy(Some(&p),None).unwrap(),bytes);
    }
    let p=Policy::for_test(bytes.len()-1,16,16,1<<24);
    let error=stream.get_plain_content_with_policy(Some(&p),None).unwrap_err();
    assert_eq!(error.resource_limit().unwrap().observed_or_requested,bytes.len() as u64);
    let mut corrupt=encoded;let last=corrupt.len()-1;corrupt[last]^=1;
    for maximum in [8191,8192,16383,32767,bytes.len()] {
        let expected=B20::decompress_zlib(&corrupt,None,Some(maximum));
        let stream=Stream::new(dictionary!{"Filter"=>"FlateDecode"},corrupt.clone());
        for p in [Policy::default(),Policy::disabled()] {
            let actual=stream.get_plain_content_with_policy(Some(&p),Some(maximum));
            match (&actual,&expected) {
                (Ok(a),Ok(b))=>assert!(a==b,"legacy limit={maximum}, got {} bytes, b20 {} bytes",a.len(),b.len()),
                (Err(a),Err(b))=>assert_eq!(a.to_string(),b.to_string()),
                _=>panic!("legacy limit={maximum}: changed success/error result"),
            }
        }
    }
}

#[test]
fn corrupt_flate_cap_at_b20_retained_prefix_preserves_that_prefix() {
    let mut input=compressed(&plain());let last=input.len()-1;input[last]^=1;
    let expected=B20::decompress_zlib(&input,None,None).unwrap();
    assert!(expected.len()>8192);
    let stream=Stream::new(dictionary!{"Filter"=>"FlateDecode"},input);
    for cap in [expected.len()+1,expected.len()] {
        let p=Policy::for_test(cap,16,16,1<<24);
        let actual=stream.get_plain_content_with_policy(Some(&p),None).unwrap();
        assert!(actual==expected,"cap={cap}, b20 retained={}, actual={}",expected.len(),actual.len());
    }
    let p=Policy::for_test(expected.len()-1,16,16,1<<24);
    assert!(stream.get_plain_content_with_policy(Some(&p),None).unwrap_err().is_resource_limit());
}

mod b20_png {
use std::convert::TryInto;
use std::io::{Error, ErrorKind, Read, Result, Write};
use std::mem;
use lopdf::filters::png::decode_row;
pub fn decode_frame(content: &[u8], bpp: usize, bytes_per_row: usize) -> Result<Vec<u8>> {
    let mut previous = Vec::new();
    previous.try_reserve(bytes_per_row)?;
    previous.resize(bytes_per_row, 0_u8);
    let mut current = Vec::new();
    current.try_reserve(bytes_per_row)?;
    current.resize(bytes_per_row, 0_u8);
    let mut decoded = Vec::new();
    let mut pos = 0;
    while pos < content.len() {
        if let Ok(filter) = content[pos].try_into() {
            pos += 1;
            (&content[pos..]).read_exact(current.as_mut_slice())?;
            pos += bytes_per_row;

            decode_row(filter, bpp, previous.as_slice(), current.as_mut_slice());
            decoded.write_all(current.as_slice())?;
            mem::swap(&mut previous, &mut current);
        } else {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("invalid PNG filter type ({})", content[pos]),
            ));
        }
    }
    Ok(decoded)
}

}

#[test]
fn png_empty_and_malformed_input_match_b20_without_scratch_allocation_failure() {
    for width in [0, 1, 2, 32] {
        for bpp in [0, 1, 2, 8] {
            let mut inputs = vec![vec![], vec![0], vec![5]];
            for filter in 0..=4 {
                let mut row = vec![filter]; row.extend(vec![3; width]);
                inputs.push(row.clone());
                row.push(5); inputs.push(row);
            }
            for input in inputs {
                let expected = b20_png::decode_frame(&input, bpp, width);
                for p in [Policy::default(), Policy::disabled()] {
                    let actual = lopdf::filters::png::decode_frame_with_policy(&input, bpp, width, Some(&p));
                    match (&actual, &expected) {
                        (Ok(a), Ok(b)) => assert_eq!(a, b),
                        (Err(a), Err(b)) => { assert_eq!(a.kind(), b.kind()); assert_eq!(a.to_string(), b.to_string()); },
                        _ => panic!("PNG ordinary behavior changed: width={width}, bpp={bpp}, input={input:?}"),
                    }
                }
            }
        }
    }
}
