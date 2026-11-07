//! Exact b20d3984 recovery methods; only the enclosing test type is new.
use super::*;
struct B20;
impl B20 {
    fn scan_object_markers(buffer: &[u8]) -> Vec<(u32, ObjectId)> {
        const STREAM_KEYWORD: &[u8] = b"stream";
        const END_STREAM_KEYWORD: &[u8] = b"endstream";

        let mut markers = Vec::new();
        let mut oversized_number_warned = false;
        let mut at_line_start = true;
        let mut pos = 0;
        while pos < buffer.len() {
            // Skip raw stream data: an uncompressed payload may embed
            // convincing `N G obj` lines whose later offsets would otherwise
            // override the genuine entries for those object numbers.
            if buffer[pos..].starts_with(STREAM_KEYWORD)
                && !buffer[..pos].ends_with(b"end")
                && matches!(buffer.get(pos + STREAM_KEYWORD.len()), Some(b'\r' | b'\n'))
            {
                let after_keyword = pos + STREAM_KEYWORD.len();
                if let Some(relative) = buffer[after_keyword..]
                    .windows(END_STREAM_KEYWORD.len())
                    .position(|window| window == END_STREAM_KEYWORD)
                {
                    pos = after_keyword + relative + END_STREAM_KEYWORD.len();
                    at_line_start = false;
                    continue;
                }
                // Damaged stream without terminator: fall back to the
                // dictionary's /Length hint so the payload cannot hide
                // line-start pseudo headers, while objects written after it
                // stay reachable.
                if let Some(resume) = Self::payload_end_by_length(buffer, pos) {
                    pos = resume;
                    at_line_start = false;
                    continue;
                }
                // Unusable /Length too: nothing bounds the payload, so keep
                // scanning byte by byte rather than dropping what follows.
            }
            // Anchor at line starts so pseudo headers buried after another
            // token (string literal, comment) are never mistaken for markers.
            if at_line_start
                && buffer[pos].is_ascii_digit()
                && let Some(id) = Self::parse_object_header(&buffer[pos..])
            {
                // A huge bogus number would inflate `size` (and thus
                // `max_id`) via `Xref::insert`; genuine numbering stays
                // within the same cap as the marker count.
                if id.0 > MAX_RECONSTRUCTED_OBJECTS as u32 {
                    if !oversized_number_warned {
                        warn!("ignoring object headers numbered above {MAX_RECONSTRUCTED_OBJECTS}");
                        oversized_number_warned = true;
                    }
                } else {
                    if markers.len() == MAX_RECONSTRUCTED_OBJECTS {
                        warn!(
                            "object marker scan stopped at the {MAX_RECONSTRUCTED_OBJECTS}-marker cap; reconstruction may be incomplete"
                        );
                        break;
                    }
                    markers.push((pos as u32, id));
                }
            }
            match buffer[pos] {
                b'\r' | b'\n' => at_line_start = true,
                b' ' | b'\t' => {}
                _ => at_line_start = false,
            }
            pos += 1;
        }
        markers
    }

    /// Resume offset past a stream payload according to a *direct* `/Length`
    /// integer in the dictionary preceding the `stream` keyword at
    /// `stream_pos`. The search is scoped to the current object (after the
    /// nearest preceding `obj` keyword) so a missing `/Length` cannot latch
    /// onto a previous object's value. Returns `None` when the hint is
    /// absent, indirect, or points outside the buffer — damaged files tend
    /// to carry wrong `/Length` values, so it must stay a hint, never a hard
    /// boundary.
    fn payload_end_by_length(buffer: &[u8], stream_pos: usize) -> Option<usize> {
        const LENGTH_KEY: &[u8] = b"/Length";
        const OBJ_KEYWORD: &[u8] = b"obj";
        const STREAM_KEYWORD_LEN: usize = b"stream".len();

        let head = &buffer[..stream_pos];
        let obj_pos = head
            .windows(OBJ_KEYWORD.len())
            .rposition(|window| window == OBJ_KEYWORD)?;
        let dict_region = &buffer[obj_pos + OBJ_KEYWORD.len()..stream_pos];
        // Nearest `/Length` in this object's dictionary; parsing it validates
        // that the match really is a key with an integer value.
        let key_pos = dict_region
            .windows(LENGTH_KEY.len())
            .rposition(|window| window == LENGTH_KEY)?;
        let rest = &dict_region[key_pos + LENGTH_KEY.len()..];
        let digits_start = rest.iter().position(u8::is_ascii_digit)?;
        let digits_len = rest[digits_start..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        if digits_len > 10 {
            return None;
        }
        // An indirect reference (`/Length 5 0 R`) continues with another
        // number token; a direct integer ends at a name, `>>`, or the keyword.
        match rest[digits_start + digits_len..]
            .iter()
            .copied()
            .find(|byte| !byte.is_ascii_whitespace())
        {
            Some(b'/') | Some(b'>') => {}
            _ => return None,
        }
        let length: usize = std::str::from_utf8(&rest[digits_start..digits_start + digits_len])
            .ok()?
            .parse()
            .ok()?;
        // Spec: the EOL after the `stream` keyword counts as CRLF or LF.
        let eol_len = match (
            buffer.get(stream_pos + STREAM_KEYWORD_LEN),
            buffer.get(stream_pos + STREAM_KEYWORD_LEN + 1),
        ) {
            (Some(b'\r'), Some(b'\n')) => 2,
            _ => 1,
        };
        let end = (stream_pos + STREAM_KEYWORD_LEN + eol_len).checked_add(length)?;
        (buffer.len() >= end).then_some(end)
    }

    /// Parse an `N G obj` header; the byte after `obj` must end the token.
    fn parse_object_header(input: &[u8]) -> Option<ObjectId> {
        // deliberate: caps at u32/u16 width bound work on pathological padding.
        fn digits(input: &[u8], cap: usize) -> Option<(&[u8], &[u8])> {
            let n = input.iter().take_while(|b| b.is_ascii_digit()).count();
            (0 < n && n <= cap).then(|| input.split_at(n))
        }
        fn spaces(input: &[u8]) -> Option<&[u8]> {
            let n = input
                .iter()
                .take_while(|&&b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
                .count();
            (n > 0).then(|| &input[n..])
        }

        let (number, rest) = digits(input, 10)?;
        let number: u32 = std::str::from_utf8(number).ok()?.parse().ok()?;
        let rest = spaces(rest)?;
        let (generation, rest) = digits(rest, 5)?;
        let generation: u16 = std::str::from_utf8(generation).ok()?.parse().ok()?;
        let rest = spaces(rest)?;
        let tail = rest.strip_prefix(b"obj")?;
        match tail.first() {
            None => {}
            Some(&byte) if !byte.is_ascii_alphanumeric() => {}
            _ => return None,
        }
        Some((number, generation))
    }

}

#[test]
fn recovery_literal_index_matches_b20_markers_and_length() {
    let mut cases: Vec<Vec<u8>> = vec![
        b"1 0 obj
<< /Length 3 >>
stream
abc
2 0 obj
".to_vec(),
        b"1 0 obj
<< /Length 5 0 R >>
stream
2 0 obj
".to_vec(),
        b"1 0 obj
<< /Length foo123 /Other 9 >>
stream\r\n2 0 obj
".to_vec(),
        b"1 0 obj
stream
7 0 obj
endstream
2 0 obj
".to_vec(),
        b"1 0 obj
stream
stream
stream
".to_vec(),
    ];
    let chunks: &[&[u8]] = &[b"obj", b"/Length", b"stream\n", b"endstream", b" 0 ", b"12", b">>", b"\r\n", b"x", b" R ", b"/", b" "];
    let mut seed = 728163u64;
    for _ in 0..500 {
        let mut bytes = Vec::new();
        for _ in 0..40 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            bytes.extend_from_slice(chunks[(seed as usize >> 16) % chunks.len()]);
        }
        cases.push(bytes);
    }
    for bytes in cases {
        assert_eq!(Reader::scan_object_markers(&bytes), B20::scan_object_markers(&bytes), "{bytes:?}");
        let index = RecoveryIndex::new(&bytes);
        for pos in 0..bytes.len() {
            assert_eq!(index.payload_end(&bytes, pos), B20::payload_end_by_length(&bytes, pos), "pos={pos}, {bytes:?}");
        }
    }
}

#[test]
fn recovery_index_scan_work_is_linear_with_shared_bad_suffix() {
    for n in [16,32,64,128] {
        let mut bytes=b"1 0 obj\n".to_vec();
        for _ in 0..n { bytes.extend_from_slice(b"/Length invalid stream\n"); }
        bytes.extend(std::iter::repeat_n(b' ',4096));
        bytes.extend_from_slice(b"12345678901234567890 >>");
        let mut reads=0usize;
        let index=RecoveryIndex::new_observed(&bytes,|n|reads+=n);
        assert!(reads <= bytes.len()*6,"reads={reads}, bytes={}",bytes.len());
        assert_eq!(index.lengths.len(),n);
        assert_eq!(Reader::scan_object_markers(&bytes),B20::scan_object_markers(&bytes));
    }
}

fn test_reader(buffer: &[u8], document: Document) -> Reader<'_> {
    Reader {
        parser_policy: None,
        real_parser: None,
        buffer,
        shared_buffer: None,
        document,
        encryption_state: None,
        raw_objects: BTreeMap::new(),
        password: None,
        strict: false,
        max_decompressed_size: None,
        normal_offsets: Vec::new(),
        loaded_object_stream_containers: HashSet::new(),
    }
}

#[test]
fn encrypted_segment_classification_is_bounded_to_sixty_four_failures() {
    let buffer = b"x";
    let mut reader = test_reader(buffer, Document::new());
    let mut used = 0usize;
    for _ in 0..65 {
        reader.record_encrypted_segment_failure(0, &mut used);
    }
    assert_eq!(reader.document.parser_diagnostics.invalid_encrypted_objects, 65);
    assert_eq!(
        reader
            .document
            .parser_diagnostics
            .invalid_encrypted_segment_both_fail,
        64
    );
    assert_eq!(used, 64);
}

#[test]
fn lazy_object_stream_container_is_parsed_once_for_missing_members() {
    let buffer = b"";
    let mut document = Document::new();
    let stream = crate::Stream::new(
        dictionary! {"Type"=>"ObjStm","N"=>1,"First"=>4},
        b"3 0 [".to_vec(),
    );
    document
        .objects
        .insert((2, 0), Object::Stream(Box::new(stream)));
    let mut xref = Xref::new(5, XrefType::CrossReferenceTable);
    xref.insert(2, XrefEntry::Normal { offset: 0, generation: 0 });
    xref.insert(3, XrefEntry::Compressed { container: 2, index: 0 });
    xref.insert(4, XrefEntry::Compressed { container: 2, index: 0 });
    document.reference_table = xref;
    document.max_id = 4;
    let mut reader = test_reader(buffer, document);
    reader.load_object((3, 0)).unwrap();
    reader.load_object((4, 0)).unwrap();
    assert_eq!(
        reader.document.parser_diagnostics.invalid_objstm_members,
        1
    );
}
