#[cfg(not(feature = "async"))]
mod sync_tests {
    use lopdf::{Document, Error, LoadOptions, Object, ParseError};
    use std::assert_matches;
    use std::fs::File;

    #[test]
    fn load_options_default() {
        let opts = LoadOptions::default();
        assert!(opts.password.is_none());
        assert!(opts.filter.is_none());
        assert!(!opts.strict);
    }

    #[test]
    fn load_options_with_password() {
        let opts = LoadOptions::with_password("secret");
        assert_eq!(opts.password.as_deref(), Some("secret"));
        assert!(opts.filter.is_none());
        assert!(!opts.strict);
    }

    #[test]
    fn load_options_with_filter() {
        fn my_filter(id: (u32, u16), obj: &mut Object) -> Option<((u32, u16), Object)> {
            Some((id, obj.clone()))
        }
        let opts = LoadOptions::with_filter(my_filter);
        assert!(opts.password.is_none());
        assert!(opts.filter.is_some());
        assert!(!opts.strict);
    }

    #[test]
    fn load_options_debug_masks_password() {
        let opts = LoadOptions::with_password("supersecret");
        let debug = format!("{:?}", opts);
        assert!(
            !debug.contains("supersecret"),
            "password should be masked in Debug output"
        );
        assert!(debug.contains("***"), "masked password should show ***");
    }

    #[test]
    fn load_options_debug_no_password() {
        let opts = LoadOptions::default();
        let debug = format!("{:?}", opts);
        assert!(debug.contains("None"));
    }

    #[test]
    fn load_with_options_default_matches_load() {
        let doc1 = Document::load("assets/example.pdf").unwrap();
        let doc2 = Document::load_with_options("assets/example.pdf", LoadOptions::default()).unwrap();
        assert_eq!(doc1.version, doc2.version);
        assert_eq!(doc1.objects.len(), doc2.objects.len());
    }

    #[test]
    fn load_from_with_options_default_matches_load_from() {
        let doc1 = Document::load_from(File::open("assets/example.pdf").unwrap()).unwrap();
        let doc2 = Document::load_from_with_options(File::open("assets/example.pdf").unwrap(), LoadOptions::default())
            .unwrap();
        assert_eq!(doc1.version, doc2.version);
        assert_eq!(doc1.objects.len(), doc2.objects.len());
    }

    #[test]
    fn load_mem_with_options_default_matches_load_mem() {
        let buf = std::fs::read("assets/example.pdf").unwrap();
        let doc1 = Document::load_mem(&buf).unwrap();
        let doc2 = Document::load_mem_with_options(&buf, LoadOptions::default()).unwrap();
        assert_eq!(doc1.version, doc2.version);
        assert_eq!(doc1.objects.len(), doc2.objects.len());
    }

    #[test]
    fn load_with_options_filter_removes_objects() {
        // Filter that drops all stream objects
        fn drop_streams(id: (u32, u16), obj: &mut Object) -> Option<((u32, u16), Object)> {
            if obj.as_stream().is_ok() {
                None
            } else {
                Some((id, obj.clone()))
            }
        }

        let full = Document::load("assets/example.pdf").unwrap();
        let filtered =
            Document::load_with_options("assets/example.pdf", LoadOptions::with_filter(drop_streams)).unwrap();

        let full_streams = full.objects.values().filter(|o| o.as_stream().is_ok()).count();
        let filtered_streams = filtered.objects.values().filter(|o| o.as_stream().is_ok()).count();

        assert!(full_streams > 0, "example.pdf should have streams");
        assert_eq!(filtered_streams, 0, "filter should have removed all streams");
        assert!(
            filtered.objects.len() < full.objects.len(),
            "filtered doc should have fewer objects"
        );
    }

    #[test]
    fn load_mem_with_options_filter() {
        // Filter that only keeps dictionary objects
        fn only_dicts(id: (u32, u16), obj: &mut Object) -> Option<((u32, u16), Object)> {
            if obj.as_dict().is_ok() {
                Some((id, obj.clone()))
            } else {
                None
            }
        }

        let buf = std::fs::read("assets/example.pdf").unwrap();
        let filtered = Document::load_mem_with_options(&buf, LoadOptions::with_filter(only_dicts)).unwrap();

        // All remaining objects should be dictionaries
        for obj in filtered.objects.values() {
            assert!(obj.as_dict().is_ok(), "all objects should be dictionaries after filter");
        }
    }

    #[test]
    fn load_with_options_strict_false_loads_normally() {
        let doc = Document::load_with_options(
            "assets/example.pdf",
            LoadOptions {
                strict: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(doc.version, "1.5");
    }

    #[test]
    fn load_with_options_strict_true_loads_valid_pdf() {
        // strict=true should still load a valid, conforming PDF
        let doc = Document::load_with_options(
            "assets/example.pdf",
            LoadOptions {
                strict: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(doc.version, "1.5");
    }

    #[test]
    fn load_with_options_password_on_encrypted_pdf() {
        let doc = Document::load_with_options("assets/encrypted.pdf", LoadOptions::with_password("")).unwrap();
        assert!(!doc.is_encrypted());
        assert!(doc.encryption_state.is_some());
        let pages = doc.get_pages();
        assert_eq!(pages.len(), 1);
    }

    #[test]
    fn load_mem_with_options_password_on_encrypted_pdf() {
        let buf = std::fs::read("assets/encrypted.pdf").unwrap();
        let doc = Document::load_mem_with_options(&buf, LoadOptions::with_password("")).unwrap();
        assert!(!doc.is_encrypted());
        assert!(doc.encryption_state.is_some());
    }

    #[test]
    fn load_with_options_combined_password_and_strict() {
        let opts = LoadOptions {
            password: Some(String::new()),
            strict: true,
            ..Default::default()
        };
        let doc = Document::load_with_options("assets/encrypted.pdf", opts).unwrap();
        assert!(!doc.is_encrypted());
        assert!(doc.encryption_state.is_some());
    }

    #[test]
    fn load_with_options_nonexistent_file() {
        let result = Document::load_with_options("nonexistent.pdf", LoadOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn strict_rejects_binary_bytes_on_header_line() {
        // Minimal PDF-like buffer with binary bytes on the header line.
        // Strict mode should reject this with InvalidFileHeader.
        let buf = b"%PDF-1.3 \xb0\x9f\x92\x9c\r%%EOF\r";
        let result = Document::load_mem_with_options(
            buf,
            LoadOptions {
                strict: true,
                ..Default::default()
            },
        );
        assert_matches!(result.unwrap_err(), Error::Parse(ParseError::InvalidFileHeader));
    }

    #[test]
    fn lenient_accepts_binary_bytes_on_header_line() {
        // Same buffer, but lenient (default) mode should parse the header
        // successfully (it will fail later because the rest isn't a real PDF,
        // but the header itself should be accepted).
        let buf = b"%PDF-1.3 \xb0\x9f\x92\x9c\r%%EOF\r";
        let result = Document::load_mem_with_options(buf, LoadOptions::default());
        // The error should NOT be InvalidFileHeader — the header parsed fine.
        if let Err(e) = &result {
            assert!(
                !e.to_string().contains("invalid file header"),
                "lenient mode should accept binary bytes on header line, got: {e}"
            );
        }
    }
}
