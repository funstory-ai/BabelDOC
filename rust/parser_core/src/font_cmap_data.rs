use std::io::{Cursor, Read};
use std::ops::Range;
use std::sync::OnceLock;

static FONT_ASSETS: LazyLock<FontAssetBundle> = LazyLock::new(|| {
    FontAssetBundle::decode(include_bytes!("../assets/font_runtime/font_tables.zst"))
});

struct FontAssetBundle {
    compressed: &'static [u8],
    fonts: HashMap<String, FontAsset>,
    cmaps: HashMap<String, FontAsset>,
}

struct FontAsset {
    frame: Range<usize>,
    json: OnceLock<Box<str>>,
    cmap: OnceLock<Arc<BuiltinCmap>>,
}

#[derive(Debug, Default)]
pub(crate) struct BuiltinCmap {
    pub(crate) vertical: Option<bool>,
    pub(crate) code2cid: Arc<BTreeMap<Vec<u8>, u32>>,
    pub(crate) unicode_h: Arc<BTreeMap<u32, String>>,
    pub(crate) unicode_v: Arc<BTreeMap<u32, String>>,
}

impl FontAssetBundle {
    fn decode(compressed: &'static [u8]) -> Self {
        let mut cursor = 0;
        let magic = take(compressed, &mut cursor, 8);
        assert_eq!(
            magic, b"BDFNTBL2",
            "invalid embedded font asset bundle magic"
        );
        let count = u32::from_be_bytes(
            take(compressed, &mut cursor, 4)
                .try_into()
                .expect("embedded font asset count must be four bytes"),
        ) as usize;
        let mut fonts = HashMap::with_capacity(count);
        let mut cmaps = HashMap::with_capacity(count);

        for _ in 0..count {
            let name_len = u16::from_be_bytes(
                take(compressed, &mut cursor, 2)
                    .try_into()
                    .expect("embedded font asset name length must be two bytes"),
            ) as usize;
            let frame_len = u64::from_be_bytes(
                take(compressed, &mut cursor, 8)
                    .try_into()
                    .expect("embedded font asset frame length must be eight bytes"),
            );
            let frame_len = usize::try_from(frame_len)
                .expect("embedded font asset frame is too large for this platform");
            let name = std::str::from_utf8(take(compressed, &mut cursor, name_len))
                .expect("embedded font asset name is not UTF-8");
            let frame_start = cursor;
            take(compressed, &mut cursor, frame_len);
            let asset = FontAsset {
                frame: frame_start..cursor,
                json: OnceLock::new(),
                cmap: OnceLock::new(),
            };

            if let Some(name) = name
                .strip_prefix("cmap/")
                .and_then(|name| name.strip_suffix(".json"))
            {
                assert!(!name.is_empty(), "embedded CMap name is empty");
                assert!(
                    cmaps.insert(name.to_owned(), asset).is_none(),
                    "duplicate embedded CMap name: {name}"
                );
            } else {
                assert!(
                    name.ends_with(".json") && !name.contains('/'),
                    "invalid embedded font asset path: {name}"
                );
                assert!(
                    fonts.insert(name.to_owned(), asset).is_none(),
                    "duplicate embedded font asset name: {name}"
                );
            }
        }

        assert_eq!(
            cursor,
            compressed.len(),
            "trailing bytes in embedded font asset bundle"
        );
        assert_eq!(fonts.len() + cmaps.len(), count);
        Self {
            compressed,
            fonts,
            cmaps,
        }
    }

    fn font_json(&self, name: &str) -> Option<&str> {
        let asset = self.fonts.get(name)?;
        Some(asset.json(&self.compressed))
    }

    fn cmap_json(&self, name: &str) -> Option<&str> {
        let asset = self.cmaps.get(name)?;
        Some(asset.json(&self.compressed))
    }

    fn cmap(&self, name: &str) -> Option<Arc<BuiltinCmap>> {
        let asset = self.cmaps.get(name)?;
        Some(Arc::clone(asset.cmap.get_or_init(|| {
            let json = asset.json(&self.compressed);
            let value = serde_json::from_str::<serde_json::Value>(json)
                .expect("embedded CMap table is not valid JSON");
            Arc::new(builtin_cmap_from_json(&value))
        })))
    }
}

fn builtin_cmap_from_json(value: &serde_json::Value) -> BuiltinCmap {
    let mut code2cid = BTreeMap::new();
    if let Some(root) = value.get("CODE2CID") {
        flatten_cmap(root, &mut Vec::new(), &mut code2cid);
    }
    let unicode = |key: &str| {
        Arc::new(
            value
                .get(key)
                .and_then(|value| value.as_object())
                .map(|map| {
                    map.iter()
                        .filter_map(|(code, text)| {
                            Some((code.parse::<u32>().ok()?, text.as_str()?.to_owned()))
                        })
                        .collect()
                })
                .unwrap_or_default(),
        )
    };
    BuiltinCmap {
        vertical: value.get("IS_VERTICAL").and_then(|value| value.as_bool()),
        code2cid: Arc::new(code2cid),
        unicode_h: unicode("CID2UNICHR_H"),
        unicode_v: unicode("CID2UNICHR_V"),
    }
}

fn flatten_cmap(
    value: &serde_json::Value,
    prefix: &mut Vec<u8>,
    output: &mut BTreeMap<Vec<u8>, u32>,
) {
    let Some(map) = value.as_object() else {
        return;
    };
    for (key, value) in map {
        let Ok(byte) = key.parse::<u8>() else {
            continue;
        };
        prefix.push(byte);
        if let Some(cid) = value.as_u64() {
            output.insert(prefix.clone(), cid as u32);
        } else {
            flatten_cmap(value, prefix, output);
        }
        prefix.pop();
    }
}

impl FontAsset {
    fn json(&self, compressed: &'static [u8]) -> &str {
        self.json
            .get_or_init(|| {
                let frame = &compressed[self.frame.clone()];
                let mut decoder = ruzstd::decoding::StreamingDecoder::new(Cursor::new(frame))
                    .unwrap_or_else(|error| {
                        panic!("failed to initialize embedded font asset frame: {error}")
                    });
                let content_size = usize::try_from(decoder.decoder.content_size())
                    .expect("embedded font asset is too large for this platform");
                assert_ne!(
                    content_size, 0,
                    "embedded font asset frame is missing its content size"
                );
                let mut bytes = Vec::with_capacity(content_size);
                decoder.read_to_end(&mut bytes).unwrap_or_else(|error| {
                    panic!("failed to decompress embedded font asset frame: {error}")
                });
                assert_eq!(
                    bytes.len(),
                    content_size,
                    "embedded font asset frame content size mismatch"
                );
                String::from_utf8(bytes)
                    .unwrap_or_else(|error| panic!("embedded font table is not UTF-8: {error}"))
                    .into_boxed_str()
            })
            .as_ref()
    }
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> &'a [u8] {
    let end = (*cursor)
        .checked_add(len)
        .expect("embedded font asset offset overflow");
    let slice = bytes
        .get(*cursor..end)
        .expect("truncated embedded font asset bundle");
    *cursor = end;
    slice
}

fn font_asset_json(name: &str) -> &'static str {
    FONT_ASSETS
        .font_json(name)
        .unwrap_or_else(|| panic!("missing embedded font asset: {name}"))
}

/// Return the converted bundled mapping JSON for a canonical CMap name.
pub fn cmap_json(name: &str) -> Option<&'static str> {
    FONT_ASSETS.cmap_json(name)
}

pub(crate) fn builtin_cmap(name: &str) -> Option<Arc<BuiltinCmap>> {
    FONT_ASSETS.cmap(name)
}
