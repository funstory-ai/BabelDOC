use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::path::Path;

use babeldoc_font_core::{
    FontFace, FontFamily, FontMapper, FontType, PrimaryFontFamily, SourceFontStyle,
};

const FONT_SIZES: [f64; 4] = [11.0, 12.5, 13.7, 18.75];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("scan-cmap") => {
            let face = FontFace::load(
                args.next().ok_or("missing font id")?,
                args.next().ok_or("missing font path")?,
            )?;
            let mut out = io::BufWriter::new(io::stdout().lock());
            for cp in 0..=0x10ffff {
                out.write_all(&face.has_glyph(cp).to_le_bytes())?;
            }
        }
        Some("scan-widths") => {
            let face = FontFace::load(
                args.next().ok_or("missing font id")?,
                args.next().ok_or("missing font path")?,
            )?;
            let mut input = String::new();
            io::stdin().read_to_string(&mut input)?;
            let mut out = io::BufWriter::new(io::stdout().lock());
            for line in input.lines() {
                let cp: u32 = line.parse()?;
                let ch = char::from_u32(cp).ok_or("invalid Unicode scalar for width scan")?;
                for size in FONT_SIZES {
                    let value = face.char_lengths(&ch.to_string(), size)[0];
                    out.write_all(&value.to_le_bytes())?;
                }
            }
        }
        Some("scan-width-text") => {
            let face = FontFace::load(
                args.next().ok_or("missing font id")?,
                args.next().ok_or("missing font path")?,
            )?;
            let sizes = args
                .map(|value| value.parse::<f64>())
                .collect::<Result<Vec<_>, _>>()?;
            if sizes.is_empty() {
                return Err("missing font size".into());
            }
            let mut text = String::new();
            io::stdin().read_to_string(&mut text)?;
            let mut out = io::BufWriter::new(io::stdout().lock());
            for size in sizes {
                for value in face.char_lengths(&text, size) {
                    out.write_all(&value.to_le_bytes())?;
                }
            }
        }
        Some("scan-outlines") => {
            let face = FontFace::load(
                args.next().ok_or("missing font id")?,
                args.next().ok_or("missing font path")?,
            )?;
            let mut out = io::BufWriter::new(io::stdout().lock());
            for has_outline in face.outline_flags()? {
                out.write_all(&[u8::from(has_outline)])?;
            }
        }
        Some("scan-traits") => {
            let face = FontFace::load(
                args.next().ok_or("missing font id")?,
                args.next().ok_or("missing font path")?,
            )?;
            let traits = face.traits();
            let flags = u8::from(traits.bold)
                | (u8::from(traits.italic) << 1)
                | (u8::from(traits.monospaced) << 2);
            io::stdout().write_all(&[flags])?;
            io::stdout().write_all(&face.glyph_count().to_le_bytes())?;
        }
        Some("scan-map") => scan_map(args.next().ok_or("missing family manifest")?.as_ref())?,
        _ => {
            return Err(
                "usage: font-core-probe scan-cmap|scan-widths|scan-width-text|scan-outlines|scan-traits|scan-map"
                    .into(),
            );
        }
    }
    Ok(())
}

fn scan_map(manifest_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = std::fs::read_to_string(manifest_path)?;
    let mut specs: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
    let mut paths = BTreeMap::new();
    for line in manifest.lines() {
        let mut fields = line.split('\t');
        let family = fields.next().ok_or("manifest family")?.to_owned();
        let kind = fields.next().ok_or("manifest kind")?.to_owned();
        let id = fields.next().ok_or("manifest id")?.to_owned();
        let path = fields.next().ok_or("manifest path")?.to_owned();
        specs
            .entry(family)
            .or_default()
            .entry(kind)
            .or_default()
            .push(id.clone());
        paths.insert(id, path);
    }
    let mut faces = BTreeMap::new();
    for (id, path) in paths {
        faces.insert(id.clone(), FontFace::load(id, path)?);
    }
    let mut mappers = Vec::new();
    for family_ix in 0..specs.len() {
        let family_name = family_name(family_ix);
        let groups = specs.get(family_name).ok_or("missing family in manifest")?;
        let load_group = |name: &str| -> Vec<_> {
            groups
                .get(name)
                .into_iter()
                .flatten()
                .map(|id| faces[id].clone())
                .collect()
        };
        let family = FontFamily {
            normal: load_group("normal"),
            script: load_group("script"),
            fallback: load_group("fallback"),
            base: load_group("base"),
        };
        for primary in [
            PrimaryFontFamily::None,
            PrimaryFontFamily::Serif,
            PrimaryFontFamily::SansSerif,
            PrimaryFontFamily::Script,
        ] {
            mappers.push(FontMapper::new(family.clone(), primary)?);
        }
    }
    let ids = faces
        .keys()
        .enumerate()
        .map(|(ix, id)| (id.clone(), ix as u16))
        .collect::<BTreeMap<_, _>>();
    let mut stdin = io::stdin().lock();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    let mut case = [0u8; 8];
    loop {
        let mut read = 0;
        while read < case.len() {
            match stdin.read(&mut case[read..])? {
                0 if read == 0 => return Ok(()),
                0 => return Err("truncated map case stream".into()),
                count => read += count,
            }
        }
        let family_ix = case[0] as usize;
        let mode = case[1];
        let primary_ix = case[2] as usize;
        let flags = case[3];
        let cp = u32::from_le_bytes(case[4..8].try_into()?);
        let ch = char::from_u32(cp).ok_or("invalid map sample scalar")?;
        if mode == 1 {
            let value = u16::from(mappers[family_ix * 4].has_char(&ch.to_string()));
            stdout.write_all(&value.to_le_bytes())?;
            continue;
        }
        let style = SourceFontStyle {
            bold: flags & 1 != 0,
            italic: flags & 2 != 0,
            monospaced: flags & 4 != 0,
            serif: flags & 8 != 0,
        };
        let mapper = &mappers[family_ix * 4 + primary_ix];
        let selected_font = if mode == 2 {
            let font_type = [
                FontType::Normal,
                FontType::Script,
                FontType::Fallback,
                FontType::Base,
            ][primary_ix];
            mapper.map_in_type(style, ch, font_type)
        } else {
            mapper.map(style, ch)
        };
        let selected = selected_font.map(|font| ids[font.id()]).unwrap_or(u16::MAX);
        stdout.write_all(&selected.to_le_bytes())?;
    }
}

fn family_name(index: usize) -> &'static str {
    ["CN", "EN", "HK", "JP", "KR", "TW"][index]
}
