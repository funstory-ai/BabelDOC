# lopdf

[![Crates.io](https://img.shields.io/crates/v/lopdf.svg)](https://crates.io/crates/lopdf)
[![CI](https://github.com/J-F-Liu/lopdf/actions/workflows/ci.yml/badge.svg)](https://github.com/J-F-Liu/lopdf/actions/workflows/ci.yml)
[![Docs]( https://docs.rs/lopdf/badge.svg)](https://docs.rs/lopdf)

A Rust library for PDF document manipulation.

A useful reference for understanding the PDF file format and the
eventual usage of this library is the
[PDF 1.7 Reference Document](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/PDF32000_2008.pdf).
The PDF 2.0 specification is available [here](https://www.pdfa.org/announcing-no-cost-access-to-iso-32000-2-pdf-2-0/).

## Requirements

- **Rust 1.85 or later** - Required for Rust 2024 edition features and object streams support
- To check your Rust version: `rustc --version`
- To update Rust: `rustup update`

## Cargo features

| Feature | Default | What it adds |
| --- | :---: | --- |
| `chrono-clock` | ✅ | `chrono` plus its `clock` feature: conversions to and from `DateTime<Local>`, the reading machine's own zone. Brings `iana-time-zone` and its per-platform chain. |
| `rayon` | ✅ | Parallel object-stream and cross-reference parsing. |
| `chrono` | | Conversions to and from `DateTime<FixedOffset>` and `DateTime<Utc>`, which is all a PDF date can express. Costs `chrono` and `num-traits`, nothing else. |
| `jiff` | | Conversions to and from `jiff::Zoned` and `jiff::Timestamp`. Resolves named zones, so it needs a timezone database — bundled into the binary on Windows and on any wasm target. |
| `time` | | Conversions to and from `time::OffsetDateTime` and `time::PrimitiveDateTime`. |
| `serde` | | `Serialize`/`Deserialize` for the object model. |
| `async` | | Tokio-based asynchronous document loading. |
| `embed_image` | | Embedding raster images, via the `image` crate. |
| `font_embedding` | | Embedding TrueType fonts, via `skrifa`. |
| `wasm_js` | | Selects `getrandom`'s `wasm_js` backend, needed for encryption on wasm. |

The date backends are alternatives, not layers: each supplies conversions for the same [`DateTime`] value, so enabling more than one only adds dependencies. Enabling none is supported too — `Object::as_datetime` needs no backend, and `DateTime::as_str` returns the raw date for a caller that would rather parse it itself.

A PDF date states a fixed offset from UT and never a named zone (ISO 32000-1, 7.9.4), so `chrono` without `clock` is enough to read one faithfully. `chrono-clock` is the default only because it is what earlier versions gave you.

## Example Code

* Create PDF document

```rust
use lopdf::dictionary;
use lopdf::{Document, Object, Stream};
use lopdf::content::{Content, Operation};

// `with_version` specifes the PDF version this document complies with.
let mut doc = Document::with_version("1.5");
// Object IDs are used for cross referencing in PDF documents.
// `lopdf` helps keep track of them for us. They are simple integers.
// Calls to `doc.new_object_id` and `doc.add_object` return an object ID.

// "Pages" is the root node of the page tree.
let pages_id = doc.new_object_id();

// Fonts are dictionaries. The "Type", "Subtype" and "BaseFont" tags
// are straight out of the PDF spec.
//
// The dictionary macro is a helper that allows complex
// key-value relationships to be represented in a simpler
// visual manner, similar to a match statement.
// A dictionary is implemented as an IndexMap of Vec<u8>, and Object
let font_id = doc.add_object(dictionary! {
    // type of dictionary
    "Type" => "Font",
    // type of font, type1 is simple postscript font
    "Subtype" => "Type1",
    // basefont is postscript name of font for type1 font.
    // See PDF reference document for more details
    "BaseFont" => "Courier",
});

// Font dictionaries need to be added into resource
// dictionaries in order to be used.
// Resource dictionaries can contain more than just fonts,
// but normally just contains fonts.
// Only one resource dictionary is allowed per page tree root.
let resources_id = doc.add_object(dictionary! {
    // Fonts are actually triplely nested dictionaries. Fun!
    "Font" => dictionary! {
        // F1 is the font name used when writing text.
        // It must be unique in the document. It does not
        // have to be F1
        "F1" => font_id,
    },
});

// `Content` is a wrapper struct around an operations struct that contains
// a vector of operations. The operations struct contains a vector of
// that match up with a particular PDF operator and operands.
// Refer to the PDF spec for more details on the operators and operands
// Note, the operators and operands are specified in a reverse order
// from how they actually appear in the PDF file itself.
let content = Content {
    operations: vec![
        // BT begins a text element. It takes no operands.
        Operation::new("BT", vec![]),
        // Tf specifies the font and font size.
        // Font scaling is complicated in PDFs.
        // Refer to the spec for more info.
        // The `into()` methods convert the types into
        // an enum that represents the basic object types in PDF documents.
        Operation::new("Tf", vec!["F1".into(), 48.into()]),
        // Td adjusts the translation components of the text matrix.
        // When used for the first time after BT, it sets the initial
        // text position on the page.
        // Note: PDF documents have Y=0 at the bottom. Thus 600 to print text near the top.
        Operation::new("Td", vec![100.into(), 600.into()]),
        // Tj prints a string literal to the page. By default, this is black text that is
        // filled in. There are other operators that can produce various textual effects and
        // colors
        Operation::new("Tj", vec![Object::string_literal("Hello World!")]),
        // ET ends the text element.
        Operation::new("ET", vec![]),
    ],
};

// Streams are a dictionary followed by a (possibly encoded) sequence of bytes.
// What that sequence of bytes represents, depends on the context.
// The stream dictionary is set internally by lopdf and normally doesn't
// need to be manually manipulated. It contains keys such as
// Length, Filter, DecodeParams, etc.
let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));

// Page is a dictionary that represents one page of a PDF file.
// Its required fields are "Type", "Parent" and "Contents".
let page_id = doc.add_object(dictionary! {
    "Type" => "Page",
    "Parent" => pages_id,
    "Contents" => content_id,
});

// Again, "Pages" is the root of the page tree. The ID was already created
// at the top of the page, since we needed it to assign to the parent element
// of the page dictionary.
//
// These are just the basic requirements for a page tree root object.
// There are also many additional entries that can be added to the dictionary,
// if needed. Some of these can also be defined on the page dictionary itself,
// and not inherited from the page tree root.
let pages = dictionary! {
    // Type of dictionary
    "Type" => "Pages",
    // Vector of page IDs in document. Normally would contain more than one ID
    // and be produced using a loop of some kind.
    "Kids" => vec![page_id.into()],
    // Page count
    "Count" => 1,
    // ID of resources dictionary, defined earlier
    "Resources" => resources_id,
    // A rectangle that defines the boundaries of the physical or digital media.
    // This is the "page size".
    "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
};

// Using `insert()` here, instead of `add_object()` since the ID is already known.
doc.objects.insert(pages_id, Object::Dictionary(pages));

// Creating document catalog.
// There are many more entries allowed in the catalog dictionary.
let catalog_id = doc.add_object(dictionary! {
    "Type" => "Catalog",
    "Pages" => pages_id,
});

// The "Root" key in trailer is set to the ID of the document catalog,
// the remainder of the trailer is set during `doc.save()`.
doc.trailer.set("Root", catalog_id);
doc.compress();

// Store file in current working directory.
// Note: Line is excluded when running tests
if false {
    // Traditional save
    doc.save("example.pdf").unwrap();
    
    // Or save with object streams for smaller file size
    let mut file = std::fs::File::create("example_compressed.pdf").unwrap();
    doc.save_modern(&mut file).unwrap();
}
```

* Merge PDF documents

```rust
use lopdf::dictionary;

use std::collections::BTreeMap;

use lopdf::content::{Content, Operation};
use lopdf::{Document, Object, ObjectId, Stream, Bookmark};

pub fn generate_fake_document() -> Document {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Courier",
    });
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => font_id,
        },
    });
    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 48.into()]),
            Operation::new("Td", vec![100.into(), 600.into()]),
            Operation::new("Tj", vec![Object::string_literal("Hello World!")]),
            Operation::new("ET", vec![]),
        ],
    };
    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
    });
    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => vec![page_id.into()],
        "Count" => 1,
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    doc
}

fn main() -> std::io::Result<()> {
    // Generate a stack of Documents to merge.
    let documents = vec![
        generate_fake_document(),
        generate_fake_document(),
        generate_fake_document(),
        generate_fake_document(),
    ];

    // Define a starting `max_id` (will be used as start index for object_ids).
    let mut max_id = 1;
    let mut pagenum = 1;
    // Collect all Documents Objects grouped by a map
    let mut documents_pages = BTreeMap::new();
    let mut documents_objects = BTreeMap::new();
    let mut document = Document::with_version("1.5");

    for mut doc in documents {
        let mut first = false;
        doc.renumber_objects_with(max_id);

        max_id = doc.max_id + 1;

        documents_pages.extend(
            doc
                    .get_pages()
                    .into_iter()
                    .map(|(_, object_id)| {
                        if !first {
                            let bookmark = Bookmark::new(String::from(format!("Page_{}", pagenum)), [0.0, 0.0, 1.0], 0, object_id);
                            document.add_bookmark(bookmark, None);
                            first = true;
                            pagenum += 1;
                        }

                        (
                            object_id,
                            doc.get_object(object_id).unwrap().to_owned(),
                        )
                    })
                    .collect::<BTreeMap<ObjectId, Object>>(),
        );
        documents_objects.extend(doc.objects);
    }

    // "Catalog" and "Pages" are mandatory.
    let mut catalog_object: Option<(ObjectId, Object)> = None;
    let mut pages_object: Option<(ObjectId, Object)> = None;

    // Process all objects except "Page" type
    for (object_id, object) in documents_objects.iter() {
        // We have to ignore "Page" (as are processed later), "Outlines" and "Outline" objects.
        // All other objects should be collected and inserted into the main Document.
        match object.type_name().unwrap_or(b"") {
            b"Catalog" => {
                // Collect a first "Catalog" object and use it for the future "Pages".
                catalog_object = Some((
                    if let Some((id, _)) = catalog_object {
                        id
                    } else {
                        *object_id
                    },
                    object.clone(),
                ));
            }
            b"Pages" => {
                // Collect and update a first "Pages" object and use it for the future "Catalog"
                // We have also to merge all dictionaries of the old and the new "Pages" object
                if let Ok(dictionary) = object.as_dict() {
                    let mut dictionary = dictionary.clone();
                    if let Some((_, ref object)) = pages_object {
                        if let Ok(old_dictionary) = object.as_dict() {
                            dictionary.extend(old_dictionary);
                        }
                    }

                    pages_object = Some((
                        if let Some((id, _)) = pages_object {
                            id
                        } else {
                            *object_id
                        },
                        Object::Dictionary(dictionary),
                    ));
                }
            }
            b"Page" => {}     // Ignored, processed later and separately
            b"Outlines" => {} // Ignored, not supported yet
            b"Outline" => {}  // Ignored, not supported yet
            _ => {
                document.objects.insert(*object_id, object.clone());
            }
        }
    }

    // If no "Pages" object found, abort.
    if pages_object.is_none() {
        println!("Pages root not found.");

        return Ok(());
    }

    // Iterate over all "Page" objects and collect into the parent "Pages" created before
    for (object_id, object) in documents_pages.iter() {
        if let Ok(dictionary) = object.as_dict() {
            let mut dictionary = dictionary.clone();
            dictionary.set("Parent", pages_object.as_ref().unwrap().0);

            document
                    .objects
                    .insert(*object_id, Object::Dictionary(dictionary));
        }
    }

    // If no "Catalog" found, abort.
    if catalog_object.is_none() {
        println!("Catalog root not found.");

        return Ok(());
    }

    let catalog_object = catalog_object.unwrap();
    let pages_object = pages_object.unwrap();

    // Build a new "Pages" with updated fields
    if let Ok(dictionary) = pages_object.1.as_dict() {
        let mut dictionary = dictionary.clone();

        // Set new pages count
        dictionary.set("Count", documents_pages.len() as u32);

        // Set new "Kids" list (collected from documents pages) for "Pages"
        dictionary.set(
            "Kids",
            documents_pages
                    .into_iter()
                    .map(|(object_id, _)| Object::Reference(object_id))
                    .collect::<Vec<_>>(),
        );

        document
                .objects
                .insert(pages_object.0, Object::Dictionary(dictionary));
    }

    // Build a new "Catalog" with updated fields
    if let Ok(dictionary) = catalog_object.1.as_dict() {
        let mut dictionary = dictionary.clone();
        dictionary.set("Pages", pages_object.0);
        dictionary.remove(b"Outlines"); // Outlines not supported in merged PDFs

        document
                .objects
                .insert(catalog_object.0, Object::Dictionary(dictionary));
    }

    document.trailer.set("Root", catalog_object.0);

    // Update the max internal ID as wasn't updated before due to direct objects insertion
    document.max_id = document.objects.len() as u32;

    // Reorder all new Document objects
    document.renumber_objects();

    // Set any Bookmarks to the First child if they are not set to a page
    document.adjust_zero_pages();

    // Set all bookmarks to the PDF Object tree then set the Outlines to the Bookmark content map.
    if let Some(n) = document.build_outline() {
        if let Ok(Object::Dictionary(dict)) = document.get_object_mut(catalog_object.0) {
            dict.set("Outlines", Object::Reference(n));
        }
    }

    document.compress();

    // Save the merged PDF.
    // Store file in current working directory.
    // Note: Line is excluded when running doc tests
    if false {
        document.save("merged.pdf").unwrap();
    }

    Ok(())
}
```

* Decrypt PDF documents

```rust
use lopdf::Document;

// Load and decrypt PDF documents with empty password
#[cfg(not(feature = "async"))]
{
    // Load an encrypted PDF - automatically attempts decryption with empty password
    let doc = Document::load("assets/encrypted.pdf").unwrap();
    
    // Check if the document is encrypted
    if doc.is_encrypted() {
        println!("Document is encrypted");
        
        // The document has been automatically decrypted if the password was empty
        if doc.encryption_state.is_some() {
            println!("Successfully decrypted with empty password");
        }
    }
    
    // Access decrypted content
    let pages = doc.get_pages();
    println!("Number of pages: {}", pages.len());
    
    // Extract text from decrypted document
    let page_numbers: Vec<u32> = pages.keys().cloned().collect();
    let text = doc.extract_text(&page_numbers).unwrap();
    println!("Extracted {} characters of text", text.len());
    
    // Access individual objects
    for i in 1..=10 {
        if let Ok(obj) = doc.get_object((i, 0)) {
            println!("Successfully accessed object ({}, 0)", i);
        }
    }
}

#[cfg(feature = "async")]
{
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("Failed to create runtime")
        .block_on(async move {
            // Async version
            let doc = Document::load("assets/encrypted.pdf").await.unwrap();
            
            if doc.is_encrypted() {
                println!("Document is encrypted");
                if doc.encryption_state.is_some() {
                    println!("Successfully decrypted with empty password");
                }
            }
            
            let pages = doc.get_pages();
            let page_numbers: Vec<u32> = pages.keys().cloned().collect();
            let text = doc.extract_text(&page_numbers).unwrap();
            println!("Extracted {} characters of text", text.len());
        });
}
```

* Modify PDF document

```rust
use lopdf::Document;

// For this example to work a parser feature needs to be enabled
#[cfg(not(feature = "async"))]
#[cfg(feature = "nom_parser")]
{
    let mut doc = Document::load("assets/example.pdf").unwrap();

    doc.version = "1.4".to_string();
    doc.replace_text(1, "Hello World!", "Modified text!", None);
    // Store file in current working directory.
    // Note: Line is excluded when running tests
    if false {
        doc.save("modified.pdf").unwrap();
    }
}

#[cfg(feature = "async")]
#[cfg(feature = "nom_parser")]
{
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("Failed to create runtime")
        .block_on(async move {
            let mut doc = Document::load("assets/example.pdf").await.unwrap();
            
            doc.version = "1.4".to_string();
            doc.replace_text(1, "Hello World!", "Modified text!", None);
            // Store file in current working directory.
            // Note: Line is excluded when running tests
            if false {
                doc.save("modified.pdf").unwrap();
            }
    });
}

// For this example to work a parser feature needs to be enabled
#[cfg(not(feature = "async"))]
#[cfg(feature = "nom_parser")]
{
    let mut doc = Document::load("assets/example.pdf").unwrap();

    doc.version = "1.4".to_string();
    
    // Replace exact text matches
    doc.replace_text(1, "Hello World!", "Modified text!", None);
    
    // Replace partial text matches (new method)
    let count = doc.replace_partial_text(1, "Hello", "Hi", None).unwrap();
    println!("Replaced {} occurrences", count);
    
    // Store file in current working directory.
    // Note: Line is excluded when running tests
    if false {
        doc.save("modified.pdf").unwrap();
    }
}

#[cfg(feature = "async")]
#[cfg(feature = "nom_parser")]
{
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("Failed to create runtime")
        .block_on(async move {
            let mut doc = Document::load("assets/example.pdf").await.unwrap();
            
            doc.version = "1.4".to_string();
            
            // Replace exact text matches
            doc.replace_text(1, "Hello World!", "Modified text!", None);
            
            // Replace partial text matches (new method)
            let count = doc.replace_partial_text(1, "Hello", "Hi", None).unwrap();
            println!("Replaced {} occurrences", count);
            
            // Store file in current working directory.
            // Note: Line is excluded when running tests
            if false {
                doc.save("modified.pdf").unwrap();
            }
    });
}
```

* Save PDF with Object Streams (Modern Format)

Object streams allow multiple non-stream objects to be compressed together, significantly reducing file size.

```rust,no_run
use lopdf::{Document, SaveOptions};

#[cfg(not(feature = "async"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load existing PDF
    let mut doc = Document::load("input.pdf")?;

    // Save with modern features (object streams + cross-reference streams)
    // This typically reduces file size by 11-38%
    let mut file = std::fs::File::create("output.pdf")?;
    doc.save_modern(&mut file)?;

    // For more control, use SaveOptions
    let options = SaveOptions::builder()
        .use_object_streams(true)        // Enable object streams
        .use_xref_streams(true)          // Enable cross-reference streams
        .max_objects_per_stream(200)     // Max objects per stream (default: 100)
        .compression_level(9)            // Compression level 0-9 (default: 6)
        .build();

    let mut file2 = std::fs::File::create("output_custom.pdf")?;
    doc.save_with_options(&mut file2, options)?;
    
    Ok(())
}

#[cfg(feature = "async")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // For async feature, you need to use tokio runtime
    println!("This example requires the async feature to be disabled");
    Ok(())
}
```

### Complete Example: Creating and Saving with Object Streams

```rust
use lopdf::{Document, SaveOptions};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create or load a document
    let mut doc = Document::with_version("1.5");
    // ... add content to document ...

    // Method 1: Quick modern save (recommended)
    let mut file = File::create("output.pdf")?;
    doc.save_modern(&mut file)?;

    // Method 2: Custom settings for maximum compression
    let options = SaveOptions::builder()
        .use_object_streams(true)
        .use_xref_streams(true)
        .max_objects_per_stream(200)
        .compression_level(9)
        .build();

    let mut file2 = File::create("output_max_compressed.pdf")?;
    doc.save_with_options(&mut file2, options)?;

    // Compare file sizes (if traditional file exists)
    if std::path::Path::new("output_traditional.pdf").exists() {
        let traditional_size = std::fs::metadata("output_traditional.pdf")?.len();
        let modern_size = std::fs::metadata("output.pdf")?.len();
        let reduction = 100.0 - (modern_size as f64 / traditional_size as f64 * 100.0);
        println!("Size reduction: {:.1}%", reduction);
    }
    
    Ok(())
}
```

For more examples, see:
- [`examples/object_streams.rs`](examples/object_streams.rs) - Creating PDFs with object streams
- [`examples/compress_existing_pdf.rs`](examples/compress_existing_pdf.rs) - Compress existing PDFs
- [`examples/analyze_object_streams.rs`](examples/analyze_object_streams.rs) - Analyze object stream usage

## Object Streams Support

lopdf now includes full support for creating and reading PDF object streams (PDF 1.5+ feature). Object streams provide significant file size reduction by compressing multiple non-stream objects together.

### Key Benefits

- **File size reduction**: 11-61% smaller PDFs depending on content
- **Modern PDF compliance**: Full PDF 1.5+ specification support
- **Backward compatibility**: All existing APIs remain unchanged
- **Performance**: <2ms to check 1000 objects for compression eligibility

### Creating Object Streams Directly

```rust
use lopdf::{Object, ObjectStream, dictionary};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an object stream with custom settings
    let mut obj_stream = ObjectStream::builder()
        .max_objects(100)      // Maximum objects per stream
        .compression_level(6)  // zlib compression level (0-9)
        .build();

    // Add objects to the stream
    obj_stream.add_object((1, 0), Object::Integer(42))?;
    obj_stream.add_object((2, 0), Object::Name(b"Example".to_vec()))?;
    obj_stream.add_object((3, 0), Object::Dictionary(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica"
    }))?;

    // Convert to a stream object
    let stream = obj_stream.to_stream_object()?;
    # Ok::<(), Box<dyn std::error::Error>>(())
# }
```

### Object Eligibility

Not all objects can be compressed into object streams. The following objects are **excluded**:

- Stream objects (content streams, image streams, etc.)
- Cross-reference streams (Type = XRef)
- Object streams themselves (Type = ObjStm)
- Encryption dictionary (when referenced by trailer's Encrypt entry)
- Objects with generation number > 0
- Document catalog in linearized PDFs only

All other objects, including structural objects (Catalog, Pages, Page) and trailer-referenced objects (except encryption), can be compressed.

### Cross-reference Streams

When using `save_modern()` or enabling `use_xref_streams(true)`, lopdf creates binary cross-reference streams instead of traditional ASCII cross-reference tables. This provides additional space savings and is part of the PDF 1.5+ specification.

### SaveOptions Reference

The `SaveOptions` builder provides fine-grained control over PDF compression:

```rust
use lopdf::SaveOptions;

let options = SaveOptions::builder()
    .use_object_streams(true)        // Enable object streams (default: false)
    .use_xref_streams(true)          // Enable xref streams (default: false)
    .max_objects_per_stream(200)     // Max objects per stream (default: 100)
    .compression_level(9)            // zlib level 0-9 (default: 6)
    .build();
```

## PDF Decryption Support

lopdf now includes enhanced support for reading encrypted PDF documents. The library can automatically decrypt PDFs that use empty passwords, which is common for many protected documents.

### Key Features

- **Automatic decryption**: PDFs encrypted with empty passwords are automatically decrypted on load
- **Object stream support**: Handles encrypted PDFs containing compressed object streams
- **Transparent access**: Once decrypted, all document methods work normally
- **Preservation of structure**: Document structure and content remain intact after decryption

### How It Works

When loading an encrypted PDF, lopdf:
1. Detects encryption via the `Encrypt` entry in the trailer
2. Extracts raw object bytes before parsing
3. Attempts authentication with an empty password
4. Decrypts all objects if authentication succeeds
5. Processes compressed objects from object streams

### Example: Working with Encrypted PDFs

```rust
use lopdf::Document;

#[cfg(not(feature = "async"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load an encrypted PDF - automatically attempts decryption
    let doc = Document::load("assets/encrypted.pdf")?;
    
    // Check encryption status
    if doc.is_encrypted() {
        println!("Document is encrypted");
        
        // Check if decryption was successful
        if doc.encryption_state.is_some() {
            println!("Successfully decrypted");
            
            // Now you can work with the document normally
            let pages = doc.get_pages();
            println!("Pages: {}", pages.len());
            
            // Extract text
            let page_nums: Vec<u32> = pages.keys().cloned().collect();
            let text = doc.extract_text(&page_nums)?;
            println!("Text length: {} chars", text.len());
            
            // Access objects
            for i in 1..=10 {
                if let Ok(_) = doc.get_object((i, 0)) {
                    println!("Object ({}, 0) accessible", i);
                }
            }
        } else {
            println!("Decryption failed - password required");
        }
    }
    
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load an encrypted PDF - automatically attempts decryption
    let doc = Document::load("assets/encrypted.pdf").await?;
    
    // Check encryption status
    if doc.is_encrypted() {
        println!("Document is encrypted");
        
        // Check if decryption was successful
        if doc.encryption_state.is_some() {
            println!("Successfully decrypted");
            
            // Now you can work with the document normally
            let pages = doc.get_pages();
            println!("Pages: {}", pages.len());
            
            // Extract text
            let page_nums: Vec<u32> = pages.keys().cloned().collect();
            let text = doc.extract_text(&page_nums)?;
            println!("Text length: {} chars", text.len());
            
            // Access objects
            for i in 1..=10 {
                if let Ok(_) = doc.get_object((i, 0)) {
                    println!("Object ({}, 0) accessible", i);
                }
            }
        } else {
            println!("Decryption failed - password required");
        }
    }
    
    Ok(())
}
```

### Limitations

- Currently only supports PDFs encrypted with empty passwords
- Password-protected PDFs require manual authentication (use `authenticate_password` method)
- Some encryption algorithms may not be fully supported

For more examples, see:
- [`examples/test_decryption.rs`](examples/test_decryption.rs) - Testing decryption functionality
- [`examples/verify_decryption.rs`](examples/verify_decryption.rs) - Comprehensive decryption verification
- [`tests/decryption.rs`](tests/decryption.rs) - Decryption test suite

## FAQ

* Why does the library keep everything in memory as high-level objects until finally serializing the entire document?

  Normally, a PDF document won't be very large, ranging from tens of KB to hundreds of MB. Memory size is not a bottle neck for today's computer.
  By keeping the whole document in memory, the stream length can be pre-calculated, no need to use a reference object for the Length entry.
  The resulting PDF file is smaller for distribution and faster for PDF consumers to process.

  Producing is a one-time effort, while consuming is many more.

* How do object streams affect memory usage?

  Object streams actually help reduce memory usage during document creation. When enabled, multiple small objects are grouped and compressed together, reducing the overall memory footprint. The compression happens during the save operation, so the in-memory representation remains the same until `save_with_options()` or `save_modern()` is called.

* What PDF versions support object streams?

  Object streams were introduced in PDF 1.5. When using `save_modern()` or object streams, lopdf automatically ensures the document version is at least 1.5. For maximum compatibility with older PDF readers, you can use the traditional `save()` method.

* Can I analyze existing PDFs to see if they use object streams?

  Yes! lopdf can read and parse object streams from existing PDFs. Use the `Document::load()` method to open any PDF, and lopdf will automatically handle object streams if present. See the examples directory for analysis tools.

## License

lopdf is available under the MIT license, with the exception of the Montserrat font.