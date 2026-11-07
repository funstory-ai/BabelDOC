use crate::Result;
use crate::{Dictionary, Document, Object, ObjectId};
#[cfg(feature = "font_embedding")]
use crate::{FontData, Stream};

impl Document {
    /// Create new PDF document with version.
    pub fn with_version<S: Into<String>>(version: S) -> Document {
        let mut document = Self::new();
        document.version = version.into();
        document
    }

    /// Create an object ID.
    pub fn new_object_id(&mut self) -> ObjectId {
        self.max_id += 1;
        (self.max_id, 0)
    }

    /// Add PDF object into document's object list.
    pub fn add_object<T: Into<Object>>(&mut self, object: T) -> ObjectId {
        self.max_id += 1;
        let id = (self.max_id, 0);
        self.objects.insert(id, object.into());
        id
    }

    pub fn set_object<T: Into<Object>>(&mut self, id: ObjectId, object: T) {
        self.objects.insert(id, object.into());
    }

    /// Remove PDF object from document's object list.
    ///
    /// Other objects may still hold references to this object! Therefore, removing the object might
    /// lead to dangling references.
    pub fn remove_object(&mut self, object_id: &ObjectId) -> Result<()> {
        self.objects.remove(object_id);
        Ok(())
    }

    /// Remove annotation from the document.
    ///
    /// References to this annotation are removed from the pages' lists of annotations. Finally, the
    /// annotation object itself is removed.
    pub fn remove_annot(&mut self, object_id: &ObjectId) -> Result<()> {
        for (_, page_id) in self.get_pages() {
            let page = self.get_object_mut(page_id)?.as_dict_mut()?;
            let annots = page.get_mut(b"Annots")?.as_array_mut()?;

            annots.retain(|object| {
                if let Ok(id) = object.as_reference() {
                    return id != *object_id;
                }

                true
            });
        }

        self.remove_object(object_id)?;

        Ok(())
    }

    /// Get the page's resource dictionary.
    ///
    /// Get Object that has the key "Resources".
    pub fn get_or_create_resources(&mut self, page_id: ObjectId) -> Result<&mut Object> {
        let resources_id = {
            let page = self.get_object(page_id).and_then(Object::as_dict)?;
            if page.has(b"Resources") {
                page.get(b"Resources").and_then(Object::as_reference).ok()
            } else {
                None
            }
        };
        if let Some(res_id) = resources_id {
            return self.get_object_mut(res_id);
        }
        let page = self.get_object_mut(page_id).and_then(Object::as_dict_mut)?;
        if !page.has(b"Resources") {
            page.set(b"Resources", Dictionary::new());
        }
        page.get_mut(b"Resources")
    }

    /// Add XObject to a page.
    ///
    /// Get Object that has the key `Resources -> XObject`.
    pub fn add_xobject<N: Into<Vec<u8>>>(
        &mut self, page_id: ObjectId, xobject_name: N, xobject_id: ObjectId,
    ) -> Result<()> {
        if let Ok(resources) = self.get_or_create_resources(page_id).and_then(Object::as_dict_mut) {
            if !resources.has(b"XObject") {
                resources.set("XObject", Dictionary::new());
            }
            let mut xobjects = resources.get_mut(b"XObject")?;
            if let Object::Reference(xobjects_ref_id) = xobjects {
                let mut xobjects_id = *xobjects_ref_id;
                while let Object::Reference(id) = self.get_object(xobjects_id)? {
                    xobjects_id = *id;
                }
                xobjects = self.get_object_mut(xobjects_id)?;
            }
            let xobjects = Object::as_dict_mut(xobjects)?;
            xobjects.set(xobject_name, Object::Reference(xobject_id));
        }
        Ok(())
    }

    /// Add Graphics State to a page.
    ///
    /// Get Object that has the key `Resources -> ExtGState`.
    pub fn add_graphics_state<N: Into<Vec<u8>>>(
        &mut self, page_id: ObjectId, gs_name: N, gs_id: ObjectId,
    ) -> Result<()> {
        if let Ok(resources) = self.get_or_create_resources(page_id).and_then(Object::as_dict_mut) {
            if !resources.has(b"ExtGState") {
                resources.set("ExtGState", Dictionary::new());
            }
            let states = resources.get_mut(b"ExtGState").and_then(Object::as_dict_mut)?;
            states.set(gs_name, Object::Reference(gs_id));
        }
        Ok(())
    }

    /// Add font to a page.
    /// # Examples
    ///
    /// ```no_run
    /// // Assuming you have a font file at "./SomeFont.ttf"
    /// use lopdf::dictionary;
    ///
    /// let font_file = std::fs::read("./SomeFont.ttf").unwrap();
    ///
    /// // Create a new FontData instance with the font file.
    /// let font_name = "SomeFont".to_string();
    /// let mut font_data = lopdf::FontData::new(&font_file, font_name.clone());
    ///
    /// // Customize the font data if needed.
    /// font_data
    ///     .set_italic_angle(10)
    ///     .set_encoding("WinAnsiEncoding".to_string());
    ///
    ///
    /// // Create a new PDF document.
    /// let mut doc = lopdf::Document::with_version("1.5");
    ///
    /// // Add the font to the document.
    /// let font_id = doc.add_font(font_data).unwrap();
    ///
    /// // Now you can use `font_id` to reference the font in your PDF content.
    /// // For example:
    /// let resources_id = doc.add_object(dictionary! {
    ///  "Font" => dictionary! {
    ///         font_name => font_id,
    ///     },
    /// });
    /// ```
    #[cfg(feature = "font_embedding")]
    pub fn add_font(&mut self, font_data: FontData) -> Result<ObjectId> {
        // Create embedded font stream
        let font_stream = Stream::new(
            dictionary! {
                "Length1" => Object::Integer(font_data.bytes().len() as i64),
            },
            font_data.bytes(),
        );
        let font_file_id = self.add_object(font_stream);
        let font_name = font_data.font_name.clone();

        // Create font descriptor dictionary
        let font_descriptor_id = self.add_object(dictionary! {
            "Type" => "FontDescriptor",
            "FontName" => Object::Name(font_name.clone().into_bytes()),
            "Flags" => Object::Integer(font_data.flags),
            "FontBBox" => Object::Array(vec![
                Object::Integer(font_data.font_bbox.0),
                Object::Integer(font_data.font_bbox.1),
                Object::Integer(font_data.font_bbox.2),
                Object::Integer(font_data.font_bbox.3),
            ]),
            "ItalicAngle" => Object::Integer(font_data.italic_angle),
            "Ascent" => Object::Integer(font_data.ascent),
            "Descent" => Object::Integer(font_data.descent),
            "CapHeight" => Object::Integer(font_data.cap_height),
            "StemV" => Object::Integer(font_data.stem_v),
            "FontFile2" => Object::Reference(font_file_id),
        });

        // Create font dictionary
        let font_id = self.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "TrueType",
            "BaseFont" => Object::Name(font_name.clone().into_bytes()),
            "FontDescriptor" => Object::Reference(font_descriptor_id),
            "Encoding" => Object::Name(font_data.encoding.into_bytes()),
        });

        if let Some(first_char) = font_data.first_char {
            self.get_dictionary_mut(font_id)
                .unwrap()
                .set("FirstChar", Object::Integer(first_char));
        }
        if let Some(last_char) = font_data.last_char {
            self.get_dictionary_mut(font_id)
                .unwrap()
                .set("LastChar", Object::Integer(last_char));
        }
        if let Some(widths_array) = font_data.widths {
            let widths = widths_array.into_iter().map(Object::Integer).collect();
            self.get_dictionary_mut(font_id)
                .unwrap()
                .set("Widths", Object::Array(widths));
        }

        Ok(font_id)
    }
}

#[cfg(test)]
pub mod tests {
    use std::path::PathBuf;

    #[cfg(feature = "font_embedding")]
    use crate::FontData;
    use crate::content::*;
    use crate::{Document, Object, Stream};

    #[cfg(not(feature = "time"))]
    pub fn get_timestamp() -> Object {
        Object::string_literal("D:19700101000000Z")
    }

    #[cfg(feature = "time")]
    pub fn get_timestamp() -> Object {
        time::OffsetDateTime::now_utc().into()
    }

    /// Create and return a document for testing
    pub fn create_document() -> Document {
        create_document_with_texts(&["Hello World!"])
    }

    pub fn create_document_with_texts(texts_for_pages: &[&str]) -> Document {
        let mut doc = Document::with_version("1.5");
        let info_id = doc.add_object(dictionary! {
            "Title" => Object::string_literal("Create PDF document example"),
            "Creator" => Object::string_literal("https://crates.io/crates/lopdf"),
            "CreationDate" => get_timestamp(),
        });
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
        let contents = texts_for_pages.iter().map(|text| Content {
            operations: vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 48.into()]),
                Operation::new("Td", vec![100.into(), 600.into()]),
                Operation::new("Tj", vec![Object::string_literal(*text)]),
                Operation::new("ET", vec![]),
            ],
        });

        let pages = contents.map(|content| {
            let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
            let page = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "Contents" => content_id,
            });
            page.into()
        });

        let pages = dictionary! {
            "Type" => "Pages",
            "Kids" => pages.collect::<Vec<Object>>(),
            "Count" => 1,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
        };
        doc.objects.insert(pages_id, Object::Dictionary(pages));
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);
        doc.trailer.set("Info", info_id);
        doc.trailer.set(
            "ID",
            Object::Array(vec![Object::string_literal(b"ABC"), Object::string_literal(b"DEF")]),
        );
        doc.compress();
        doc
    }

    /// Create a single-page document whose content stream is the
    /// caller-supplied list of operations (wrapped in `BT`/`ET` if not
    /// already). Used by tests that need to exercise specific
    /// content-stream operators not produced by `create_document_with_texts`.
    pub fn create_document_with_operations(operations: Vec<Operation>) -> Document {
        let mut doc = Document::with_version("1.5");
        let info_id = doc.add_object(dictionary! {
            "Title" => Object::string_literal("Create PDF document example"),
            "Creator" => Object::string_literal("https://crates.io/crates/lopdf"),
            "CreationDate" => get_timestamp(),
        });
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
        let content = Content { operations };
        let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
        let page = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
        });
        let pages = dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page.into()],
            "Count" => 1,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
        };
        doc.objects.insert(pages_id, Object::Dictionary(pages));
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);
        doc.trailer.set("Info", info_id);
        doc.trailer.set(
            "ID",
            Object::Array(vec![Object::string_literal(b"ABC"), Object::string_literal(b"DEF")]),
        );
        doc.compress();
        doc
    }

    /// Save a document
    pub fn save_document(file_path: &PathBuf, doc: &mut Document) {
        let res = doc.save(file_path);

        assert!(match res {
            Ok(_file) => true,
            Err(_e) => false,
        });
    }

    #[test]
    fn save_created_document() {
        // Create temporary folder to store file.
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_1_create.pdf");

        let mut doc = create_document();
        // Save to file
        save_document(&file_path, &mut doc);
        assert!(file_path.exists());
    }

    #[cfg(feature = "font_embedding")]
    #[test]
    fn test_add_font_embeds_font_correctly() {
        // Create a dummy TTF font in memory (fake content, just to test structure)
        let font_file = std::fs::read("./assets/fonts/Montserrat-Regular.ttf").unwrap();

        // Construct FontData manually
        let mut font_data = FontData::new(&font_file, "MyFont".to_string());
        font_data
            .set_flags(32)
            .set_font_bbox((0, -200, 1000, 800))
            .set_italic_angle(0)
            .set_ascent(750)
            .set_descent(-250)
            .set_cap_height(700)
            .set_stem_v(80)
            .set_encoding("WinAnsiEncoding".to_string());

        // Create PDF document
        let mut doc = Document::with_version("1.5");

        // Create dummy page
        let page_id = doc.new_object_id();
        doc.set_object(page_id, dictionary! {});

        // Add font
        let font_id = doc.add_font(font_data.clone()).unwrap();

        // Font dictionary must exist
        let font_obj = doc.get_object(font_id).unwrap();
        let font_dict = font_obj.as_dict().unwrap();

        // Check base font name
        assert_eq!(font_dict.get(b"BaseFont").unwrap(), &Object::Name(b"MyFont".to_vec()));

        // Check encoding
        assert_eq!(
            font_dict.get(b"Encoding").unwrap(),
            &Object::Name(b"WinAnsiEncoding".to_vec())
        );

        // Check font descriptor exists and is referenced
        let descriptor_ref = font_dict.get(b"FontDescriptor").unwrap().as_reference().unwrap();
        let descriptor_obj = doc.get_object(descriptor_ref).unwrap().as_dict().unwrap();
        assert_eq!(
            descriptor_obj.get(b"FontName").unwrap(),
            &Object::Name(b"MyFont".to_vec())
        );

        // Check font file is embedded
        let font_file_ref = descriptor_obj.get(b"FontFile2").unwrap().as_reference().unwrap();
        let font_stream = doc.get_object(font_file_ref).unwrap().as_stream().unwrap();
        assert_eq!(font_stream.content, font_file);
    }
}
