use crate::{Dictionary, Document, Object, ObjectId, Result};

#[derive(Debug, Clone)]
pub struct IncrementalDocument {
    /// The raw data for the files read from input.
    bytes_documents: Vec<u8>,

    /// The combined result of `bytes_documents`.
    /// Do not edit this document as it will not be saved.
    prev_documents: Document,

    /// A new document appended to the previously loaded file.
    pub new_document: Document,
}

impl IncrementalDocument {
    /// Create new PDF document.
    pub fn new() -> Self {
        Self {
            bytes_documents: Vec::new(),
            prev_documents: Document::new(),
            new_document: Document::new(),
        }
    }

    /// Create new `IncrementalDocument` from the bytes and document.
    ///
    /// The function expects the bytes and previous document to match.
    /// If they do not match exactly this might result in broken PDFs.
    pub fn create_from(prev_bytes: Vec<u8>, prev_documents: Document) -> Self {
        Self {
            bytes_documents: prev_bytes,
            new_document: Document::new_from_prev(&prev_documents),
            prev_documents,
        }
    }

    /// Get the structure of the previous documents (all prev incremental updates combined.)
    pub fn get_prev_documents(&self) -> &Document {
        &self.prev_documents
    }

    /// Get the bytes of the previous documents.
    pub fn get_prev_documents_bytes(&self) -> &[u8] {
        &self.bytes_documents
    }

    /// Clone Object from previous document to new document.
    /// If the object already exists nothing is done.
    ///
    /// This function can be used to clone an object so it can be changed in the incremental updates.
    pub fn opt_clone_object_to_new_document(&mut self, object_id: ObjectId) -> Result<()> {
        if !self.new_document.has_object(object_id) {
            let old_object = self.prev_documents.get_object(object_id)?;
            self.new_document.set_object(object_id, old_object.clone());
        }
        Ok(())
    }

    /// Get the page's resource dictionary (only in new document).
    ///
    /// Get Object that has the key `Resources`.
    pub fn get_or_create_resources(&mut self, page_id: ObjectId) -> Result<&mut Object> {
        self.opt_clone_object_to_new_document(page_id)?;
        let resources_id = {
            let page = self.new_document.get_object(page_id).and_then(Object::as_dict)?;
            if page.has(b"Resources") {
                page.get(b"Resources").and_then(Object::as_reference).ok()
            } else {
                None
            }
        };
        if let Some(res_id) = resources_id {
            self.opt_clone_object_to_new_document(res_id)?;
            return self.new_document.get_object_mut(res_id);
        }
        let page = self
            .new_document
            .get_object_mut(page_id)
            .and_then(Object::as_dict_mut)?;
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
                while let Object::Reference(id) = self.new_document.get_object(xobjects_id)? {
                    xobjects_id = *id;
                }
                xobjects = self.new_document.get_object_mut(xobjects_id)?;
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
}

impl Default for IncrementalDocument {
    fn default() -> Self {
        Self::new()
    }
}
