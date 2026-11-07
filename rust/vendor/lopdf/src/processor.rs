use crate::Result;
use crate::{Document, Object, ObjectId};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;

impl Document {
    /// Change producer of document information dictionary.
    pub fn change_producer(&mut self, producer: &str) {
        if let Ok(info) = self.trailer.get_mut(b"Info")
            && let Some(dict) = match info {
                Object::Dictionary(dict) => Some(dict),
                Object::Reference(id) => self.objects.get_mut(id).and_then(|o| o.as_dict_mut().ok()),
                _ => None,
            }
        {
            dict.set("Producer", Object::string_literal(producer));
        }
    }

    /// Compress PDF stream objects.
    pub fn compress(&mut self) {
        for object in self.objects.values_mut() {
            if let Object::Stream(stream) = object
                && stream.allows_compression
            {
                // Ignore any error and continue to compress other streams.
                let _ = stream.compress();
            }
        }
    }

    /// Decompress PDF stream objects.
    pub fn decompress(&mut self) {
        for object in self.objects.values_mut() {
            if let Object::Stream(stream) = object {
                let _ = stream.decompress();
            }
        }
    }

    /// Delete pages.
    pub fn delete_pages(&mut self, page_numbers: &[u32]) {
        let pages = self.get_pages();
        for page_number in page_numbers {
            if let Some(page) = pages.get(page_number).and_then(|page_id| self.delete_object(*page_id)) {
                let mut page_tree_ref = page
                    .as_dict()
                    .and_then(|dict| dict.get(b"Parent"))
                    .and_then(Object::as_reference);
                while let Ok(page_tree_id) = page_tree_ref {
                    if let Some(page_tree) = self.objects.get_mut(&page_tree_id).and_then(|pt| pt.as_dict_mut().ok()) {
                        if let Ok(count) = page_tree.get(b"Count").and_then(Object::as_i64) {
                            page_tree.set("Count", count - 1);
                        }
                        page_tree_ref = page_tree.get(b"Parent").and_then(Object::as_reference);
                    } else {
                        break;
                    }
                }
            }
        }
    }

    /// Prune all unused objects.
    pub fn prune_objects(&mut self) -> Vec<ObjectId> {
        let mut ids = vec![];
        let refs = self.traverse_objects(|_| {});
        for id in self.objects.keys() {
            if !refs.contains(id) {
                ids.push(*id);
            }
        }

        for id in &ids {
            self.objects.remove(id);
        }

        ids
    }

    /// Delete object by object ID.
    pub fn delete_object(&mut self, id: ObjectId) -> Option<Object> {
        let action = |object: &mut Object| match object {
            Object::Array(array) => {
                if let Some(index) = array.iter().position(|item: &Object| match *item {
                    Object::Reference(ref_id) => ref_id == id,
                    _ => false,
                }) {
                    array.remove(index);
                }
            }
            Object::Dictionary(dict) => {
                let keys: Vec<Vec<u8>> = dict
                    .iter()
                    .filter(|&(_, item): &(&Vec<u8>, &Object)| match *item {
                        Object::Reference(ref_id) => ref_id == id,
                        _ => false,
                    })
                    .map(|(k, _)| k.clone())
                    .collect();
                for key in keys {
                    dict.remove(&key);
                }
            }
            _ => {}
        };
        self.traverse_objects(action);
        self.objects.remove(&id)
    }

    /// Delete zero length stream objects.
    pub fn delete_zero_length_streams(&mut self) -> Vec<ObjectId> {
        let mut ids = vec![];
        for id in self.objects.keys() {
            if self
                .objects
                .get(id)
                .and_then(|o| Object::as_stream(o).ok())
                .map(|stream| stream.content.is_empty())
                .unwrap_or(false)
            {
                ids.push(*id);
            }
        }

        for id in &ids {
            self.delete_object(*id);
        }

        ids
    }

    /// Renumber objects, normally called after delete_unused_objects.
    pub fn renumber_objects(&mut self) {
        self.renumber_objects_with(1)
    }

    fn update_bookmark_pages(&mut self, bookmarks: &[u32], old: &ObjectId, new: &ObjectId) {
        for id in bookmarks {
            let (children, page) = match self.bookmark_table.get(id) {
                Some(n) => (n.children.clone(), n.page),
                None => return,
            };

            if page == *old {
                let bookmark = self.bookmark_table.get_mut(id).unwrap();
                bookmark.page = *new;
            }

            if !children.is_empty() {
                self.update_bookmark_pages(&children[..], old, new);
            }
        }
    }

    pub fn renumber_bookmarks(&mut self, old: &ObjectId, new: &ObjectId) {
        if !self.bookmarks.is_empty() {
            self.update_bookmark_pages(&self.bookmarks.clone(), old, new);
        }
    }

    /// Renumber objects with a custom starting id, this is very useful in case of multiple
    /// document object insertions in a single main document
    pub fn renumber_objects_with(&mut self, starting_id: u32) {
        let mut replace = BTreeMap::new();
        let mut new_id = starting_id;
        let mut i = 0;

        // Check if we need to order the pages first, as this means the first page doesn't have a lower ID.
        // So it ends up in a random spot based on its ID. We check first to avoid double traversal, unless we have too.

        let mut page_order: Vec<(i32, (u32, u16))> = self
            .page_iter()
            .map(|id| {
                i += 1;
                (i, id)
            })
            .collect();

        page_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        i = 0;

        let needs_ordering = page_order.iter().any(|a| {
            i += 1;
            a.0 != i
        });

        if needs_ordering {
            let mut pages = page_order.clone();
            pages.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut objects = BTreeMap::new();

            for (old, new) in pages.iter().zip(page_order) {
                if let Some(object) = self.objects.remove(&old.1) {
                    objects.insert((new.1.0, old.1.1), object);
                    replace.insert(old.1, (new.1.0, old.1.1));
                }

                if old.1 != new.1 {
                    self.renumber_bookmarks(&old.1, &(new.1.0, old.1.1));
                }
            }

            for (new, object) in objects {
                self.objects.insert(new, object);
            }

            let action = |object: &mut Object| {
                if let Object::Reference(id) = object
                    && let Some(new_id) = replace.get(id)
                {
                    *id = *new_id;
                }
            };

            self.traverse_objects(action);
            replace.clear();
        }

        let mut ids = self.objects.keys().cloned().collect::<Vec<ObjectId>>();
        ids.sort_unstable();

        for id in ids {
            if id.0 != new_id {
                replace.insert(id, (new_id, id.1));
            }

            new_id += 1;
        }

        let mut objects = BTreeMap::new();

        // remove and collect all removed objects
        for (old, new) in &replace {
            if let Some(object) = self.objects.remove(old) {
                objects.insert(*new, object);
            }

            if old != new {
                self.renumber_bookmarks(old, new);
            }
        }

        // insert new replaced keys objects
        for (new, object) in objects {
            self.objects.insert(new, object);
        }

        let action = |object: &mut Object| {
            if let Object::Reference(id) = object
                && let Some(new_id) = replace.get(id)
            {
                *id = *new_id;
            }
        };

        self.traverse_objects(action);

        self.max_id = new_id - 1;
    }

    pub fn change_content_stream(&mut self, stream_id: ObjectId, content: Vec<u8>) {
        if let Some(Object::Stream(stream)) = self.objects.get_mut(&stream_id) {
            stream.set_plain_content(content);
            // Ignore any compression error.
            let _ = stream.compress();
        }
    }

    pub fn change_page_content(&mut self, page_id: ObjectId, content: Vec<u8>) -> Result<()> {
        let contents = self.get_dictionary(page_id).and_then(|page| page.get(b"Contents"))?;
        match contents {
            Object::Reference(id) => self.change_content_stream(*id, content),
            Object::Array(arr) => {
                if arr.len() == 1 {
                    if let Ok(id) = arr[0].as_reference() {
                        self.change_content_stream(id, content)
                    }
                } else {
                    let new_stream = self.add_object(super::Stream::new(dictionary! {}, content));
                    if let Ok(Object::Dictionary(dict)) = self.get_object_mut(page_id) {
                        dict.set("Contents", new_stream);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn extract_stream(&self, stream_id: ObjectId, decompress: bool) -> Result<()> {
        let mut file = File::create(format!("{stream_id:?}.bin"))?;
        if let Ok(Object::Stream(stream)) = self.get_object(stream_id) {
            if decompress {
                if let Ok(data) = stream.decompressed_content() {
                    file.write_all(&data)?;
                } else {
                    file.write_all(&stream.content)?;
                }
            } else {
                file.write_all(&stream.content)?;
            }
        }
        Ok(())
    }
}
