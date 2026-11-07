use super::{Dictionary, Document, Object, ObjectId};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Bookmark {
    /// Children, Must be a Collection that allows for insertion of the same page ID.
    pub children: Vec<u32>,
    pub title: String,
    /// 0, 1 for italic, 2 for bold, 3 for italic bold
    pub format: u32,
    /// R,G,B
    pub color: [f32; 3],
    pub page: ObjectId,
    pub id: u32,
}

impl Bookmark {
    pub fn new(title: String, color: [f32; 3], format: u32, page: ObjectId) -> Bookmark {
        Bookmark {
            children: Vec::new(),
            title,
            format,
            color,
            page,
            id: 0,
        }
    }
}

impl Document {
    pub fn add_bookmark(&mut self, mut bookmark: Bookmark, parent: Option<u32>) -> u32 {
        self.max_bookmark_id += 1;
        let id = self.max_bookmark_id;

        bookmark.id = id;

        if let Some(p) = parent {
            if let Some(b) = self.bookmark_table.get_mut(&p) {
                b.children.push(id);
            }
        } else {
            self.bookmarks.push(id);
        }

        self.bookmark_table.insert(id, bookmark);
        id
    }

    fn outline_child(
        &self, maxid: &mut u32, parent: (ObjectId, &[u32]), processed: &mut HashMap<ObjectId, Dictionary>,
    ) -> (Option<ObjectId>, Option<ObjectId>, i64) {
        let mut first: Option<ObjectId> = None;
        let mut last: Option<ObjectId> = None;
        let count = parent.1.len();
        for i in parent.1 {
            let mut child = Dictionary::new();
            *maxid += 1;
            let id: ObjectId = (*maxid, 0);
            *maxid += 1;
            let info_id: ObjectId = (*maxid, 0);
            let bookmark = self.bookmark_table.get(i).unwrap();

            let info = dictionary! {
                "D" =>  vec![bookmark.page.into(), Object::Name("Fit".into())],
                "S" => "GoTo",
            };

            let title_bytes = if bookmark.title.is_ascii() {
                bookmark.title.as_bytes().to_vec()
            } else {
                // If the title contains non-ASCII characters:
                // Create a new vector with the UTF-16 Byte Order Mark (BOM) for UTF-16BE.
                let mut bom = vec![0xFE, 0xFF];
                let utf16_title = bookmark.title.encode_utf16();
                // Append the UTF-16BE encoded bytes of the title to the BOM.
                bom.extend(utf16_title.flat_map(u16::to_be_bytes));
                bom
            };

            child.set("Parent", parent.0);
            child.set("Title", Object::string_literal(title_bytes));
            child.set("A", info_id);
            child.set("F", Object::Integer(bookmark.format.into()));
            child.set(
                "C",
                vec![
                    bookmark.color[0].into(),
                    bookmark.color[1].into(),
                    bookmark.color[2].into(),
                ],
            );

            if first.is_none() {
                first = Some(id);
            } else if let Some(x) = last {
                let inner_object = processed.get_mut(&x).unwrap();
                inner_object.set("Next", id);
                child.set("Prev", x);
            }

            last = Some(id);

            if !bookmark.children.is_empty() {
                let (c_first, c_last, c_count) = self.outline_child(maxid, (id, &bookmark.children[..]), processed);

                if let Some(n) = c_first {
                    child.set("First", n);
                }

                if let Some(n) = c_last {
                    child.set("Last", n);
                }

                child.set("Count", c_count);
            }

            processed.insert(id, child);
            processed.insert(info_id, info);
        }

        (first, last, count as i64)
    }

    pub fn build_outline(&mut self) -> Option<ObjectId> {
        let mut processed: HashMap<ObjectId, Dictionary> = HashMap::new();

        if !self.bookmarks.is_empty() {
            let mut outline = Dictionary::new();
            let mut maxid = self.max_id;
            maxid += 1;
            let id: ObjectId = (maxid, 0);

            let (first, last, count) = self.outline_child(&mut maxid, (id, &self.bookmarks[..]), &mut processed);

            if let Some(n) = first {
                outline.set("First", n);
            }

            if let Some(n) = last {
                outline.set("Last", n);
            }

            outline.set("Count", Object::Integer(count));

            for (obj_id, obj) in processed.drain() {
                self.objects.insert(obj_id, obj.into());
            }

            self.objects.insert(id, outline.into());
            self.max_id = maxid;
            return Some(id);
        }

        None
    }
}
