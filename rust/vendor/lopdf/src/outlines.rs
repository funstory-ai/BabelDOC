use indexmap::IndexMap;

use super::{Destination, Dictionary, Document, Error, Object, ObjectId, Result};

#[derive(Debug, Clone)]
pub enum Outline {
    Destination(Destination),
    SubOutlines(Vec<Outline>),
}

impl Document {
    pub fn get_outline(
        &self, node: &Dictionary, named_destinations: &mut IndexMap<Vec<u8>, Destination>,
    ) -> Result<Option<Outline>> {
        let action = match self.get_dict_in_dict(node, b"A") {
            Ok(a) => a,
            Err(_) => {
                return self.build_outline_result(node.get(b"Dest")?, node.get(b"Title")?, named_destinations);
            }
        };
        let command = action.get(b"S")?.as_name()?;
        if command != b"GoTo" && command != b"GoToR" {
            return Err(Error::InvalidOutline("Expected GoTo or GoToR".to_string()));
        }
        let title_obj = node.get(b"Title")?;
        let title_ref = match title_obj.as_reference() {
            Ok(o) => o,
            Err(_) => match title_obj.as_str() {
                Ok(_) => return self.build_outline_result(action.get(b"D")?, title_obj, named_destinations),
                Err(err) => return Err(err),
            },
        };
        self.build_outline_result(action.get(b"D")?, self.get_object(title_ref)?, named_destinations)
    }

    pub fn get_outlines(
        &self, node: Option<Object>, outlines: Option<Vec<Outline>>,
        named_destinations: &mut IndexMap<Vec<u8>, Destination>,
    ) -> Result<Option<Vec<Outline>>> {
        self.get_outlines_impl(node, outlines, named_destinations, 0)
    }

    fn get_outlines_impl(
        &self, mut node: Option<Object>, mut outlines: Option<Vec<Outline>>,
        named_destinations: &mut IndexMap<Vec<u8>, Destination>, depth: usize,
    ) -> Result<Option<Vec<Outline>>> {
        if depth >= crate::reader::MAX_NESTING_DEPTH {
            return Err(Error::RecursionLimit);
        }
        if outlines.is_none() {
            outlines = Some(Vec::new());
            let catalog = self.catalog()?;
            let mut dict_node = self.get_dict_in_dict(catalog, b"Outlines")?;
            let first = self.get_dict_in_dict(dict_node, b"First");
            if let Ok(first) = first {
                dict_node = first;
            }
            let mut tree = self.get_dict_in_dict(catalog, b"Dests");
            if tree.is_err() {
                let names = self.get_dict_in_dict(catalog, b"Names");
                if let Ok(names) = names {
                    let dests = self.get_dict_in_dict(names, b"Dests");
                    if dests.is_ok() {
                        tree = dests;
                    }
                }
            }
            if let Ok(tree) = tree {
                self.get_named_destinations(tree, named_destinations)?;
            }
            node = Some(Object::Dictionary(dict_node.clone()));
        }
        if node.is_none() {
            return Ok(outlines);
        }
        let node = node.unwrap();
        let mut node = match node.as_dict() {
            Ok(n) => n,
            Err(_) => self.get_object(node.as_reference()?)?.as_dict()?,
        };
        loop {
            if let Ok(Some(outline)) = self.get_outline(node, named_destinations)
                && let Some(ref mut outlines) = outlines
            {
                outlines.push(outline);
            }
            if let Ok(first) = node.get(b"First") {
                let sub_outlines = Vec::new();
                let sub_outlines =
                    self.get_outlines_impl(Some(first.clone()), Some(sub_outlines), named_destinations, depth + 1)?;
                if let Some(sub_outlines) = sub_outlines
                    && !sub_outlines.is_empty()
                    && let Some(ref mut outlines) = outlines
                {
                    outlines.push(Outline::SubOutlines(sub_outlines));
                }
            }
            node = match self.get_dict_in_dict(node, b"Next") {
                Ok(n) => n,
                Err(_) => break,
            };
        }
        Ok(outlines)
    }

    pub fn delete_outlines(&mut self) -> Result<()> {
        if let Some(outlines_id) = self
            .catalog()?
            .get(b"Outlines")
            .ok()
            .and_then(|v| v.as_reference().ok())
        {
            let outlines = self.get_dictionary(outlines_id)?;
            if let Ok(Object::Reference(first)) = outlines.get(b"First") {
                self.walk_outlines_del(*first, 0)?;
            }
            self.delete_object(outlines_id);
        }

        Ok(())
    }

    fn build_outline_result(
        &self, dest: &Object, title: &Object, named_destinations: &mut IndexMap<Vec<u8>, Destination>,
    ) -> Result<Option<Outline>> {
        let outline = match dest {
            Object::Array(obj_array) => Outline::Destination(Destination::new(
                title.to_owned(),
                obj_array[0].clone(),
                obj_array[1].clone(),
            )),
            Object::String(key, _fmt) => {
                if let Some(destination) = named_destinations.get_mut(key) {
                    destination.set(b"Title", title.to_owned());
                    Outline::Destination(destination.clone())
                } else {
                    return Ok(None);
                }
            }
            Object::Reference(object_id) => {
                return self.build_outline_result(self.get_object(*object_id)?, title, named_destinations);
            }
            _ => return Err(Error::InvalidOutline(format!("Unexpected destination {dest:?}"))),
        };
        Ok(Some(outline))
    }

    fn walk_outlines_del(&mut self, outline: ObjectId, depth: usize) -> Result<()> {
        if depth >= crate::reader::MAX_NESTING_DEPTH {
            return Err(Error::RecursionLimit);
        }
        let mut cur = Some(outline);

        while let Some(id) = cur.take() {
            let outline = self.get_dictionary(id)?;
            let first = outline.get(b"First").ok().and_then(|v| v.as_reference().ok());
            let next = outline.get(b"Next").ok().and_then(|v| v.as_reference().ok());

            if let Some(first) = first {
                self.walk_outlines_del(first, depth + 1)?;
            }
            cur = next;
            self.delete_object(id);
        }

        Ok(())
    }
}
