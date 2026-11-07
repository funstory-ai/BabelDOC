use super::{Dictionary, Document, Object, Result};
use indexmap::IndexMap;
#[derive(Debug, Clone)]
pub struct Destination(Dictionary);

impl Destination {
    pub fn new(title: Object, page: Object, typ: Object) -> Self {
        let mut dict = Dictionary::new();
        dict.set(b"Title", title);
        dict.set(b"Page", page);
        dict.set(b"Type", typ);
        Destination(dict)
    }

    pub fn set<K, V>(&mut self, key: K, value: V)
    where
        K: Into<Vec<u8>>,
        V: Into<Object>,
    {
        self.0.set(key, value);
    }

    pub fn title(&self) -> Result<&Object> {
        self.0.get(b"Title")
    }

    pub fn page(&self) -> Result<&Object> {
        self.0.get(b"Page")
    }
}

impl Document {
    pub fn get_named_destinations(
        &self, tree: &Dictionary, named_destinations: &mut IndexMap<Vec<u8>, Destination>,
    ) -> Result<()> {
        self.get_named_destinations_impl(tree, named_destinations, 0)
    }

    fn get_named_destinations_impl(
        &self, tree: &Dictionary, named_destinations: &mut IndexMap<Vec<u8>, Destination>, depth: usize,
    ) -> Result<()> {
        if depth >= crate::reader::MAX_NESTING_DEPTH {
            return Err(crate::Error::RecursionLimit);
        }
        if let Ok(kids) = tree.get(b"Kids") {
            for kid in kids.as_array()? {
                if let Ok(kid) = kid.as_reference().and_then(move |id| self.get_dictionary(id)) {
                    self.get_named_destinations_impl(kid, named_destinations, depth + 1)?;
                }
            }
        }
        if let Ok(names) = tree.get(b"Names") {
            let mut names = names.as_array()?.iter();
            loop {
                let key = names.next();
                if key.is_none() {
                    break;
                }
                let val = names.next();
                if val.is_none() {
                    break;
                }
                if let Ok(obj_ref) = val.unwrap().as_reference() {
                    if let Ok(dict) = self.get_dictionary(obj_ref) {
                        let val = dict.get(b"D").as_ref().unwrap().as_array()?;
                        let dest = Destination::new(key.unwrap().clone(), val[0].clone(), val[1].clone());
                        named_destinations.insert(key.unwrap().as_str().unwrap().to_vec(), dest);
                    } else if let Ok(Object::Array(val)) = self.get_object(obj_ref) {
                        let dest = Destination::new(key.unwrap().clone(), val[0].clone(), val[1].clone());
                        named_destinations.insert(key.unwrap().as_str().unwrap().to_vec(), dest);
                    }
                } else if let Ok(dict) = val.unwrap().as_dict() {
                    let val = dict.get(b"D").as_ref().unwrap().as_array()?;
                    let dest = Destination::new(key.unwrap().clone(), val[0].clone(), val[1].clone());
                    named_destinations.insert(key.unwrap().as_str().unwrap().to_vec(), dest);
                } else {
                    // TODO: Log error: Unpexpected node type
                }
            }
        }
        Ok(())
    }
}
