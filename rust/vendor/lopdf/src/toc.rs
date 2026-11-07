use indexmap::IndexMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{Document, Error, Object, ObjectId, Outline, Result};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TocType {
    pub level: usize,
    pub title: String,
    pub page: usize,
}

#[allow(dead_code)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct Toc {
    pub toc: Vec<TocType>,
    pub errors: Vec<String>,
}

impl Toc {
    pub fn new() -> Self {
        Toc {
            toc: Vec::new(),
            errors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Destination {
    map: IndexMap<Vec<u8>, Object>,
}

#[allow(dead_code)]
impl Destination {
    pub fn new(title: Object, page: Object, typ: Object) -> Self {
        let mut map = IndexMap::new();
        map.insert(b"Title".to_vec(), title);
        map.insert(b"Page".to_vec(), page);
        map.insert(b"Type".to_vec(), typ);
        Destination { map }
    }

    pub fn set(&mut self, key: Vec<u8>, value: Object) {
        self.map.insert(key, value);
    }

    pub fn title(&self) -> Option<&Object> {
        self.map.get(b"Title".as_slice())
    }

    pub fn page(&self) -> Option<&Object> {
        self.map.get(b"Page".as_slice())
    }
}

type OutlinePageIds = Vec<(Vec<u8>, ObjectId, usize)>;

fn setup_outline_page_ids<'a>(
    outlines: &'a Vec<Outline>, result: &mut OutlinePageIds, level: usize,
) -> Result<&'a Vec<Outline>> {
    for outline in outlines.iter() {
        match outline {
            Outline::Destination(destination) => {
                result.push((
                    destination.title()?.as_str()?.to_vec(),
                    destination.page()?.as_reference()?,
                    level,
                ));
            }
            Outline::SubOutlines(sub_outlines) => {
                setup_outline_page_ids(sub_outlines, result, level + 1)?;
            }
        }
    }
    Ok(outlines)
}

impl Document {
    fn setup_page_id_to_num(&self) -> IndexMap<(u32, u16), u32> {
        let mut result = IndexMap::new();
        for (page_num, page_id) in self.get_pages() {
            result.insert(page_id, page_num);
        }
        result
    }

    pub fn get_toc(&self) -> Result<Toc> {
        let mut toc: Toc = Toc {
            toc: Vec::new(),
            errors: Vec::new(),
        };
        let mut named_destinations = IndexMap::new();

        let Some(outlines) = self.get_outlines(None, None, &mut named_destinations)? else {
            return Err(Error::NoOutline);
        };

        let mut outline_page_ids = Vec::new();
        setup_outline_page_ids(&outlines, &mut outline_page_ids, 1)?;
        let page_id_to_page_numbers = self.setup_page_id_to_num();
        for (title, page_id, level) in outline_page_ids {
            if let Some(page_num) = page_id_to_page_numbers.get(&page_id) {
                let s = if title.len() < 2 {
                    String::from_utf8_lossy(&title).to_string()
                } else if title[0] == 0xfe && title[1] == 0xff {
                    if title.len() & 1 != 0 {
                        toc.errors
                            .push(format!("Title encoded UTF16_BE {title:?} has invalid length!"));
                        continue;
                    }
                    let t16: Vec<u16> = title
                        .chunks(2)
                        .skip(1)
                        .map(|x| ((x[0] as u16) << 8) | x[1] as u16)
                        .collect();
                    String::from_utf16_lossy(&t16)
                } else if title[0] == 0xff && title[1] == 0xfe {
                    if title.len() & 1 != 0 {
                        toc.errors
                            .push(format!("Title encoded UTF16_LE {title:?} has invalid length!"));
                        continue;
                    }
                    let t16: Vec<u16> = title
                        .chunks(2)
                        .skip(1)
                        .map(|x| ((x[1] as u16) << 8) | x[0] as u16)
                        .collect();
                    String::from_utf16_lossy(&t16)
                } else {
                    String::from_utf8_lossy(&title).to_string()
                };
                toc.toc.push(TocType {
                    level,
                    title: s,
                    page: *page_num as usize,
                });
            }
        }
        Ok(toc)
    }
}

#[cfg(not(feature = "async"))]
#[cfg(test)]
mod tests {
    use crate::{Document, TocType};

    #[test]
    fn parse_toc() {
        let expected = vec![
            TocType {
                level: 1,
                title: String::from("1. Flesh Fruits"),
                page: 1,
            },
            TocType {
                level: 2,
                title: String::from("1.1. Stone Fruits"),
                page: 2,
            },
            TocType {
                level: 3,
                title: String::from("1.1.1. Peaches"),
                page: 3,
            },
            TocType {
                level: 3,
                title: String::from("1.1.2. Plums"),
                page: 6,
            },
            TocType {
                level: 2,
                title: String::from("1.2. Pomes"),
                page: 28,
            },
            TocType {
                level: 3,
                title: String::from("1.2.1. Apples"),
                page: 30,
            },
            TocType {
                level: 3,
                title: String::from("1.2.2. Pears"),
                page: 35,
            },
            TocType {
                level: 2,
                title: String::from("Summary"),
                page: 36,
            },
            TocType {
                level: 1,
                title: String::from("2. Berries & Hesperidia"),
                page: 40,
            },
            TocType {
                level: 2,
                title: String::from("2.1. True Berries"),
                page: 40,
            },
            TocType {
                level: 2,
                title: String::from("Summary"),
                page: 41,
            },
            TocType {
                level: 1,
                title: String::from("3. The End"),
                page: 100,
            },
        ];

        let doc = Document::load("assets/test.pdf").unwrap();
        let toc = doc.get_toc().unwrap();
        assert_eq!(toc.toc, expected);
    }
}
