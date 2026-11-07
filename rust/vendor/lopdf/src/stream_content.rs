use std::ops::{Deref, Range};
use std::sync::Arc;

#[derive(Clone)]
pub struct StreamContent {
    storage: StreamStorage,
}

#[derive(Debug, Clone)]
enum StreamStorage {
    Owned(Vec<u8>),
    Shared { bytes: Arc<Vec<u8>>, range: Range<usize> },
}

impl StreamContent {
    pub fn as_slice(&self) -> &[u8] {
        self
    }
    pub fn len(&self) -> usize {
        self.as_ref().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Owned allocation capacity; shared ranges have no per-stream allocation.
    pub fn capacity(&self) -> usize {
        match &self.storage {
            StreamStorage::Owned(v) => v.capacity(),
            StreamStorage::Shared { .. } => 0,
        }
    }
    pub fn to_vec(&self) -> Vec<u8> {
        self.as_ref().to_vec()
    }
    fn make_mut(&mut self) -> &mut Vec<u8> {
        if let StreamStorage::Shared { bytes, range } = &self.storage {
            self.storage = StreamStorage::Owned(bytes[range.clone()].to_vec());
        }
        match &mut self.storage {
            StreamStorage::Owned(v) => v,
            StreamStorage::Shared { .. } => unreachable!(),
        }
    }
    pub(crate) fn set_shared(&mut self, bytes: Arc<Vec<u8>>, range: Range<usize>) {
        self.storage = StreamStorage::Shared { bytes, range };
    }
}

impl From<Vec<u8>> for StreamContent {
    fn from(value: Vec<u8>) -> Self {
        Self {
            storage: StreamStorage::Owned(value),
        }
    }
}
impl AsRef<[u8]> for StreamContent {
    fn as_ref(&self) -> &[u8] {
        self
    }
}
impl Deref for StreamContent {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match &self.storage {
            StreamStorage::Owned(v) => v,
            StreamStorage::Shared { bytes, range } => &bytes[range.clone()],
        }
    }
}
impl std::ops::DerefMut for StreamContent {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.make_mut().as_mut_slice()
    }
}
impl std::fmt::Debug for StreamContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Preserve the old Vec debug value; never print an entire backing PDF.
        self.as_slice().fmt(f)
    }
}
impl PartialEq for StreamContent {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl PartialEq<Vec<u8>> for StreamContent {
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.as_ref() == other.as_slice()
    }
}
impl<const N: usize> PartialEq<[u8; N]> for StreamContent {
    fn eq(&self, other: &[u8; N]) -> bool {
        self.as_ref() == other
    }
}
impl<const N: usize> PartialEq<&[u8; N]> for StreamContent {
    fn eq(&self, other: &&[u8; N]) -> bool {
        self.as_ref() == other.as_slice()
    }
}

#[cfg(test)]
mod stream_content_tests {
    use super::{StreamContent, StreamStorage};
    use std::sync::Arc;

    #[test]
    fn shared_range_clones_without_copy_and_mutation_detaches() {
        let input = Arc::new(b"prefix-image-suffix".to_vec());
        let mut content = StreamContent::from(b"image".to_vec());
        content.set_shared(input.clone(), 7..12);
        let clone = content.clone();
        assert_eq!(format!("{content:?}"), format!("{:?}", b"image".to_vec()));
        let mut other = StreamContent::from(Vec::new());
        other.set_shared(Arc::new(b"image".to_vec()), 0..5);
        assert_eq!(content, other);
        assert_eq!(content.as_slice(), b"image");
        assert_eq!(content.capacity(), 0);
        match &content.storage {
            StreamStorage::Shared { bytes, .. } => assert!(Arc::ptr_eq(bytes, &input)),
            StreamStorage::Owned(_) => panic!("expected shared range"),
        }
        content[0] = b'I';
        assert_eq!(content.as_slice(), b"Image");
        assert_eq!(clone.as_slice(), b"image");
        assert_eq!(&input[7..12], b"image");
        assert!(matches!(content.storage, StreamStorage::Owned(_)));
        assert_eq!(clone, StreamContent::from(b"image".to_vec()));
    }
}
