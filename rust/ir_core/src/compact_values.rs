//! Lossless private storage for path scalars and immutable curve CTMs.
use super::PythonValue;
use serde::{Serialize, Serializer, ser::SerializeSeq};
use std::{borrow::Cow, ops::Deref, sync::Arc};

#[derive(Clone, Debug)]
pub enum PathOp {
    Empty,
    Move,
    Line,
    Curve,
    Close,
    Rectangle,
    Other(Arc<String>),
}
impl PathOp {
    fn common(s: &str) -> Option<Self> {
        Some(match s {
            "" => Self::Empty,
            "m" => Self::Move,
            "l" => Self::Line,
            "c" => Self::Curve,
            "h" => Self::Close,
            "re" => Self::Rectangle,
            _ => return None,
        })
    }
}
impl From<&str> for PathOp {
    fn from(s: &str) -> Self {
        Self::common(s).unwrap_or_else(|| Self::Other(Arc::new(s.to_owned())))
    }
}
impl From<String> for PathOp {
    fn from(s: String) -> Self {
        Self::common(&s).unwrap_or_else(|| Self::Other(Arc::new(s)))
    }
}
impl Deref for PathOp {
    type Target = str;
    fn deref(&self) -> &str {
        match self {
            Self::Empty => "",
            Self::Move => "m",
            Self::Line => "l",
            Self::Curve => "c",
            Self::Close => "h",
            Self::Rectangle => "re",
            Self::Other(s) => s.as_str(),
        }
    }
}
impl Serialize for PathOp {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self)
    }
}

#[derive(Default)]
pub struct DataPdfPath {
    coordinates: [u64; 2],
    pub op: Option<PathOp>,
    presence: u8,
}
impl DataPdfPath {
    pub fn new(x: Option<f64>, y: Option<f64>, op: Option<PathOp>, has_xy: Option<bool>) -> Self {
        Self {
            coordinates: [
                x.map(f64::to_bits).unwrap_or(0),
                y.map(f64::to_bits).unwrap_or(0),
            ],
            op,
            presence: u8::from(x.is_some())
                | (u8::from(y.is_some()) << 1)
                | (u8::from(has_xy.is_some()) << 2)
                | (u8::from(has_xy.unwrap_or(false)) << 3),
        }
    }
    pub fn x(&self) -> Option<f64> {
        (self.presence & 1 != 0).then(|| f64::from_bits(self.coordinates[0]))
    }
    pub fn y(&self) -> Option<f64> {
        (self.presence & 2 != 0).then(|| f64::from_bits(self.coordinates[1]))
    }
    pub fn has_xy(&self) -> Option<bool> {
        (self.presence & 4 != 0).then_some(self.presence & 8 != 0)
    }
}

#[derive(Clone, Debug)]
enum CtmValues {
    SixFloats([f64; 6]),
    General(Vec<PythonValue>),
}
/// Shared payloads are immutable; every write replaces the curve's complete value.
#[derive(Clone, Debug, Default)]
pub struct CurveCtm(Option<Arc<CtmValues>>);
impl CurveCtm {
    pub fn from_matrix(values: [f64; 6]) -> Self {
        Self(Some(Arc::new(CtmValues::SixFloats(values))))
    }
    pub fn len(&self) -> usize {
        match self.0.as_deref() {
            None => 0,
            Some(CtmValues::SixFloats(_)) => 6,
            Some(CtmValues::General(v)) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Cow<'_, PythonValue>> {
        (0..self.len()).map(|i| match self.0.as_deref().expect("nonempty CTM") {
            CtmValues::SixFloats(v) => Cow::Owned(PythonValue::Float(v[i])),
            CtmValues::General(v) => Cow::Borrowed(&v[i]),
        })
    }
    pub fn to_vec(&self) -> Vec<PythonValue> {
        self.iter().map(Cow::into_owned).collect()
    }
    pub fn utf8_bytes(&self) -> usize {
        match self.0.as_deref() {
            Some(CtmValues::General(v)) => v.iter().map(PythonValue::utf8_bytes).sum(),
            _ => 0,
        }
    }
}
impl From<Vec<PythonValue>> for CurveCtm {
    fn from(values: Vec<PythonValue>) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        if values.len() == 6 && values.iter().all(|v| matches!(v, PythonValue::Float(_))) {
            return Self::from_matrix(std::array::from_fn(|i| match &values[i] {
                PythonValue::Float(n) => *n,
                _ => unreachable!(),
            }));
        }
        Self(Some(Arc::new(CtmValues::General(values))))
    }
}
impl Serialize for CurveCtm {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for v in self.iter() {
            seq.serialize_element(v.as_ref())?;
        }
        seq.end()
    }
}

#[cfg(test)]
mod compact_values_tests;
