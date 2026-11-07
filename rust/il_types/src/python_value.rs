//! Owned scalar values at the Python/M boundary. Numeric values never pass
//! through JSON; JSON artifacts are a reversible diagnostic encoding only.
use serde::{Deserialize, Serialize, Serializer};

#[derive(Clone, Debug)]
pub enum PythonValue {
    Null,
    Bool(bool),
    Int(String),
    Float(f64),
    String(String),
    Unicode(Vec<u32>),
    List(Vec<Self>),
    Tuple(Vec<Self>),
    Dict(Vec<(Self, Self)>),
    Bytes(Vec<u8>),
    ByteArray(Vec<u8>),
    Set(Vec<Self>),
    FrozenSet(Vec<Self>),
}
impl Default for PythonValue {
    fn default() -> Self {
        Self::Null
    }
}
impl From<f64> for PythonValue {
    fn from(n: f64) -> Self {
        Self::Float(n)
    }
}
impl PythonValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(n) => Some(*n),
            Self::Int(s) => integer_float(s),
            _ => None,
        }
    }
    pub fn utf8_bytes(&self) -> usize {
        match self {
            Self::String(s) => s.len(),
            Self::Unicode(c) => c.len() * 4,
            Self::Int(_) => 0,
            Self::List(v) | Self::Tuple(v) | Self::Set(v) | Self::FrozenSet(v) => {
                v.iter().map(Self::utf8_bytes).sum()
            }
            Self::Dict(v) => v.iter().map(|(k, v)| k.utf8_bytes() + v.utf8_bytes()).sum(),
            _ => 0,
        }
    }
    pub fn artifact(&self) -> serde_json::Value {
        use serde_json::{Value, json};
        // Container tags cannot collide with user dictionaries: every dictionary
        // is encoded as ordered key/value pairs, even if its keys are strings.
        match self {
            Self::Null => Value::Null,
            Self::Bool(b) => json!(b),
            Self::String(s) => json!(s),
            Self::Int(s) => {
                if let Some(n) = integer_i64(s) {
                    json!(n)
                } else {
                    json!({"$py":"int","value":s})
                }
            }
            Self::Float(n) => json!({"$py":"float","bits":format!("{:016x}",n.to_bits())}),
            Self::Unicode(v) => json!({"$py":"unicode","value":v}),
            Self::List(v) => Value::Array(v.iter().map(Self::artifact).collect()),
            Self::Tuple(v) => {
                json!({"$py":"tuple","value":v.iter().map(Self::artifact).collect::<Vec<_>>()})
            }
            Self::Dict(v) => {
                json!({"$py":"dict","value":v.iter().map(|(k,v)|vec![k.artifact(),v.artifact()]).collect::<Vec<_>>()})
            }
            Self::Bytes(v) => json!({"$py":"bytes","value":v}),
            Self::ByteArray(v) => json!({"$py":"bytearray","value":v}),
            Self::Set(v) => {
                json!({"$py":"set","value":v.iter().map(Self::artifact).collect::<Vec<_>>()})
            }
            Self::FrozenSet(v) => {
                json!({"$py":"frozenset","value":v.iter().map(Self::artifact).collect::<Vec<_>>()})
            }
        }
    }
}
impl Serialize for PythonValue {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.artifact().serialize(s)
    }
}

fn integer_i64(s: &str) -> Option<i64> {
    let (negative, hex) = s
        .strip_prefix("-0x")
        .map(|s| (true, s))
        .or_else(|| s.strip_prefix("0x").map(|s| (false, s)))?;
    let magnitude = u64::from_str_radix(hex, 16).ok()?;
    if negative {
        i64::try_from(-(magnitude as i128)).ok()
    } else {
        i64::try_from(magnitude).ok()
    }
}
fn integer_float(s: &str) -> Option<f64> {
    // Decimal parsing supplies correctly rounded IEEE conversion once generated.
    // Python conversion is used only by writer's bridge-free numeric consumer;
    // decimal digits are produced by elementary base conversion, without bigint.
    let (negative, hex) = s
        .strip_prefix("-0x")
        .map(|s| (true, s))
        .or_else(|| s.strip_prefix("0x").map(|s| (false, s)))?;
    let mut digits = vec![0u8];
    for ch in hex.bytes() {
        let mut carry = (ch as char).to_digit(16)? as u32;
        for d in &mut digits {
            let n = *d as u32 * 16 + carry;
            *d = (n % 10) as u8;
            carry = n / 10;
        }
        while carry > 0 {
            digits.push((carry % 10) as u8);
            carry /= 10;
        }
    }
    let mut text = if negative {
        "-".to_string()
    } else {
        String::new()
    };
    text.extend(digits.into_iter().rev().map(|n| (b'0' + n) as char));
    text.parse::<f64>().ok().filter(|n| n.is_finite())
}

impl<'de> Deserialize<'de> for PythonValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(d)?;
        Self::from_artifact(v).map_err(serde::de::Error::custom)
    }
}
impl PythonValue {
    fn from_artifact(v: serde_json::Value) -> Result<Self, String> {
        use serde_json::Value as J;
        let values = |v: J| -> Result<Vec<Self>, String> {
            v.as_array()
                .ok_or("expected array")?
                .iter()
                .cloned()
                .map(Self::from_artifact)
                .collect()
        };
        Ok(match v {
            J::Null => Self::Null,
            J::Bool(b) => Self::Bool(b),
            J::String(s) => Self::String(s),
            J::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Self::Int(if i < 0 {
                        format!("-0x{:x}", i.unsigned_abs())
                    } else {
                        format!("{i:#x}")
                    })
                } else {
                    return Err("untagged numeric artifact".into());
                }
            }
            J::Array(v) => Self::List(
                v.into_iter()
                    .map(Self::from_artifact)
                    .collect::<Result<_, _>>()?,
            ),
            J::Object(mut m) => {
                let tag = m
                    .remove("$py")
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .ok_or("missing carrier tag")?;
                let v = m.remove("value").unwrap_or(J::Null);
                match tag.as_str() {
                    "int" => Self::Int(v.as_str().ok_or("expected integer spelling")?.into()),
                    "float" => {
                        let b = m.remove("bits").ok_or("missing bits")?;
                        Self::Float(f64::from_bits(
                            u64::from_str_radix(b.as_str().ok_or("expected bits string")?, 16)
                                .map_err(|e| e.to_string())?,
                        ))
                    }
                    "tuple" => Self::Tuple(values(v)?),
                    "set" => Self::Set(values(v)?),
                    "frozenset" => Self::FrozenSet(values(v)?),
                    "bytes" => Self::Bytes(serde_json::from_value(v).map_err(|e| e.to_string())?),
                    "bytearray" => {
                        Self::ByteArray(serde_json::from_value(v).map_err(|e| e.to_string())?)
                    }
                    "unicode" => {
                        Self::Unicode(serde_json::from_value(v).map_err(|e| e.to_string())?)
                    }
                    "dict" => {
                        let mut pairs = vec![];
                        for pair in v.as_array().ok_or("expected pairs")? {
                            let pair = pair.as_array().ok_or("expected pair")?;
                            if pair.len() != 2 {
                                return Err("expected two values".into());
                            }
                            pairs.push((
                                Self::from_artifact(pair[0].clone())?,
                                Self::from_artifact(pair[1].clone())?,
                            ));
                        }
                        Self::Dict(pairs)
                    }
                    _ => return Err(format!("unknown carrier tag {tag}")),
                }
            }
        })
    }
}
#[cfg(test)]
mod python_value_tests;
