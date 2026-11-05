//! Tool definitions retain wire key order and explicit null fields for Qwen3's
//! Transformers `tojson` filter. Do not enable preserve_order workspace-wide:
//! legacy Coder rendering intentionally retains its existing sorted-map behavior.
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fmt;

#[derive(Debug)]
pub(super) enum OrderedJson {
    Scalar(Value),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}

impl<'de> Deserialize<'de> for OrderedJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct OrderedVisitor;
        impl<'de> Visitor<'de> for OrderedVisitor {
            type Value = OrderedJson;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON without duplicate object keys")
            }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(OrderedJson::Scalar(Value::Null))
            }
            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                self.visit_unit()
            }
            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(OrderedJson::Scalar(value.into()))
            }
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(OrderedJson::Scalar(value.into()))
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(OrderedJson::Scalar(value.into()))
            }
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                let number = serde_json::Number::from_f64(value)
                    .ok_or_else(|| E::custom("nonfinite JSON number"))?;
                Ok(OrderedJson::Scalar(Value::Number(number)))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(OrderedJson::Scalar(Value::String(value.into())))
            }
            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(OrderedJson::Scalar(Value::String(value)))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element()? {
                    values.push(value);
                }
                Ok(OrderedJson::Array(values))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                let mut keys = BTreeSet::new();
                while let Some((key, value)) = map.next_entry::<String, Self::Value>()? {
                    if !keys.insert(key.clone()) {
                        return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
                    }
                    values.push((key, value));
                }
                Ok(OrderedJson::Object(values))
            }
        }
        deserializer.deserialize_any(OrderedVisitor)
    }
}

impl OrderedJson {
    /// Preserve integer versus floating-point lexical kind. serde_json otherwise
    /// silently converts out-of-range integers (and integer -0) to f64. Refuse
    /// integers outside its exact i64/u64 range instead of changing schema bounds
    /// or generated arguments. RawValue checks syntax first; the final typed
    /// deserialization applies serde_json's recursion limit.
    pub(super) fn parse(text: &str) -> Result<Self, serde_json::Error> {
        let raw: Box<serde_json::value::RawValue> = serde_json::from_str(text)?;
        let bytes = raw.get().as_bytes();
        let mut normalized = String::new();
        let mut at = 0;
        let mut copied = 0;
        while at < bytes.len() {
            match bytes[at] {
                b'"' => {
                    at += 1;
                    while bytes[at] != b'"' {
                        if bytes[at] == b'\\' {
                            at += 1;
                        }
                        at += 1;
                    }
                    at += 1;
                }
                b'-' | b'0'..=b'9' => {
                    let start = at;
                    while at < bytes.len()
                        && matches!(bytes[at], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                    {
                        at += 1;
                    }
                    let token = &raw.get()[start..at];
                    if !token.contains(['.', 'e', 'E']) {
                        if token.parse::<i64>().is_err() && token.parse::<u64>().is_err() {
                            return Err(de::Error::custom(
                                "JSON integer is outside the exact i64/u64 range",
                            ));
                        }
                        if token == "-0" {
                            normalized.push_str(&raw.get()[copied..start]);
                            normalized.push('0');
                            copied = at;
                        }
                    }
                }
                _ => at += 1,
            }
        }
        normalized.push_str(&raw.get()[copied..]);
        serde_json::from_str(&normalized)
    }

    pub(super) fn into_value(self) -> Value {
        match self {
            Self::Scalar(value) => value,
            Self::Array(values) => Value::Array(values.into_iter().map(Self::into_value).collect()),
            Self::Object(values) => Value::Object(
                values
                    .into_iter()
                    .map(|(key, value)| (key, value.into_value()))
                    .collect(),
            ),
        }
    }

    /// Match Transformers v4.51.3: json.dumps(ensure_ascii=False,
    /// indent=None, separators=None, sort_keys=False). Python's scientific
    /// notation uses an explicit sign and at least two exponent digits.
    pub(super) fn to_python_json(&self) -> String {
        match self {
            Self::Scalar(Value::Number(number)) if number.is_f64() => {
                let value = format!("{:?}", number.as_f64().expect("float"));
                if let Some((mantissa, exponent)) = value.split_once('e') {
                    let exponent: i32 = exponent.parse().expect("float exponent");
                    format!("{mantissa}e{exponent:+03}")
                } else {
                    value
                }
            }
            Self::Scalar(value) => serde_json::to_string(value).expect("finite scalar"),
            Self::Array(values) => format!(
                "[{}]",
                values
                    .iter()
                    .map(Self::to_python_json)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Object(values) => format!(
                "{{{}}}",
                values
                    .iter()
                    .map(|(key, value)| format!(
                        "{}: {}",
                        serde_json::to_string(key).expect("string"),
                        value.to_python_json()
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}
