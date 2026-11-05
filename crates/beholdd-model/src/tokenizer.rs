//! Qwen-family GPT-2 byte-level BPE tokenizers as stored in GGUF metadata.
//!
//! `qwen2` and `qwen35` use distinct pretokenizers. Qwen3.5 joins Unicode combining
//! marks to letter runs; it is not an alias for the older segmentation rule.
//!
//! Token pieces are bytes, not independently valid UTF-8 strings. In particular, one
//! Unicode scalar may be split across several tokens. Use [`Qwen2Tokenizer::decode_bytes`]
//! while streaming, and validate UTF-8 only after complete pieces have been joined.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Context, Result};
use beholdd_format::gguf::{Gguf, Value};
use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

const TOKEN_NORMAL: u64 = 1;
const TOKEN_UNKNOWN: u64 = 2;
const TOKEN_CONTROL: u64 = 3;
const TOKEN_USER_DEFINED: u64 = 4;
const TOKEN_UNUSED: u64 = 5;
const TOKEN_BYTE: u64 = 6;
const MAX_SPECIAL_TOKENS: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

#[derive(Debug)]
struct Token {
    text: String,
    kind: TokenKind,
}

#[derive(Debug)]
struct SpecialToken {
    text: String,
    id: u32,
    kind: TokenKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreTokenizer {
    Qwen2,
    Qwen35,
}

/// A Qwen GPT-2 byte-level BPE tokenizer loaded from GGUF metadata.
///
/// The historical type name is retained; the exact `qwen2` or `qwen35` pretokenizer
/// is selected by metadata. No Unicode normalization is introduced: the pinned
/// llama.cpp GGUF BPE path splits the original text, unlike some HF JSON tokenizers.
///
/// `encode` obeys the checkpoint's `add_bos_token` and `add_eos_token` flags. The
/// Qwen3-Coder checkpoint used by beholdd sets `add_bos_token = false`, has no usable
/// BOS ID, and declares `<|im_end|>` (151645) as EOS.
#[derive(Debug)]
pub struct Qwen2Tokenizer {
    pretokenizer: PreTokenizer,
    tokens: Vec<Token>,
    token_to_id: HashMap<String, u32>,
    merge_ranks: HashMap<(u32, u32), usize>,
    specials: Vec<SpecialToken>,
    bos_token_id: Option<u32>,
    eos_token_id: u32,
    padding_token_id: Option<u32>,
    add_bos_token: bool,
    add_eos_token: bool,
    chat_template: Option<String>,
}

impl Qwen2Tokenizer {
    /// Load the tokenizer vocabulary, merge ranks, and token policy from a GGUF.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let meta = gguf.metadata();
        let model = required_str(meta, "tokenizer.ggml.model")?;
        if model != "gpt2" {
            bail!("unsupported tokenizer.ggml.model {model:?}; expected \"gpt2\"");
        }
        let pre = required_str(meta, "tokenizer.ggml.pre")?;
        let pretokenizer = match pre {
            "qwen2" => PreTokenizer::Qwen2,
            "qwen35" => PreTokenizer::Qwen35,
            _ => bail!("unsupported tokenizer.ggml.pre {pre:?}; expected \"qwen2\" or \"qwen35\""),
        };

        let token_values = required_array(meta, "tokenizer.ggml.tokens")?;
        let type_values = required_array(meta, "tokenizer.ggml.token_type")?;
        if token_values.len() != type_values.len() {
            bail!(
                "tokenizer has {} token strings but {} token types",
                token_values.len(),
                type_values.len()
            );
        }
        let _ = u32::try_from(token_values.len())
            .context("tokenizer vocabulary has more than u32::MAX entries")?;

        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(token_values.len())
            .map_err(|error| anyhow!("cannot allocate tokenizer vocabulary: {error}"))?;
        let mut token_to_id = HashMap::new();
        token_to_id
            .try_reserve(token_values.len())
            .map_err(|error| anyhow!("cannot allocate tokenizer token index: {error}"))?;
        let mut specials = Vec::new();
        specials
            .try_reserve(token_values.len().min(256))
            .map_err(|error| anyhow!("cannot allocate tokenizer special-token index: {error}"))?;

        for (index, (text_value, type_value)) in
            token_values.iter().zip(type_values.iter()).enumerate()
        {
            let text = value_str(text_value, "tokenizer.ggml.tokens", index)?;
            let token_type = type_value.as_u64().ok_or_else(|| {
                anyhow!("tokenizer.ggml.token_type[{index}] is not a non-negative integer")
            })?;
            let id = u32::try_from(index).context("token ID does not fit u32")?;
            let kind = match token_type {
                TOKEN_NORMAL => TokenKind::Normal,
                TOKEN_UNKNOWN => TokenKind::Unknown,
                TOKEN_CONTROL => TokenKind::Control,
                TOKEN_USER_DEFINED => TokenKind::UserDefined,
                TOKEN_UNUSED => TokenKind::Unused,
                TOKEN_BYTE => TokenKind::Byte,
                0 => TokenKind::Unused,
                _ => bail!("unsupported GGUF token type {token_type} at ID {id}"),
            };
            let owned = clone_string(text, "token text")?;
            if token_to_id.insert(owned.clone(), id).is_some() {
                bail!("duplicate tokenizer token text at ID {id}: {text:?}");
            }
            if matches!(
                kind,
                TokenKind::Unknown | TokenKind::Control | TokenKind::UserDefined
            ) {
                if specials.len() == MAX_SPECIAL_TOKENS {
                    bail!(
                        "tokenizer has more than {MAX_SPECIAL_TOKENS} special tokens; refusing pathological partition cost"
                    );
                }
                if text.is_empty() {
                    bail!("special token ID {id} has empty text");
                }
                specials.push(SpecialToken {
                    text: clone_string(text, "special token text")?,
                    id,
                    kind,
                });
            }
            tokens.push(Token { text: owned, kind });
        }
        specials.sort_unstable_by(|left, right| {
            right
                .text
                .len()
                .cmp(&left.text.len())
                .then_with(|| left.id.cmp(&right.id))
        });

        let merge_values = required_array(meta, "tokenizer.ggml.merges")?;
        let mut merge_ranks = HashMap::new();
        merge_ranks
            .try_reserve(merge_values.len())
            .map_err(|error| anyhow!("cannot allocate tokenizer merge table: {error}"))?;
        for (rank, value) in merge_values.iter().enumerate() {
            let merge = value_str(value, "tokenizer.ggml.merges", rank)?;
            let (left, right) = merge.split_once(' ').ok_or_else(|| {
                anyhow!("tokenizer.ggml.merges[{rank}] is not a two-token merge: {merge:?}")
            })?;
            if left.is_empty() || right.is_empty() || right.contains(' ') {
                bail!("tokenizer.ggml.merges[{rank}] is not a two-token merge: {merge:?}");
            }
            let &left_id = token_to_id
                .get(left)
                .ok_or_else(|| anyhow!("merge {rank} references unknown left token {left:?}"))?;
            let &right_id = token_to_id
                .get(right)
                .ok_or_else(|| anyhow!("merge {rank} references unknown right token {right:?}"))?;
            let merged = format!("{left}{right}");
            if !token_to_id.contains_key(&merged) {
                bail!("merge {rank} produces unknown token {merged:?}");
            }
            if merge_ranks.insert((left_id, right_id), rank).is_some() {
                bail!("duplicate tokenizer merge pair at rank {rank}: {merge:?}");
            }
        }

        let eos_token_id = required_token_id(meta, "tokenizer.ggml.eos_token_id", tokens.len())?;
        let bos_token_id = optional_token_id(meta, "tokenizer.ggml.bos_token_id", tokens.len())?;
        let padding_token_id =
            optional_token_id(meta, "tokenizer.ggml.padding_token_id", tokens.len())?;
        let add_bos_token = optional_bool(meta, "tokenizer.ggml.add_bos_token")?.unwrap_or(false);
        let add_eos_token = optional_bool(meta, "tokenizer.ggml.add_eos_token")?.unwrap_or(false);
        if add_bos_token && bos_token_id.is_none() {
            bail!("tokenizer requests a BOS token but has no tokenizer.ggml.bos_token_id");
        }
        let chat_template = meta
            .get("tokenizer.chat_template")
            .map(|value| {
                let template = value
                    .as_str()
                    .ok_or_else(|| anyhow!("tokenizer.chat_template is not a string"))?;
                clone_string(template, "chat template")
            })
            .transpose()?;

        Ok(Self {
            pretokenizer,
            tokens,
            token_to_id,
            merge_ranks,
            specials,
            bos_token_id,
            eos_token_id,
            padding_token_id,
            add_bos_token,
            add_eos_token,
            chat_template,
        })
    }

    /// Encode UTF-8 text into token IDs.
    ///
    /// If `parse_special` is true, control-token spellings such as `<|im_start|>` are
    /// recognized as one token. User-defined tokens are recognized in either mode, as
    /// required by the GGUF token-type contract and llama.cpp behavior.
    pub fn encode(&self, text: &str, parse_special: bool) -> Result<Vec<u32>> {
        let mut output = Vec::new();
        output
            .try_reserve(text.len().saturating_add(2))
            .map_err(|error| anyhow!("cannot allocate token output: {error}"))?;
        if self.add_bos_token {
            output.push(self.bos_token_id.expect("validated BOS metadata"));
        }

        let mut cursor = 0;
        while cursor < text.len() {
            let next = self.next_special(&text[cursor..], parse_special);
            let Some((relative, special)) = next else {
                self.encode_plain(&text[cursor..], &mut output)?;
                break;
            };
            let position = cursor
                .checked_add(relative)
                .context("special-token position overflow")?;
            self.encode_plain(&text[cursor..position], &mut output)?;
            output.push(special.id);
            cursor = position
                .checked_add(special.text.len())
                .context("special-token end overflow")?;
        }
        if self.add_eos_token {
            output.push(self.eos_token_id);
        }
        Ok(output)
    }

    /// Decode token IDs to their original bytes.
    ///
    /// With `render_special = false`, control and unknown tokens (including EOS) are
    /// omitted. User-defined tokens remain literal in both modes. Undefined and unused
    /// vocabulary entries never render.
    pub fn decode_bytes(&self, ids: &[u32], render_special: bool) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for &id in ids {
            let token = self
                .tokens
                .get(usize::try_from(id).context("token ID does not fit usize")?)
                .ok_or_else(|| anyhow!("token ID {id} is outside the vocabulary"))?;
            match token.kind {
                TokenKind::Normal => {
                    bytes
                        .try_reserve(token.text.len())
                        .map_err(|error| anyhow!("cannot allocate decoded bytes: {error}"))?;
                    for character in token.text.chars() {
                        let byte = unicode_to_byte(character).ok_or_else(|| {
                            anyhow!(
                                "normal token ID {id} contains non-GPT-2 character {character:?}"
                            )
                        })?;
                        bytes.push(byte);
                    }
                }
                TokenKind::Byte => bytes.push(parse_byte_token(&token.text, id)?),
                TokenKind::UserDefined => append_utf8(&mut bytes, &token.text)?,
                TokenKind::Unknown | TokenKind::Control if render_special => {
                    append_utf8(&mut bytes, &token.text)?;
                }
                TokenKind::Unknown | TokenKind::Control | TokenKind::Unused => {}
            }
        }
        Ok(bytes)
    }

    /// Decode token IDs and require the complete result to be valid UTF-8.
    pub fn decode(&self, ids: &[u32], render_special: bool) -> Result<String> {
        String::from_utf8(self.decode_bytes(ids, render_special)?)
            .context("decoded token sequence is not valid UTF-8")
    }

    /// Number of entries in the vocabulary.
    pub fn vocab_len(&self) -> usize {
        self.tokens.len()
    }

    /// Exact GGUF pretokenizer profile, useful when reporting tokenizer provenance.
    pub fn pretokenizer(&self) -> &'static str {
        match self.pretokenizer {
            PreTokenizer::Qwen2 => "qwen2",
            PreTokenizer::Qwen35 => "qwen35",
        }
    }

    /// Optional BOS token ID declared by the checkpoint.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// EOS token ID declared by the checkpoint.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Optional padding token ID declared by the checkpoint.
    pub fn padding_token_id(&self) -> Option<u32> {
        self.padding_token_id
    }

    /// Whether the checkpoint requests automatic BOS insertion.
    pub fn adds_bos_token(&self) -> bool {
        self.add_bos_token
    }

    /// Whether the checkpoint requests automatic EOS insertion.
    pub fn adds_eos_token(&self) -> bool {
        self.add_eos_token
    }

    /// The checkpoint's raw Jinja chat template, if present.
    ///
    /// This crate does not interpret Jinja. Callers needing tools or arbitrary chat
    /// template behavior must render this template before passing the result to
    /// [`Self::encode`] with `parse_special = true`.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    fn next_special<'a>(
        &'a self,
        text: &str,
        parse_special: bool,
    ) -> Option<(usize, &'a SpecialToken)> {
        let mut best: Option<(usize, &SpecialToken)> = None;
        for special in &self.specials {
            if !parse_special && matches!(special.kind, TokenKind::Unknown | TokenKind::Control) {
                continue;
            }
            let Some(position) = text.find(&special.text) else {
                continue;
            };
            match best {
                None => best = Some((position, special)),
                Some((best_position, best_special))
                    if position < best_position
                        || (position == best_position
                            && special.text.len() > best_special.text.len()) =>
                {
                    best = Some((position, special));
                }
                _ => {}
            }
        }
        best
    }

    fn encode_plain(&self, text: &str, output: &mut Vec<u32>) -> Result<()> {
        let words = match self.pretokenizer {
            PreTokenizer::Qwen2 => qwen2_words(text),
            PreTokenizer::Qwen35 => qwen35_words(text),
        };
        for word in words {
            let byte_map = byte_to_unicode();
            let mut symbols = Vec::new();
            symbols
                .try_reserve(word.len())
                .map_err(|error| anyhow!("cannot allocate BPE symbols: {error}"))?;
            for &byte in word.as_bytes() {
                let character = byte_map[usize::from(byte)];
                let mut encoded = String::new();
                encoded.push(character);
                let id = *self.token_to_id.get(&encoded).ok_or_else(|| {
                    anyhow!("tokenizer vocabulary is missing byte token {byte:#04x}")
                })?;
                symbols.push(Symbol { text: encoded, id });
            }

            loop {
                let mut best: Option<(usize, usize)> = None;
                for index in 0..symbols.len().saturating_sub(1) {
                    let Some(&rank) = self
                        .merge_ranks
                        .get(&(symbols[index].id, symbols[index + 1].id))
                    else {
                        continue;
                    };
                    if best.is_none_or(|(best_rank, best_index)| {
                        rank < best_rank || (rank == best_rank && index < best_index)
                    }) {
                        best = Some((rank, index));
                    }
                }
                let Some((_, index)) = best else {
                    break;
                };
                let right = symbols.remove(index + 1);
                symbols[index]
                    .text
                    .try_reserve(right.text.len())
                    .map_err(|error| anyhow!("cannot allocate merged BPE token: {error}"))?;
                symbols[index].text.push_str(&right.text);
                symbols[index].id =
                    *self.token_to_id.get(&symbols[index].text).ok_or_else(|| {
                        anyhow!("BPE merge produced unknown token {:?}", symbols[index].text)
                    })?;
            }
            output
                .try_reserve(symbols.len())
                .map_err(|error| anyhow!("cannot grow token output: {error}"))?;
            output.extend(symbols.into_iter().map(|symbol| symbol.id));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Symbol {
    text: String,
    id: u32,
}

fn required_str<'a>(
    meta: &'a std::collections::BTreeMap<String, Value>,
    key: &str,
) -> Result<&'a str> {
    meta.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing or non-string GGUF metadata {key}"))
}

fn required_array<'a>(
    meta: &'a std::collections::BTreeMap<String, Value>,
    key: &str,
) -> Result<&'a [Value]> {
    meta.get(key)
        .and_then(Value::as_arr)
        .ok_or_else(|| anyhow!("missing or non-array GGUF metadata {key}"))
}

fn value_str<'a>(value: &'a Value, key: &str, index: usize) -> Result<&'a str> {
    value
        .as_str()
        .ok_or_else(|| anyhow!("{key}[{index}] is not a string"))
}

fn required_token_id(
    meta: &std::collections::BTreeMap<String, Value>,
    key: &str,
    vocab_len: usize,
) -> Result<u32> {
    optional_token_id(meta, key, vocab_len)?.ok_or_else(|| anyhow!("missing GGUF metadata {key}"))
}

fn optional_token_id(
    meta: &std::collections::BTreeMap<String, Value>,
    key: &str,
    vocab_len: usize,
) -> Result<Option<u32>> {
    let Some(value) = meta.get(key) else {
        return Ok(None);
    };
    let raw = value
        .as_u64()
        .ok_or_else(|| anyhow!("GGUF metadata {key} is not a non-negative integer"))?;
    let id = u32::try_from(raw).with_context(|| format!("GGUF metadata {key} does not fit u32"))?;
    if usize::try_from(id).context("token ID does not fit usize")? >= vocab_len {
        bail!("GGUF metadata {key}={id} is outside vocabulary length {vocab_len}");
    }
    Ok(Some(id))
}

fn optional_bool(
    meta: &std::collections::BTreeMap<String, Value>,
    key: &str,
) -> Result<Option<bool>> {
    match meta.get(key) {
        None => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => bail!("GGUF metadata {key} is not a boolean"),
    }
}

fn clone_string(value: &str, description: &str) -> Result<String> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(value.len())
        .map_err(|error| anyhow!("cannot allocate {description}: {error}"))?;
    cloned.push_str(value);
    Ok(cloned)
}

fn append_utf8(output: &mut Vec<u8>, value: &str) -> Result<()> {
    output
        .try_reserve(value.len())
        .map_err(|error| anyhow!("cannot allocate decoded bytes: {error}"))?;
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn parse_byte_token(text: &str, id: u32) -> Result<u8> {
    let digits = text
        .strip_prefix("<0x")
        .and_then(|rest| rest.strip_suffix('>'))
        .filter(|digits| digits.len() == 2)
        .with_context(|| format!("byte token ID {id} is not written as <0xXX>: {text:?}"))?;
    u8::from_str_radix(digits, 16)
        .with_context(|| format!("byte token ID {id} has invalid hexadecimal text {text:?}"))
}

fn byte_to_unicode() -> &'static [char; 256] {
    static MAP: OnceLock<[char; 256]> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = ['\0'; 256];
        for byte in 0x21u8..=0x7e {
            map[usize::from(byte)] = char::from(byte);
        }
        for byte in 0xa1u8..=0xac {
            map[usize::from(byte)] = char::from(byte);
        }
        for byte in 0xaeu8..=0xff {
            map[usize::from(byte)] = char::from(byte);
        }
        let mut extra = 0u32;
        for byte in 0u8..=u8::MAX {
            if map[usize::from(byte)] == '\0' {
                map[usize::from(byte)] =
                    char::from_u32(0x100 + extra).expect("GPT-2 byte map is valid Unicode");
                extra += 1;
            }
        }
        map
    })
}

fn unicode_to_byte(character: char) -> Option<u8> {
    byte_to_unicode()
        .iter()
        .position(|&candidate| candidate == character)
        .and_then(|index| u8::try_from(index).ok())
}

fn is_letter(character: char) -> bool {
    character.general_category_group() == GeneralCategoryGroup::Letter
}

fn is_number(character: char) -> bool {
    character.general_category_group() == GeneralCategoryGroup::Number
}

fn qwen2_words(text: &str) -> Vec<&str> {
    qwen_words(text, false)
}

fn qwen35_words(text: &str) -> Vec<&str> {
    qwen_words(text, true)
}

/// Exact ordered alternatives from llama.cpp b11223 (4da633776), llama-vocab.cpp.
/// Qwen35 changes the letter-run body from L to L|M and excludes M from punctuation.
/// The optional prefix still excludes only CR, LF, L and N, not M. Contractions are
/// explicitly ASCII case-insensitive; numbers remain individual Unicode N scalars.
fn qwen_words(text: &str, include_marks: bool) -> Vec<&str> {
    let is_word = |character: char| {
        is_letter(character)
            || (include_marks && character.general_category_group() == GeneralCategoryGroup::Mark)
    };
    let characters: Vec<(usize, char)> = text.char_indices().collect();
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut position = 0usize;

    let end_byte = |index: usize| characters.get(index).map_or(text.len(), |&(byte, _)| byte);
    let character = |index: usize| characters.get(index).map(|&(_, value)| value);
    let push = |end: usize, start: &mut usize, ranges: &mut Vec<(usize, usize)>| {
        let end = end_byte(end);
        if *start < end {
            ranges.push((*start, end));
        }
        *start = end;
    };

    while position < characters.len() {
        let current = characters[position].1;

        if current == '\'' {
            if let Some(next) = character(position + 1).map(|value| value.to_ascii_lowercase()) {
                if matches!(next, 's' | 't' | 'm' | 'd') {
                    position += 2;
                    push(position, &mut start, &mut ranges);
                    continue;
                }
                if let Some(after) = character(position + 2).map(|value| value.to_ascii_lowercase())
                {
                    if matches!((next, after), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                        position += 3;
                        push(position, &mut start, &mut ranges);
                        continue;
                    }
                }
            }
        }

        if current != '\r'
            && current != '\n'
            && !is_number(current)
            && (is_word(current) || character(position + 1).is_some_and(is_word))
        {
            position += 1;
            while character(position).is_some_and(is_word) {
                position += 1;
            }
            push(position, &mut start, &mut ranges);
            continue;
        }

        if is_number(current) {
            position += 1;
            push(position, &mut start, &mut ranges);
            continue;
        }

        let first_non_space = if current == ' ' {
            character(position + 1)
        } else {
            Some(current)
        };
        if first_non_space
            .is_some_and(|value| !value.is_whitespace() && !is_word(value) && !is_number(value))
        {
            if current == ' ' {
                position += 1;
            }
            while character(position)
                .is_some_and(|value| !value.is_whitespace() && !is_word(value) && !is_number(value))
            {
                position += 1;
            }
            while matches!(character(position), Some('\r' | '\n')) {
                position += 1;
            }
            push(position, &mut start, &mut ranges);
            continue;
        }

        let mut whitespace_count = 0usize;
        let mut last_newline_end = None;
        while let Some(value) =
            character(position + whitespace_count).filter(|value| value.is_whitespace())
        {
            whitespace_count += 1;
            if matches!(value, '\r' | '\n') {
                last_newline_end = position.checked_add(whitespace_count);
            }
        }
        if let Some(end) = last_newline_end {
            position = end;
            push(position, &mut start, &mut ranges);
            continue;
        }
        if whitespace_count > 1 && position + whitespace_count < characters.len() {
            position += whitespace_count - 1;
            push(position, &mut start, &mut ranges);
            continue;
        }
        if whitespace_count > 0 {
            position += whitespace_count;
            push(position, &mut start, &mut ranges);
            continue;
        }

        position += 1;
        push(position, &mut start, &mut ranges);
    }
    ranges
        .into_iter()
        .map(|(begin, end)| &text[begin..end])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TokenizerFixture {
        path: std::path::PathBuf,
    }

    impl Drop for TokenizerFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    impl TokenizerFixture {
        fn build() -> Result<Self> {
            Self::with_profile("qwen2", &[])
        }

        fn with_profile(pre: &str, extra_merges: &[(String, String)]) -> Result<Self> {
            static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

            let mut tokens = byte_to_unicode()
                .iter()
                .map(char::to_string)
                .collect::<Vec<_>>();
            tokens.push("ab".to_owned());
            let mut merges = vec!["a b".to_owned()];
            for (left, right) in extra_merges {
                tokens.push(format!("{left}{right}"));
                merges.push(format!("{left} {right}"));
            }
            tokens.push("<eos>".to_owned());
            let mut token_types = vec![u32::try_from(TOKEN_NORMAL)?; tokens.len()];
            *token_types
                .last_mut()
                .context("fixture vocabulary is empty")? = u32::try_from(TOKEN_CONTROL)?;
            let eos = u32::try_from(tokens.len() - 1)?;

            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"GGUF");
            bytes.extend_from_slice(&3u32.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
            bytes.extend_from_slice(&10u64.to_le_bytes());
            push_string_metadata(&mut bytes, "tokenizer.ggml.model", "gpt2");
            push_string_metadata(&mut bytes, "tokenizer.ggml.pre", pre);
            push_string_array_metadata(&mut bytes, "tokenizer.ggml.tokens", &tokens);
            push_u32_array_metadata(&mut bytes, "tokenizer.ggml.token_type", &token_types);
            push_string_array_metadata(&mut bytes, "tokenizer.ggml.merges", &merges);
            push_u32_metadata(&mut bytes, "tokenizer.ggml.eos_token_id", eos);
            push_u32_metadata(&mut bytes, "tokenizer.ggml.padding_token_id", eos);
            push_bool_metadata(&mut bytes, "tokenizer.ggml.add_bos_token", false);
            push_bool_metadata(&mut bytes, "tokenizer.ggml.add_eos_token", true);
            push_string_metadata(
                &mut bytes,
                "tokenizer.chat_template",
                "fixture {{ messages }}",
            );

            let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "beholdd-tokenizer-{}-{id}.gguf",
                std::process::id()
            ));
            std::fs::write(&path, bytes)?;
            Ok(Self { path })
        }
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_key(bytes: &mut Vec<u8>, key: &str, value_type: u32) {
        push_string(bytes, key);
        bytes.extend_from_slice(&value_type.to_le_bytes());
    }

    fn push_string_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_key(bytes, key, 8);
        push_string(bytes, value);
    }

    fn push_u32_metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_key(bytes, key, 4);
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_bool_metadata(bytes: &mut Vec<u8>, key: &str, value: bool) {
        push_key(bytes, key, 7);
        bytes.push(u8::from(value));
    }

    fn push_string_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[String]) {
        push_key(bytes, key, 9);
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            push_string(bytes, value);
        }
    }

    fn push_u32_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[u32]) {
        push_key(bytes, key, 9);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    #[test]
    fn gpt2_byte_map_round_trips_every_byte() {
        for byte in 0u8..=u8::MAX {
            assert_eq!(
                unicode_to_byte(byte_to_unicode()[usize::from(byte)]),
                Some(byte)
            );
        }
    }

    #[test]
    fn qwen2_pretokenizer_matches_reference_boundaries() {
        assert_eq!(
            qwen2_words("I'm can't won't we're I'd you've shouldn't've"),
            [
                "I", "'m", " can", "'t", " won", "'t", " we", "'re", " I", "'d", " you", "'ve",
                " shouldn", "'t", "'ve"
            ]
        );
        assert_eq!(
            qwen2_words(" leading and  multiple spaces\tline\n"),
            [
                " leading",
                " and",
                " ",
                " multiple",
                " spaces",
                "\tline",
                "\n"
            ]
        );
        assert_eq!(qwen2_words("012345"), ["0", "1", "2", "3", "4", "5"]);
        assert_eq!(
            qwen2_words("line1\r\nline2\n\n"),
            ["line", "1", "\r\n", "line", "2", "\n\n"]
        );
    }

    /// Goldens were generated by an independent regex engine from the pinned llama.cpp
    /// expressions, not by this scanner. Includes Mn/Mc/Me marks and decomposed input.
    #[test]
    fn both_pretokenizers_match_independent_unicode_boundary_goldens() -> Result<()> {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("tokenizer_fixtures/qwen35_oracle.json"))?;
        for case in fixture["cases"].as_array().context("oracle cases")? {
            let text = case["text"].as_str().context("oracle text")?;
            for (key, actual) in [
                ("qwen2Words", qwen2_words(text)),
                ("qwen35Words", qwen35_words(text)),
            ] {
                let expected: Vec<&str> = case[key]
                    .as_array()
                    .context("oracle words")?
                    .iter()
                    .map(|v| v.as_str().expect("oracle string"))
                    .collect();
                assert_eq!(actual, expected, "{key}: {text:?}");
                assert_eq!(
                    actual.concat(),
                    text,
                    "segmentation must preserve all input bytes"
                );
            }
        }
        assert_eq!(qwen2_words("a\u{301}b"), ["a", "\u{301}b"]);
        assert_eq!(qwen35_words("a\u{301}b"), ["a\u{301}b"]);
        Ok(())
    }

    #[test]
    fn qwen35_bpe_can_merge_across_marks_without_changing_qwen2_or_normalizing() -> Result<()> {
        let first = byte_to_unicode()[0xcc].to_string();
        let second = byte_to_unicode()[0x81].to_string();
        let merges = [
            ("e".to_owned(), first.clone()),
            (format!("e{first}"), second),
        ];
        for pre in ["qwen2", "qwen35"] {
            let file = TokenizerFixture::with_profile(pre, &merges)?;
            let tokenizer = Qwen2Tokenizer::from_gguf(&Gguf::open(&file.path)?)?;
            assert_eq!(tokenizer.pretokenizer(), pre);
            let expected = if pre == "qwen35" {
                vec![258, 259]
            } else {
                vec![101, 204, 129, 259]
            };
            assert_eq!(tokenizer.encode("e\u{301}", false)?, expected);
            assert_eq!(tokenizer.decode(&expected, false)?, "e\u{301}");
            assert_eq!(tokenizer.encode("é", false)?, [195, 169, 259]);
            assert_eq!(tokenizer.encode("<eos>", true)?, [259, 259]);
            assert_ne!(tokenizer.encode("<eos>", false)?, [259, 259]);
        }
        let file = TokenizerFixture::with_profile("qwen35-unknown", &[])?;
        assert!(Qwen2Tokenizer::from_gguf(&Gguf::open(&file.path)?)
            .unwrap_err()
            .to_string()
            .contains("unsupported tokenizer.ggml.pre"));
        Ok(())
    }

    /// Uses only tokenizer metadata from the GGUF. Expected IDs come from the independent
    /// regex/BPE oracle script, which never reads weights or creates an inference context.
    #[test]
    #[ignore = "requires BEHOLDD_QWEN35_MODEL; reads only tokenizer metadata"]
    fn real_qwen35_matches_independent_gguf_token_id_goldens() -> Result<()> {
        let path = std::path::PathBuf::from(
            std::env::var_os("BEHOLDD_QWEN35_MODEL").context("set BEHOLDD_QWEN35_MODEL")?,
        );
        let tokenizer = Qwen2Tokenizer::from_gguf(&Gguf::open(path)?)?;
        assert_eq!(tokenizer.pretokenizer(), "qwen35");
        assert_eq!(tokenizer.vocab_len(), 248_320);
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("tokenizer_fixtures/qwen35_oracle.json"))?;
        for case in fixture["cases"].as_array().context("oracle cases")? {
            let text = case["text"].as_str().context("oracle text")?;
            for (key, parse_special) in [("specialIds", true), ("literalIds", false)] {
                let expected: Vec<u32> = case[key]
                    .as_array()
                    .context("oracle IDs")?
                    .iter()
                    .map(|id| {
                        u32::try_from(id.as_u64().expect("oracle integer")).expect("oracle u32")
                    })
                    .collect();
                assert_eq!(
                    tokenizer.encode(text, parse_special)?,
                    expected,
                    "{key}: {text:?}"
                );
                assert_eq!(tokenizer.decode(&expected, true)?, text);
            }
        }
        assert_ne!(
            tokenizer.encode("é", false)?,
            tokenizer.encode("e\u{301}", false)?
        );
        Ok(())
    }

    #[test]
    fn byte_pieces_are_joined_before_utf8_validation() {
        let tokenizer = Qwen2Tokenizer {
            pretokenizer: PreTokenizer::Qwen2,
            tokens: vec![
                Token {
                    text: [0xf0u8, 0x90, 0x8d]
                        .into_iter()
                        .map(|byte| byte_to_unicode()[usize::from(byte)])
                        .collect::<String>(),
                    kind: TokenKind::Normal,
                },
                Token {
                    text: byte_to_unicode()[usize::from(0x88u8)].to_string(),
                    kind: TokenKind::Normal,
                },
            ],
            token_to_id: HashMap::new(),
            merge_ranks: HashMap::new(),
            specials: Vec::new(),
            bos_token_id: None,
            eos_token_id: 0,
            padding_token_id: None,
            add_bos_token: false,
            add_eos_token: false,
            chat_template: None,
        };
        assert_eq!(tokenizer.decode(&[0, 1], false).unwrap(), "𐍈");
        assert!(tokenizer.decode(&[0], false).is_err());
    }

    #[test]
    fn token_kinds_follow_gguf_decode_semantics() {
        let tokenizer = Qwen2Tokenizer {
            pretokenizer: PreTokenizer::Qwen2,
            tokens: vec![
                Token {
                    text: "<unk>".into(),
                    kind: TokenKind::Unknown,
                },
                Token {
                    text: "<control>".into(),
                    kind: TokenKind::Control,
                },
                Token {
                    text: "<user>".into(),
                    kind: TokenKind::UserDefined,
                },
                Token {
                    text: "[UNUSED]".into(),
                    kind: TokenKind::Unused,
                },
                Token {
                    text: "<0xF0>".into(),
                    kind: TokenKind::Byte,
                },
            ],
            token_to_id: HashMap::new(),
            merge_ranks: HashMap::new(),
            specials: Vec::new(),
            bos_token_id: None,
            eos_token_id: 0,
            padding_token_id: None,
            add_bos_token: false,
            add_eos_token: false,
            chat_template: None,
        };
        assert_eq!(
            tokenizer.decode_bytes(&[0, 1, 2, 3, 4], false).unwrap(),
            b"<user>\xf0"
        );
        assert_eq!(
            tokenizer.decode_bytes(&[0, 1, 2, 3, 4], true).unwrap(),
            b"<unk><control><user>\xf0"
        );
        assert!(parse_byte_token("not-a-byte", 9).is_err());
    }

    #[test]
    fn loads_and_executes_bpe_from_a_metadata_only_gguf() -> Result<()> {
        let fixture = TokenizerFixture::build()?;
        let gguf = Gguf::open(&fixture.path)?;
        let tokenizer = Qwen2Tokenizer::from_gguf(&gguf)?;

        assert_eq!(tokenizer.vocab_len(), 258);
        assert_eq!(tokenizer.bos_token_id(), None);
        assert!(!tokenizer.adds_bos_token());
        assert_eq!(tokenizer.eos_token_id(), 257);
        assert_eq!(tokenizer.padding_token_id(), Some(257));
        assert!(tokenizer.adds_eos_token());
        assert_eq!(tokenizer.chat_template(), Some("fixture {{ messages }}"));

        assert_eq!(tokenizer.encode("ab", false)?, [256, 257]);
        assert_eq!(tokenizer.decode(&[256, 257], false)?, "ab");
        assert_eq!(tokenizer.decode(&[256, 257], true)?, "ab<eos>");
        assert_eq!(tokenizer.encode("<eos>", true)?, [257, 257]);
        assert_ne!(tokenizer.encode("<eos>", false)?, [257, 257]);
        Ok(())
    }

    /// Real-checkpoint probe against frozen oracle values; no llama.cpp runtime is used.
    /// Kept out of the ordinary suite because the checkpoint may be on a remote mount.
    #[test]
    #[ignore = "requires BEHOLDD_QWEN3_CODER_MODEL pointing to Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf"]
    fn real_qwen3_coder_matches_recorded_llama_cpp_oracle() -> Result<()> {
        let model_path = std::path::PathBuf::from(
            std::env::var_os("BEHOLDD_QWEN3_CODER_MODEL").context(
                "set BEHOLDD_QWEN3_CODER_MODEL to the Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf checkpoint",
            )?,
        );
        if !model_path.is_file() {
            bail!(
                "BEHOLDD_QWEN3_CODER_MODEL is not a checkpoint file: {}",
                model_path.display()
            );
        }
        let gguf = Gguf::open(&model_path)?;
        let tokenizer = Qwen2Tokenizer::from_gguf(&gguf)?;

        assert_eq!(tokenizer.vocab_len(), 151_936);
        assert_eq!(tokenizer.bos_token_id(), None);
        assert!(!tokenizer.adds_bos_token());
        assert_eq!(tokenizer.eos_token_id(), 151_645);
        assert_eq!(tokenizer.padding_token_id(), Some(151_643));
        assert_eq!(tokenizer.encode("", true)?, Vec::<u32>::new());
        assert_eq!(
            tokenizer.encode("Hello, world!", true)?,
            [9707, 11, 1879, 0]
        );
        assert_eq!(
            tokenizer.encode(" leading and  multiple spaces\tline\n", true)?,
            [6388, 323, 220, 5248, 12621, 27109, 198]
        );
        assert_eq!(
            tokenizer.encode("I'm can't won't we're I'd you've shouldn't've", true)?,
            [40, 2776, 646, 944, 2765, 944, 582, 2299, 358, 4172, 498, 3003, 13133, 944, 3003]
        );
        assert_eq!(
            tokenizer.encode("café naïve — 中文🙂", true)?,
            [924, 58858, 94880, 586, 1959, 72858, 16744, 145080]
        );
        assert_eq!(tokenizer.encode("𐍈", true)?, [126190, 230]);
        assert_eq!(tokenizer.decode(&[126190, 230], false)?, "𐍈");

        let chat = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
        assert_eq!(
            tokenizer.encode(chat, true)?,
            [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]
        );
        assert_eq!(
            tokenizer.encode(chat, false)?,
            [
                27, 91, 318, 4906, 91, 29, 872, 198, 9707, 27, 91, 318, 6213, 91, 397, 27, 91, 318,
                4906, 91, 29, 77091, 198
            ]
        );
        Ok(())
    }
}
