These model-free fixtures compare the Qwen3 renderer against the committed
checkpoint Jinja template, using the `tojson` filter from Transformers v4.51.3.
The generator pins the template's SHA-256 and records the Python/Jinja versions,
filter defaults, and generator hash in the resulting JSON.

Run from the repository root, with Python 3 and Jinja2 already available:

```sh
python3 crates/beholdd-server/tests/fixtures/qwen3-tools/generate.py \
  --output crates/beholdd-server/tests/fixtures/qwen3-tools/golden.json
```

The output path must not already exist. The generator loads no model, opens no
listener, and does not import the production renderer. For test comparisons,
deserialize each `request_text` directly into `ChatRequest`; parsing it through a
generic JSON object first can change the tool key ordering being tested.

`cases` contains accepted request fixtures. It includes direct request text for
the integer token `-0` (which Python normalizes to integer `0`) and the exact
`i64::MIN` and `u64::MAX` integer boundaries. These tokens must not first be
round-tripped through a floating-point representation.

`rejected_cases` records the intended stricter admission boundaries, with an
`expected_rejection_stage` for each case. Null tool-result content must fail prompt
admission: the unguarded Jinja template would otherwise stringify it as `None`.
Integer literals outside the server's `i64`/`u64` range must fail deserialization,
rather than silently rounding into floats; Python itself accepts those arbitrary
precision integers. Explicit floating-point tokens remain part of the supported
finite `f64` subset. Each rejected case's `expected_prompt` documents the unguarded
reference behavior, not an admission expectation.

The filter source is
[Transformers v4.51.3 chat_template_utils.py](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/utils/chat_template_utils.py).
The exact defaults are `json.dumps(..., ensure_ascii=False, indent=None,
separators=None, sort_keys=False)`. Thus HTML and Unicode remain literal, key
insertion order remains intact, and JSON punctuation contains the Python default
spaces. Missing function fields and fields explicitly set to null remain distinct.
