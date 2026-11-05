#!/usr/bin/env python3
"""Generate independent Qwen3 template goldens without loading a model.

Requires Python 3 and Jinja2. No transformers installation is needed: the
Transformers v4.51.3 ``tojson`` filter is reproduced below, with attribution.
The committed template is the only production input; renderer code is not used.
Output is created exclusively so an existing evidence file cannot be replaced.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment


TEMPLATE_SHA256 = "5da44855ab7e0641774d5ed0ff0b4f89671d01df7d316e54d06557a962211cb4"
TEMPLATE_RELATIVE = "crates/beholdd-server/src/templates/qwen3-0.6b.jinja"
FILTER_SOURCE = (
    "https://github.com/huggingface/transformers/blob/v4.51.3/"
    "src/transformers/utils/chat_template_utils.py"
)
MISSING = object()


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # Transformers v4.51.3 _compile_jinja_template() overrides Jinja's HTML-safe
    # filter with these json.dumps defaults. In particular: no HTML escaping,
    # no key sorting, spaces after separators, and unescaped non-ASCII text.
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def request(messages, tools=MISSING):
    result = {"model": "qwen3-0.6b", "max_tokens": 32, "messages": messages}
    if tools is not MISSING:
        result["tools"] = tools
    return result


def user(text="Inspect the workspace."):
    return {"role": "user", "content": text}


def function(name="inspect", description=MISSING, parameters=MISSING):
    body = {"name": name}
    if description is not MISSING:
        body["description"] = description
    if parameters is not MISSING:
        body["parameters"] = parameters
    return {"type": "function", "function": body}


def call(call_id="call_one", name="inspect", arguments='{"path":"."}'):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def assistant_calls(calls, content=MISSING):
    result = {"role": "assistant"}
    if content is not MISSING:
        result["content"] = content
    result["tool_calls"] = calls
    return result


def result(call_id="call_one", content="README.md", name=MISSING):
    message = {"role": "tool", "tool_call_id": call_id, "content": content}
    if name is not MISSING:
        message["name"] = name
    return message


def admitted_requests():
    """Use wire-representable requests accepted by the intended strict subset."""
    base = function(
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }
    )
    cases = []

    def add(name, wire):
        cases.append((name, wire))

    add("no_tools_omitted", request([user()]))
    add("no_tools_null", request([user()], None))
    add("no_tools_empty", request([user()], []))
    add("tools_without_system", request([user()], [base]))
    add(
        "tools_with_system",
        request([{"role": "system", "content": "You edit carefully.\n"}, user()], [base]),
    )
    add(
        "tools_with_empty_system",
        request([{"role": "system", "content": ""}, user()], [base]),
    )
    add(
        "tool_insertion_order",
        request(
            [user()],
            [
                {
                    "function": {
                        "parameters": {
                            "required": ["zeta", "alpha"],
                            "properties": {
                                "zeta": {"description": "last first", "type": "string"},
                                "alpha": {"minimum": 1, "type": "integer"},
                            },
                            "additionalProperties": False,
                            "type": "object",
                        },
                        "description": "Deliberately nonalphabetical keys",
                        "name": "ordered",
                    },
                    "type": "function",
                },
                function("second"),
            ],
        ),
    )
    add(
        "omitted_and_null_function_fields",
        request(
            [user()],
            [
                function("omitted"),
                function("null_description", description=None),
                function("null_parameters", parameters=None),
                function("both_null", description=None, parameters=None),
                function("empty_description", description="", parameters={"type": "object"}),
            ],
        ),
    )
    text = "雪 🐈 café <tag> & 'quoted' \"double\" /slash\\backslash\n\r\t\b\f\x00\x1f\x85\u2028\u2029"
    add(
        "unicode_html_and_control_escapes",
        request(
            [{"role": "system", "content": "系统 <&>"}, user("Read café.py 🐈")],
            [
                function(
                    "describe",
                    description=text,
                    parameters={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": text, "enum": [text, "雪"]}
                        },
                    },
                )
            ],
        ),
    )
    minima = [
        0,
        -1,
        -9223372036854775808,
        9007199254740993,
        18446744073709551615,
        0.0,
        -0.0,
        1.0,
        0.0001,
        0.00001,
        0.000001,
        0.0000001,
        1000000000000000.0,
        10000000000000000.0,
        1e20,
        1e21,
        1.2345678901234567,
        5e-324,
        2.2250738585072014e-308,
        1.7976931348623157e308,
    ]
    add(
        "minimum_integer_and_float_boundaries",
        request(
            [user()],
            [
                function(
                    "numbers",
                    parameters={
                        "type": "object",
                        "properties": {
                            f"n{index}": {"minimum": value, "type": "number"}
                            for index, value in enumerate(minima)
                        },
                    },
                )
            ],
        ),
    )

    for name, content in [("missing", MISSING), ("null", None), ("empty", "")]:
        add(
            f"assistant_tool_replay_content_{name}",
            request([user(), assistant_calls([call()], content), result()], [base]),
        )
    add(
        "empty_tool_result",
        request([user(), assistant_calls([call()], ""), result(content="")], [base]),
    )
    add(
        "raw_arguments_whitespace_and_order",
        request(
            [
                user(),
                assistant_calls(
                    [call(arguments=' {\n  "z": "雪",\n  "a": 1e+02\n } ')],
                    "I will inspect it.\n",
                ),
                result(),
            ],
            [function(parameters={"type": "object"})],
        ),
    )
    add(
        "multiple_calls_and_reverse_results",
        request(
            [
                user(),
                assistant_calls(
                    [call("first", arguments='{"path":"a"}'), call("second", arguments='{"path":"b"}')],
                    "Inspecting both paths.",
                ),
                result("second", "B result", name="inspect"),
                result("first", "A result"),
            ],
            [base],
        ),
    )
    add(
        "repeated_ids_across_completed_batches",
        request(
            [
                user(),
                assistant_calls([call("reused")], None),
                result("reused", "first result"),
                assistant_calls([call("reused", arguments='{"path":"src"}')], "Next path."),
                result("reused", "second result"),
            ],
            [base],
        ),
    )
    add(
        "previous_reasoning_removed_and_current_reasoning_retained",
        request(
            [
                user("Previous question"),
                assistant_calls([call("old")], "<think>\nprevious analysis\n</think>\nPrevious visible text"),
                result("old", "previous result"),
                {"role": "assistant", "content": "<think>previous final analysis</think>Previous answer"},
                user("Current question"),
                assistant_calls([call("new")], "<think>\ncurrent analysis\n</think>\nCurrent visible text"),
                result("new", "current result"),
                {"role": "assistant", "content": "<think>\nfinal analysis\n</think>\nFinal answer"},
            ],
            [base],
        ),
    )
    add(
        "intermediate_empty_reasoning_and_final_empty_assistant",
        request(
            [
                user(),
                assistant_calls([call()], "<think>\n\x1c\t \n</think>\nVisible text"),
                result(),
                {"role": "assistant", "content": None},
            ],
            [base],
        ),
    )
    add(
        "thinking_split_boundaries_with_tools",
        request(
            [
                user(),
                assistant_calls([call()], "discard<think>\nkeep\n</think>drop</think>\nfinal"),
                result(),
            ],
            [base],
        ),
    )
    add(
        "user_tool_response_wrapper_does_not_become_last_query",
        request(
            [
                user("Actual query"),
                assistant_calls([call()], "<think>retain this reasoning</think>"),
                result(),
                user("<tool_response>wrapped user result</tool_response>"),
                {"role": "assistant", "content": "Done"},
            ],
            [base],
        ),
    )
    for suffix, tools in [("omitted", MISSING), ("null", None), ("empty", [])]:
        add(
            f"completed_history_tools_removed_{suffix}",
            request([user(), assistant_calls([call()], None), result(), user("Summarize")], tools),
        )
    add(
        "historical_function_absent_from_current_tools",
        request(
            [user(), assistant_calls([call(name="removed_tool")], None), result(name="removed_tool"), user("Continue")],
            [function("new_tool", parameters={"type": "object"})],
        ),
    )
    add(
        "historical_arguments_do_not_use_current_schema",
        request(
            [
                user(),
                assistant_calls([call(arguments='{"old_property": true}')], None),
                result(),
                user("Use the revised tool if needed"),
            ],
            [base],
        ),
    )
    add(
        "nested_maximum_and_max_items",
        request(
            [user()],
            [function("bounded", parameters={
                "type": "object",
                "properties": {
                    "limit": {"maximum": 9007199254740993, "minimum": 0, "type": ["integer", "null"]},
                    "items": {"maxItems": 2.0, "type": "array", "items": {"type": "number", "maximum": 1e20}},
                    "empty": {"type": "array", "maxItems": -0.0},
                },
                "additionalProperties": {"type": "number", "maximum": -0.5},
            })],
        ),
    )
    return cases


def numeric_request_text(token, keyword="minimum"):
    """Keep the exact incoming numeric token, including the integer token -0."""
    marker = "__RAW_BOUND_LITERAL__"
    wire = request(
        [user()],
        [
            function(
                "numeric_literal",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "array" if keyword == "maxItems" else "number", keyword: marker}},
                },
            )
        ],
    )
    request_text = json.dumps(wire, ensure_ascii=False, allow_nan=False)
    quoted_marker = json.dumps(marker)
    if request_text.count(quoted_marker) != 1:
        raise AssertionError("Numeric fixture marker must occur exactly once")
    return request_text.replace(quoted_marker, token)


def admitted_lexical_requests():
    # Building these from Python values would already normalize -0 into 0;
    # direct wire text exercises the Rust JSON decoder as well as the renderer.
    return [
        ("minimum_integer_negative_zero_literal", numeric_request_text("-0")),
        ("minimum_i64_minimum_literal", numeric_request_text("-9223372036854775808")),
        ("minimum_u64_maximum_literal", numeric_request_text("18446744073709551615")),
        ("maximum_integer_negative_zero_literal", numeric_request_text("-0", "maximum")),
        ("maximum_u64_boundary_literal", numeric_request_text("18446744073709551615", "maximum")),
        ("max_items_u64_boundary_literal", numeric_request_text("18446744073709551615", "maxItems")),
        ("max_items_integral_float_literal", numeric_request_text("2.0", "maxItems")),
        ("max_items_negative_zero_float_literal", numeric_request_text("-0.0", "maxItems")),
    ]


def rejected_requests():
    # The checkpoint template would stringify a null tool result as "None".
    # Strict admission must reject it rather than silently accepting that text.
    return [
        (
            "null_tool_result_requires_admission_rejection",
            request([user(), assistant_calls([call()], None), result(content=None)], [function()]),
            {
                "expected_rejection_stage": "render_prompt",
                "expected_admission_error": {"code": "invalid_request", "param": "messages"},
            },
        )
    ] + [
        (
            name,
            numeric_request_text(token),
            {"expected_rejection_stage": "deserialize"},
        )
        for name, token in [
            ("minimum_below_i64_integer_bound_rejected", "-9223372036854775809"),
            ("minimum_above_u64_integer_bound_rejected", "18446744073709551616"),
            ("minimum_large_positive_integer_rejected", "100000000000000000000"),
            ("minimum_large_negative_integer_rejected", "-100000000000000000000"),
        ]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New golden JSON file (exclusive create)")
    args = parser.parse_args()
    template_path = Path(__file__).resolve().parents[3] / "src/templates/qwen3-0.6b.jinja"
    template_bytes = template_path.read_bytes()
    observed_hash = hashlib.sha256(template_bytes).hexdigest()
    if observed_hash != TEMPLATE_SHA256:
        raise SystemExit(f"Pinned template hash mismatch: {observed_hash}")
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    environment.filters["tojson"] = tojson
    template = environment.from_string(template_bytes.decode("utf-8"))

    def render_case(name, wire):
        # Round-trip the exact request string before rendering. Rust consumers
        # must likewise deserialize request_text directly, not first a Value,
        # whose map implementation might discard the ordering under test.
        request_text = wire if isinstance(wire, str) else json.dumps(wire, ensure_ascii=False, allow_nan=False)
        parsed = json.loads(request_text)
        before = copy.deepcopy(parsed)
        prompt = template.render(
            messages=parsed["messages"],
            tools=parsed.get("tools"),
            add_generation_prompt=True,
        )
        if parsed != before:
            raise AssertionError(f"Oracle mutated request: {name}")
        return {"name": name, "request_text": request_text, "expected_prompt": prompt}

    output = {
        "schema_version": 1,
        "provenance": {
            "template_path": TEMPLATE_RELATIVE,
            "template_sha256": TEMPLATE_SHA256,
            "filter_source": FILTER_SOURCE,
            "filter_version": "transformers v4.51.3",
            "filter_defaults": {
                "ensure_ascii": False,
                "indent": None,
                "separators": None,
                "sort_keys": False,
            },
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python_version": platform.python_version(),
            "jinja2_version": importlib.metadata.version("jinja2"),
            "model_loaded": False,
            "request_representation": "Deserialize request_text directly to preserve wire key order.",
        },
        "cases": [
            render_case(name, wire)
            for name, wire in admitted_requests() + admitted_lexical_requests()
        ],
        "rejected_cases": [
            {
                **render_case(name, wire),
                **expectation,
            }
            for name, wire, expectation in rejected_requests()
        ],
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(output, target, ensure_ascii=False, allow_nan=False, indent=2)
        target.write("\n")
    print(f"Wrote {len(output['cases'])} admitted and {len(output['rejected_cases'])} rejected cases to {args.output}")


if __name__ == "__main__":
    main()
