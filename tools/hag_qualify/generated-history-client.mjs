#!/usr/bin/env node
/** One chat turn in the frozen, CPU-first generated-history qualification.
 * Python authenticates the parent response artifact and exact prompt inspection
 * before invoking this adapter. This adapter checks the concrete wire recipe;
 * it does not claim to authenticate a response hash supplied in the envelope.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { PROJECT_EXPECTED_REQUEST, runClient, main as groundedMain }
  from './antimatter-grounded-client.mjs';

export const PROFILE = 'antimatter-generated-history-c2048';
export const FOLLOWUP = 'Reply with that same one word again. /no_think';
export const CAPACITY = 2048;
export const REQUEST_SECONDS = 240;
export const FIRST_PROMPT_TOKENS = 467;
export const SECOND_MAXIMUM_PROMPT_TOKENS = 491;
const SHA256 = /^[a-f0-9]{64}$/;

function check(condition, message) {
  if (!condition) throw new Error(message);
}

export function validateEnvelope(envelope) {
  check(envelope && typeof envelope === 'object' && !Array.isArray(envelope), 'Invalid generated-history envelope.');
  assert.deepStrictEqual(Object.keys(envelope).sort(),
    ['history_parent', 'model_descriptor', 'profile', 'prompt_tokens', 'request', 'turn']);
  check(envelope.profile === PROFILE, 'Wrong generated-history profile.');
  check(envelope.turn === 1 || envelope.turn === 2, 'Invalid generated-history turn.');
  check(Number.isSafeInteger(envelope.prompt_tokens)
    && (envelope.turn === 1 ? envelope.prompt_tokens === FIRST_PROMPT_TOKENS
      : envelope.prompt_tokens > 0 && envelope.prompt_tokens <= SECOND_MAXIMUM_PROMPT_TOKENS),
  'Prompt inspection is outside the fixed turn bound.');

  const request = { ...structuredClone(PROJECT_EXPECTED_REQUEST), stream: false };
  if (envelope.turn === 1) {
    check(envelope.history_parent === null, 'First turn cannot contain a history parent.');
  } else {
    const parent = envelope.history_parent;
    check(parent && typeof parent === 'object' && !Array.isArray(parent), 'Second turn requires its completed parent.');
    assert.deepStrictEqual(Object.keys(parent).sort(), ['response_sha256', 'text', 'text_sha256']);
    check(typeof parent.response_sha256 === 'string' && SHA256.test(parent.response_sha256)
      && typeof parent.text_sha256 === 'string' && SHA256.test(parent.text_sha256), 'Invalid parent identity.');
    check(typeof parent.text === 'string' && parent.text.trim().length > 0
      && Buffer.byteLength(parent.text) <= 65536 && Buffer.from(parent.text).toString('utf8') === parent.text,
    'Parent text must be bounded nonempty UTF-8.');
    check(createHash('sha256').update(parent.text, 'utf8').digest('hex') === parent.text_sha256,
      'Parent text differs from its exact byte identity.');
    request.messages.push({ role: 'assistant', content: parent.text }, { role: 'user', content: FOLLOWUP });
  }
  assert.deepStrictEqual(envelope.request, request, 'Generated history must preserve the exact recipe and raw parent.');
  const descriptor = envelope.model_descriptor;
  check(descriptor && typeof descriptor === 'object' && !Array.isArray(descriptor)
    && descriptor.id === request.model && descriptor.object === 'model' && descriptor.owned_by === 'beholdd'
    && descriptor.loaded === (envelope.turn === 2)
    && descriptor.execution_limits?.max_evaluated_tokens === CAPACITY,
  'Generated-history discovery or cold/warm state differs.');
  return request;
}

function generatedRequestPolicy(envelope) {
  const request = validateEnvelope(envelope);
  return { request, settings: { capacity: CAPACITY, promptTokens: envelope.prompt_tokens,
    maximumEvaluations: envelope.prompt_tokens + 127, requestSeconds: REQUEST_SECONDS,
    preflightOutputTokens: CAPACITY } };
}

export function runGeneratedClient(options) {
  return runClient({ ...options, requestPolicy: generatedRequestPolicy });
}

export function main(args = process.argv.slice(2)) {
  return groundedMain(args, generatedRequestPolicy);
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().then((code) => { process.exitCode = code; }, (error) => {
    console.error(String(error?.message ?? error).slice(0, 512));
    process.exitCode = 1;
  });
}
