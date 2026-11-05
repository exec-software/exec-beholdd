/** Model-free generated-history replay; optional frozen provider uses mocked fetch only. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { PROFILE, FOLLOWUP, validateEnvelope, runGeneratedClient, main }
  from './hag_qualify/generated-history-client.mjs';
import { PROJECT_EXPECTED_REQUEST, HISTORY_EXPECTED_REQUEST, TEMPLATE_HISTORY_PROFILE,
  profileSettings, runClient } from './hag_qualify/antimatter-grounded-client.mjs';

const hash = (value) => createHash('sha256').update(value).digest('hex');
const TEXT = '  <think>é🙂</think>\n\nAlternate.\n';
function envelope(turn = 1, text = TEXT, prompt = turn === 1 ? 467 : 489) {
  const request = { ...structuredClone(PROJECT_EXPECTED_REQUEST), stream: false };
  if (turn === 2) request.messages.push({ role: 'assistant', content: text }, { role: 'user', content: FOLLOWUP });
  return { profile: PROFILE, turn, prompt_tokens: prompt, request,
    model_descriptor: { id: request.model, object: 'model', owned_by: 'beholdd', loaded: turn === 2,
      execution_limits: { max_evaluated_tokens: 2048 } },
    history_parent: turn === 1 ? null : { response_sha256: hash('retained first artifact'), text_sha256: hash(text), text } };
}
function payload(input, text = TEXT, completion = 7) {
  return { id: 'generated-reply', object: 'chat.completion', model: input.request.model,
    choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: text } }],
    usage: { prompt_tokens: input.prompt_tokens, completion_tokens: completion,
      total_tokens: input.prompt_tokens + completion } };
}
function response(raw, status = 200) {
  const bytes = Buffer.from(raw); let offset = 0;
  return new Response(new ReadableStream({ pull(controller) {
    if (offset === bytes.length) { controller.close(); return; }
    controller.enqueue(bytes.subarray(offset, ++offset));
  } }), { status, headers: { 'content-type': 'application/json' } });
}
class ReplayProvider {
  constructor(config, track, prepare) { Object.assign(this, { config, track, prepare }); }
  async chat(request, signal) {
    const estimate = request.messages.reduce((total, m) => total + Math.ceil(m.content.length / 3.2) + 4, 0);
    if (estimate + request.maxTokens - 1 > this.config.models[0].maxEvaluatedTokens) {
      throw new Error('Estimated capacity exceeded (context_length_exceeded)');
    }
    const end = this.track();
    try {
      await this.prepare();
      const result = await fetch(`${this.config.baseUrl}/chat/completions`, { method: 'POST', redirect: 'error', signal,
        body: JSON.stringify({ model: request.model, messages: request.messages.map(({ role, content }) => ({ role, content })),
          max_tokens: request.maxTokens, temperature: request.temperature, stream: false }) });
      return (await result.json()).choices?.[0]?.message?.content ?? '';
    } finally { end(); }
  }
  async stream() { throw new Error('Generated-history profile must never stream.'); }
}
const providers = [['test-double', ReplayProvider]];
if (process.env.HAG_ANTIMATTER_PROVIDER_BUNDLE) {
  const file = path.resolve(process.env.HAG_ANTIMATTER_PROVIDER_BUNDLE);
  assert.ok((await fs.lstat(file)).isFile());
  const { OpenAICompatibleProvider } = await import(pathToFileURL(file).href);
  providers.push(['committed-antimatter', OpenAICompatibleProvider]);
}
async function replay(Provider, input = envelope(), options = {}) {
  let calls = 0;
  const original = globalThis.fetch;
  const evidence = await runGeneratedClient({ Provider, envelope: input, port: 12345,
    fetchImpl: async (url, init) => {
      calls++;
      assert.equal(url, 'http://127.0.0.1:12345/v1/chat/completions');
      assert.deepEqual(JSON.parse(init.body), input.request);
      return response(JSON.stringify(payload(input)));
    }, ...options });
  assert.equal(globalThis.fetch, original);
  return { evidence, calls };
}

test('generated recipe is distinct and does not alter any seeded profile setting', () => {
  const before = profileSettings(TEMPLATE_HISTORY_PROFILE);
  for (const text of [TEXT, 'different', '<think>\n\n</think>\n\nREADY.']) {
    const input = envelope(2, text);
    assert.deepEqual(validateEnvelope(input), input.request);
    assert.equal(input.request.messages[2].content, text);
    assert.equal(input.request.messages[3].content, FOLLOWUP);
  }
  assert.deepEqual(profileSettings(TEMPLATE_HISTORY_PROFILE), before);
  assert.deepEqual(before.request, HISTORY_EXPECTED_REQUEST);
  assert.equal(before.promptTokens, 491);
  assert.equal(before.preflightOutputTokens, 1559);
});

test('generated envelope refuses all recipe, lineage, descriptor and scalar substitutions before hooks', async () => {
  const changes = [
    (v) => { v.profile = TEMPLATE_HISTORY_PROFILE; }, (v) => { v.turn = true; },
    (v) => { v.prompt_tokens = '489'; }, (v) => { v.prompt_tokens = 492; }, (v) => { v.prompt_tokens = 0; },
    (v) => { v.request.stream = true; }, (v) => { v.request.max_tokens = 127; },
    (v) => { v.request.temperature = 0.1; }, (v) => { v.request.tools = []; },
    (v) => { v.request.messages.reverse(); }, (v) => { v.request.messages.splice(2, 1); },
    (v) => { v.request.messages[2].content = v.history_parent.text.trim(); },
    (v) => { v.request.messages[2].role = 'user'; },
    (v) => { v.request.messages[0].content += '\n'; }, (v) => { v.request.messages[3].content += '\n'; },
    (v) => { v.history_parent = null; }, (v) => { v.history_parent.text_sha256 = '0'.repeat(64); },
    (v) => { v.history_parent.response_sha256 = '../file'; }, (v) => { v.history_parent.extra = true; },
    (v) => { v.model_descriptor.loaded = false; },
    (v) => { v.model_descriptor.execution_limits.max_evaluated_tokens = 512; },
    (v) => { v.model_descriptor.owned_by = 'other'; },
    (v) => { v.requestPolicy = {}; },
  ];
  for (const change of changes) {
    const input = envelope(2); change(input);
    assert.throws(() => validateEnvelope(input));
    const { evidence, calls } = await replay(ReplayProvider, input);
    assert.equal(calls, 0); assert.notEqual(evidence.client.error, null);
    assert.equal(evidence.client.attempt_count, 0);
    assert.deepEqual(evidence.client.hooks, { track_calls: 0, prepare_calls: 0, track_end_calls: 0 });
  }
});

test('turn one has no parent and only its exact measured prompt count', () => {
  assert.deepEqual(validateEnvelope(envelope()), envelope().request);
  for (const mutate of [(v) => { v.prompt_tokens = 466; }, (v) => { v.model_descriptor.loaded = true; },
    (v) => { v.history_parent = envelope(2).history_parent; }, (v) => { v.request.messages.push({ role: 'assistant', content: TEXT }); }]) {
    const input = envelope(); mutate(input);
    assert.throws(() => validateEnvelope(input));
  }
});

test('parent is exact bounded UTF-8, never silently normalized or replaced', () => {
  for (const text of ['', ' \n', 'x'.repeat(65537), '\ud800']) assert.throws(() => validateEnvelope(envelope(2, text)));
  const input = envelope(2, 'é');
  input.history_parent.text_sha256 = hash('e\u0301');
  assert.throws(() => validateEnvelope(input));
  assert.doesNotThrow(() => validateEnvelope(envelope(2, 'x'.repeat(65536))));
});

for (const [name, Provider] of providers) {
  test(`${name}: both turns retain raw chat, exact usage, one POST and one complete hook cycle`, async () => {
    for (const turn of [1, 2]) {
      const input = envelope(turn);
      const { evidence, calls } = await replay(Provider, input, { timeoutMs: 60001 });
      assert.equal(calls, 1); assert.equal(evidence.client.error, null);
      assert.equal(evidence.client.mode, 'chat'); assert.equal(evidence.client.result, TEXT);
      assert.deepEqual(evidence.client.request, input.request);
      assert.equal(evidence.client.request_count, 1); assert.equal(evidence.client.attempt_count, 1);
      assert.equal(evidence.client.response_complete, true);
      assert.deepEqual(evidence.client.hooks, { track_calls: 1, prepare_calls: 1, track_end_calls: 1 });
      assert.deepEqual(evidence.client.preflight, { capacity: 2048, requested_output_tokens: 2048,
        fetch_attempts: 0, track_calls: 0, prepare_calls: 0, error_code: 'context_length_exceeded' });
      assert.deepEqual(evidence.events, []); assert.deepEqual(evidence.client.callbacks, []);
      assert.equal(Buffer.from(evidence.raw_body_base64, 'base64').toString(), JSON.stringify(payload(input)));
    }
  });

  test(`${name}: second POST receives the actual first provider result rather than any seed`, async () => {
    const firstInput = envelope();
    const first = await replay(Provider, firstInput, { fetchImpl: async () => response(JSON.stringify(payload(firstInput, '  unique-é.\n'))) });
    assert.equal(first.evidence.client.error, null);
    const secondInput = envelope(2, first.evidence.client.result, 480);
    secondInput.history_parent.response_sha256 = hash(JSON.stringify(first.evidence));
    let sent;
    const second = await replay(Provider, secondInput, { fetchImpl: async (_url, init) => {
      sent = JSON.parse(init.body);
      return response(JSON.stringify(payload(secondInput, 'unique-é.')));
    } });
    assert.equal(second.evidence.client.error, null);
    assert.equal(sent.messages[2].content, first.evidence.client.result);
    assert.notEqual(sent.messages[2].content, HISTORY_EXPECTED_REQUEST.messages[2].content);
  });

  test(`${name}: wrong prompt/EOS/usage and malformed responses preserve raw failure evidence`, async () => {
    const input = envelope(2);
    const good = payload(input);
    const cases = ['{broken',
      JSON.stringify({ ...good, usage: { ...good.usage, prompt_tokens: 491 } }),
      JSON.stringify({ ...good, usage: { ...good.usage, total_tokens: 0 } }),
      JSON.stringify(payload(input, TEXT, 128)),
      JSON.stringify({ ...good, choices: [{ ...good.choices[0], finish_reason: 'length' }] }),
      JSON.stringify({ ...good, choices: [{ ...good.choices[0], message: { ...good.choices[0].message, tool_calls: [] } }] })];
    for (const raw of cases) {
      const { evidence } = await replay(Provider, input, { fetchImpl: async () => response(raw) });
      assert.notEqual(evidence.client.error, null);
      assert.equal(Buffer.from(evidence.raw_body_base64, 'base64').toString(), raw);
    }
  });
}

test('new policy is unavailable to the old adapter default or its request JSON', async () => {
  let sent = 0;
  const evidence = await runClient({ Provider: ReplayProvider, envelope: envelope(), port: 12345,
    fetchImpl: async () => { sent++; throw new Error('must not fetch'); } });
  assert.equal(sent, 0); assert.notEqual(evidence.client.error, null);
});

test('new adapter refuses an extended deadline and an actual provider retry', async () => {
  const tooLong = await replay(ReplayProvider, envelope(), { timeoutMs: 240001 });
  assert.equal(tooLong.calls, 0); assert.match(tooLong.evidence.client.error, /deadline/);
  class RetryProvider extends ReplayProvider {
    async chat(request, signal) {
      const value = await super.chat(request, signal);
      await fetch(`${this.config.baseUrl}/chat/completions`, { method: 'POST', redirect: 'error', signal, body: '{}' });
      return value;
    }
  }
  const retry = await replay(RetryProvider);
  assert.equal(retry.calls, 1); assert.equal(retry.evidence.client.request_count, 1);
  assert.equal(retry.evidence.client.attempt_count, 2); assert.match(retry.evidence.client.error, /second provider request/);
});

test('generated CLI retains a failed envelope exclusively without loading a model', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'hag-generated-client-'));
  try {
    await fs.mkdir(path.join(root, 'client'));
    await fs.writeFile(path.join(root, 'client/provider.mjs'), 'export class OpenAICompatibleProvider {}\n');
    const request = path.join(root, 'request.json'), output = path.join(root, 'response.json');
    const input = envelope(); input.request.stream = true;
    await fs.writeFile(request, JSON.stringify(input));
    const args = ['--root', root, '--port', '12345', '--request', request, '--output', output];
    assert.equal(await main(args), 1);
    const raw = await fs.readFile(output, 'utf8');
    assert.notEqual(JSON.parse(raw).client.error, null);
    assert.equal(JSON.parse(raw).client.request_count, 0);
    await assert.rejects(main(args), /EEXIST/);
    assert.equal(await fs.readFile(output, 'utf8'), raw);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
