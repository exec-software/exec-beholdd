/** Model-free C2048 replay; optionally exercise the frozen committed 78147b7 provider. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { BODY_LIMIT, CAPACITY, PROFILE, PROJECT_PROFILE, PROJECT_EXPECTED_REQUEST, REQUEST_ID,
  inspectChat, profileSettings, runClient, validateEnvelope } from './hag_qualify/antimatter-grounded-client.mjs';

const TEXT = '<think></think>\n\nReady.';
const envelope = (stream) => ({ profile: PROJECT_PROFILE,
  request: { ...structuredClone(PROJECT_EXPECTED_REQUEST), stream },
  model_descriptor: { id: PROJECT_EXPECTED_REQUEST.model, object: 'model',
    execution_limits: { max_evaluated_tokens: 2048 } } });
const chat = (content = TEXT) => ({ id: 'project-1', object: 'chat.completion',
  model: PROJECT_EXPECTED_REQUEST.model,
  choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 467, completion_tokens: 7, total_tokens: 474 } });
const frame = (delta = {}, finish = null) => `data: ${JSON.stringify({ id: 'project-1',
  object: 'chat.completion.chunk', model: PROJECT_EXPECTED_REQUEST.model,
  choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
const sse = () => frame({ role: 'assistant' }) + frame({ content: TEXT }) + frame({}, 'stop') + 'data: [DONE]\n\n';

class ReplayProvider {
  constructor(config, track, prepare) { Object.assign(this, { config, track, prepare }); }
  guard(request) {
    const prompt = request.messages.reduce((sum, m) => sum + Math.ceil(m.content.length / 3.2) + 4, 0);
    if (prompt + request.maxTokens - 1 > this.config.models[0].maxEvaluatedTokens) {
      throw new Error('Estimated request exceeds capacity (context_length_exceeded)');
    }
  }
  async response(request, stream, signal) {
    return fetch(`${this.config.baseUrl}/chat/completions`, { method: 'POST', redirect: 'error', signal,
      body: JSON.stringify({ model: request.model, messages: request.messages.map(({ role, content }) => ({ role, content })),
        max_tokens: request.maxTokens, temperature: request.temperature, stream }) });
  }
  async chat(request, signal) {
    this.guard(request);
    const end = this.track();
    try { await this.prepare(); return (await (await this.response(request, false, signal)).json()).choices[0].message.content; }
    finally { end(); }
  }
  async stream(request, callback, _alive, signal) {
    this.guard(request);
    const end = this.track();
    try {
      await this.prepare();
      const raw = await (await this.response(request, true, signal)).text();
      let content = '', finishReason, servedModel;
      for (const line of raw.split('\n')) {
        if (line === 'data: [DONE]') break;
        if (!line.startsWith('data: ')) continue;
        const event = JSON.parse(line.slice(6)), choice = event.choices[0];
        servedModel = event.model;
        if (choice.delta.content) { content += choice.delta.content; callback({ id: request.requestId, content: choice.delta.content }); }
        if (choice.finish_reason) finishReason = choice.finish_reason;
      }
      return { content, finishReason, servedModel, reasoning: '', toolCalls: [], toolCallsCutOff: false };
    } finally { end(); }
  }
}

const providers = [['test-double', ReplayProvider]];
if (process.env.HAG_ANTIMATTER_PROVIDER_BUNDLE) {
  const file = path.resolve(process.env.HAG_ANTIMATTER_PROVIDER_BUNDLE);
  assert.ok((await fs.lstat(file)).isFile());
  const { OpenAICompatibleProvider } = await import(pathToFileURL(file));
  providers.push(['committed-antimatter', OpenAICompatibleProvider]);
}

function response(raw, stream) {
  const bytes = Buffer.from(raw); let position = 0;
  return new Response(new ReadableStream({ pull(controller) {
    if (position === bytes.length) { controller.close(); return; }
    controller.enqueue(bytes.subarray(position, ++position));
  } }), { headers: { 'content-type': stream ? 'text/event-stream' : 'application/json' } });
}

async function replay(Provider, stream, options = {}) {
  let calls = 0;
  const original = globalThis.fetch;
  const result = await runClient({ Provider, envelope: envelope(stream), port: 12345,
    fetchImpl: async (url, init) => {
      calls++;
      assert.equal(url, 'http://127.0.0.1:12345/v1/chat/completions');
      assert.deepEqual(JSON.parse(init.body), envelope(stream).request);
      return response(stream ? sse() : JSON.stringify(chat()), stream);
    }, ...options });
  assert.equal(globalThis.fetch, original);
  return { result, calls };
}

test('C2048 fixture reproduces the measured instruction wrapper and preserves C512 exports', () => {
  const system = PROJECT_EXPECTED_REQUEST.messages[0].content;
  assert.equal(Buffer.byteLength(system), 2593);
  assert.equal(createHash('sha256').update(system).digest('hex'),
    '91e1e04229504e4a2c6d5d58bf1aa9ffd77037b676c63dae31ed5e4bc6e3dfa7');
  assert.equal(CAPACITY, 512);
  assert.equal(profileSettings(PROFILE).requestSeconds, 60);
  assert.equal(profileSettings(PROJECT_PROFILE).requestSeconds, 240);
  assert.equal(profileSettings(PROJECT_PROFILE).preflightOutputTokens, 1583);
  assert.deepEqual(validateEnvelope(envelope(false)), envelope(false).request);
});

for (const [name, Provider] of providers) {
  for (const stream of [false, true]) {
    test(`${name}: project ${stream ? 'stream' : 'chat'} preserves one POST and complete lifecycle`, async () => {
      // Greater than the old 60-second bound, with an immediate fake response.
      const { result, calls } = await replay(Provider, stream, { timeoutMs: 60001 });
      assert.equal(calls, 1); assert.equal(result.client.error, null);
      assert.equal(result.client.max_evaluated_tokens, 2048);
      assert.equal(result.client.mode, stream ? 'stream' : 'chat');
      assert.equal(result.client.request_count, 1); assert.equal(result.client.attempt_count, 1);
      assert.deepEqual(result.client.preflight, { capacity: 2048, requested_output_tokens: 1583,
        fetch_attempts: 0, track_calls: 0, prepare_calls: 0, error_code: 'context_length_exceeded' });
      assert.deepEqual(result.client.hooks, { track_calls: 1, prepare_calls: 1, track_end_calls: 1 });
      assert.deepEqual(result.client.request, envelope(stream).request);
      assert.equal(result.client.response_complete, true);
      if (stream) {
        assert.equal(result.events.map((e) => e.line).join(''), sse());
        assert.equal(result.client.result.content, TEXT);
        assert.equal(result.client.result.finishReason, 'stop');
        assert.equal(result.client.callbacks.map((c) => c.content).join(''), TEXT);
        assert.ok(result.client.callbacks.every((c) => c.id === REQUEST_ID));
      } else {
        assert.equal(result.client.result, TEXT);
        assert.equal(Buffer.from(result.raw_body_base64, 'base64').toString(), JSON.stringify(chat()));
        assert.deepEqual(result.client.callbacks, []);
      }
    });

    test(`${name}: local output1218 passes estimation while1219 refuses before hooks`, async () => {
      const request = { model: PROJECT_EXPECTED_REQUEST.model, providerId: 'local', requestId: REQUEST_ID,
        messages: PROJECT_EXPECTED_REQUEST.messages.map((m, i) => ({ id: `m${i}`, ...m })), temperature: 0 };
      assert.equal(request.messages.reduce((sum, m) => sum + Math.ceil(m.content.length / 3.2) + 4, 0), 831);
      for (const output of [1218, 1219, 1582, 1583]) {
        let tracked = 0, prepared = 0, fetched = 0;
        const admitted = new Error('intercepted admitted request before preparation');
        const original = globalThis.fetch;
        globalThis.fetch = async () => { fetched++; throw new Error('must not fetch'); };
        try {
          const provider = new Provider({ id: 'local', name: 'local', apiKey: '', baseUrl: 'http://127.0.0.1:1/v1',
            models: [{ id: request.model, name: request.model, maxEvaluatedTokens: 2048 }],
            supportsStreaming: true, kind: 'beholdd', protocol: 'openai-chat' },
          () => { tracked++; throw admitted; }, async () => { prepared++; throw new Error('must not prepare'); });
          await assert.rejects(stream ? provider.stream({ ...request, maxTokens: output }, () => {})
            : provider.chat({ ...request, maxTokens: output }),
          output === 1218 ? (error) => error === admitted : /context_length_exceeded/);
          assert.equal(tracked, output === 1218 ? 1 : 0);
          assert.equal(prepared, 0); assert.equal(fetched, 0);
        } finally { globalThis.fetch = original; }
      }
    });
  }

  test(`${name}: malformed, wrong usage and unfinished responses retain bounded evidence`, async () => {
    for (const raw of ['{broken', JSON.stringify({ ...chat(), usage: { prompt_tokens: 93, completion_tokens: 7, total_tokens: 100 } }),
      JSON.stringify({ ...chat(), choices: [{ ...chat().choices[0], finish_reason: 'length' }] })]) {
      const { result } = await replay(Provider, false, { fetchImpl: async () => response(raw, false) });
      assert.notEqual(result.client.error, null);
      assert.equal(Buffer.from(result.raw_body_base64, 'base64').toString(), raw);
      assert.ok(Buffer.byteLength(result.raw_body_base64) <= 2 * BODY_LIMIT);
    }
    for (const raw of [sse().replace('data: [DONE]\n\n', ''), sse() + frame({ content: 'extra' })]) {
      const { result } = await replay(Provider, true, { fetchImpl: async () => response(raw, true) });
      assert.notEqual(result.client.error, null);
      assert.equal(result.events.map((event) => event.line).join(''), raw);
    }
  });
}

test('project scope refuses modified fixture, wrong capacity, tools and deadline before transmission', async () => {
  for (const mutate of [(v) => { v.profile = PROFILE; }, (v) => { v.profile = undefined; },
    (v) => { v.model_descriptor.execution_limits.max_evaluated_tokens = 512; },
    (v) => { v.request.messages[0].content += '\n'; }, (v) => { v.request.max_tokens = 129; },
    (v) => { v.request.tools = []; }]) {
    const input = envelope(false); mutate(input);
    assert.throws(() => validateEnvelope(input));
    const { result, calls } = await replay(ReplayProvider, false, { envelope: input });
    assert.equal(calls, 0); assert.notEqual(result.client.error, null);
    assert.equal(result.client.hooks.track_calls, 0);
  }
  const { result, calls } = await replay(ReplayProvider, false, { timeoutMs: 240001 });
  assert.equal(calls, 0); assert.match(result.client.error, /deadline/);
});

test('project chat validates exact prompt, bounded EOS completion and full usage sum', () => {
  assert.equal(inspectChat(Buffer.from(JSON.stringify(chat())), PROJECT_EXPECTED_REQUEST.model, PROJECT_PROFILE), TEXT);
  for (const usage of [{ prompt_tokens: 467, completion_tokens: 128, total_tokens: 595 },
    { prompt_tokens: 467, completion_tokens: 0, total_tokens: 467 },
    { prompt_tokens: 467, completion_tokens: 7, total_tokens: 473 },
    { prompt_tokens: '467', completion_tokens: 7, total_tokens: 474 }]) {
    assert.throws(() => inspectChat(Buffer.from(JSON.stringify({ ...chat(), usage })),
      PROJECT_EXPECTED_REQUEST.model, PROJECT_PROFILE));
  }
});
