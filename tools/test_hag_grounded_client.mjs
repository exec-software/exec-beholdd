/** Hardware-free replay; supply the frozen 78147b7 bundle to exercise the real provider too. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { BODY_LIMIT, CAPACITY, EXPECTED_REQUEST, PROFILE, PROJECT_PROFILE, PROJECT_EXPECTED_REQUEST,
  HISTORY_PROFILE, HISTORY_EXPECTED_REQUEST, TEMPLATE_HISTORY_PROFILE, TEMPLATE_HISTORY_EXPECTED_REQUEST,
  REQUEST_ID, inspectChat,
  main, parseArguments, profileSettings, runClient, validateEnvelope } from './hag_qualify/antimatter-grounded-client.mjs';

const TEXT = '<think>é🙂</think>\n\nReady.';
const descriptor = () => ({ id: EXPECTED_REQUEST.model, object: 'model', loaded: true,
  execution_limits: { max_evaluated_tokens: CAPACITY } });
const envelope = (stream = true) => ({ profile: PROFILE,
  request: { ...structuredClone(EXPECTED_REQUEST), stream }, model_descriptor: descriptor() });
const chatPayload = (content = TEXT) => ({ id: 'reply-1', object: 'chat.completion',
  model: EXPECTED_REQUEST.model, choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 93, completion_tokens: 7, total_tokens: 100 } });
const frame = (delta = {}, finish = null) => 'data: ' + JSON.stringify({
  id: 'reply-1', object: 'chat.completion.chunk', model: EXPECTED_REQUEST.model,
  choices: [{ index: 0, delta, finish_reason: finish }],
}) + '\n\n';
const sseText = () => frame({ role: 'assistant' }) + frame({ content: TEXT }) + frame({}, 'stop') + 'data: [DONE]\n\n';

class ReplayProvider {
  constructor(config, track, prepare) { Object.assign(this, { config, track, prepare }); }
  guard(request) {
    if (request.maxTokens === this.config.models[0].maxEvaluatedTokens) {
      throw new Error('The estimated prompt plus output exceeds capacity (context_length_exceeded)');
    }
  }
  async post(request, stream, signal) {
    return fetch(`${this.config.baseUrl}/chat/completions`, { method: 'POST', redirect: 'error', signal,
      body: JSON.stringify({ model: request.model,
        messages: request.messages.map(({ role, content }) => ({ role, content })),
        temperature: request.temperature, max_tokens: request.maxTokens, stream }) });
  }
  async chat(request, signal) {
    this.guard(request);
    const end = this.track();
    try {
      await this.prepare();
      return (await (await this.post(request, false, signal)).json()).choices?.[0]?.message?.content ?? '';
    } finally { end(); }
  }
  async stream(request, callback, _alive, signal) {
    this.guard(request);
    const end = this.track();
    try {
      await this.prepare();
      const reader = (await this.post(request, true, signal)).body.getReader(), decoder = new TextDecoder();
      let pending = '', content = '', finishReason, servedModel, done = false;
      try {
        while (!done) {
          const part = await reader.read();
          if (part.done) break;
          pending += decoder.decode(part.value, { stream: true });
          let newline;
          while ((newline = pending.indexOf('\n')) !== -1) {
            const line = pending.slice(0, newline).trim();
            pending = pending.slice(newline + 1);
            if (!line.startsWith('data:')) continue;
            const data = line.slice(5).trim();
            if (data === '[DONE]') { done = true; break; }
            let value;
            try { value = JSON.parse(data); } catch { continue; }
            servedModel = value.model;
            const choice = value.choices?.[0];
            if (choice?.delta?.content) {
              content += choice.delta.content;
              callback({ id: request.requestId, content: choice.delta.content });
            }
            if (choice?.finish_reason) finishReason = choice.finish_reason;
          }
        }
      } finally { await reader.cancel(); reader.releaseLock(); }
      return { content, toolCalls: [], reasoning: '', finishReason, servedModel, toolCallsCutOff: false };
    } finally { end(); }
  }
}

const providers = [['test-double', ReplayProvider]];
if (process.env.HAG_ANTIMATTER_PROVIDER_BUNDLE) {
  const modulePath = path.resolve(process.env.HAG_ANTIMATTER_PROVIDER_BUNDLE);
  assert.ok((await fs.lstat(modulePath)).isFile());
  const { OpenAICompatibleProvider } = await import(pathToFileURL(modulePath).href);
  assert.equal(typeof OpenAICompatibleProvider, 'function');
  providers.push(['committed-antimatter', OpenAICompatibleProvider]);
}

function response(raw, stream, { chunkSize = 1, status = 200, type } = {}) {
  const bytes = Buffer.isBuffer(raw) ? raw : Buffer.from(raw);
  let offset = 0;
  return new Response(new ReadableStream({ pull(controller) {
    if (offset === bytes.length) { controller.close(); return; }
    const end = Math.min(bytes.length, offset + chunkSize);
    controller.enqueue(bytes.subarray(offset, end));
    offset = end;
  } }), { status, headers: { 'content-type': type ?? (stream ? 'text/event-stream' : 'application/json') } });
}

async function replay(Provider, stream, options = {}) {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  const evidence = await runClient({ Provider, envelope: envelope(stream), port: 12345,
    fetchImpl: async (url, init) => {
      calls++;
      assert.equal(url, 'http://127.0.0.1:12345/v1/chat/completions');
      assert.deepEqual(JSON.parse(init.body), envelope(stream).request);
      return response(stream ? sseText() : JSON.stringify(chatPayload()), stream);
    }, ...options });
  assert.equal(globalThis.fetch, originalFetch);
  return { evidence, calls };
}

for (const [name, Provider] of providers) {
  for (const stream of [false, true]) {
    test(`${name}: ${stream ? 'stream' : 'chat'} proves preflight before hooks and preserves exact response`, async () => {
      const { evidence, calls } = await replay(Provider, stream);
      assert.equal(calls, 1);
      assert.equal(evidence.client.error, null);
      assert.equal(evidence.client.mode, stream ? 'stream' : 'chat');
      assert.equal(evidence.client.max_evaluated_tokens, CAPACITY);
      assert.deepEqual(evidence.client.preflight, { capacity: 512, requested_output_tokens: 512,
        fetch_attempts: 0, track_calls: 0, prepare_calls: 0, error_code: 'context_length_exceeded' });
      assert.deepEqual(evidence.client.hooks, { track_calls: 1, prepare_calls: 1, track_end_calls: 1 });
      assert.equal(evidence.client.attempt_count, 1);
      assert.equal(evidence.client.request_count, 1);
      assert.equal(evidence.client.response_complete, true);
      assert.deepEqual(evidence.client.request, envelope(stream).request);
      if (stream) {
        assert.equal(evidence.events.map(({ line }) => line).join(''), sseText());
        assert.equal(evidence.client.result.content, TEXT);
        assert.equal(evidence.client.callbacks.map(({ content }) => content).join(''), TEXT);
        assert.ok(evidence.client.callbacks.every(({ id }) => id === REQUEST_ID));
        let previous = 0;
        for (const { received_seconds: at } of evidence.events) { assert.ok(at >= previous); previous = at; }
      } else {
        assert.equal(Buffer.from(evidence.raw_body_base64, 'base64').toString(), JSON.stringify(chatPayload()));
        assert.equal(evidence.client.result, TEXT);
        assert.deepEqual(evidence.client.callbacks, []);
        assert.deepEqual(evidence.events, []);
        assert.ok(evidence.elapsed_seconds >= 0);
      }
    });
  }

  test(`${name}: chat failures retain bounded original bytes and real hooks finish`, async () => {
    const cases = [
      '{broken',
      JSON.stringify({ ...chatPayload(), model: 'wrong' }),
      JSON.stringify({ ...chatPayload(), usage: { prompt_tokens: 92, completion_tokens: 8, total_tokens: 100 } }),
      JSON.stringify({ ...chatPayload(), choices: [{ ...chatPayload().choices[0], finish_reason: 'length' }] }),
      JSON.stringify({ ...chatPayload(), choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: TEXT, tool_calls: [] } }] }),
      JSON.stringify(chatPayload('')),
    ];
    for (const raw of cases) {
      const { evidence } = await replay(Provider, false, { fetchImpl: async () => response(raw, false) });
      assert.notEqual(evidence.client.error, null);
      assert.equal(Buffer.from(evidence.raw_body_base64, 'base64').toString(), raw);
      assert.equal(evidence.client.request_count, 1);
      assert.equal(evidence.client.hooks.track_end_calls, 1);
    }
  });

  test(`${name}: SSE malformed frames, missing DONE, wrong model, length and trailing data refuse`, async () => {
    for (const raw of [sseText().replace('data: [DONE]\n\n', ''), 'data: [DONE]\n\n',
      sseText().replace('data: [DONE]', 'data: {broken\n\ndata: [DONE]'),
      sseText().replaceAll(EXPECTED_REQUEST.model, 'wrong'),
      sseText().replace('"finish_reason":"stop"', '"finish_reason":"length"'),
      sseText() + frame({ content: 'trailing' })]) {
      const { evidence } = await replay(Provider, true, { fetchImpl: async () => response(raw, true) });
      assert.notEqual(evidence.client.error, null);
      assert.ok(evidence.events.length > 0);
      assert.equal(evidence.client.hooks.track_end_calls, 1);
    }
  });

  test(`${name}: provider stops at DONE but adapter drains delayed EOF`, async () => {
    let reads = 0, cancelled = false;
    const incoming = new Response(new ReadableStream({
      async pull(controller) {
        if (reads++ === 0) { controller.enqueue(Buffer.from(sseText())); return; }
        await new Promise((resolve) => setTimeout(resolve, 5));
        controller.enqueue(Buffer.from(': final comment\n\n')); controller.close();
      }, cancel() { cancelled = true; },
    }), { headers: { 'content-type': 'text/event-stream' } });
    const { evidence } = await replay(Provider, true, { fetchImpl: async () => incoming });
    assert.equal(evidence.client.error, null);
    assert.equal(cancelled, false);
    assert.equal(evidence.client.response_complete, true);
    assert.equal(evidence.events.map(({ line }) => line).join(''), sseText() + ': final comment\n\n');
  });

  test(`${name}: unexpected status/type preserves body and prevents actual retry`, async () => {
    for (const stream of [false, true]) {
      let transmissions = 0;
      const raw = '{"error":{"type":"model_loading"}}';
      const { evidence } = await replay(Provider, stream, { fetchImpl: async () => {
        transmissions++;
        return response(raw, stream, { status: 503, type: 'application/json' });
      } });
      assert.equal(transmissions, 1);
      assert.equal(evidence.status, 503);
      assert.equal(Buffer.from(evidence.raw_body_base64, 'base64').toString(), raw);
      assert.notEqual(evidence.client.error, null);
    }
  });

  test(`${name}: response limits and invalid UTF-8 preserve bounded failure evidence`, async () => {
    for (const stream of [false, true]) for (const raw of [Buffer.alloc(BODY_LIMIT + 1, 120), Buffer.from([0xff])]) {
      const { evidence } = await replay(Provider, stream, { fetchImpl: async () => response(raw, stream, { chunkSize: BODY_LIMIT + 1 }) });
      assert.notEqual(evidence.client.error, null);
      assert.ok(Buffer.byteLength(evidence.events.map(({ line }) => line).join('')) <= BODY_LIMIT);
      if (evidence.raw_body_base64) assert.ok(Buffer.from(evidence.raw_body_base64, 'base64').length <= BODY_LIMIT);
    }
  });
}

test('exact profile, mode, message, descriptor and output ceiling are bound before any provider hooks', async () => {
  for (const mutate of [
    (input) => { input.profile = 'backend-parity-c128'; },
    (input) => { input.request.max_tokens = 127; },
    (input) => { input.request.stream = 'true'; },
    (input) => { input.request.messages[0].content += ' changed'; },
    (input) => { input.request.tools = []; },
    (input) => { input.model_descriptor.id = 'other'; },
    (input) => { input.model_descriptor.execution_limits.max_evaluated_tokens = 128; },
    (input) => { input.model_descriptor.execution_limits.max_evaluated_tokens = '512'; },
    (input) => { delete input.model_descriptor.execution_limits; },
    (input) => { input.extra = true; },
  ]) {
    const input = envelope(); mutate(input);
    assert.throws(() => validateEnvelope(input));
    const { evidence, calls } = await replay(ReplayProvider, true, { envelope: input });
    assert.equal(calls, 0);
    assert.equal(evidence.client.hooks.track_calls, 0);
    assert.notEqual(evidence.client.error, null);
  }
});

test('missing provider guard refuses before transmission or managed preparation can escape', async () => {
  class NoGuard extends ReplayProvider { guard() {} }
  for (const stream of [false, true]) {
    const { evidence, calls } = await replay(NoGuard, stream);
    assert.equal(calls, 0);
    assert.notEqual(evidence.client.error, null);
    assert.equal(evidence.client.preflight.fetch_attempts, 1);
    assert.equal(evidence.client.preflight.track_calls, 1);
    assert.equal(evidence.client.preflight.prepare_calls, 1);
  }
});

test('repeated POST and changed body/destination/redirect are blocked before a second transmission', async () => {
  for (const alteration of ['repeat', 'body', 'destination', 'redirect']) {
    class Altered extends ReplayProvider {
      async post(request, stream, signal) {
        if (alteration === 'repeat') await super.post(request, stream, signal);
        return fetch(alteration === 'destination' ? 'https://example.invalid/v1/chat/completions' : `${this.config.baseUrl}/chat/completions`, {
          method: 'POST', redirect: alteration === 'redirect' ? 'follow' : 'error', signal,
          body: JSON.stringify({ ...envelope(stream).request, ...(alteration === 'body' ? { max_tokens: 129 } : {}) }),
        });
      }
    }
    const { evidence, calls } = await replay(Altered, false);
    assert.equal(calls, alteration === 'repeat' ? 1 : 0);
    assert.notEqual(evidence.client.error, null);
  }
});

test('provider result or callbacks cannot replace the raw evidence', async () => {
  for (const stream of [false, true]) {
    class Altered extends ReplayProvider {
      async chat(...args) { await super.chat(...args); return 'different'; }
      async stream(request, callback, alive, signal) {
        const result = await super.stream(request, (value) => callback({ ...value, content: 'different' }), alive, signal);
        return { ...result, content: 'different' };
      }
    }
    const { evidence } = await replay(Altered, stream);
    assert.notEqual(evidence.client.error, null);
  }
});

test('overall deadline and loopback port validation are enforced', async () => {
  for (const options of [{ port: 0 }, { port: 65536 }, { port: '12345' }, { timeoutMs: 60001 }]) {
    const { evidence, calls } = await replay(ReplayProvider, false, options);
    assert.equal(calls, 0);
    assert.notEqual(evidence.client.error, null);
  }
  const { evidence } = await replay(ReplayProvider, false, { timeoutMs: 5, fetchImpl: () => new Promise(() => {}) });
  assert.match(evidence.client.error, /request deadline/);
});

test('chat usage and extra choices are rejected independently from provider parsing', () => {
  for (const payload of [
    { ...chatPayload(), choices: [...chatPayload().choices, ...chatPayload().choices] },
    { ...chatPayload(), usage: { prompt_tokens: 93, completion_tokens: 129, total_tokens: 222 } },
    { ...chatPayload(), usage: { prompt_tokens: 93, completion_tokens: 1, total_tokens: 95 } },
  ]) assert.throws(() => inspectChat(Buffer.from(JSON.stringify(payload)), EXPECTED_REQUEST.model));
});

test('CLI paths and exclusive evidence creation refuse malformed inputs and overwrites', async (context) => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'hag-grounded-adapter-test-'));
  context.after(() => fs.rm(root, { recursive: true, force: true }));
  const request = path.join(root, 'request.json'), output = path.join(root, 'output.json');
  const args = ['--root', root, '--port', '12345', '--request', request, '--output', output];
  assert.deepEqual(parseArguments(args), { root, port: 12345, request, output });
  assert.throws(() => parseArguments([...args, '--other', 'x']));
  assert.throws(() => parseArguments(args.map((value) => value === request ? '/outside.json' : value)));
  await fs.writeFile(request, '{bad');
  assert.equal(await main(args), 1);
  const first = await fs.readFile(output, 'utf8');
  assert.notEqual(JSON.parse(first).client.error, null);
  await assert.rejects(main(args), /EEXIST/);
  assert.equal(await fs.readFile(output, 'utf8'), first);
});

const historyEnvelope = (stream = true) => ({ profile: HISTORY_PROFILE,
  request: { ...structuredClone(HISTORY_EXPECTED_REQUEST), stream },
  model_descriptor: { id: EXPECTED_REQUEST.model, object: 'model',
    execution_limits: { max_evaluated_tokens: 2048 } } });
const historyChat = () => ({ ...chatPayload(),
  usage: { prompt_tokens: 495, completion_tokens: 7, total_tokens: 502 } });

class HistoryReplayProvider extends ReplayProvider {
  guard(request) {
    const estimate = request.messages.reduce((sum, message) => sum + Math.ceil(message.content.length / 3.2) + 4, 0);
    if (estimate + request.maxTokens - 1 > this.config.models[0].maxEvaluatedTokens) {
      throw new Error('Estimated request exceeds capacity (context_length_exceeded)');
    }
  }
}

test('history fixture pins the original system and unstripped prior reply, with independent cloned settings', () => {
  const settings = profileSettings(HISTORY_PROFILE);
  assert.deepEqual(settings, { capacity: 2048, promptTokens: 495, maximumEvaluations: 622,
    requestSeconds: 240, preflightOutputTokens: 1555, request: HISTORY_EXPECTED_REQUEST });
  assert.equal(createHash('sha256').update(settings.request.messages[0].content).digest('hex'),
    '91e1e04229504e4a2c6d5d58bf1aa9ffd77037b676c63dae31ed5e4bc6e3dfa7');
  assert.deepEqual(settings.request.messages.slice(0, 2), PROJECT_EXPECTED_REQUEST.messages);
  assert.deepEqual(settings.request.messages.slice(2), [
    { role: 'assistant', content: '<think>\n\n</think>\n\nready.' },
    { role: 'user', content: 'Reply with that same one word again. /no_think' },
  ]);
  settings.request.messages[0].content = 'changed';
  settings.request.messages[2].content = 'ready.';
  settings.request.messages.reverse();
  assert.deepEqual(profileSettings(HISTORY_PROFILE).request, HISTORY_EXPECTED_REQUEST);
  assert.deepEqual(profileSettings(PROJECT_PROFILE).request, PROJECT_EXPECTED_REQUEST);
  assert.equal(profileSettings(PROJECT_PROFILE).promptTokens, 467);
  assert.equal(profileSettings(PROJECT_PROFILE).preflightOutputTokens, 1583);
});

test('history envelope rejects changed, dropped, duplicated and reordered roles/content before hooks', async () => {
  const changes = [
    (v) => { v.profile = PROJECT_PROFILE; },
    (v) => { v.profile = PROFILE; },
    (v) => { v.request.messages.reverse(); },
    (v) => { [v.request.messages[1], v.request.messages[2]] = [v.request.messages[2], v.request.messages[1]]; },
    (v) => { v.request.messages.splice(2, 0, { ...v.request.messages[2] }); },
    (v) => { v.request.messages[2].content = 'ready.'; },
    (v) => { v.request.messages[2].content = '<think></think>\n\nready.'; },
    (v) => { v.request.messages[2].content = '<think>\n\n</think>\n\nReady.'; },
    (v) => { v.request.messages[2].content = '<think>\n\n</think>\n\nready'; },
    (v) => { v.request.messages[2].content = v.request.messages[2].content.replaceAll('\n', '\r\n'); },
    (v) => { v.request.messages[2].reasoning = ''; },
    (v) => { v.request.max_tokens = 127; },
    (v) => { v.request.tools = []; },
    (v) => { v.model_descriptor.execution_limits.max_evaluated_tokens = 512; },
  ];
  for (let index = 0; index < 4; index++) changes.push(
    (v) => { v.request.messages.splice(index, 1); },
    (v) => { v.request.messages[index].role = index === 1 || index === 3 ? 'assistant' : 'user'; },
    (v) => { v.request.messages[index].content += '\n'; },
  );
  for (const mutate of changes) {
    const input = historyEnvelope(); mutate(input);
    assert.throws(() => validateEnvelope(input));
    const { evidence, calls } = await replay(HistoryReplayProvider, true, { envelope: input });
    assert.equal(calls, 0);
    assert.equal(evidence.client.attempt_count, 0);
    assert.deepEqual(evidence.client.hooks, { track_calls: 0, prepare_calls: 0, track_end_calls: 0 });
    assert.notEqual(evidence.client.error, null);
  }
});

for (const [name, Base] of providers) {
  const Provider = Base === ReplayProvider ? HistoryReplayProvider : Base;
  for (const stream of [false, true]) {
    test(`${name}: history ${stream ? 'stream' : 'chat'} sends exact four-message bytes and preserves the response`, async () => {
      const input = historyEnvelope(stream);
      let captured;
      const raw = stream ? sseText() : JSON.stringify(historyChat());
      const { evidence } = await replay(Provider, stream, { envelope: input, timeoutMs: 60001,
        fetchImpl: async (url, init) => {
          assert.equal(url, 'http://127.0.0.1:12345/v1/chat/completions');
          assert.equal(captured, undefined, 'Only one transport request is allowed.');
          captured = Buffer.from(init.body);
          const expected = { model: input.request.model, messages: input.request.messages,
            temperature: 0, max_tokens: 128, stream };
          assert.deepEqual(captured, Buffer.from(JSON.stringify(expected)));
          return response(raw, stream);
        } });
      assert.ok(captured);
      assert.equal(evidence.client.error, null);
      assert.equal(evidence.client.max_evaluated_tokens, 2048);
      assert.deepEqual(evidence.client.request, input.request);
      assert.deepEqual(evidence.client.preflight, { capacity: 2048, requested_output_tokens: 1555,
        fetch_attempts: 0, track_calls: 0, prepare_calls: 0, error_code: 'context_length_exceeded' });
      assert.deepEqual(evidence.client.hooks, { track_calls: 1, prepare_calls: 1, track_end_calls: 1 });
      assert.equal(evidence.client.attempt_count, 1);
      assert.equal(evidence.client.request_count, 1);
      assert.equal(evidence.client.response_complete, true);
      if (stream) {
        assert.equal(evidence.events.map(({ line }) => line).join(''), raw);
        assert.equal(evidence.client.result.content, TEXT);
        assert.equal(evidence.client.result.finishReason, 'stop');
        assert.equal(evidence.client.callbacks.map(({ content }) => content).join(''), TEXT);
        assert.ok(evidence.client.callbacks.every(({ id }) => id === REQUEST_ID));
      } else {
        assert.equal(Buffer.from(evidence.raw_body_base64, 'base64').toString(), raw);
        assert.equal(evidence.client.result, TEXT);
        assert.deepEqual(evidence.client.callbacks, []);
      }
    });

    test(`${name}: history local1187 admits but1188 and exact-server limits refuse before source hooks`, async () => {
      const request = { model: HISTORY_EXPECTED_REQUEST.model, requestId: REQUEST_ID, temperature: 0,
        messages: HISTORY_EXPECTED_REQUEST.messages.map((message, index) => ({ id: `history-${index}`, ...message })) };
      assert.equal(request.messages.reduce((sum, m) => sum + Math.ceil(m.content.length / 3.2) + 4, 0), 862);
      for (const maxTokens of [1187, 1188, 1554, 1555]) {
        let tracked = 0, prepared = 0, fetched = 0;
        const admitted = new Error('admitted plan intercepted before preparation');
        const originalFetch = globalThis.fetch;
        globalThis.fetch = async () => { fetched++; throw new Error('must not fetch'); };
        try {
          const provider = new Provider({ id: 'history', name: 'history', apiKey: '',
            baseUrl: 'http://127.0.0.1:1/v1', models: [{ id: request.model, name: request.model, maxEvaluatedTokens: 2048 }],
            supportsStreaming: true, kind: 'beholdd', protocol: 'openai-chat' },
          () => { tracked++; throw admitted; }, async () => { prepared++; throw new Error('must not prepare'); });
          await assert.rejects(stream ? provider.stream({ ...request, maxTokens }, () => {})
            : provider.chat({ ...request, maxTokens }),
          maxTokens === 1187 ? (error) => error === admitted : /context_length_exceeded/);
          assert.equal(tracked, maxTokens === 1187 ? 1 : 0);
          assert.equal(prepared, 0);
          assert.equal(fetched, 0);
        } finally { globalThis.fetch = originalFetch; }
      }
    });
  }
}

test('history wire transformations are blocked before transmission even after a valid envelope', async () => {
  for (const mutate of [
    (messages) => { messages.splice(2, 1); },
    (messages) => { messages[2].content = 'ready.'; },
    (messages) => { messages[2].role = 'user'; },
    (messages) => { messages.reverse(); },
  ]) {
    class AlteredHistory extends HistoryReplayProvider {
      async post(request, stream, signal) {
        const changed = structuredClone(request);
        mutate(changed.messages);
        return super.post(changed, stream, signal);
      }
    }
    for (const stream of [false, true]) {
      const { evidence, calls } = await replay(AlteredHistory, stream, { envelope: historyEnvelope(stream) });
      assert.equal(calls, 0);
      assert.equal(evidence.client.attempt_count, 1);
      assert.equal(evidence.client.request_count, 0);
      assert.equal(evidence.client.hooks.track_end_calls, 1);
      assert.notEqual(evidence.client.error, null);
    }
  }
});

test('history response accounting rejects old prompt counts, full output ceiling and unbounded deadline', async () => {
  assert.equal(inspectChat(Buffer.from(JSON.stringify(historyChat())), EXPECTED_REQUEST.model, HISTORY_PROFILE), TEXT);
  for (const usage of [{ prompt_tokens: 467, completion_tokens: 7, total_tokens: 474 },
    { prompt_tokens: 495, completion_tokens: 128, total_tokens: 623 },
    { prompt_tokens: 495, completion_tokens: 0, total_tokens: 495 },
    { prompt_tokens: 495, completion_tokens: 7, total_tokens: 501 },
    { prompt_tokens: '495', completion_tokens: 7, total_tokens: 502 }]) {
    assert.throws(() => inspectChat(Buffer.from(JSON.stringify({ ...historyChat(), usage })),
      EXPECTED_REQUEST.model, HISTORY_PROFILE));
  }
  const { evidence, calls } = await replay(HistoryReplayProvider, false,
    { envelope: historyEnvelope(false), timeoutMs: 240001 });
  assert.equal(calls, 0);
  assert.match(evidence.client.error, /deadline/);
});

const templateHistoryEnvelope = (stream = true) => ({ ...historyEnvelope(stream), profile: TEMPLATE_HISTORY_PROFILE });
const templateHistoryChat = () => ({ ...chatPayload(),
  usage: { prompt_tokens: 491, completion_tokens: 7, total_tokens: 498 } });

test('checkpoint-rendered history adds measured bounds without changing any wire history or old profile', () => {
  const selected = profileSettings(TEMPLATE_HISTORY_PROFILE);
  assert.deepEqual(selected, { capacity: 2048, promptTokens: 491, maximumEvaluations: 618,
    requestSeconds: 240, preflightOutputTokens: 1559, request: HISTORY_EXPECTED_REQUEST });
  assert.deepEqual(TEMPLATE_HISTORY_EXPECTED_REQUEST, HISTORY_EXPECTED_REQUEST);
  assert.notEqual(TEMPLATE_HISTORY_EXPECTED_REQUEST, HISTORY_EXPECTED_REQUEST);
  assert.equal(selected.request.messages.reduce((sum, m) => sum + Math.ceil(m.content.length / 3.2) + 4, 0), 862);
  assert.equal(profileSettings(HISTORY_PROFILE).promptTokens, 495);
  assert.equal(profileSettings(HISTORY_PROFILE).preflightOutputTokens, 1555);
  selected.request.messages[2].content = 'ready.';
  assert.deepEqual(profileSettings(TEMPLATE_HISTORY_PROFILE).request, HISTORY_EXPECTED_REQUEST);
  assert.deepEqual(profileSettings(HISTORY_PROFILE).request, HISTORY_EXPECTED_REQUEST);
});

for (const [name, Base] of providers) {
  const Provider = Base === ReplayProvider ? HistoryReplayProvider : Base;
  for (const stream of [false, true]) {
    test(`${name}: checkpoint-rendered history ${stream ? 'stream' : 'chat'} preserves exact wire and preflight1559`, async () => {
      const input = templateHistoryEnvelope(stream);
      const raw = stream ? sseText() : JSON.stringify(templateHistoryChat());
      let captured = 0;
      const { evidence } = await replay(Provider, stream, { envelope: input, timeoutMs: 60001,
        fetchImpl: async (url, init) => {
          captured++;
          assert.equal(url, 'http://127.0.0.1:12345/v1/chat/completions');
          assert.deepEqual(Buffer.from(init.body), Buffer.from(JSON.stringify({ model: input.request.model,
            messages: input.request.messages, temperature: 0, max_tokens: 128, stream })));
          return response(raw, stream);
        } });
      assert.equal(captured, 1);
      assert.equal(evidence.client.error, null);
      assert.deepEqual(evidence.client.request, input.request);
      assert.deepEqual(evidence.client.preflight, { capacity: 2048, requested_output_tokens: 1559,
        fetch_attempts: 0, track_calls: 0, prepare_calls: 0, error_code: 'context_length_exceeded' });
      assert.deepEqual(evidence.client.hooks, { track_calls: 1, prepare_calls: 1, track_end_calls: 1 });
      assert.equal(evidence.client.response_complete, true);
      assert.equal(evidence.client.attempt_count, 1);
      assert.equal(evidence.client.request_count, 1);
      if (stream) {
        assert.equal(evidence.events.map(({ line }) => line).join(''), raw);
        assert.equal(evidence.client.result.content, TEXT);
        assert.equal(evidence.client.callbacks.map(({ content }) => content).join(''), TEXT);
      } else {
        assert.equal(Buffer.from(evidence.raw_body_base64, 'base64').toString(), raw);
        assert.equal(evidence.client.result, TEXT);
        assert.deepEqual(evidence.client.callbacks, []);
      }
    });
  }
}

test('checkpoint-rendered history refuses old usage and altered raw history without relaxing deadlines', async () => {
  assert.equal(inspectChat(Buffer.from(JSON.stringify(templateHistoryChat())), EXPECTED_REQUEST.model,
    TEMPLATE_HISTORY_PROFILE), TEXT);
  assert.throws(() => inspectChat(Buffer.from(JSON.stringify(historyChat())), EXPECTED_REQUEST.model,
    TEMPLATE_HISTORY_PROFILE));
  assert.throws(() => inspectChat(Buffer.from(JSON.stringify(templateHistoryChat())), EXPECTED_REQUEST.model,
    HISTORY_PROFILE));
  for (const change of [(input) => { input.request.messages[2].content = 'ready.'; },
    (input) => { input.request.messages.splice(2, 1); },
    (input) => { input.request.messages.reverse(); }]) {
    const input = templateHistoryEnvelope(); change(input);
    const { evidence, calls } = await replay(HistoryReplayProvider, true, { envelope: input });
    assert.equal(calls, 0);
    assert.deepEqual(evidence.client.hooks, { track_calls: 0, prepare_calls: 0, track_end_calls: 0 });
    assert.notEqual(evidence.client.error, null);
  }
  const { evidence, calls } = await replay(HistoryReplayProvider, false,
    { envelope: templateHistoryEnvelope(false), timeoutMs: 240001 });
  assert.equal(calls, 0);
  assert.match(evidence.client.error, /deadline/);
});
