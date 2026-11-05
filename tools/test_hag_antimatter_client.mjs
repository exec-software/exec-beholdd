/** Hardware-free provider replay. Set HAG_ANTIMATTER_PROVIDER_BUNDLE to test the frozen real provider too. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { BODY_LIMIT, EXPECTED_REQUEST, REQUEST_ID, main, parseArguments, runClient } from './hag_qualify/antimatter-client.mjs';

const TEXT = '<think>é🙂</think>\n\n1, 2, 3, 4, 5, 6, 7, 8.';
const frame = (delta = {}, finish = null) => 'data: ' + JSON.stringify({
  id: 'test-response', object: 'chat.completion.chunk', model: EXPECTED_REQUEST.model,
  choices: [{ index: 0, delta, finish_reason: finish }],
}) + '\n\n';
const streamText = (text = TEXT) => frame({ role: 'assistant' }) + frame({ content: text })
  + frame({}, 'stop') + 'data: [DONE]\n\n';
const body = (request) => ({ model: request.model,
  messages: request.messages.map(({ role, content }) => ({ role, content })),
  temperature: request.temperature, max_tokens: request.maxTokens, stream: true });

/** Tiny test double mirrors the real provider's cancellation at DONE, not the SSE validation gate. */
class ReplayProvider {
  constructor(config) { this.config = config; }
  async stream(request, callback, _alive, signal) {
    const response = await fetch(`${this.config.baseUrl}/chat/completions`, {
      method: 'POST', redirect: 'error', signal, body: JSON.stringify(body(request)),
    });
    const reader = response.body.getReader(), decoder = new TextDecoder();
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
  }
}

const providers = [['test-double', ReplayProvider]];
if (process.env.HAG_ANTIMATTER_PROVIDER_BUNDLE) {
  const bundle = path.resolve(process.env.HAG_ANTIMATTER_PROVIDER_BUNDLE);
  assert.ok((await fs.lstat(bundle)).isFile(), 'The explicit real-provider bundle must exist.');
  const { OpenAICompatibleProvider } = await import(pathToFileURL(bundle).href);
  assert.equal(typeof OpenAICompatibleProvider, 'function');
  providers.push(['committed-antimatter', OpenAICompatibleProvider]);
}

function response(text, chunkSize = 31) {
  const bytes = Buffer.isBuffer(text) ? text : Buffer.from(text);
  let offset = 0;
  return new Response(new ReadableStream({
    pull(controller) {
      if (offset === bytes.length) { controller.close(); return; }
      const end = Math.min(bytes.length, offset + chunkSize);
      controller.enqueue(bytes.subarray(offset, end));
      offset = end;
    },
  }), { headers: { 'content-type': 'text/event-stream; charset=utf-8' } });
}

async function replay(Provider, raw = streamText(), options = {}) {
  let calls = 0;
  const before = globalThis.fetch;
  const evidence = await runClient({ Provider, request: structuredClone(EXPECTED_REQUEST), port: 12345,
    fetchImpl: async (url, init) => {
      calls++;
      assert.equal(url, 'http://127.0.0.1:12345/v1/chat/completions');
      assert.deepEqual(JSON.parse(init.body), EXPECTED_REQUEST);
      return response(raw, 1);
    }, ...options });
  assert.equal(globalThis.fetch, before, 'The test transport must be restored.');
  return { evidence, calls };
}

for (const [name, Provider] of providers) {
  test(`${name}: fragmented UTF-8, physical lines, callbacks and EOS retain exact raw bytes`, async () => {
    const raw = streamText();
    const { evidence, calls } = await replay(Provider, raw);
    assert.equal(calls, 1);
    assert.equal(evidence.client.error, null);
    assert.equal(evidence.client.request_count, 1);
    assert.equal(evidence.client.attempt_count, 1);
    assert.deepEqual(evidence.client.request, EXPECTED_REQUEST);
    assert.equal(evidence.events.map((event) => event.line).join(''), raw);
    assert.equal(evidence.client.result.content, TEXT);
    assert.equal(evidence.client.callbacks.map((event) => event.content).join(''), TEXT);
    assert.ok(evidence.client.callbacks.every((event) => event.id === REQUEST_ID));
    assert.equal(evidence.client.response_complete, true);
    let previous = 0;
    for (const event of evidence.events) {
      assert.ok(event.received_seconds >= previous);
      previous = event.received_seconds;
    }
  });

  test(`${name}: provider DONE cancellation still drains a delayed original EOF`, async () => {
    let chunks = 0, originalCancelled = false;
    const raw = streamText();
    const incoming = new Response(new ReadableStream({
      async pull(controller) {
        if (chunks++ === 0) { controller.enqueue(Buffer.from(raw)); return; }
        await new Promise((resolve) => setTimeout(resolve, 5));
        controller.enqueue(Buffer.from(': after DONE\n\n'));
        controller.close();
      },
      cancel() { originalCancelled = true; },
    }), { headers: { 'content-type': 'text/event-stream' } });
    const { evidence } = await replay(Provider, raw, { fetchImpl: async () => incoming });
    assert.equal(evidence.client.error, null);
    assert.equal(evidence.client.response_complete, true);
    assert.equal(originalCancelled, false);
    assert.equal(evidence.events.map((event) => event.line).join(''), raw + ': after DONE\n\n');
  });

  test(`${name}: EOF, malformed SSE, early DONE, wrong model and post-DONE data fail`, async () => {
    const cases = [
      frame({ role: 'assistant' }) + frame({ content: TEXT }) + frame({}, 'stop'),
      frame({ role: 'assistant' }) + 'data: {bad\n\n' + frame({ content: TEXT }) + frame({}, 'stop') + 'data: [DONE]\n\n',
      'data: [DONE]\n\n',
      streamText().replaceAll(EXPECTED_REQUEST.model, 'different-model'),
      streamText() + frame({ content: 'trailing data' }),
      streamText().replace('"finish_reason":"stop"', '"finish_reason":"length"'),
    ];
    for (const raw of cases) {
      const { evidence } = await replay(Provider, raw);
      assert.equal(typeof evidence.client.error, 'string');
      assert.ok(evidence.client.error.length <= 512);
      assert.equal(evidence.client.request_count, 1);
      assert.ok(evidence.events.length > 0);
    }
  });

  test(`${name}: non-SSE HTTP failures preserve status and forbid retry transmissions`, async () => {
    for (const [status, type, raw] of [[200, 'application/json', '{}'],
      [503, 'application/json', '{"error":{"type":"model_loading"}}']]) {
      let calls = 0;
      const { evidence } = await replay(Provider, '', { fetchImpl: async () => {
        calls++; return new Response(raw, { status, headers: { 'content-type': type } });
      } });
      assert.equal(calls, 1);
      assert.equal(evidence.status, status);
      assert.equal(evidence.content_type, type);
      assert.notEqual(evidence.client.error, null);
      assert.equal(evidence.client.response_complete, false);
    }
  });

  test(`${name}: byte, event and UTF-8 limits preserve bounded failure evidence`, async () => {
    for (const raw of [Buffer.alloc(BODY_LIMIT + 1, 120), '\n'.repeat(4097), Buffer.from([0xff])]) {
      const { evidence } = await replay(Provider, raw, { fetchImpl: async () => response(raw, BODY_LIMIT + 1) });
      assert.notEqual(evidence.client.error, null);
      assert.ok(evidence.events.length <= 4096);
      assert.ok(Buffer.byteLength(evidence.events.map((event) => event.line).join('')) <= BODY_LIMIT);
    }
  });
}

test('second provider POST is refused before any second network transmission', async () => {
  class Repeater extends ReplayProvider {
    async stream(request, callback, alive, signal) {
      await super.stream(request, callback, alive, signal);
      await fetch(`${this.config.baseUrl}/chat/completions`, { method: 'POST', redirect: 'error', signal, body: JSON.stringify(body(request)) });
    }
  }
  const { evidence, calls } = await replay(Repeater);
  assert.equal(calls, 1);
  assert.equal(evidence.client.request_count, 1);
  assert.equal(evidence.client.attempt_count, 2);
  assert.match(evidence.client.error, /second provider request/);
});

test('changed destination, body or redirect policy fails before transmission', async () => {
  for (const alteration of ['destination', 'body', 'redirect']) {
    class Altered extends ReplayProvider {
      async stream(request, _callback, _alive, signal) {
        return fetch(alteration === 'destination' ? 'https://example.invalid/chat/completions' : `${this.config.baseUrl}/chat/completions`, {
          method: 'POST', signal, redirect: alteration === 'redirect' ? 'follow' : 'error',
          body: JSON.stringify({ ...body(request), ...(alteration === 'body' ? { max_tokens: 81 } : {}) }),
        });
      }
    }
    const { evidence, calls } = await replay(Altered);
    assert.equal(calls, 0);
    assert.equal(evidence.client.request_count, 0);
    assert.notEqual(evidence.client.error, null);
  }
});

test('request schema, port and overall deadline are enforced without broadening the fixture', async () => {
  for (const options of [{ port: 0 }, { port: 65536 }, { port: '12345' },
    { request: { ...EXPECTED_REQUEST, max_tokens: 81 } },
    { request: { ...EXPECTED_REQUEST, messages: [{ role: 'user', content: 'Different' }] } },
    { request: { ...EXPECTED_REQUEST, tools: [] } }]) {
    const { evidence, calls } = await replay(ReplayProvider, streamText(), options);
    assert.equal(calls, 0);
    assert.notEqual(evidence.client.error, null);
  }
  const { evidence } = await replay(ReplayProvider, '', { timeoutMs: 5, fetchImpl: () => new Promise(() => {}) });
  assert.match(evidence.client.error, /request deadline/);
});

test('bounded provider result and callback discrepancies cannot be reported as success', async () => {
  for (const change of ['result', 'tools', 'callback']) {
    class Misreporting extends ReplayProvider {
      async stream(request, callback, alive, signal) {
        const result = await super.stream(request, change === 'callback'
          ? (value) => callback({ ...value, content: 'different' }) : callback, alive, signal);
        if (change === 'result') result.content = 'different';
        if (change === 'tools') result.toolCalls = [{ name: 'unexpected' }];
        return result;
      }
    }
    const { evidence } = await replay(Misreporting);
    assert.notEqual(evidence.client.error, null);
    assert.equal(evidence.client.response_complete, true);
  }
});

test('CLI validates canonical evidence paths and preserves failures without overwriting', async (context) => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'hag-antimatter-adapter-test-'));
  context.after(() => fs.rm(root, { recursive: true, force: true }));
  const request = path.join(root, 'request.json'), output = path.join(root, 'output.json');
  const args = ['--root', root, '--port', '12345', '--request', request, '--output', output];
  assert.deepEqual(parseArguments(args), { root, port: 12345, request, output });
  for (const bad of [args.slice(0, -1), [...args, '--other', 'value'],
    args.map((value) => value === '12345' ? '0' : value),
    args.map((value) => value === request ? '/outside/request.json' : value)]) {
    assert.throws(() => parseArguments(bad));
  }
  await fs.writeFile(request, '{malformed');
  assert.equal(await main(args), 1);
  const first = await fs.readFile(output, 'utf8');
  const evidence = JSON.parse(first);
  assert.equal(evidence.client.request_count, 0);
  assert.equal(evidence.client.attempt_count, 0);
  assert.notEqual(evidence.client.error, null);
  await assert.rejects(main(args), /EEXIST/);
  assert.equal(await fs.readFile(output, 'utf8'), first);
});
