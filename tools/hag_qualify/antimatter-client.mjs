#!/usr/bin/env node
/** One bounded call through a frozen Antimatter provider; no app configuration or model discovery. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

export const BODY_LIMIT = 1024 * 1024;
export const REQUEST_ID = 'hag-client-list';
export const REQUEST_SECONDS = 30;
export const EXPECTED_REQUEST = {
  model: 'hag-context-qwen3-0.6b',
  messages: [{ role: 'user', content: 'List the integers from 1 to 8, separated by commas. No explanation. /no_think' }],
  max_tokens: 80, temperature: 0, stream: true,
};

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function copyBounded(value) {
  const text = JSON.stringify(value);
  check(typeof text === 'string' && Buffer.byteLength(text) <= BODY_LIMIT, 'Oversized client result.');
  return JSON.parse(text);
}

function emptyEvidence() {
  return { status: null, content_type: '', clock: 'monotonic_elapsed_from_request_start', events: [],
    client: { kind: 'antimatter', pid: process.pid, request_count: 0, attempt_count: 0, request: null,
      result: null, callbacks: [], response_complete: false, error: null } };
}

/** Independently require the raw stream's complete EOS framing, including bytes after DONE. */
export function inspectSse(events, model) {
  let content = '', identifier, roles = 0, finishes = 0, done = 0;
  for (const event of events) {
    const line = event.line.replace(/[\r\n]+$/, '');
    if (!line || line.startsWith(':')) continue;
    check(!done && line.startsWith('data: '), 'Unexpected SSE line or data after DONE.');
    const data = line.slice(6);
    if (data === '[DONE]') {
      check(roles === 1 && finishes === 1 && content.length > 0, 'Premature SSE DONE.');
      done++;
      continue;
    }
    const frame = JSON.parse(data);
    check(frame?.object === 'chat.completion.chunk' && frame.model === model
      && typeof frame.id === 'string' && frame.id.length > 0 && frame.id.length <= 256,
    'Unexpected SSE response identity.');
    identifier ??= frame.id;
    check(identifier === frame.id, 'Mixed SSE response identities.');
    check(Array.isArray(frame.choices) && frame.choices.length === 1 && frame.choices[0]?.index === 0,
      'Unexpected SSE choice.');
    const choice = frame.choices[0], delta = choice.delta;
    check(delta && typeof delta === 'object' && !Array.isArray(delta)
      && Object.keys(delta).every((key) => key === 'role' || key === 'content'), 'Unexpected SSE delta.');
    if ('role' in delta) {
      check(delta.role === 'assistant' && roles === 0 && finishes === 0 && !content, 'Misplaced SSE role.');
      roles++;
    }
    if ('content' in delta) {
      check(roles === 1 && finishes === 0 && typeof delta.content === 'string', 'Misplaced SSE content.');
      content += delta.content;
      check(Buffer.byteLength(content) <= 65536, 'Oversized answer content.');
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
      check(roles === 1 && finishes === 0 && choice.finish_reason === 'stop', 'SSE did not finish once at EOS.');
      finishes++;
    }
  }
  check(roles === 1 && finishes === 1 && done === 1, 'SSE ended before EOS and DONE.');
  return content;
}

/** Injectable only for hardware-free replay tests. CLI always imports the frozen provider module. */
export async function runClient({ Provider, request, port, fetchImpl = globalThis.fetch,
  timeoutMs = REQUEST_SECONDS * 1000, now = () => performance.now() }) {
  const evidence = emptyEvidence();
  const client = evidence.client, originalFetch = globalThis.fetch, controller = new AbortController();
  let timer, incomingReader, drain, cancelled = false, delivery, receivedBytes = 0, callbackBytes = 0;
  let pending = '', previous = 0, transportError;
  const started = now();
  const elapsed = () => {
    const value = (now() - started) / 1000;
    check(Number.isFinite(value) && value >= previous, 'Client clock moved backwards.');
    previous = value;
    return value;
  };
  const capture = (text, at, final = false) => {
    pending += text;
    let newline;
    while ((newline = pending.indexOf('\n')) !== -1) {
      check(evidence.events.length < 4096, 'Too many SSE lines.');
      evidence.events.push({ received_seconds: at, line: pending.slice(0, newline + 1) });
      pending = pending.slice(newline + 1);
    }
    if (final && pending) {
      check(evidence.events.length < 4096, 'Too many SSE lines.');
      evidence.events.push({ received_seconds: at, line: pending });
      pending = '';
    }
  };
  try {
    assert.deepStrictEqual(request, EXPECTED_REQUEST, 'Only the frozen bounded list request is permitted.');
    check(Number.isInteger(port) && port > 0 && port <= 65535, 'Invalid loopback port.');
    check(Number.isFinite(timeoutMs) && timeoutMs > 0 && timeoutMs <= REQUEST_SECONDS * 1000, 'Invalid client deadline.');
    const baseUrl = `http://127.0.0.1:${port}/v1`;
    const deadline = new Promise((_resolve, reject) => {
      timer = setTimeout(() => {
        controller.abort();
        reject(new Error('Antimatter client exceeded its 30-second request deadline.'));
      }, timeoutMs);
    });
    globalThis.fetch = async (input, init) => {
      client.attempt_count++;
      check(client.attempt_count === 1, 'A second provider request was refused before transmission.');
      check(input === `${baseUrl}/chat/completions` && init?.method === 'POST'
        && init.redirect === 'error' && init.signal === controller.signal, 'Unexpected provider request destination or controls.');
      check(typeof init.body === 'string' && Buffer.byteLength(init.body) <= BODY_LIMIT, 'Invalid provider request body.');
      const body = JSON.parse(init.body);
      assert.deepStrictEqual(body, request, 'Provider changed the frozen request body.');
      client.request = copyBounded(body);
      client.request_count++;
      const response = await fetchImpl(input, init);
      evidence.status = response.status;
      evidence.content_type = response.headers.get('content-type') ?? '';
      check(response.status === 200 && /^text\/event-stream(?:;|$)/i.test(evidence.content_type)
        && response.body, 'Provider response is not successful SSE.');
      const declared = response.headers.get('content-length');
      check(declared === null || (/^[0-9]+$/.test(declared) && Number(declared) <= BODY_LIMIT), 'Oversized declared SSE body.');
      incomingReader = response.body.getReader();
      const decoder = new TextDecoder('utf-8', { fatal: true });
      const bodyStream = new ReadableStream({
        start(value) { delivery = value; },
        cancel() { cancelled = true; }, // Still drain the original response to its real EOF below.
      });
      drain = (async () => {
        try {
          for (;;) {
            const { done, value } = await incomingReader.read();
            const at = elapsed();
            if (done) {
              capture(decoder.decode(), at, true);
              client.response_complete = true;
              if (!cancelled) delivery.close();
              return;
            }
            check(value instanceof Uint8Array, 'Unexpected response byte stream.');
            const remaining = BODY_LIMIT - receivedBytes;
            receivedBytes += value.byteLength;
            capture(decoder.decode(value.subarray(0, Math.max(0, remaining)), { stream: true }), at);
            check(receivedBytes <= BODY_LIMIT, 'SSE body exceeded 1 MiB.');
            if (!cancelled) delivery.enqueue(value);
          }
        } catch (error) {
          transportError = error;
          if (!cancelled) delivery.error(error);
          throw error;
        } finally {
          incomingReader.releaseLock();
        }
      })();
      drain.catch(() => {}); // Observed below even if the provider ends or rejects early.
      return new Response(bodyStream, { status: response.status, headers: response.headers });
    };
    const provider = new Provider({ id: 'isolated-hag-client', name: 'Frozen Antimatter HAG client', apiKey: '',
      baseUrl, models: [{ id: request.model, name: request.model }], supportsStreaming: true,
      kind: 'beholdd', protocol: 'openai-chat' });
    const operation = (async () => {
      const result = await provider.stream({ model: request.model, providerId: 'isolated-hag-client',
        requestId: REQUEST_ID, messages: request.messages.map((message, index) => ({ id: `fixture-${index}`, ...message })),
        maxTokens: 80, temperature: 0 }, (callback) => {
        check(callback && typeof callback === 'object' && !Array.isArray(callback)
          && Object.keys(callback).every((key) => ['id', 'content', 'reasoning'].includes(key))
          && callback.id === REQUEST_ID && typeof callback.content === 'string'
          && (callback.reasoning === undefined || typeof callback.reasoning === 'string'), 'Unexpected provider callback.');
        callbackBytes += Buffer.byteLength(JSON.stringify(callback));
        check(callbackBytes <= BODY_LIMIT && client.callbacks.length < 4096, 'Oversized provider callbacks.');
        client.callbacks.push(copyBounded(callback));
      }, undefined, controller.signal);
      client.result = copyBounded(result);
      await drain;
      if (transportError) throw transportError;
      check(client.request_count === 1 && client.attempt_count === 1 && client.response_complete,
        'Provider did not complete exactly one response.');
      const content = inspectSse(evidence.events, request.model);
      check(result?.content === content && result.finishReason === 'stop' && result.servedModel === request.model
        && Array.isArray(result.toolCalls) && result.toolCalls.length === 0 && result.toolCallsCutOff === false
        && (result.reasoning === undefined || result.reasoning === ''), 'Provider result differs from complete tool-free EOS stream.');
      check(client.callbacks.every((callback) => !callback.reasoning)
        && client.callbacks.map((callback) => callback.content).join('') === content,
      'Provider callbacks differ from its complete result.');
    })();
    await Promise.race([operation, deadline]);
  } catch (error) {
    client.error = String(error?.message ?? error).slice(0, 512);
    controller.abort();
    if (pending && evidence.events.length < 4096) {
      evidence.events.push({ received_seconds: previous, line: pending });
      pending = '';
    }
    incomingReader?.cancel().catch(() => {});
  } finally {
    clearTimeout(timer);
    globalThis.fetch = originalFetch;
  }
  return evidence;
}

export function parseArguments(args) {
  assert.deepStrictEqual(args.filter((_value, index) => index % 2 === 0), ['--root', '--port', '--request', '--output']);
  check(args.length === 8 && /^[1-9][0-9]{0,4}$/.test(args[3]), 'Expected four explicit client arguments.');
  const [root, port, request, output] = [args[1], Number(args[3]), args[5], args[7]];
  check(port <= 65535, 'Invalid loopback port.');
  for (const value of [root, request, output]) check(path.isAbsolute(value) && path.resolve(value) === value, 'Client paths must be canonical absolute paths.');
  for (const value of [request, output]) check(path.dirname(value) === root, 'Request and output must be direct evidence files.');
  check(request !== output, 'Request and output paths must differ.');
  return { root, port, request, output };
}

export async function main(args = process.argv.slice(2)) {
  const options = parseArguments(args);
  check(await fs.realpath(options.root) === options.root, 'Evidence root must not traverse a symlink.');
  const output = await fs.open(options.output, 'wx'); // Refuse overwrites before importing or calling the provider.
  try {
    let evidence;
    try {
      const requestInfo = await fs.lstat(options.request);
      check(requestInfo.isFile() && requestInfo.size <= BODY_LIMIT, 'Invalid or oversized request file.');
      const input = await fs.open(options.request, 'r');
      let bytes;
      try {
        bytes = Buffer.alloc(BODY_LIMIT + 1);
        const { bytesRead } = await input.read(bytes, 0, bytes.length, 0);
        check(bytesRead === requestInfo.size && bytesRead <= BODY_LIMIT, 'Request file changed while reading.');
        bytes = bytes.subarray(0, bytesRead);
      } finally { await input.close(); }
      const modulePath = path.join(options.root, 'client/provider.mjs');
      check(await fs.realpath(modulePath) === modulePath && (await fs.lstat(modulePath)).isFile(), 'Invalid frozen provider module.');
      const { OpenAICompatibleProvider } = await import(pathToFileURL(modulePath).href);
      evidence = await runClient({ Provider: OpenAICompatibleProvider, request: JSON.parse(bytes.toString('utf8')), port: options.port });
    } catch (error) {
      evidence = emptyEvidence();
      evidence.client.error = String(error?.message ?? error).slice(0, 512);
    }
    await output.writeFile(JSON.stringify(evidence, null, 2) + '\n');
    return evidence.client.error === null ? 0 : 1;
  } finally { await output.close(); }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().then((code) => { process.exitCode = code; }, (error) => {
    console.error(String(error?.message ?? error).slice(0, 512));
    process.exitCode = 1;
  });
}
