#!/usr/bin/env node
/** One bounded grounded request through the frozen capacity-aware Antimatter provider. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { BODY_LIMIT, inspectSse, parseArguments } from './antimatter-client.mjs';

export { BODY_LIMIT, parseArguments };
export const PROFILE = 'antimatter-grounded-c512';
export const PROJECT_PROFILE = 'antimatter-project-c2048';
export const HISTORY_PROFILE = 'antimatter-history-c2048';
export const TEMPLATE_HISTORY_PROFILE = 'antimatter-history-template-c2048';
export const CAPACITY = 512;
export const REQUEST_SECONDS = 60;
export const REQUEST_ID = 'hag-client-grounded';
export const EXPECTED_REQUEST = {
  model: 'hag-context-qwen3-0.6b',
  messages: [
    { role: 'system', content: "You are the assistant inside Antimatter, a desktop application on the user's own machine.\n\n"
      + 'You have no tools connected in this conversation, so you cannot read or change files or run\n'
      + 'commands. Answer from what you are told, and say when something would need a tool you do not\n'
      + 'have rather than describing what you would have done.' },
    { role: 'user', content: 'Reply with one word: ready. /no_think' },
  ],
  max_tokens: 128, temperature: 0, stream: true,
};
// Literal rendering from committed renderInstructions/withSituation at 78147b7.
// This is synthetic qualification input, never a live project's AGENTS.md.
export const PROJECT_EXPECTED_REQUEST = {
  ...EXPECTED_REQUEST,
  messages: [
    { role: 'system', content:
      "The project's own instructions follow, from AGENTS.md. The people who work on it wrote them "
      + 'for anyone doing so, you included: follow them unless the user tells you otherwise.\n\n'
      + '<file path="AGENTS.md">\n'
      + 'Use the existing project conventions.\n'.repeat(60).slice(0, 2048)
      + '\n</file>\n\n---\n\n' + EXPECTED_REQUEST.messages[0].content },
    { ...EXPECTED_REQUEST.messages[1] },
  ],
};
// Seeded history from the qualified first reply. The pinned provider preserves
// literal thinking markup in content; it only separates reasoning wire fields.
export const HISTORY_EXPECTED_REQUEST = {
  ...EXPECTED_REQUEST,
  messages: [
    ...PROJECT_EXPECTED_REQUEST.messages.map((message) => ({ ...message })),
    { role: 'assistant', content: '<think>\n\n</think>\n\nready.' },
    { role: 'user', content: 'Reply with that same one word again. /no_think' },
  ],
};
// The wire history is unchanged. Only the server's checkpoint-selected rendering
// and its independently measured token count distinguish this additive profile.
export const TEMPLATE_HISTORY_EXPECTED_REQUEST = structuredClone(HISTORY_EXPECTED_REQUEST);

export function profileSettings(profile = PROFILE) {
  if (profile === TEMPLATE_HISTORY_PROFILE) return { capacity: 2048, promptTokens: 491,
    maximumEvaluations: 618, requestSeconds: 240, preflightOutputTokens: 1559,
    request: structuredClone(TEMPLATE_HISTORY_EXPECTED_REQUEST) };
  if (profile === HISTORY_PROFILE) return { capacity: 2048, promptTokens: 495,
    maximumEvaluations: 622, requestSeconds: 240, preflightOutputTokens: 1555,
    request: structuredClone(HISTORY_EXPECTED_REQUEST) };
  if (profile === PROJECT_PROFILE) return { capacity: 2048, promptTokens: 467,
    maximumEvaluations: 594, requestSeconds: 240, preflightOutputTokens: 1583,
    request: structuredClone(PROJECT_EXPECTED_REQUEST) };
  check(profile === PROFILE, 'Unknown grounded client profile.');
  return { capacity: CAPACITY, promptTokens: 93, maximumEvaluations: 220,
    requestSeconds: REQUEST_SECONDS, preflightOutputTokens: CAPACITY,
    request: structuredClone(EXPECTED_REQUEST) };
}

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function copyBounded(value) {
  const text = JSON.stringify(value);
  check(typeof text === 'string' && Buffer.byteLength(text) <= BODY_LIMIT, 'Oversized client value.');
  return JSON.parse(text);
}

function emptyEvidence() {
  return { status: null, content_type: '', clock: 'monotonic_elapsed_from_request_start', events: [],
    client: { kind: 'antimatter', pid: process.pid, mode: null, max_evaluated_tokens: null,
      request_count: 0, attempt_count: 0, request: null, result: null, callbacks: [],
      response_complete: false, error: null,
      preflight: { capacity: CAPACITY, requested_output_tokens: CAPACITY,
        fetch_attempts: 0, track_calls: 0, prepare_calls: 0, error_code: null },
      hooks: { track_calls: 0, prepare_calls: 0, track_end_calls: 0 } } };
}

export function validateEnvelope(envelope) {
  check(envelope && typeof envelope === 'object' && !Array.isArray(envelope), 'Invalid request envelope.');
  assert.deepStrictEqual(Object.keys(envelope).sort(), ['model_descriptor', 'profile', 'request']);
  check(typeof envelope.profile === 'string', 'Missing grounded profile.');
  const settings = profileSettings(envelope.profile);
  check(typeof envelope.request?.stream === 'boolean', 'Wrong grounded profile or mode.');
  const expected = { ...settings.request, stream: envelope.request.stream };
  assert.deepStrictEqual(envelope.request, expected, 'Only the frozen bounded grounded request is permitted.');
  const descriptor = envelope.model_descriptor;
  check(descriptor && typeof descriptor === 'object' && !Array.isArray(descriptor)
    && descriptor.id === expected.model && descriptor.execution_limits?.max_evaluated_tokens === settings.capacity,
  'Discovery descriptor does not advertise this model with the frozen capacity.');
  copyBounded(envelope);
  return expected;
}

/** Strict independent wire checks supplement the provider's ordinary chat parser. */
export function inspectChat(raw, model, profile = PROFILE) {
  return inspectChatWithSettings(raw, model, profileSettings(profile));
}

function inspectChatWithSettings(raw, model, settings) {
  const parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(raw));
  check(parsed?.object === 'chat.completion' && parsed.model === model
    && typeof parsed.id === 'string' && parsed.id.length > 0 && parsed.id.length <= 256
    && Array.isArray(parsed.choices) && parsed.choices.length === 1,
  'Unexpected chat response identity.');
  const choice = parsed.choices[0], message = choice?.message;
  check(choice?.index === 0 && choice.finish_reason === 'stop'
    && message && !Array.isArray(message) && message.role === 'assistant'
    && Object.keys(message).every((key) => key === 'role' || key === 'content')
    && typeof message.content === 'string' && message.content.trim().length > 0
    && Buffer.byteLength(message.content) <= 65536,
  'Chat did not finish once with a bounded tool-free EOS answer.');
  const usage = parsed.usage;
  check(usage && usage.prompt_tokens === settings.promptTokens && Number.isSafeInteger(usage.completion_tokens)
    && usage.completion_tokens > 0 && usage.completion_tokens < 128
    && usage.total_tokens === usage.prompt_tokens + usage.completion_tokens
    && Number.isSafeInteger(usage.total_tokens) && usage.total_tokens <= settings.maximumEvaluations,
  'Unexpected chat usage.');
  return message.content;
}

function groundedRequestPolicy(envelope) {
  return { request: validateEnvelope(envelope), settings: profileSettings(envelope.profile) };
}

/** Injectable for hardware-free replay; the CLI imports only its frozen provider bundle. */
export async function runClient({ Provider, envelope, port, fetchImpl = globalThis.fetch,
  timeoutMs, now = () => performance.now(), requestPolicy = groundedRequestPolicy }) {
  const evidence = emptyEvidence(), client = evidence.client;
  const originalFetch = globalThis.fetch, controller = new AbortController();
  let timer, reader, drain, delivery, providerCancelled = false, pending = '', previous = 0;
  let phase = 'preflight', started, receivedBytes = 0, callbackBytes = 0, transportError;
  const rawBytes = Buffer.alloc(BODY_LIMIT);
  const capturedBytes = () => rawBytes.subarray(0, Math.min(receivedBytes, BODY_LIMIT));
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
    // Only a compiled adapter can select another policy. No envelope field is
    // interpreted as policy or permits a caller to change transport controls.
    const { request, settings } = requestPolicy(envelope), stream = request.stream;
    timeoutMs ??= settings.requestSeconds * 1000;
    client.preflight.capacity = settings.capacity;
    client.preflight.requested_output_tokens = settings.preflightOutputTokens;
    client.mode = stream ? 'stream' : 'chat';
    client.max_evaluated_tokens = envelope.model_descriptor.execution_limits.max_evaluated_tokens;
    check(Number.isInteger(port) && port > 0 && port <= 65535, 'Invalid loopback port.');
    check(Number.isFinite(timeoutMs) && timeoutMs > 0 && timeoutMs <= settings.requestSeconds * 1000, 'Invalid client deadline.');
    const baseUrl = `http://127.0.0.1:${port}/v1`;
    const deadline = new Promise((_resolve, reject) => {
      timer = setTimeout(() => {
        controller.abort();
        reject(new Error(`Antimatter grounded client exceeded its ${settings.requestSeconds}-second request deadline.`));
      }, timeoutMs);
    });
    globalThis.fetch = async (input, init) => {
      client.attempt_count++;
      if (phase === 'preflight') {
        client.preflight.fetch_attempts++;
        throw new Error('Capacity preflight attempted fetch; transmission refused.');
      }
      check(client.attempt_count === 1, 'A second provider request was refused before transmission.');
      check(input === `${baseUrl}/chat/completions` && init?.method === 'POST'
        && init.redirect === 'error' && init.signal === controller.signal,
      'Unexpected provider request destination or controls.');
      check(typeof init.body === 'string' && Buffer.byteLength(init.body) <= BODY_LIMIT, 'Invalid provider request body.');
      const body = JSON.parse(init.body);
      assert.deepStrictEqual(body, request, 'Provider changed the frozen request body.');
      client.request = copyBounded(body);
      client.request_count++;
      const response = await fetchImpl(input, init);
      evidence.status = response.status;
      evidence.content_type = response.headers.get('content-type') ?? '';
      const declared = response.headers.get('content-length');
      check(declared === null || (/^[0-9]+$/.test(declared) && Number(declared) <= BODY_LIMIT),
        'Oversized declared response body.');
      check(response.body, 'Provider returned no response body.');
      reader = response.body.getReader();
      const expectedType = stream ? /^text\/event-stream(?:;|$)/i : /^application\/json(?:;|$)/i;
      const validTransport = response.status === 200 && expectedType.test(evidence.content_type);
      const decoder = new TextDecoder('utf-8', { fatal: true });
      const bodyStream = new ReadableStream({
        start(value) { delivery = value; },
        cancel() { providerCancelled = true; }, // Still drain the original body to real EOF.
      });
      drain = (async () => {
        try {
          for (;;) {
            const { done, value } = await reader.read();
            const at = elapsed();
            if (done) {
              if (stream && validTransport) capture(decoder.decode(), at, true);
              client.response_complete = true;
              if (!providerCancelled) delivery.close();
              return;
            }
            check(value instanceof Uint8Array && value.byteLength > 0, 'Unexpected response byte stream.');
            const bounded = value.subarray(0, Math.max(0, BODY_LIMIT - receivedBytes));
            rawBytes.set(bounded, receivedBytes);
            receivedBytes += value.byteLength;
            if (stream && validTransport) capture(decoder.decode(bounded, { stream: true }), at);
            check(receivedBytes <= BODY_LIMIT, 'Response body exceeded 1 MiB.');
            if (!providerCancelled) delivery.enqueue(value);
          }
        } catch (error) {
          transportError = error;
          if (!providerCancelled) delivery.error(error);
          throw error;
        } finally {
          if (!stream || !validTransport) evidence.raw_body_base64 = capturedBytes().toString('base64');
          reader.releaseLock();
        }
      })();
      drain.catch(() => {});
      if (!validTransport) {
        // Keep the bounded error response even if the provider would retry or accept a fallback.
        providerCancelled = true;
        await drain;
        throw new Error('Provider response has unexpected status or content type.');
      }
      return new Response(bodyStream, { status: response.status, headers: response.headers });
    };
    const hooks = client.hooks;
    const provider = new Provider({ id: 'isolated-hag-grounded', name: 'Frozen Antimatter grounded HAG client',
      apiKey: '', baseUrl, models: [{ id: request.model, name: request.model, maxEvaluatedTokens: settings.capacity }],
      supportsStreaming: true, kind: 'beholdd', protocol: 'openai-chat' },
    () => {
      hooks.track_calls++;
      if (phase === 'preflight') client.preflight.track_calls++;
      check(hooks.track_calls === 1, 'Provider tracked more than one request.');
      return () => { hooks.track_end_calls++; check(hooks.track_end_calls === 1, 'Provider ended tracking more than once.'); };
    }, async () => {
      hooks.prepare_calls++;
      if (phase === 'preflight') client.preflight.prepare_calls++;
      check(hooks.prepare_calls === 1 && hooks.track_calls === 1 && hooks.track_end_calls === 0,
        'Provider preparation did not occur inside one tracked request.');
      // Deliberately ephemeral: no managed app/source configuration is touched.
      return { baseUrl, apiKey: '' };
    });
    const mapped = { model: request.model, providerId: 'isolated-hag-grounded', requestId: REQUEST_ID,
      messages: request.messages.map((message, index) => ({ id: `fixture-${index}`, ...message })),
      maxTokens: 128, temperature: 0 };
    const operation = (async () => {
      try {
        const oversized = { ...mapped, maxTokens: settings.preflightOutputTokens };
        if (stream) await provider.stream(oversized, () => { throw new Error('Preflight emitted a callback.'); }, undefined, controller.signal);
        else await provider.chat(oversized, controller.signal);
        throw new Error('Provider accepted the impossible capacity preflight.');
      } catch (error) {
        check(/\(context_length_exceeded\)/.test(String(error?.message ?? error)),
          'Provider did not reject the local capacity preflight with context_length_exceeded.');
        client.preflight.error_code = 'context_length_exceeded';
      }
      check(client.preflight.fetch_attempts === 0 && hooks.track_calls === 0
        && hooks.prepare_calls === 0 && hooks.track_end_calls === 0,
      'Capacity preflight reached transport or source lifecycle hooks.');
      phase = 'request';
      started = now();
      const callback = (value) => {
        check(value && typeof value === 'object' && !Array.isArray(value)
          && Object.keys(value).every((key) => ['id', 'content', 'reasoning'].includes(key))
          && value.id === REQUEST_ID && typeof value.content === 'string'
          && (value.reasoning === undefined || typeof value.reasoning === 'string'), 'Unexpected provider callback.');
        callbackBytes += Buffer.byteLength(JSON.stringify(value));
        check(callbackBytes <= BODY_LIMIT && client.callbacks.length < 4096, 'Oversized provider callbacks.');
        client.callbacks.push(copyBounded(value));
      };
      const result = stream ? await provider.stream(mapped, callback, undefined, controller.signal)
        : await provider.chat(mapped, controller.signal);
      client.result = copyBounded(result);
      await drain;
      if (transportError) throw transportError;
      if (!stream) evidence.elapsed_seconds = elapsed();
      check(client.request_count === 1 && client.attempt_count === 1 && client.response_complete,
        'Provider did not complete exactly one response.');
      assert.deepStrictEqual(hooks, { track_calls: 1, prepare_calls: 1, track_end_calls: 1 },
        'Provider lifecycle hooks did not complete once.');
      if (stream) {
        const content = inspectSse(evidence.events, request.model);
        check(result?.content === content && result.finishReason === 'stop' && result.servedModel === request.model
          && Array.isArray(result.toolCalls) && result.toolCalls.length === 0 && result.toolCallsCutOff === false
          && (result.reasoning === undefined || result.reasoning === ''), 'Provider result differs from complete tool-free EOS stream.');
        check(client.callbacks.every((value) => !value.reasoning)
          && client.callbacks.map((value) => value.content).join('') === content,
        'Provider callbacks differ from its complete result.');
      } else {
        const content = inspectChatWithSettings(capturedBytes(), request.model, settings);
        check(typeof result === 'string' && result === content, 'Provider chat result differs from raw EOS response.');
      }
    })();
    await Promise.race([operation, deadline]);
  } catch (error) {
    client.error = String(error?.message ?? error).slice(0, 512);
    controller.abort();
    if (pending && evidence.events.length < 4096) evidence.events.push({ received_seconds: previous, line: pending });
    reader?.cancel().catch(() => {});
  } finally {
    clearTimeout(timer);
    if (client.mode === 'chat') {
      evidence.raw_body_base64 = capturedBytes().toString('base64');
      evidence.elapsed_seconds ??= started === undefined ? 0 : previous;
    }
    globalThis.fetch = originalFetch;
  }
  return evidence;
}

export async function main(args = process.argv.slice(2), requestPolicy = groundedRequestPolicy) {
  const options = parseArguments(args);
  check(await fs.realpath(options.root) === options.root, 'Evidence root must not traverse a symlink.');
  const output = await fs.open(options.output, 'wx');
  try {
    let evidence;
    try {
      const info = await fs.lstat(options.request);
      check(info.isFile() && info.size <= BODY_LIMIT, 'Invalid or oversized request file.');
      const input = await fs.open(options.request, 'r');
      let bytes;
      try {
        bytes = Buffer.alloc(BODY_LIMIT + 1);
        const { bytesRead } = await input.read(bytes, 0, bytes.length, 0);
        check(bytesRead === info.size && bytesRead <= BODY_LIMIT, 'Request file changed while reading.');
        bytes = bytes.subarray(0, bytesRead);
      } finally { await input.close(); }
      const modulePath = path.join(options.root, 'client/provider.mjs');
      check(await fs.realpath(modulePath) === modulePath && (await fs.lstat(modulePath)).isFile(), 'Invalid frozen provider module.');
      const { OpenAICompatibleProvider } = await import(pathToFileURL(modulePath).href);
      const envelope = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
      evidence = await runClient({ Provider: OpenAICompatibleProvider, envelope, port: options.port, requestPolicy });
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
