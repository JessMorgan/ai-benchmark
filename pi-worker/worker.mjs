#!/usr/bin/env node
/**
 * One-request Pi SDK worker for ai-benchmark.
 *
 * stdin: one JSON request
 * stdout: pi-worker-v1 NDJSON events
 * stderr: diagnostics only (never credentials or request bodies)
 */
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import readline from "node:readline";
import { Agent } from "@earendil-works/pi-agent-core";
import {
  ModelRuntime,
  createCodingTools,
  createReadOnlyTools,
} from "@earendil-works/pi-coding-agent";

const PROTOCOL = "pi-worker-v1";
const WORKER_VERSION = "1.0.0";
let activeAgent;
let outputChain = Promise.resolve();

if (process.argv.includes("--preflight")) {
  process.stdout.write(JSON.stringify({
    protocol: PROTOCOL,
    worker_version: WORKER_VERSION,
    sdk_version: "@earendil-works/pi-coding-agent@0.84.2",
  }) + "\n");
  process.exit(0);
}

function emit(event, attempt, data = {}) {
  const record = {
    protocol: PROTOCOL,
    event,
    attempt: Number.isInteger(attempt) ? attempt : 1,
    timestamp: Date.now() / 1000,
    data,
  };
  outputChain = outputChain.then(async () => {
    process.stdout.write(`${JSON.stringify(record)}\n`);
    if (process.stdout.flush) process.stdout.flush();
  });
  return outputChain;
}

function baseUrl(apiUrl) {
  const parsed = new URL(apiUrl);
  let path = parsed.pathname.replace(/\/+$/u, "");
  for (const suffix of ["/chat/completions", "/completions"]) {
    if (path.endsWith(suffix)) {
      path = path.slice(0, -suffix.length);
      break;
    }
  }
  parsed.pathname = path.replace(/\/+$/u, "");
  parsed.search = "";
  parsed.hash = "";
  return parsed.toString().replace(/\/$/u, "");
}

function redactedError(error) {
  const message = error instanceof Error ? error.message : String(error);
  return message
    .replace(/Bearer\s+[^\s"']+/giu, "Bearer [REDACTED]")
    .replace(/sk-[A-Za-z0-9_-]+/gu, "[REDACTED]");
}

function providerConfig(request, providerId) {
  const source = request.source ?? {};
  const headers = source.headers && typeof source.headers === "object" ? source.headers : {};
  const customHeaders = {};
  let apiKey = typeof source.api_key === "string" && source.api_key
    ? source.api_key : "ai-benchmark";
  let authHeader = Boolean(source.api_key);
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() === "authorization") {
      const match = /^Bearer\s+(.+)$/iu.exec(String(value));
      if (match) {
        apiKey = match[1];
        authHeader = true;
      } else {
        customHeaders[key] = String(value);
      }
    } else if (key.toLowerCase() !== "content-type") {
      customHeaders[key] = String(value);
    }
  }
  const compat = source.pi?.compat ?? {
    supportsDeveloperRole: false,
    supportsReasoningEffort: false,
  };
  const model = {
    id: String(request.api_model),
    name: String(request.api_model),
    reasoning: Boolean(request.reasoning),
    input: ["text"],
    contextWindow: Number(request.context_window) || 131072,
    maxTokens: Number(request.max_tokens),
    compat,
    ...(request.temperature === undefined || request.temperature === null
      ? {}
      : { samplingParams: { temperature: Number(request.temperature) } }),
  };
  return {
    providerId,
    config: {
      name: `AI Benchmark (${request.source?.name ?? "source"})`,
      baseUrl: baseUrl(String(source.api_url)),
      api: "openai-completions",
      apiKey,
      authHeader,
      headers: customHeaders,
      models: [model],
    },
  };
}

function selectedTools(request) {
  const names = Array.isArray(request.tools) ? request.tools.map(String) : [];
  if (names.length === 0) return [];
  const cwd = typeof request.cwd === "string" && request.cwd ? request.cwd : process.cwd();
  const all = [...createCodingTools(cwd), ...createReadOnlyTools(cwd)];
  const byName = new Map(all.map((tool) => [tool.name, tool]));
  const permissions = request.permissions && typeof request.permissions === "object"
    ? request.permissions
    : {};
  const tools = [];
  for (const name of names) {
    if (!byName.has(name)) throw new Error(`Unsupported Pi tool: ${name}`);
    if (permissions[name] === "deny") continue;
    if (!tools.some((tool) => tool.name === name)) tools.push(byName.get(name));
  }
  return tools;
}

function textFromMessage(message) {
  if (!message || !Array.isArray(message.content)) return "";
  return message.content
    .filter((part) => part?.type === "text")
    .map((part) => part.text ?? "")
    .join("");
}

function thinkingFromMessage(message) {
  if (!message || !Array.isArray(message.content)) return "";
  return message.content
    .filter((part) => part?.type === "thinking")
    .map((part) => part.thinking ?? "")
    .join("");
}

async function run(request) {
  const attempt = Number.isInteger(request.attempt) ? request.attempt : 1;
  const source = request.source;
  if (!source || !source.api_url || !request.api_model || !request.prompt) {
    throw new Error("Pi worker request requires source.api_url, api_model, and prompt");
  }
  const providerId = `benchmark-${String(source.name ?? "source")
    .toLowerCase().replace(/[^a-z0-9]+/gu, "-").replace(/^-|-$/gu, "") || "source"}`;
  const provider = providerConfig(request, providerId);
  const agentDir = await mkdtemp(join(tmpdir(), "ai-benchmark-pi-"));
  const modelsPath = join(agentDir, "models.json");
  await writeFile(modelsPath, JSON.stringify({ providers: { [provider.providerId]: provider.config } }), { mode: 0o600 });
  let runtime;
  try {
    emit("worker_started", attempt, {
      worker_version: WORKER_VERSION,
      sdk_version: "@earendil-works/pi-coding-agent@0.84.2",
    });
    runtime = await ModelRuntime.create({
      authPath: join(agentDir, "auth.json"),
      modelsPath,
      allowModelNetwork: false,
      refreshOnCreate: false,
    });
    const model = runtime.getModel(provider.providerId, String(request.api_model));
    if (!model) throw new Error("Pi could not register the configured provider/model");
    const tools = selectedTools(request);
    const toolNames = tools.map((tool) => tool.name);
    let toolCalled = false;
    let toolCallCount = 0;
    const maxToolCalls = Math.max(0, Number(request.max_tool_calls) || 50);
    let finalMessage;
    let sawTextDelta = false;
    let sawThinkingDelta = false;
    activeAgent = new Agent({
      initialState: {
        systemPrompt: typeof request.system_prompt === "string" ? request.system_prompt : "",
        model,
        thinkingLevel: request.reasoning ? "high" : "off",
        tools,
      },
      streamFn: (streamModel, context, options = {}) => runtime.streamSimple(streamModel, context, {
        ...options,
        maxTokens: Number(request.max_tokens),
        timeoutMs: Number(request.timeout_ms) || undefined,
        maxRetries: 0,
        temperature: request.temperature === null ? undefined : request.temperature,
        thinkingBudgets: request.thinking_budgets,
      }),
      beforeToolCall: async ({ toolCall }) => {
        toolCalled = true;
        toolCallCount += 1;
        const permissions = request.permissions && typeof request.permissions === "object"
          ? request.permissions : {};
        if (toolCallCount > maxToolCalls) {
          activeAgent.abort();
          return { block: true, reason: "Pi tool-call budget exhausted", terminate: true };
        }
        if (permissions[toolCall.name] === "deny") {
          return { block: true, reason: `Pi tool ${toolCall.name} is denied by benchmark policy`, terminate: true };
        }
        await emit("tool_started", attempt, { name: toolCall.name, tool_call_id: toolCall.id });
        return undefined;
      },
    });
    activeAgent.subscribe(async (event) => {
      if (event.type === "message_update") {
        const update = event.assistantMessageEvent;
        if (update?.type === "text_delta" && update.delta) {
          sawTextDelta = true;
          await emit("text_delta", attempt, { text: update.delta });
        } else if (update?.type === "thinking_delta" && update.delta) {
          sawThinkingDelta = true;
          await emit("reasoning_delta", attempt, { text: update.delta });
        }
      } else if (event.type === "tool_execution_end") {
        await emit("tool_finished", attempt, {
          tool_call_id: event.toolCallId,
          is_error: Boolean(event.isError),
        });
      } else if (event.type === "message_end" && event.message?.role === "assistant") {
        finalMessage = event.message;
        await emit("usage", attempt, { usage: event.message.usage ?? {} });
      } else if (event.type === "agent_end") {
        const messages = Array.isArray(event.messages) ? event.messages : [];
        finalMessage = [...messages].reverse().find((message) => message?.role === "assistant") ?? finalMessage;
      }
    });
    await emit("session_started", attempt, {
      provider: provider.providerId,
      model: request.api_model,
      tools: toolNames,
      prompt_altered: request.prompt_altered ?? "none",
      max_tokens: Number(request.max_tokens),
      max_tool_calls: maxToolCalls,
    });
    await emit("prompt_started", attempt, {});
    await activeAgent.prompt(String(request.prompt));
    const text = textFromMessage(finalMessage);
    const thinkText = thinkingFromMessage(finalMessage);
    if (!sawTextDelta && text) await emit("text_delta", attempt, { text });
    if (!sawThinkingDelta && thinkText) await emit("reasoning_delta", attempt, { text: thinkText });
    const usage = finalMessage?.usage ?? {};
    const finishReason = finalMessage?.stopReason ?? "stop";
    await emit("finish", attempt, {
      text_length: text.length,
      thinking_length: thinkText.length,
      usage,
      finish_reason: finishReason,
      truncated: finishReason === "length",
      tool_called: toolCalled,
      tools: toolNames,
      provider: provider.providerId,
      model: request.api_model,
    });
    await emit("worker_finished", attempt, { ok: !finalMessage?.errorMessage });
    return 0;
  } finally {
    activeAgent = undefined;
    await rm(agentDir, { recursive: true, force: true });
  }
}

let input = "";
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  if (!input) input = line;
});
rl.on("close", async () => {
  let request = {};
  try {
    request = JSON.parse(input || "{}");
    await run(request);
    await outputChain;
    process.exitCode = 0;
  } catch (error) {
    const attempt = Number.isInteger(request.attempt) ? request.attempt : 1;
    await emit("error", attempt, {
      message: redactedError(error),
      error_type: error?.constructor?.name ?? "Error",
    });
    await emit("worker_finished", attempt, { ok: false });
    await outputChain;
    process.exitCode = 1;
  }
});

const abort = () => {
  if (activeAgent) activeAgent.abort();
};
process.on("SIGTERM", abort);
process.on("SIGINT", abort);
