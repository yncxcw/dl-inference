"use strict";

// Chat UI for the DLI OpenAI-compatible server. No build step, no libraries.

const STORAGE_KEY = "dli-chat-v1";
const DEFAULT_SETTINGS = {
  "system-prompt": "",
  temperature: "0.7",
  "top-p": "0.8",
  "max-tokens": "1024",
  thinking: false,
  "api-key": "",
};

const $ = (id) => document.getElementById(id);
const store = load();
let model = null;
let controller = null;

function load() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && Array.isArray(saved.messages)) {
      return {
        messages: saved.messages.filter((m) => !m.pending),
        settings: { ...DEFAULT_SETTINGS, ...saved.settings },
      };
    }
  } catch (_) {
    // Storage can be unavailable (private windows); the page still works.
  }
  return { messages: [], settings: { ...DEFAULT_SETTINGS } };
}

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch (_) {
    // Ignore: persistence is a convenience.
  }
}

function headers() {
  const result = { "Content-Type": "application/json" };
  if (store.settings["api-key"]) result.Authorization = `Bearer ${store.settings["api-key"]}`;
  return result;
}

// Rendering: escaped text with fenced code blocks, inline code and bold.

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function renderText(text) {
  return text
    .split("```")
    .map((part, index) => {
      if (index % 2 === 1) {
        const newline = part.indexOf("\n");
        return `<pre><code>${escapeHtml(newline >= 0 ? part.slice(newline + 1) : part)}</code></pre>`;
      }
      return escapeHtml(part)
        .replace(/`([^`\n]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");
    })
    .join("");
}

// Qwen thinking mode wraps reasoning in <think>…</think>; the prompt may
// already contain the opening tag, so a lone closing tag also counts.
function splitThinking(text) {
  const open = text.indexOf("<think>");
  const close = text.indexOf("</think>");
  if (close < 0) {
    return open < 0 ? { reasoning: "", answer: text } : { reasoning: text.slice(open + 7), answer: "" };
  }
  return { reasoning: text.slice(open < 0 ? 0 : open + 7, close), answer: text.slice(close + 8).replace(/^\s+/, "") };
}

function messageHtml(message) {
  if (message.role !== "assistant") return renderText(message.content);
  const { reasoning, answer } = splitThinking(message.content);
  const thinking = reasoning.trim()
    ? `<details class="thinking"${answer ? "" : " open"}><summary>Thinking</summary>${renderText(reasoning.trim())}</details>`
    : "";
  return thinking + renderText(answer) + (message.pending ? '<span class="cursor"></span>' : "");
}

function renderMessage(element, message) {
  element.className = `message ${message.role}${message.error ? " error" : ""}`;
  element.querySelector(".body").innerHTML = messageHtml(message);
  element.querySelector(".meta").textContent = message.stats || "";
}

function render() {
  const container = $("messages");
  container.querySelectorAll(".message").forEach((node) => node.remove());
  $("empty-state").hidden = store.messages.length > 0;
  for (const message of store.messages) {
    const element = document.createElement("div");
    element.innerHTML = '<div class="body"></div><div class="meta"></div>';
    renderMessage(element, message);
    container.append(element);
  }
  container.scrollTop = container.scrollHeight;
}

function updateLast() {
  const container = $("messages");
  const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
  const elements = container.querySelectorAll(".message");
  renderMessage(elements[elements.length - 1], store.messages[store.messages.length - 1]);
  if (nearBottom) container.scrollTop = container.scrollHeight;
}

// Server communication.

async function errorMessage(response) {
  try {
    const body = await response.json();
    return body.error?.message || `HTTP ${response.status}`;
  } catch (_) {
    return `HTTP ${response.status}`;
  }
}

async function* serverSentEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of block.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        yield JSON.parse(data);
      }
    }
  }
}

async function connect() {
  const dot = $("status-dot");
  try {
    const response = await fetch("/v1/models", { headers: headers() });
    if (!response.ok) throw new Error(await errorMessage(response));
    const models = await response.json();
    model = models.data[0].id;
    $("model-name").textContent = `${model} · ${models.data[0].max_model_len ?? "?"} token context`;
    dot.className = "dot ok";
  } catch (error) {
    model = null;
    $("model-name").textContent = `Not connected: ${error.message}`;
    dot.className = "dot bad";
  }
  refreshStats();
}

async function refreshStats() {
  try {
    const stats = await (await fetch("/stats")).json();
    const cache = stats.prefix_cache;
    const rows = {
      Requests: stats.requests,
      "Active / pending": `${stats.active} / ${stats.pending}`,
      "Prompt tokens": stats.prompt_tokens,
      "Cached prompt tokens": stats.cached_tokens,
      "Generated tokens": stats.generated_tokens,
      "Mean step": stats.mean_step_ms == null ? "–" : `${stats.mean_step_ms} ms`,
      "Prefix cache": `${cache.entries}/${cache.capacity} entries`,
      "Cache hits / misses": `${cache.hits} / ${cache.misses}`,
    };
    $("server-stats").innerHTML = Object.entries(rows)
      .map(([name, value]) => `<dt>${escapeHtml(name)}</dt><dd>${escapeHtml(String(value))}</dd>`)
      .join("");
  } catch (_) {
    $("server-stats").textContent = "";
  }
}

function requestBody() {
  const settings = store.settings;
  const messages = [];
  if (settings["system-prompt"].trim()) messages.push({ role: "system", content: settings["system-prompt"].trim() });
  for (const message of store.messages) {
    if (message.pending || message.error || !message.content) continue;
    // Earlier reasoning is not sent back, as Qwen recommends.
    const content = message.role === "assistant" ? splitThinking(message.content).answer : message.content;
    messages.push({ role: message.role, content });
  }
  return {
    model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    temperature: Number(settings.temperature),
    top_p: Number(settings["top-p"]),
    max_tokens: Number(settings["max-tokens"]),
    chat_template_kwargs: { enable_thinking: Boolean(settings.thinking) },
  };
}

function describe(usage, started, firstToken, finished, finishReason) {
  if (!usage) return finishReason || "";
  const parts = [`${usage.completion_tokens} tokens`];
  if (firstToken) {
    parts.push(`first token ${((firstToken - started) / 1000).toFixed(2)} s`);
    const seconds = (finished - firstToken) / 1000;
    if (seconds > 0 && usage.completion_tokens > 1) parts.push(`${((usage.completion_tokens - 1) / seconds).toFixed(1)} tok/s`);
  }
  const cached = usage.prompt_tokens_details?.cached_tokens ?? 0;
  parts.push(`${cached}/${usage.prompt_tokens} prompt tokens cached`);
  if (finishReason && finishReason !== "stop") parts.push(`finish: ${finishReason}`);
  return parts.join(" · ");
}

async function send(text) {
  if (!model) await connect();
  if (!model) return;
  store.messages.push({ role: "user", content: text });
  const reply = { role: "assistant", content: "", pending: true };
  store.messages.push(reply);
  render();
  setBusy(true);
  controller = new AbortController();
  const started = performance.now();
  let firstToken = null;
  let usage = null;
  let finishReason = null;
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify(requestBody()),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    for await (const event of serverSentEvents(response)) {
      if (event.error) throw new Error(event.error.message);
      if (event.usage) usage = event.usage;
      for (const choice of event.choices || []) {
        if (choice.delta?.content) {
          firstToken ??= performance.now();
          reply.content += choice.delta.content;
          updateLast();
        }
        if (choice.finish_reason) finishReason = choice.finish_reason;
      }
    }
    reply.stats = describe(usage, started, firstToken, performance.now(), finishReason);
  } catch (error) {
    if (error.name === "AbortError") {
      reply.stats = "Stopped";
    } else {
      reply.error = true;
      reply.stats = `Error: ${error.message}`;
    }
  } finally {
    delete reply.pending;
    controller = null;
    setBusy(false);
    updateLast();
    save();
    refreshStats();
  }
}

// Controls.

function setBusy(busy) {
  const button = $("send");
  button.textContent = busy ? "Stop" : "Send";
  button.classList.toggle("stop", busy);
  button.type = busy ? "button" : "submit";
}

function bindSettings() {
  for (const [key, fallback] of Object.entries(DEFAULT_SETTINGS)) {
    const input = $(key);
    const output = $(`${key}-value`);
    if (typeof fallback === "boolean") input.checked = Boolean(store.settings[key]);
    else input.value = store.settings[key];
    if (output) output.textContent = input.value;
    input.addEventListener("input", () => {
      store.settings[key] = typeof fallback === "boolean" ? input.checked : input.value;
      if (output) output.textContent = input.value;
      save();
    });
  }
  $("api-key").addEventListener("change", connect);
}

function autoResize() {
  const prompt = $("prompt");
  prompt.style.height = "auto";
  prompt.style.height = `${prompt.scrollHeight}px`;
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = $("prompt");
  const text = prompt.value.trim();
  if (!text || controller) return;
  prompt.value = "";
  autoResize();
  send(text);
});

$("send").addEventListener("click", () => {
  if (controller) controller.abort();
});

$("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});
$("prompt").addEventListener("input", autoResize);

$("new-chat").addEventListener("click", () => {
  if (controller) controller.abort();
  store.messages = [];
  save();
  render();
  $("prompt").focus();
});

bindSettings();
render();
connect();
