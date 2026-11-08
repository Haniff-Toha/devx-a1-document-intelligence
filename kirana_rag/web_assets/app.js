"use strict";

let apiOrigin = "";
const apiUrl = (path) => apiOrigin + path;
const byId = (id) => document.getElementById(id);
const question = byId("question");
const form = byId("question-form");
const askButton = byId("ask");
const status = byId("chat-status");
const transcript = byId("transcript");
const conversationScroll = byId("conversation-scroll");
const pageImage = byId("page-image");
const pdfScroll = byId("pdf-scroll");
const paper = byId("paper");
const highlight = byId("highlight");
const previewMessage = byId("preview-message");
const previousPage = byId("previous-page");
const nextPage = byId("next-page");
const openPdf = byId("open-pdf");
const showChat = byId("show-chat");
const showReadme = byId("show-readme");
const compact = matchMedia("(max-width: 52rem)");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const stages = {
  figures: "Finding visuals in the PDF…",
  search: "Searching the report…",
  links: "Following linked evidence…",
  draft: "Preparing an answer from the evidence…",
  verify: "Checking claims and citations…",
};

let documentInfo;
let currentPage = 1;
let pageRequest = 0;
let activeRequest;
let activeButton;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function requestJson(url, options) {
  const response = await fetch(apiUrl(url), options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The local server could not complete the request.");
  return data;
}

function selectView(readme) {
  byId("chat-view").hidden = readme;
  byId("readme-view").hidden = !readme;
  showChat.setAttribute("aria-pressed", String(!readme));
  showReadme.setAttribute("aria-pressed", String(readme));
  const frame = byId("readme");
  if (readme && !frame.hasAttribute("src")) frame.src = apiUrl("/docs/README.md");
}

function markCitation(button) {
  if (activeButton) activeButton.setAttribute("aria-pressed", "false");
  activeButton = button;
  if (activeButton) activeButton.setAttribute("aria-pressed", "true");
}

async function showPage(number, citation) {
  if (!documentInfo || number < 1 || number > documentInfo.pages) return;
  const revision = ++pageRequest;
  currentPage = number;
  byId("page-number").textContent = `Page ${number} of ${documentInfo.pages}`;
  previousPage.disabled = number === 1;
  nextPage.disabled = number === documentInfo.pages;
  openPdf.hidden = false;
  openPdf.href = apiUrl(`/source.pdf#page=${number}`);
  byId("citation-details").hidden = !citation;
  if (citation) {
    byId("citation-title").textContent = citation.label;
    byId("citation-quote").textContent = citation.quote;
  }
  highlight.hidden = true;
  paper.hidden = true;
  previewMessage.textContent = "Loading page preview…";
  previewMessage.hidden = false;
  pageImage.alt = `Page ${number} of ${documentInfo.name}`;
  pageImage.src = apiUrl(`/api/pages/${number}`);
  try {
    await pageImage.decode();
  } catch {
    if (revision === pageRequest) {
      previewMessage.textContent = "The preview could not load. Use Open PDF to view this page.";
    }
    return;
  }
  if (revision !== pageRequest) return;
  previewMessage.hidden = true;
  paper.hidden = false;
  pdfScroll.scrollTop = 0;
  if (citation?.highlight) {
    for (const [key, value] of Object.entries(citation.highlight)) {
      highlight.style[key] = `${100 * value}%`;
    }
    highlight.hidden = false;
    const offset = highlight.getBoundingClientRect().top - pdfScroll.getBoundingClientRect().top;
    pdfScroll.scrollTop = Math.max(0, offset - pdfScroll.clientHeight / 3);
  }
}

function addMessage(label, className) {
  const message = element("article", "", `message ${className}`);
  message.append(element("h3", label));
  transcript.append(message);
  return message;
}

function bindSource(button, cite) {
  button.type = "button";
  button.title = cite.label;
  button.setAttribute("aria-label", `Page ${cite.page} · ${cite.label}`);
  button.setAttribute("aria-pressed", "false");
  button.addEventListener("click", async () => {
    markCitation(button);
    await showPage(cite.page, cite);
    if (compact.matches && activeButton === button) {
      byId("evidence-panel").scrollIntoView({block: "start"});
      byId("evidence-heading").focus({preventScroll: true});
    }
  });
}

function followConversation(update, anchor) {
  const view = conversationScroll;
  const follow = view.scrollHeight - view.clientHeight - view.scrollTop < 72;
  update();
  if (follow) {
    view.scrollTop = anchor
      ? view.scrollTop + anchor.getBoundingClientRect().top - view.getBoundingClientRect().top - 12
      : view.scrollHeight;
  }
}

function revealText(node, text, signal) {
  const words = text.match(/\S+\s*/g) || [];
  if (reducedMotion.matches || words.length < 3) {
    followConversation(() => { node.textContent = text; });
    return;
  }
  return new Promise((resolve) => {
    const duration = Math.min(520, words.length * 21);
    let start;
    function frame(now) {
      start ??= now;
      const fraction = signal.aborted ? 1 : Math.min(1, (now - start) / duration);
      followConversation(() => {
        node.textContent = fraction === 1 ? text : words.slice(0, 1 + Math.floor(fraction * words.length)).join("");
      });
      if (fraction === 1) resolve();
      else requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  });
}

async function addClaim(message, claim, catalog, signal) {
  const block = element("div", "", "claim");
  const statement = element("p");
  block.append(statement);
  followConversation(() => message.append(block));
  await revealText(statement, claim.statement, signal);
  if (catalog) return;
  const links = element("div", "", "citations");
  const seen = new Set();
  for (const cite of claim.citations) {
    const key = `${cite.page}:${cite.label}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const label = cite.label.split(" / ").at(-1);
    const button = element("button", `Page ${cite.page} · ${label}`, "citation");
    bindSource(button, cite);
    links.append(button);
  }
  followConversation(() => block.append(links));
}

function addVisuals(message, visuals) {
  const gallery = element("div", "", "visual-gallery");
  gallery.setAttribute("role", "group");
  gallery.setAttribute("aria-label", "Original PDF visuals");
  for (const visual of visuals) {
    const button = element("button", "", "visual-card");
    bindSource(button, visual.citation);
    const thumb = element("span", "", "visual-thumb");
    const img = element("img");
    img.src = apiUrl(visual.image);
    img.alt = visual.title;
    img.loading = "lazy";
    thumb.append(img);
    button.append(thumb, element("span", visual.title, "visual-title"),
                  element("span", `View on page ${visual.page}`, "visual-meta"));
    gallery.append(button);
  }
  followConversation(() => message.append(gallery), message);
}

async function* streamAnswer(text, signal) {
  const response = await fetch(apiUrl("/api/ask/stream"), {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({question: text}),
    signal,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || "The local server could not start the answer.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      pending += decoder.decode(value, {stream: true});
      const lines = pending.split("\n");
      pending = lines.pop();
      for (const line of lines) {
        if (line) yield JSON.parse(line);
      }
    }
  } finally {
    reader.releaseLock();
  }
}

async function submit(event) {
  event.preventDefault();
  const text = question.value.trim();
  if (!documentInfo || activeRequest || !text) return;
  const request = new AbortController();
  activeRequest = request;
  askButton.type = "button";
  askButton.textContent = "Stop";
  askButton.dataset.stop = "true";
  status.textContent = "";
  byId("welcome").hidden = true;
  addMessage("You", "message-user").append(element("p", text));
  const reply = addMessage("Assistant", "message-answer");
  const progress = element("div", "Reading your question…", "reply-progress");
  progress.setAttribute("role", "status");
  reply.append(progress);
  question.value = "";
  conversationScroll.scrollTop = conversationScroll.scrollHeight;
  try {
    let catalog = false;
    let complete = false;
    for await (const event of streamAnswer(text, request.signal)) {
      if (request.signal.aborted) break;
      if (event.type === "progress") {
        progress.textContent = stages[event.stage] || progress.textContent;
      } else if (event.type === "answer") {
        progress.remove();
        catalog = event.catalog;
        if (event.status === "refused") {
          followConversation(() => reply.append(element("p", event.message, "refusal")));
        }
      } else if (event.type === "claim") {
        await addClaim(reply, event.claim, catalog, request.signal);
      } else if (event.type === "visuals") {
        addVisuals(reply, event.visuals);
      } else if (event.type === "error") {
        throw new Error(event.message);
      } else if (event.type === "done") {
        complete = true;
      }
    }
    if (!complete && !request.signal.aborted) throw new Error("The answer was interrupted. Try asking again.");
  } catch (error) {
    if (!request.signal.aborted) {
      followConversation(() => reply.append(element("p", error.message, "message-error")));
      if (!question.value) question.value = text;
    }
  } finally {
    followConversation(() => {
      progress.remove();
      if (request.signal.aborted) reply.append(element("p", "Response stopped.", "refusal"));
    });
    activeRequest = null;
    askButton.type = "submit";
    askButton.textContent = "Ask";
    delete askButton.dataset.stop;
  }
}

showChat.addEventListener("click", () => selectView(false));
showReadme.addEventListener("click", () => selectView(true));
form.addEventListener("submit", submit);
askButton.addEventListener("click", () => activeRequest?.abort());
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!activeRequest) form.requestSubmit();
  }
});
for (const button of document.querySelectorAll("[data-question]")) {
  button.addEventListener("click", () => {
    question.value = button.dataset.question;
    question.focus();
  });
}
previousPage.addEventListener("click", () => {
  markCitation();
  showPage(currentPage - 1);
});
nextPage.addEventListener("click", () => {
  markCitation();
  showPage(currentPage + 1);
});

requestJson("/config.json").then((config) => {
  apiOrigin = config.apiBaseUrl;
  return requestJson("/api/document");
}).then((data) => {
  documentInfo = data;
  byId("document-name").textContent = data.name.replace(/\.pdf$/i, "").replaceAll("_", " ");
  askButton.disabled = false;
  status.textContent = "";
  showPage(1);
}).catch((error) => {
  byId("document-name").textContent = "Report unavailable";
  previewMessage.textContent = error.message;
  status.textContent = "Check that the local server is running, then reload.";
});
