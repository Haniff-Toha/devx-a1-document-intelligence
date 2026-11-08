"use strict";

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

let documentInfo;
let currentPage = 1;
let pageRequest = 0;
let busy = false;
let activeButton;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
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
  if (readme && !frame.hasAttribute("src")) frame.src = "/docs/README.md";
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
  openPdf.href = `/source.pdf#page=${number}`;
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
  pageImage.src = `/api/pages/${number}`;
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

function addAnswer(result) {
  const message = addMessage("Assistant", "message-answer");
  if (result.status === "refused") message.append(element("p", result.message, "refusal"));
  for (const claim of result.claims) {
    const block = element("div", "", "claim");
    block.append(element("p", claim.statement));
    const links = element("div", "", "citations");
    for (const cite of claim.citations) {
      const button = element("button", `Page ${cite.page}`, "citation");
      button.type = "button";
      button.title = cite.label;
      button.setAttribute("aria-label", `Page ${cite.page} · ${cite.label}`);
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", async () => {
        markCitation(button);
        await showPage(cite.page, cite);
        if (activeButton === button && matchMedia("(max-width: 52rem)").matches) {
          byId("evidence-panel").scrollIntoView({block: "start"});
        }
      });
      links.append(button);
    }
    block.append(links);
    message.append(block);
  }
}

async function submit(event) {
  event.preventDefault();
  const text = question.value.trim();
  if (!documentInfo || busy || !text) return;
  busy = true;
  askButton.disabled = true;
  status.textContent = "Searching the report and checking citations…";
  byId("welcome").hidden = true;
  addMessage("You", "message-user").append(element("p", text));
  question.value = "";
  conversationScroll.scrollTop = conversationScroll.scrollHeight;
  try {
    const result = await requestJson("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: text}),
    });
    addAnswer(result);
  } catch (error) {
    addMessage("Could not answer", "message-error").append(element("p", error.message));
    if (!question.value) question.value = text;
  } finally {
    busy = false;
    askButton.disabled = false;
    status.textContent = "";
    conversationScroll.scrollTop = conversationScroll.scrollHeight;
  }
}

showChat.addEventListener("click", () => selectView(false));
showReadme.addEventListener("click", () => selectView(true));
form.addEventListener("submit", submit);
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
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

requestJson("/api/document").then((data) => {
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
