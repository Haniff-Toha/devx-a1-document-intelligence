# Kirana Document Intelligence

**devx AI Labs take-home, A1.** Ask questions about Kirana Logistics Group's annual report and get answers tied to the PDF page, table row, chart, or passage. The assistant can search again, follow a relationship in the evidence graph, or decline when it cannot support the answer.

[Customer summary](CUSTOMER_SUMMARY.md) · [Evaluation report](eval/REPORT.md) · [Benchmark questions](eval/questions.json) · [Per-question results](eval/results.json)

## Run it

Use [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.11–3.13, and an [OpenRouter key](https://openrouter.ai/keys) with credits. Save the supplied annual report as `Kirana_Logistics_Annual_Report_2025.pdf` in the repository root or point the ingest command at its location.

From the repository root in PowerShell, run the first command, then set `OPENROUTER_API_KEY` in the resulting `.env` before ingesting. Git ignores the key, original PDF, and generated index.

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
uv sync --locked
uv run --locked python -m kirana_rag ingest "Kirana_Logistics_Annual_Report_2025.pdf"
uv run --locked python -m kirana_rag web "Kirana_Logistics_Annual_Report_2025.pdf"
```

Open [the local chatbot](http://127.0.0.1:8765) in your browser. Use **Chat** to ask a standalone question and **README** to read this guide. Select a page citation to jump to the physical PDF page and highlight the indexed evidence region. For scans and figures, the highlight can cover the whole image. **Open PDF** opens the original file at the selected page in the browser's PDF viewer.

The HTML interface uses the same agent and index as the CLI. It binds to loopback, and OpenRouter calls use the key on the server. For a second document, ingest it first, then restart the viewer with that PDF. Pass `--index .rag/second.sqlite3` to `ingest` and `web` or `ask` to keep separate indexes. The viewer checks the PDF's hash against the index before showing citations.

```powershell
uv run --locked python -m kirana_rag ask "How many electric trucks were added in 2025? Explain the conflicting figures." --trace
uv run --locked python -m kirana_rag ask "What is the 2026 standard cold-chain pallet storage rate, and does it include VAT?" --json
uv run --locked python -m kirana_rag ask "What volume discount rate does Schedule C grant for 100 pallets?"
uv run --locked python -m kirana_rag inspect --page 13
uv run --locked python -m kirana_rag evaluate
uv run --locked python -m unittest discover -v
```

Use `search "<query>" --mode keyword` for raw BM25 hits and `--mode hybrid` for the fused ranking. Add `--json` to an answer to see exact indexed quotes, bounding boxes, extraction methods, and the search trace. `ingest "<PDF>" --local-only` plus keyword search gives a CPU-only evidence browser that works without a key.

## Architecture and decisions

```text
PDF → native text, tables, OCR, and vision → SQLite evidence and graph
Question → BM25 + cosine search → agent searches again or follows links
Agent draft → quote, number, and claim checks → cited answer or refusal
Citation → original PDF page and highlighted evidence region
```

| Choice | Why |
|---|---|
| **PyMuPDF + RapidOCR** | Native layout preserves text and table geometry without rasterizing each page. A CPU OCR model, bundled with the package, provides a separate transcription of scanned text. I considered Docling for layout detection, TableFormer, and page provenance. The native parser recovered the report's five named tables, including the two-page facilities table, so this build uses the smaller dependency and model footprint. [Docling PDF pipeline documentation](https://docling-project.github.io/docling/usage/advanced_options/) |
| **OpenRouter for embeddings, chat, and vision** | These are the model-heavy steps. Hosted inference lets the app run on a CPU-only machine with limited RAM and disk, and keeps model weights and serving infrastructure off the PC. The same API and key serve all three workloads. The trade-offs are internet access, variable provider latency, API costs, and sending document excerpts, selected image crops, and questions to hosted providers. |
| **Gemini Flash Lite and Text Embedding 3 Small** | Gemini 2.5 Flash Lite selects retrieval actions, drafts answers, and reviews claims. Gemini 3.1 Flash Lite transcribes visual evidence at ingestion. The 1,536-dimensional embeddings let semantic matches compete with exact entity names and values. Model IDs are configurable in `.env`. Re-ingest when the embedding model changes. [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs) · [Embeddings API](https://openrouter.ai/docs/api_reference/embeddings) |
| **SQLite + NumPy** | A single report is small enough for an exact cosine scan. A single local file stores the document hash, evidence, vectors, and typed links. This keeps the demo easy to reset and makes every graph hop inspectable. |

### From PDF to answer

1. **Extract evidence.** Walk every page. Split long prose into up to 120-word chunks with 20-word overlap. Keep short paragraphs intact, carry section names, and index each table as rows with headers plus a page-level table view. Repeated table headers connect consecutive pages. Raster figures and the scan retain both local OCR and targeted vision records, each marked with its extraction method and region.
2. **Build the graph.** Evidence nodes link to their sections, neighboring passages, table rows and continuations, figure notes, and shared identifiers. Errata can link back to an earlier passage when the opening correction text shares a value and topic. Named corroborating sections use a separate edge type. These relationships are generated from the current PDF.
3. **Retrieve and investigate.** Take the top seven results using BM25 and exact cosine similarity, combined with reciprocal-rank fusion. Preserve two strong keyword hits for names and other exact terms, and diversify repeated table or figure hits. Add directly connected corrections, continued tables, or figure notes. The model can then request up to two more steps through a revised search or a graph expansion. `--trace` shows the chosen action, IDs, and links.
4. **Check the answer.** Each claim needs an evidence ID. Code uses the exact indexed record or verifies the quoted substring, confirms each numeric value appears in the supporting text, and attaches page numbers and labels. A separate model call reviews whether the draft answers the question and follows from the cited row, entity, unit, and period. Source text is treated as data, and the agent can only search or follow links. If evidence is insufficient or validation fails, it withholds the claim.

## Evaluation and limitations

The [evaluation report](eval/REPORT.md) compares keyword, hybrid, and graph retrieval on the same questions and scores answer quality and citation coverage separately. The [raw results](eval/results.json) show each answer, citation, and trace. Run the benchmark on the supplied Kirana report. The replacement-PDF test generates a different two-page document with a raster-only second page.

- The current scope is one English-language PDF per index. Table grouping is tuned for native rows with regular column alignment. Heavily merged tables and irregular reading order can lose structure.
- Raster figures and scanned pages are supported through OCR and vision. Vector-only diagrams, very small images, unprinted chart values, and ambiguous relationships can be missed. Check important visual values against the page.
- Exact quotations, number checks, and a separate review catch different classes of citation error. Semantic support still depends on model interpretation, including whether a number is associated with the right row, unit, or year.
- Graph edges for corrections are heuristic. An unfamiliar erratum layout can be missed, and weak links can introduce extra context. Retrieval comparisons use this report's development benchmark, so the measured differences are specific to these questions.
- OpenRouter rate limits and network failures can interrupt a run. Requests use bounded retries. A failed re-ingestion preserves the previous index, and `inspect` shows which PDF is loaded.

### With another week

I would label a held-out set of different report layouts and compare Docling's layout and table extraction against this pipeline. I would also add a citation viewer that jumps to the stored bounding box and measure whether independent verification reduces unsupported claims on cross-page and visual questions.

### AI tool use

Codex helped inspect the brief, design and implement the pipeline, write tests and evaluation cases, and review measured failures. The reported scores come from running the checked-in evaluator against the supplied PDF. The application's OpenRouter models perform vision transcription, embeddings, and bounded answer generation during runtime.
