# A1 evaluation

The benchmark contains 18 development questions written against the supplied Kirana report. Four are deliberately unanswerable. The other 14 cover prose, native tables, a table spanning pages, an infographic, bar and line charts, a network diagram, a scanned tariff, and an erratum. Expected answers and evidence requirements are in [questions.json](questions.json). [results.json](results.json) contains the responses, quotes, and traces. [ingestion.json](ingestion.json) records the input hash and page coverage.

## Extraction coverage

| Evidence in the source | Verified in the index |
|---|---:|
| PDF pages with indexed evidence | 13/13 |
| Native table rows across five tables | 41/41 |
| KPI tiles on page 3 | 6/6 |
| Labeled values across the four charts | 25/25 |
| Labeled diagram connections on page 7 | 7/7 |
| Scanned tariff rows and notes on page 13 | 7/7 rows and 2/2 notes |

The table check compares headers and body fingerprints against an independent Poppler text extraction. It also checks that individual indexed rows carry the correct column labels. Visual checks use expectations read from the rendered PDF and look for the labeled values in the indexed vision records. The route check covers endpoints and printed rates. The original drawing shows direction and layout.

The audit caught a blank first header cell in Table 4. The previous parser treated the Scope 1 data row as the header, which lost the year-to-value mapping. The corrected parser retains the 2023–2025 headers and all three rows. The coverage check rejects the old index, and a live question about all three Scope 1 years returned the correct values with a page 9 citation.

## Retrieval

The four stages use the same ingested corpus. A page hit requires a gold page. An evidence hit requires a snippet on a gold page containing the expected value and its identifying terms. A case is complete when every required evidence group is present. Some cases require two groups. The four unanswerable questions are excluded from retrieval scoring.

| Stage | All required pages | All required evidence | Evidence groups found |
|---|---:|---:|---:|
| BM25, top 7 | 14/14 | 13/14 | 17/18 |
| BM25 + vector fusion, top 7 | 13/14 | 12/14 | 16/18 |
| Hybrid with up to 3 initial graph neighbors | 14/14 | 14/14 | 18/18 |
| Final agent context | 14/14 | 14/14 | 18/18 |

The graph restored evidence that the top-seven hybrid ranking missed. It connected the two pages of the facilities table and attached the tariff's qualifications to the rate. Keyword search alone was strong on this small report, and vector fusion alone lost one complete case relative to BM25. The graph stage had room for three extra records, so the table measures the behavior of the shipped retrieval stages at their configured context budgets.

## Answers and failures

| Check | Result |
|---|---:|
| Correct answer text | 13/14 |
| Correct answer with all required citation evidence | 13/14 |
| Intentional refusal for an unanswerable question | 4/4 |
| Provider errors surfaced to the evaluator | 0/18 |

Answer text is scored against the case's value, unit, or wording patterns. Citation coverage is scored separately against the contents of the returned quotes. The checks are case-specific and can miss subtle wording or entailment errors. The per-question outputs are available for inspection.

- **A12, safety rate.** Retrieval returned the correct passage on page 10. The model repeatedly added the current year to its answer even though that passage states the prior year explicitly and leaves the current one implicit. The numeric guard withheld the draft. Resolving an implicit period from report and section context is the next improvement.
- **Earlier chart and scan failures.** In a pilot, the retriever missed “Nov” for a query using “November.” It also ranked the cold-chain rate without its VAT qualification, and a draft associated the frozen-storage rate with the cold-chain request. Calendar label normalization, links to figure notes and scan transcripts, and claim review address those errors. In the recorded run the November and scanned-tariff questions both passed.
- **New document.** The same CLI indexed a two-page synthetic Northstar report. It answered **43%** from native text on page 1 and **THB 74 per order** from the raster-only tariff on page 2. The automated test separately verifies that re-ingesting a different PDF into the same index removes stale evidence. This is a smoke test for the refresh flow.

## Resource use

On the Windows test machine with Python 3.11, the audited run produced **154 evidence records** and **794 directed graph links** across all 13 pages. Ingestion took **91.6 seconds**, and the index occupied **1.27 MiB**. An earlier resource profiling run sampled a peak working set of **439 MiB** across the Python process tree and an installed environment of **289 MiB**, excluding the dependency cache.

OpenRouter reported **$0.00817** for the recorded ingestion and **$0.00816** for the evaluation. The evaluation took **25.8 seconds** wall time with three workers and a **3.43-second** median per question after the batch query embedding. These figures describe the recorded runs and vary with document size, agent actions, and provider latency.

## Reproduce

```powershell
uv run --locked python -m kirana_rag ingest "Kirana_Logistics_Annual_Report_2025.pdf"
uv run --locked python -m eval.coverage
uv run --locked python -m kirana_rag evaluate
uv run --locked python -m unittest discover -v
```

The benchmark uses the supplied report as a development set. The scores describe one OpenRouter run with the model IDs recorded in the results. Provider routing and model responses can vary. A synthetic two-page PDF with different facts exercises the replacement flow independently.
