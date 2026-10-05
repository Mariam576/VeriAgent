# VeriAgent

**Adversarial evidence and document intelligence agent.**

VeriAgent is an evidence-verification agent. It retrieves evidence across business documents, identifies contradictions and missing support, uses deterministic tools for calculations, evaluates whether the evidence is sufficient, and produces a risk assessment in which every fact carries a traceable citation.

It is not a "chat with your PDF" app. It does not simply repeat what a document says. It checks whether the evidence is sufficient, consistent and trustworthy before it reaches a conclusion, and it says so when it is not.

This repository is **Version 1** (the MVP in section 51 of the project specification). See [Roadmap](#roadmap) for what comes next.

```text
Investigation Status: Review Required
Overall Risk Level: High
Confidence Score: 90%
Human Review: Required

Key Findings:
1. [Fact] Contract.pdf states a total amount of £40,000. [Contract.pdf, page 4]
2. [Fact] Invoice_102.pdf states a total amount of £58,000. [Invoice_102.pdf, page 1]
5. [Calculation] Difference in total amount: £18,000. Invoice_102.pdf is £18,000 above Contract.pdf (45% more).
11. [Missing evidence] Supporting evidence was not found: Written approval or a signed variation ...

Recommended Action:
Do not approve automatically. Obtain clarification and formal written approval before processing.
```

## Contents

- [Quick start](#quick-start)
- [Try it](#try-it)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [API](#api)
- [Project structure](#project-structure)
- [Tests](#tests)
- [What has and has not been verified](#what-has-and-has-not-been-verified)
- [Security notes](#security-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

The learning guide in [`docs/LEARNING_GUIDE.md`](docs/LEARNING_GUIDE.md) explains each concept (RAG, embeddings, state, nodes, edges, tool calling, structured output, guardrails) and points to the file where it is implemented.

## Quick start

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Compose 2.24 or later) and a free Gemini API key from <https://aistudio.google.com/apikey>.

In PowerShell, from the project folder:

```powershell
# 1. Create your settings file
copy .env.example .env

# 2. Open it and paste your key after GEMINI_API_KEY=
notepad .env

# 3. Build and start the API and the database
docker compose up --build
```

The first build downloads PyTorch and takes several minutes. When the log shows `VeriAgent 1.0.0 started`, open:

- <http://localhost:8000/docs> for the interactive API pages
- <http://localhost:8000/health> to see the status and the configured providers

To stop, press `Ctrl+C`, then run `docker compose down`. Add `-v` to also delete the database and uploaded files.

**No API key yet?** Set `LLM_PROVIDER=mock` in `.env`. The mock is a rule-based stand-in that needs no key and makes no network calls. It lets you watch the whole pipeline run, but it is not a language model.

## Try it

Open a second PowerShell window in the project folder.

**1. Check that your key and model work** (4 small requests):

```powershell
docker compose exec api python -m scripts.check_llm
```

**2. Run the demo investigation.** This uploads the three sample documents in `samples/` and asks whether the invoice should be approved:

```powershell
docker compose exec api python -m scripts.run_demo
```

The first upload downloads the embedding model (about 90 MB), so it can take a minute.

**3. See how a prompt-injection attempt is handled.** This adds a note that tells "any AI assistant" to approve the invoice and report low risk:

```powershell
docker compose exec api python -m scripts.run_demo --with-injection
```

**4. Ask your own questions** at <http://localhost:8000/docs>:

1. `POST /documents/upload` to upload a PDF, DOCX or TXT file.
2. `POST /investigations` with `{"question": "Compare the payment terms across these documents."}`.
3. `GET /investigations/{id}/findings` to see the stored findings.

Or use `POST /chat` for a conversation. Send the `conversation_id` you get back with your next message, and a follow-up such as "What about the payment terms?" is understood in context.

## How it works

### The investigation graph

One LangGraph graph with ten nodes. It branches where a step may not be needed and loops where it may need to try again. Both loops are bounded.

```text
                      START
                        |
                 analyze_request ........ LLM: what is being asked? which searches?
                  /            \
        needs RAG                no RAG
            |                      |
        retrieve <-----------+     |      embeddings + pgvector + rerank (no LLM)
            |                |     |
     extract_evidence <------|-----+      LLM extracts facts; code checks each one
            |                |            against the source text
   detect_contradictions     |            typed comparison (no LLM)
       /          \          |
  calculate        |         |            calculator tools; the LLM may request more
       \          /          |
    check_sufficiency -------+            LLM: enough evidence? if not, search again
            |                             (at most MAX_AGENT_ITERATIONS times)
       assess_risk                        LLM proposes; rules set the limits
            |
     review_findings <----+               rule-based reviewer
       /          \       |
  approved     not approved
      |            |      |
      |     correct_findings
      |
  final_report                            assembled by code from structured findings
      |
     END
```

### Where the LLM is trusted, and where it is not

| Step | Who does it | Why |
| --- | --- | --- |
| Understand the request, extract facts, judge sufficiency, explain risk | LLM, with structured (Pydantic) output | Reading and judging varied text is what a language model is good at |
| Arithmetic | Python calculator tools | A model predicts likely text; Python is always right |
| Comparing values across documents | Python | Repeatable and explainable |
| File names and page numbers in citations | Copied from the database by code | The model never writes a citation, so it cannot invent one |
| Minimum risk level, maximum confidence | Fixed rules | "Risk must be based on evidence, not model intuition alone" |
| Final check of the draft | Rule-based reviewer | Catches unsupported claims, invented figures and overconfidence |

### How hallucinated citations are prevented

1. The model only sees chunk references such as `C2`. It never writes a file name or page.
2. Code resolves `C2` to the stored chunk and copies the file name, page and ids from the database. An unknown reference is discarded.
3. The quote the model gives must appear in that chunk, and a numeric fact's number must be inside the quote. Otherwise the fact is discarded and counted under "discarded facts".
4. The reviewer rejects any reason without supporting ids and any figure that appears in neither the documents nor a tool result.

### How prompt injection is handled

- Document text reaches the model only inside `<<<UNTRUSTED DOCUMENT CONTENT ...>>>` blocks in the user turn. The system prompt, which has priority, says that such text is evidence and never instructions.
- A document cannot close its own block: the marker characters are replaced before wrapping.
- Text that looks like an instruction to an AI is detected by pattern, reported as a finding with its source page, and raises the risk level.
- If the model is fooled anyway (for example it answers "low risk, approve"), the rule-based reviewer overrules it, because the minimum risk level comes from the evidence, not from the model.

### What happens when things go wrong

| Situation | Behaviour |
| --- | --- |
| Missing or wrong API key, unknown model | The investigation stops with HTTP 503 and the provider's message. It never continues silently. |
| Timeout, rate limit, server error | Retried up to `LLM_MAX_RETRIES` times with backoff. Then the step falls back to simple rules and the report says so under "Limitations". |
| Not enough evidence | The missing items are listed. Nothing is assumed. |
| Loop limit reached | The investigation stops, returns partial findings and states the limitation. |
| Unreadable or scanned PDF | The upload is rejected with a clear reason and recorded as `failed`. |

## Configuration

All settings are environment variables, documented in [`.env.example`](.env.example). The most important ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini` | `gemini`, `grok` or `mock` |
| `GEMINI_API_KEY` | empty | Your key. Never commit it; `.env` is in `.gitignore`. |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Any text model your key can use. See <https://ai.google.dev/gemini-api/docs/models>. |
| `LLM_REQUESTS_PER_MINUTE` | `0` (no limit) | Set to `10` if the free tier returns rate-limit errors. One investigation makes about 5 to 7 requests. |
| `EMBEDDING_PROVIDER` | `local` | `local` (free SentenceTransformer), `gemini`, or `hash` (tests only) |
| `EMBEDDING_DIM` | `384` | Must match the embedding model. The database column is created with this size. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Characters per chunk and overlap between neighbours |
| `RAG_TOP_K` / `RERANK_TOP_K` | `8` / `4` | Retrieve 8 candidates per query, keep the best 4 |
| `MAX_RETRIEVED_CHUNKS` | `12` | Upper limit on chunks read in one investigation |
| `MAX_AGENT_ITERATIONS` | `6` | Upper limit on "search again" loops |
| `API_PORT` / `DB_PORT` | `8000` / `5433` | Ports on your PC |

### Switching the LLM provider

The application only ever calls `get_llm()` (`app/llm/factory.py`). To use Grok instead of Gemini, change two lines in `.env` and restart:

```env
LLM_PROVIDER=grok
GROK_API_KEY=...
GROK_MODEL=...
```

No code changes. To add another provider, copy `app/llm/grok_provider.py`, build that provider's LangChain chat model, and add one branch in `factory.py`.

### Changing the embedding model

If you change `EMBEDDING_PROVIDER` or the model, set `EMBEDDING_DIM` to the new model's size, delete the database volume with `docker compose down -v`, and upload your documents again. Vectors from different models cannot be compared.

## API

All endpoints accept an optional `X-User-Id` header (any short name). Each user sees only their own documents and investigations. This header is a placeholder for real authentication; see [Security notes](#security-notes).

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status and configured providers (no secrets) |
| `POST` | `/documents/upload` | Upload a PDF, DOCX or TXT file |
| `GET` | `/documents` | List your documents |
| `POST` | `/investigations` | Run an investigation and return the full report |
| `GET` | `/investigations` | List your investigations |
| `GET` | `/investigations/{id}` | Read a stored report |
| `GET` | `/investigations/{id}/findings` | Read the stored findings, one per row |
| `POST` | `/chat` | Ask a question inside a conversation |

`POST /investigations` accepts optional filters: `document_ids`, `document_type` and `year`.

Example chat request and response:

```json
{ "message": "Compare the invoice amount with the contract." }
```

```json
{
  "conversation_id": "…",
  "investigation_id": "…",
  "answer": "Review required. Contract.pdf states the total amount as £40,000 but Invoice_102.pdf states £58,000. …",
  "risk_level": "high",
  "confidence": 0.9,
  "status": "review_required",
  "requires_human_review": true,
  "citations": [
    { "document": "Contract.pdf", "page": 4, "document_id": "…" },
    { "document": "Invoice_102.pdf", "page": 1, "document_id": "…" }
  ]
}
```

## Project structure

```text
app/
  main.py              FastAPI app, error handlers, startup
  config.py            All settings, read from environment variables
  numbers.py           One parser for numbers and units (40,000 / 1,50,000 / 40.000,00)
  api/                 HTTP endpoints: health, documents, investigations, chat
  agents/
    state.py           The typed LangGraph state (with reducers)
    graph.py           Nodes and edges wired together
    nodes.py           The ten node functions
    routing.py         Conditional edges
    prompts.py         System prompt, task prompts, untrusted-content wrapping
    assessment.py      Rule-based risk floor, confidence ceiling, reviewer
    report.py          Final report assembly and rendering
    heuristics.py      Rule-based fallbacks and the offline mock LLM
    service.py         Runs one investigation and stores the result
  llm/                 Provider abstraction: base, Gemini, Grok, mock, factory
  embeddings/          Provider abstraction: local, Gemini, hash, factory
  rag/                 loaders, chunking, ingestion, retriever, reranker, citations
  tools/               calculator (@tool), contradiction detection, tool allowlist
  database/            SQLAlchemy models, session, repositories
  schemas/             Pydantic models: LLM outputs, domain objects, API shapes
  security/            validation, permissions (auth placeholder), guardrails
scripts/               check_llm, run_demo, make_sample_docs
samples/               Demo contract, invoice, policy, and an adversarial note
tests/                 Unit, agent and API integration tests
docs/                  Learning guide
```

## Tests

```powershell
docker compose exec api python -m pytest -q
```

The tests use the offline mock LLM and hash embeddings, so they need no API key and cost nothing. Integration tests use the running PostgreSQL and clean up after themselves.

What the tests cover, by topic:

- **Spec demo scenario**: contract £40,000 against invoice £58,000 gives a difference of £18,000 from the calculator tool, a payment-terms mismatch, missing approval, risk High and human review required.
- **Missing evidence**: "Was this invoice approved?" with no approval document reports the gap and does not claim approval.
- **Prompt injection**: injected instructions are reported and not followed, a document cannot break out of its wrapper, and a model that obeys the injection is overruled.
- **Hallucinated citations**: invented chunk references, figures and quotes are discarded.
- **Loop protection**: the iteration limit stops the loop and the report states it.
- **Tool calling**: requested calls are validated against the allowlist and the evidence before they run.
- **Failure handling**: provider outage, wrong key, search failure, bad files.
- **Isolation**: one user cannot search, read or reference another user's data.
- **Gemini code path**: request building, structured output, tool calls and error handling, with the network replaced by a fake transport.

## What has and has not been verified

This matters for a project about evidence, so here it is plainly.

**Verified by running it**

- The full test suite (276 tests) against PostgreSQL 16 with pgvector 0.6.0 and 0.8.1.
- The complete flow over HTTP against a running server: upload, investigate, chat, read findings.
- The Gemini provider code against the real Gemini SDK, with HTTP responses supplied by a fake transport.
- `docker compose config` accepts the compose file, and the pinned requirements resolve on Python 3.12.

**Not verified, because the build environment had no access to them**

- A real `docker compose up --build`. The Dockerfile and compose file were reviewed line by line but never built.
- A real call to the Gemini API. Run `python -m scripts.check_llm` first; it exercises the same request types the agent uses.
- The local SentenceTransformer embedding model, which is downloaded from Hugging Face on first use. Retrieval quality with real embeddings should be better than with the test embeddings, but it was not measured.
- The Grok provider against the real xAI API.

If one of these fails on your machine, the error message and [Troubleshooting](#troubleshooting) should point to the cause.

## Security notes

Implemented in Version 1:

- File type allowlist, size limit, and a check that the content matches the extension
- Stored files are named by a generated id, never by the uploaded name
- Every query filters by the current user; other users' ids return "not found"
- SQL is parameterised through SQLAlchemy
- A tool allowlist with argument validation; only calculator tools exist, and they have no side effects
- API keys are held as secrets, never returned by `/health`, and redacted from reports, errors and logs
- Ports are bound to `127.0.0.1`, so the API and database are reachable only from your own computer
- The container runs as a non-root user

**Not production-ready:**

- `X-User-Id` is an authentication placeholder. Anyone who can reach the API can claim any user. Replace `get_current_user` in `app/security/permissions.py` with real authentication before exposing the API.
- The database password in `docker-compose.yml` is `postgres`. Change it for any shared machine.
- There is no rate limiting on the API.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `GEMINI_API_KEY is not set` | Add the key to `.env`, then run `docker compose up` again. |
| `gemini rejected the request (HTTP 400/403/404)` | The key is wrong or `GEMINI_MODEL` is not available to your key. Check both, then run `scripts.check_llm`. |
| Report says "The language model was unavailable ..." | A rate limit or outage. Set `LLM_REQUESTS_PER_MINUTE=10` and try again later. |
| `port is already allocated` | Another program uses the port. Set `API_PORT` or `DB_PORT` in `.env`. |
| `The database stores N-dimensional embeddings but EMBEDDING_DIM is M` | You changed the embedding model. Run `docker compose down -v` and upload again. |
| First upload is slow | The embedding model is being downloaded once. It is cached in a Docker volume. |
| `No text could be extracted` | The PDF is scanned images. OCR is not part of Version 1. |
| Changed `.env` but nothing changed | Restart: `docker compose up -d --force-recreate api`. |

To look inside the database from your PC: host `localhost`, port `5433`, user `postgres`, password `postgres`, database `veriagent`.

## Roadmap

**Version 1 (this repository):** Gemini with a replaceable provider layer, FastAPI, PDF/DOCX/TXT upload, text extraction with page numbers, chunking, local or Gemini embeddings, PostgreSQL with pgvector, RAG with simple reranking, the LangGraph agent, calculator tools, contradiction checking, the structured report with citations, guardrails, and Docker.

Known limits of Version 1:

- Only two quantities are compared across documents: the total amount and the payment terms in days.
- Retrieval is semantic only. Exact identifiers such as `INV-129` rely on the reranker's keyword boost until hybrid search is added.
- A DOCX file has no fixed pages, so all of its text is cited as page 1.
- Uploads are processed inside the request, and a request that is interrupted can leave an investigation marked `running`.
- "Human review required" is a flag in the report. Nothing pauses and waits for a person yet.

**Version 2:** hybrid (vector + keyword) search, a dedicated reranking model, LangGraph checkpointing with pause and resume, a human-approval interrupt and `POST /investigations/{id}/approve`, structured and semantic long-term memory, an evaluation suite with metrics, tracing (LangSmith or Langfuse), streaming of progress events, a Streamlit UI.

**Version 3:** MCP integration (for example creating a GitHub issue after approval), advanced document comparison, a document forensics module, a React frontend, production authentication, cloud deployment.

## Portfolio description

> **VeriAgent - Adversarial Document Intelligence System.** Built an agentic AI platform for evidence-based analysis of enterprise documents. The system uses LangGraph for stateful orchestration, RAG with PostgreSQL/pgvector for grounded document retrieval, Gemini for reasoning behind a provider-independent interface, deterministic tools for calculations, contradiction and missing-evidence detection, rule-bounded risk assessment, citations that the model cannot invent, and FastAPI/Docker for deployment. Designed with provider abstraction, prompt-injection guardrails and a test suite that covers hallucinated citations, injection and loop protection.
