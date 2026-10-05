# VeriAgent learning guide

This guide explains the ideas behind VeriAgent in the order the project was built, and shows where each idea lives in the code. Read a section, open the file it names, then run the "Try it" step.

Every commit in the git history is one milestone, so `git log --oneline` is also a table of contents.

- [Milestone 1: skeleton, settings, database, Docker](#milestone-1-skeleton-settings-database-docker)
- [Milestone 2: LLM and embedding providers](#milestone-2-llm-and-embedding-providers)
- [Milestone 3: the document pipeline](#milestone-3-the-document-pipeline)
- [Milestone 4: retrieval, citations and tools](#milestone-4-retrieval-citations-and-tools)
- [Milestone 5: the LangGraph agent](#milestone-5-the-langgraph-agent)
- [Milestone 6: the API](#milestone-6-the-api)
- [Milestone 7: hardening](#milestone-7-hardening)
- [Concepts planned for later versions](#concepts-planned-for-later-versions)
- [Interview questions](#interview-questions)

All commands assume the stack is running (`docker compose up`) and are typed in a second PowerShell window in the project folder.

---

## Milestone 1: skeleton, settings, database, Docker

**What was built:** the folder structure, settings, database tables, a `/health` endpoint and the Docker files.

**Why:** everything else needs a place to live, a way to read configuration and a database to store into.

| File | Role |
| --- | --- |
| `app/config.py` | Every setting, read from environment variables. No model name or key is hardcoded anywhere else. |
| `app/database/models.py` | The tables. `document_chunks.embedding` is the pgvector column. |
| `app/database/session.py` | The async engine, one session per request, table creation at startup. |
| `Dockerfile`, `docker-compose.yml` | Two containers: `api` (FastAPI) and `db` (PostgreSQL with pgvector). |

**Concept: async.** Talking to a database or an LLM is mostly waiting. With `async`/`await`, the server handles other requests while one request waits, instead of blocking a thread. CPU-heavy work (parsing a PDF, running a local embedding model) gains nothing from `async`, so it is moved to a worker thread with `asyncio.to_thread`.

**Try it:**

```powershell
curl.exe http://localhost:8000/health
```

Expected: `"status":"ok"`, `"database":"ok"`, and your provider names. No key appears.

---

## Milestone 2: LLM and embedding providers

**What was built:** `get_llm()` and `get_embedding_provider()`, each with replaceable providers behind one interface.

**Why:** the rest of the application must not care whether Gemini, Grok or a test double answers.

| File | Role |
| --- | --- |
| `app/llm/base.py` | The interface (`structured`, `choose_tools`) plus retries, timeouts, rate limiting and fallback, written once. |
| `app/llm/langchain_adapter.py` | Turns any LangChain chat model into that interface. |
| `app/llm/gemini_provider.py`, `grok_provider.py` | Each only builds the right chat model. |
| `app/llm/factory.py` | `get_llm()`: the single place a provider is chosen. |
| `app/embeddings/` | The same pattern for embeddings: `embed_text` and `embed_documents`. |

**Concept: provider abstraction.** Search the code for `ChatGoogleGenerativeAI`: it appears in one file. Everything else depends on `BaseLLM`. That is why switching provider is a change to `.env`.

**Concept: structured output.** Instead of asking for prose and parsing it, the model must fill in a Pydantic schema. `llm.structured(RiskAssessment, ...)` returns an object whose `level` is one of four allowed words. Decisions are read from typed fields, never guessed from sentences. Open `app/schemas/investigation.py` and note that every field is required: a provider may skip optional fields, and a skipped field would silently disable a later step.

**Concept: embeddings.** An embedding model turns text into a list of numbers. Texts with similar meaning get similar lists. Comparing lists is how "search by meaning" works.

**Concept: bounded retries.** `_call_with_retries` in `base.py` tries at most `LLM_MAX_RETRIES + 1` times, with a timeout per attempt. It distinguishes two failures. A temporary one (timeout, rate limit) is retried. A permanent one (wrong key, unknown model) stops immediately, because retrying cannot help and carrying on would hide the cause.

**Try it:**

```powershell
docker compose exec api python -m scripts.check_llm
```

Expected: four lines ending in `OK`. Then put a wrong character in your key, restart, and run it again to see the clear error.

---

## Milestone 3: the document pipeline

**What was built:** upload, validation, text extraction, chunking, embedding and storage.

```text
upload -> validate -> save metadata -> extract text per page -> clean
       -> chunk -> embed -> store chunks + vectors + metadata
```

| File | Role |
| --- | --- |
| `app/security/validation.py` | Extension allowlist, size limit, and a check that the bytes really are that format. |
| `app/rag/loaders.py` | PDF (PyMuPDF), DOCX (python-docx), TXT. Returns text per page. |
| `app/rag/chunking.py` | Recursive chunking with overlap. |
| `app/rag/ingestion.py` | Runs the pipeline; records failures. |

**Concept: why chunk?** One vector for a 40-page contract is a blurry average of everything in it. One vector per paragraph is sharp: a search for "payment terms" lands on the paragraph about payment terms. Small chunks also keep prompts short.

**Concept: recursive chunking and overlap.** Split on paragraphs first; if a piece is still too long, split on lines, then sentences, then words. Neighbouring chunks share about 120 characters, so a sentence cut at a boundary is still whole in one of them.

**Concept: never lose the source.** The page number is attached when the text is extracted and stays with every chunk. Without it, a citation like `[Contract.pdf, page 4]` would be impossible later.

**Try it:** open <http://localhost:8000/docs>, use `POST /documents/upload` with `samples/Contract.pdf`, then `GET /documents`. Expected: `page_count` 5 and `status` `ready`. Upload it again: `duplicate` is `true` and nothing is re-embedded.

---

## Milestone 4: retrieval, citations and tools

**What was built:** semantic search with filters, a reranker, citation and grounding checks, calculator tools and contradiction detection.

| File | Role |
| --- | --- |
| `app/rag/retriever.py` | One SQL statement: metadata filters plus vector distance. |
| `app/rag/reranker.py` | Re-scores candidates and keeps the best few. |
| `app/rag/citations.py` | Builds citations in code and checks quotes against source text. |
| `app/numbers.py` | Reads numbers in the formats documents use. |
| `app/tools/calculator.py` | Calculator functions decorated with `@tool`. |
| `app/tools/registry.py` | The tool allowlist and the only function that executes a tool. |
| `app/tools/contradiction.py` | Typed comparison of values across documents. |

**Concept: RAG.** Retrieval-augmented generation means: embed the question, fetch the closest chunks, and give only those to the model. The model answers from evidence that we chose and can cite, and the prompt stays small.

**Concept: pgvector.** A `vector` column plus a distance operator (`<=>`) inside PostgreSQL. Users, documents, chunks and findings live in one database, and a search can combine "closest vectors" with ordinary `WHERE` conditions such as "only this user's documents".

**Concept: reranking.** Fetch more than you need (8), score them more carefully, pass on the best (4). Here the second score mixes vector similarity with the share of query words found in the chunk, and exact codes like `INV-102` count double because embeddings are weak at exact matches.

**Concept: deterministic tools.** `58,000 - 40,000` usually comes out right when a model does it, but nothing guarantees it. Python is always right. The model decides which calculation is needed; Python performs it.

**Concept: @tool.** The decorator turns a function into something a model can request. The function name, type hints and docstring are sent to the model as the tool's description, so the docstrings in `calculator.py` are written for the model to read.

**Try it:**

```powershell
docker compose exec api python -c "from app.tools.registry import execute_tool; print(execute_tool('calculate_difference', {'value_a': 58000, 'value_b': 40000}))"
docker compose exec api python -c "from app.tools.registry import execute_tool; print(execute_tool('delete_everything', {}))"
```

Expected: the first prints `status='ok'` with `{'value': 18000.0}`; the second prints `status='rejected'` because the tool is not on the allowlist.

---

## Milestone 5: the LangGraph agent

**What was built:** the state, ten nodes, the conditional edges, loop protection, guardrails and the report.

| File | Role |
| --- | --- |
| `app/agents/state.py` | What the investigation knows at each moment. |
| `app/agents/nodes.py` | One function per step. |
| `app/agents/routing.py` | The decisions between steps. |
| `app/agents/graph.py` | Nodes and edges wired together, with a diagram. |
| `app/agents/prompts.py` | The system prompt and the task for each step. |
| `app/agents/assessment.py` | The rules that bound the model: risk floor, confidence ceiling, reviewer. |
| `app/security/guardrails.py` | Untrusted wrapping, injection detection, secret redaction. |

**Concept: state.** The state is the investigation's shared notebook. A node receives it, does one job and returns only the keys it changed. Nodes never call each other; they communicate through the state.

**Concept: reducers.** By default a returned key replaces the old value. A key with a reducer is combined instead: `llm_calls: Annotated[int, operator.add]` means a node can return `{"llm_calls": 1}` and the counter goes up by one. `messages` uses `add_messages`, which appends.

**Concept: nodes and edges.** A node is a step. An edge says which step comes next. Together they are the control flow, drawn as a graph instead of hidden in a prompt.

**Concept: conditional edges.** After some nodes, a small routing function looks at the state and returns the name of the next node. That is how the graph branches ("is a search needed?") and loops ("not enough evidence, search again").

**Concept: agent loop and loop protection.** The graph has two loops. "Retrieve again" is counted in `iteration` and stops at `MAX_AGENT_ITERATIONS`. "Correct and review again" stops at `MAX_REVIEW_ROUNDS`. LangGraph's own `recursion_limit` is a third, outer limit. When a limit is hit, the investigation returns what it has and says so. It does not invent an answer.

**Concept: tool calling.** In `calculate`, the model receives the calculator tools and the numeric evidence. It answers with a request such as `calculate_ratio(58000, 40000)`. Python then checks that the tool is allowed, that the arguments have the right types, and that each number really exists in the evidence, and only then runs it. The result goes back into the message history as a `ToolMessage`.

**Concept: guardrails.** Controls that are plain code, so they hold even if the model misbehaves: maximum iterations, maximum chunks, file limits, the tool allowlist, schema validation, the risk floor and the reviewer.

**Concept: prompt injection.** A document may say "ignore previous instructions". VeriAgent treats that as content. Document text is wrapped in clearly marked untrusted blocks, the system prompt outranks it, and the attempt itself is reported as a risk.

**Try it:**

```powershell
docker compose exec api python -m scripts.run_demo
docker compose exec api python -m scripts.run_demo --with-injection
```

Compare the two reports. In the second one, look for the note about instructions to an AI system, and check that the recommendation did not change to "approve".

Then read the server log in the first window. Each line starting with `node_entered` shows the path the graph took, including the second pass through `retrieve`.

---

## Milestone 6: the API

**What was built:** `POST /investigations`, `POST /chat`, the read endpoints and storage of results.

| File | Role |
| --- | --- |
| `app/agents/service.py` | Runs the graph for one question and stores the report, findings and tool calls. |
| `app/api/investigations.py`, `chat.py` | The endpoints. |
| `app/security/permissions.py` | The current user (authentication placeholder). |
| `app/database/repositories.py` | Every query, each filtered by user. |

**Concept: short-term memory (Version 1 form).** `POST /chat` stores each turn. The next question is analysed together with the last few turns, and rewritten as a standalone question. "What about the payment terms?" becomes a question about the contract and invoice discussed before.

**Concept: observability.** Each tool call is stored in the `tool_calls` table with its arguments, result, status and duration. The log records each node entered and left, each LLM call with its latency, and the ids of retrieved chunks.

**Try it:** at <http://localhost:8000/docs>, call `POST /chat` with `{"message": "Compare the invoice amount with the contract."}`. Copy the `conversation_id` from the answer and call it again with `{"conversation_id": "...", "message": "What about the payment terms?"}`.

---

## Milestone 7: hardening

A second reviewer read the code and tried to break it. The fixes are a useful lesson in how such systems fail in practice:

- **Silent degradation.** A wrong API key used to produce a rule-based report that looked like success. A fallback that hides the real problem is worse than an error.
- **Grounding that is too loose.** "The number appears somewhere in the chunk" accepted an invented sentence. The number now has to be inside the quoted passage.
- **Grounding that is too strict.** Real PDF text has line breaks in the middle of words. A strict check discards true facts, which also lowers confidence.
- **Number formats.** `1,50,000` and `40.000,00` are normal in real documents. One shared parser now reads them.
- **False alarms.** Pattern-based injection detection flagged "please disregard the previous invoice". Each pattern now needs wording aimed at instructions or at an AI.

The tests added for each of these are in `tests/`. Reading a test next to the code it protects is a good way to understand both.

---

## Concepts planned for later versions

These are in the specification but not in Version 1. Knowing where they will go is part of understanding the design.

- **Checkpointing.** LangGraph can save the state after every node under a `thread_id`. The investigation id is already passed around for this purpose. The state holds only plain data so that it can be saved.
- **Human-in-the-loop.** With checkpointing, a node can call `interrupt()`, the graph pauses, a person approves or rejects, and the graph resumes. Version 1 only sets the flag `requires_human_review`.
- **Long-term memory.** Stable preferences belong in a normal key-value table. Unstructured memories ("we investigated Vendor ABC for duplicate invoicing") belong in a table with embeddings.
- **Hybrid search.** Combining vector search with keyword search, for exact identifiers.
- **Evaluation.** A dataset of questions with expected sources and answers, and metrics such as citation accuracy and unsupported-claim rate. The target for hallucinated citations is zero.
- **Streaming.** The `events` list in the state already holds safe progress messages ("Searching documents...") that a user interface can show. Model reasoning is never streamed.

---

## Interview questions

**Why use RAG instead of passing whole PDFs?**
Cost, focus and traceability. A prompt with only the four most relevant chunks is cheaper, distracts the model less, and every chunk has a known file and page, so each claim can be cited.

**Why pgvector?**
The vectors live next to the relational data. One query combines similarity with ownership and metadata filters, and there is one database to run, back up and secure.

**Why chunk documents?**
A vector for a whole document is too vague to search with, and whole documents do not fit usefully into a prompt. Chunks give precise matches and precise citations.

**Why rerank?**
The first search is fast and approximate. Fetching a few extra candidates and re-scoring them keeps the prompt small while improving what goes into it.

**Why LangGraph?**
The task branches and loops. A graph makes the control flow, the state and the loop limits explicit and testable, instead of hoping a single prompt follows a procedure.

**What is stored in state?**
The question, the retrieved chunks, the evidence, contradictions, calculations, the sufficiency judgement, the draft assessment, counters and limits, and an audit trail of events and tool calls. See `app/agents/state.py`.

**How does tool calling work?**
The model is shown tool descriptions and replies with a request: a tool name and arguments. The application validates the request and runs the function. The model never executes anything itself.

**Why deterministic calculator tools?**
A wrong figure in a financial review is unacceptable, and a language model cannot guarantee arithmetic. Python can.

**How does contradiction detection work?**
Facts are extracted with a typed key (`total_amount`, `payment_terms_days`) and a numeric value. Code groups them by key and compares the values across documents. The contract or purchase order is treated as the reference.

**How do you prevent hallucinated citations?**
The model never writes a file name or page. It points to a chunk reference; code copies the source from the database, and checks that the quote, and any number, really is in that chunk.

**What happens when evidence is missing?**
The sufficiency step lists what is missing and may trigger another search. If it is still missing, the report says so, lowers confidence, and requires human review. Nothing is assumed.

**Why is human approval needed?**
A contradiction is a reason to look, not proof of wrongdoing, and the consequences of approving or rejecting a payment belong to a person.

**How do you prevent prompt injection?**
Document text is wrapped as untrusted data under a system prompt that outranks it, suspicious text is detected and reported, and rule-based limits apply whatever the model says.

**Why not multiple agents?**
One graph with ordinary nodes does the job. More agents add cost, latency and failure modes. They would be justified only by a step that needs a truly independent view.

**How would you switch Gemini to Grok?**
Set `LLM_PROVIDER=grok` with its key and model. Only `app/llm/factory.py` knows which provider is in use.

**How do you evaluate the system?**
Version 1 has scenario tests for each behaviour the specification names. Version 2 adds a dataset and metrics, with hallucinated citations as the number to hold at zero.

**How would you deploy it?**
The API is a stateless container and PostgreSQL holds all state, so it suits any container platform with a managed PostgreSQL that has pgvector. Before that: real authentication, secrets from a secret store, and migrations with Alembic.
