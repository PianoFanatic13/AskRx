# [AskRx] — Product Requirements Document

## Overview

A web-based AI agent that helps users understand their medications. Users ask natural
language questions about drugs — what a medication does, what side effects to watch for,
whether two drugs are safe to take together — and receive plain-language, sourced answers
drawn from the FDA's DailyMed database.

The core technical pattern is a RAG pipeline over a large corpus of FDA drug labels,
orchestrated by a ReAct agent capable of multi-hop reasoning across multiple drug labels.
This allows the agent to handle complex queries like "I take lisinopril and ibuprofen, is
that safe?" by synthesizing information across multiple sources rather than returning a
single retrieved chunk.

---

## Problem Statement

People are frequently prescribed medications they don't fully understand. Finding reliable
information means either navigating dense FDA documents not written for patients, or
searching the web and landing on inconsistent, sometimes alarming results. The FDA
publishes comprehensive, authoritative drug information through DailyMed, but in its raw
form it is effectively inaccessible to non-experts. This project makes that information
queryable in plain language.

---

## Target Users

General public — patients, caregivers, or anyone wanting to understand medications they or
someone they care for is taking. No medical background assumed.

---

## Core Features (MVP)

### Natural Language Query Interface

Users interact through a simple chat-style interface. Queries can be about a single drug
or describe a multi-drug situation. No structured input required — users type naturally
and the agent handles interpretation.

### Drug Name Resolution

Resolves brand names to generic names (e.g. "Tylenol" to "acetaminophen") and handles
common variations. Uses RxNorm for standardized drug name mapping. This runs before
retrieval so queries are matched against the right documents regardless of how a user
names a drug.

### Drug Information Retrieval

Retrieves relevant information from FDA DailyMed drug labels covering key sections:
indications and usage, warnings and precautions, adverse reactions, and dosing
considerations. Returns information in plain language with source attribution showing
which drug label the answer came from. Powered by a hybrid retrieval pipeline combining
keyword and semantic search.

### Drug Interaction Information

Handles queries involving multiple drugs. The agent retrieves the Drug Interactions
section (LOINC 34073-7) from each relevant drug's label and surfaces that text to the
user with explicit framing that this is what the labels say, not a complete interaction
check, and that the user should consult a pharmacist.

This is a deliberate scoping decision. NLM permanently retired the RxNav Drug-Drug
Interaction API in January 2024, and there is no free authoritative interaction
adjudication endpoint available. Rather than ship an unreliable adjudicator, the MVP
honestly surfaces label-sourced interaction text. The tool must never assert that two
drugs do not interact — it can only report what the labels state or note that a label
does not mention a specific interaction. Authoritative pairwise interaction adjudication
(via a licensed source like DrugBank, Lexicomp, or Micromedex) is a post-MVP path.

### ReAct Agent Orchestration

A ReAct agent decides which tools to call and in what order based on the user's query.
Three tools in the MVP:

- **Drug name resolution** — maps brand or informal names to standardized generics via RxNorm
- **Drug info retrieval** — hybrid RAG retrieval over FDA label sections
- **Interaction information retrieval** — targeted retrieval of the Drug Interactions section from each relevant label, surfaced with pharmacist-routing language (not authoritative adjudication)

For multi-drug queries the agent chains multiple tool calls, pulling from separate drug
labels and synthesizing across them before generating a response.

### Source Attribution

Every chunk in the index carries structured provenance metadata, and every factual claim
in a generated answer cites a specific source. The citation format is the document SETID,
the nearest ancestor LOINC section code, and the section title path (e.g. "Dosage and
Administration → Renal Impairment"). Subsections often lack their own LOINC code, so each
chunk inherits the nearest ancestor's code while the section title path carries the
specificity. This guarantees every chunk has a traceable citation.

### Hallucination Guardrails and Safety Routing

Because this is a medical tool, safety behaviors are core MVP features, not afterthoughts.
The primary failure class to guard against is the confident false negative — the model
asserting two drugs do not interact when they do. Required behaviors:

- **Citation-enforced generation** — the model cites a specific retrieved source for every
  factual claim, or explicitly states it does not have a reliable source for that claim.
- **"I don't know" fallback** — when retrieved chunks do not support an answer with
  confidence, the model declines to answer rather than generating from parametric memory.
- **No confident interaction negatives** — the model never asserts that drugs do not
  interact; it only reports what a label says or notes that the label is silent on it.
- **Persistent disclaimer** — every response carries a short disclaimer that the
  information comes from FDA labels, is informational only, and is not medical advice.
- **Pharmacist routing on high-risk questions** — interaction questions, dosing changes
  or missed doses for high-risk drugs (anticoagulants, insulin, antiepileptics,
  immunosuppressants), pregnancy and lactation, pediatric dosing, and overdose questions
  always include explicit "consult your pharmacist or doctor" language regardless of what
  retrieval returns.

### Conversation Memory

The agent retains context across turns within a session. Users can describe their
situation once ("I take metformin and lisinopril") and ask follow-up questions without
restating that context each time. Session state is stored in Redis and passed to the
agent on each turn. This is what makes the experience genuinely differentiated from a
standard drug information search — the agent understands the user's ongoing situation
rather than treating each message in isolation.

---

## Technical Architecture

> The exact tools and libraries may shift during development. This reflects the current
> direction and will be updated as decisions are finalized.

### Data Layer

- **Primary source:** FDA DailyMed bulk drug label dataset (SPL XML)
- **Raw data storage:** S3 (FDA data files stored before processing)
- **Drug name standardization:** RxNorm — `findRxcuiByString` for exact/normalized lookup,
  `getApproximateMatch` as typo-tolerant fallback (called in sequence: exact first, fuzzy
  second). Consider running RxNav-in-a-Box locally to avoid the 20 req/sec rate limit
  during bulk ingestion. Explicit fallback behavior required when RXCUI lookup fails
  (supplements, herbals, compounded drugs are not covered) so a failed lookup never
  silently produces an empty or wrong retrieval.

### Retrieval Layer

- **Embeddings:** A medical-domain embedding model is required rather than a general-purpose
  default, since general embeddings underperform on medical terminology. Candidates:
  BGE-M3 or BGE-large (open weights, strong general + medical), MedCPT (NLM, trained on
  medical literature), or a managed option like Voyage medical. Final choice at
  implementation time, but it must be domain-aware. Run locally where possible to avoid cost.
- **Vector store:** pgvector (PostgreSQL extension) or Qdrant
- **Keyword search:** BM25
- **Retrieval strategy:** Hybrid BM25 + dense retrieval with RRF fusion. (Cross-encoder
  reranking is deferred to post-MVP — it layers on top without restructuring anything, and
  deferring it yields a clean before/after eval comparison.)
- **Chunking:** LOINC-keyed, section-aware chunking. Chunk on LOINC code + section title;
  subsections inherit the nearest ancestor LOINC code. Drop non-clinical sections
  (Description, Nonclinical Toxicology, How Supplied, labeling/display panels) from the
  index. Merge fragments under ~100 tokens with sibling/parent content before embedding.
  Tag each chunk's section type in metadata (Medication Guide, PPI, OTC Drug Facts, Patient
  Counseling Information, etc.) at ingestion so patient-facing section boosting can be added
  post-MVP without re-ingesting — the boosting logic itself is deferred, the metadata hook
  is not.
- **Deduplication:** DailyMed has 157k+ labels but far fewer unique drugs. Select one
  canonical label per active ingredient (prefer the reference listed drug / original brand)
  to avoid conflicting chunks across manufacturers.

### Agent Layer

- **Orchestration framework:** LangChain + LangGraph
- **Agent pattern:** ReAct
- **LLM:** Groq (primary, free tier); Ollama for local development

### Application Layer

- **Backend:** FastAPI
- **Database:** PostgreSQL (application data)
- **Cache:** Redis (repeated query caching, session state between agent turns)
- **Frontend:** React or Next.js

### Infrastructure

- **Deployment:** AWS EC2 (free tier / AWS credits)
- **CI/CD:** GitHub Actions
- **RAG evaluation:** RAGAS

---

## Development Phases

### Phase 1 — Data Pipeline

Build the pipeline that downloads, processes, and indexes FDA drug label data.

- Download FDA DailyMed bulk SPL data and store in S3
- Parse drug label XML into LOINC-coded sections
- Implement LOINC-keyed chunking with nearest-ancestor inheritance, non-clinical section
  dropping, and small-fragment merging
- Attach structured provenance metadata (SETID, LOINC code, section title path) to each chunk
- Implement deduplication: select one canonical label per active ingredient
- Run ingestion pipeline to populate the vector store and BM25 index

### Phase 2 — Retrieval Pipeline

Build and validate the core retrieval system before introducing agent complexity.

- Implement BM25 keyword search over drug label sections
- Implement dense semantic search using a medical-domain embedding model
- Combine into hybrid retrieval with RRF fusion
- Integrate RxNorm name resolution (exact + fuzzy fallback) with explicit handling for
  coverage gaps so a failed lookup never produces a silent bad result
- Manually validate retrieval quality on a set of representative queries

### Phase 3 — Agent

Wire the retrieval pipeline into a ReAct agent and handle multi-drug queries.

- Define the three tool schemas and implement tool logic (name resolution, drug info
  retrieval, interaction information retrieval)
- Build the LangGraph orchestration graph
- Handle multi-drug queries requiring sequential tool calls
- Implement the safety prompt layer: citation-enforced generation, "I don't know"
  fallbacks, no confident interaction negatives, and pharmacist routing on high-risk
  question types
- Test agent behavior and reasoning across varied query types, with confident
  interaction false-negatives as a specific test target

### Phase 4 — API and Frontend

Expose the agent through an API and build the user-facing interface.

- Build FastAPI backend with a conversational query endpoint
- Add Redis caching for repeated queries and session state storage for conversation memory
- Wire conversation history into the agent so prior turns inform each new response
- Build chat interface with source attribution display and conversation history visible to the user
- Integrate persistent medical disclaimer on every response

### Phase 5 — Evaluation and Deployment

Validate system quality, set up automation, and ship.

- Build a ground truth evaluation dataset of drug Q&A pairs with known correct answers,
  including a set of known drug interactions specifically to test against confident
  false negatives
- Run RAGAS evaluation to measure retrieval accuracy and answer faithfulness
- Identify and address retrieval failures based on eval results
- Set up GitHub Actions pipeline with eval running on push
- Deploy to AWS and verify production behavior

---

## Future Features

### Immediate post-MVP (deferred from MVP, low cost to add later)

These were deliberately deferred because they layer on top of the pipeline without
restructuring it, so adding them later is no harder than building them in now. The
ingestion-time metadata hooks they depend on are kept in the MVP.

**Cross-encoder reranking** — add a reranking layer on top of hybrid retrieval. Deferring
it gives a clean before/after eval number showing its impact. Note the memory cost: it
runs as a second local model alongside the embedding model, so the deployment instance
must be sized for both.

**Patient-facing section boosting** — boost Medication Guides, Patient Package Inserts,
OTC Drug Facts, and Patient Counseling Information in retrieval for common patient
questions. The section-type metadata is already tagged at ingestion, so this is purely a
ranking-logic addition.

**MedlinePlus Connect link-out** — pass the resolved RXCUI to the Connect API and surface
a "Read the full patient guide" link to the AHFS consumer drug page. Self-contained UI
feature; the AHFS text is copyrighted so link-out is the correct licensing posture.

### Larger future work

**Supplementary corpora** — add the NIH Dietary Supplement Label Database (DSLD) to cover
the supplement gap (drug-supplement interaction questions like fish oil + warfarin), and
the MedlinePlus Health Topics web service for condition-context questions ("what is atrial
fibrillation?"). Both are free and ingestible. Deferred from the MVP because each adds a
corpus to ingest and entity-routing logic to decide whether a query is about a drug,
supplement, or condition — kept out of scope to ship a focused FDA drug-label tool first.

**Authoritative interaction adjudication** — upgrade the interaction tool from surfacing
label text to severity-ranked pairwise adjudication, which requires licensing a commercial
source (DrugBank Clinical API, Lexicomp, First Databank, or Micromedex).

**Structured fact extraction** — extract typed facts (indications, dosing, adverse
reactions with incidence) from labels at ingestion time so common factual queries become
deterministic lookups rather than prose retrieval, with prose RAG as the fallback for
nuanced questions. A larger ingestion-time engineering investment, justified by eval
results if common factual queries show reliability issues.

**Saved medication list** — users can save their current medications and ask questions
relative to their personal list without typing drug names each time.

**User accounts** — persistent profiles, medication history, and saved queries across
sessions. Not in the MVP; the initial version requires no authentication.

**Expanded corpus** — supplement FDA labels with clinical guidelines, curated drug
interaction databases, and condition-specific resources to improve answer depth and
coverage.

**Additional agent tools** — section-specific retrieval (e.g. dosing only), condition-
to-drug search, and more granular interaction lookup to support a wider range of query
types.

**Evaluation dashboard** — track retrieval and answer quality metrics over time as the
system evolves, making it easy to see whether a retrieval change improved or regressed
performance.

**Observability** — integrate LangSmith for tracing agent runs, inspecting tool call
sequences, and debugging retrieval failures in production.

**Retrieval experimentation** — systematic A/B testing of chunking strategies, embedding
models, and reranking approaches against the eval dataset, with results tracked across
experiments.
