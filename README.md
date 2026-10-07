# HealSearch

### Context-Aware Self-Healing Agentic Research System with Session-Based RAG

HealSearch is a learning-focused GenAI research system that combines
**agentic web research, context-aware query understanding, self-healing
workflows, and session-based Retrieval-Augmented Generation (RAG)**.

Instead of blindly accepting search results, HealSearch checks whether
sources are relevant to the user's research intent, extracts useful
content, stores it in session-specific ChromaDB memory, and evaluates
whether the accumulated evidence is sufficient before generating an
answer.

------------------------------------------------------------------------

## ✨ Features

-   **Context-aware query understanding**
    -   Uses recent conversation history and the previous topic.
    -   Detects ambiguous queries such as `Python`.
    -   Can ask whether the user means Python the programming language
        or Python the snake.
    -   Resolves follow-up queries using session context.
-   **Two-agent architecture**
    -   **Agent 1 --- Research Agent:** searches the web, validates
        source relevance, extracts content, and recovers from research
        failures.
    -   **Agent 2 --- Evidence Analyst:** evaluates retrieved evidence,
        identifies missing information or contradictions, and decides
        whether more research is required.
-   **Self-healing pipeline**
    -   Execute → Validate → Diagnose → Recover → Retry.
    -   Refines search queries when results are irrelevant.
    -   Falls back to another extraction method when direct extraction
        fails.
    -   Performs fresh research when ChromaDB evidence is weak.
    -   Uses bounded retries to avoid infinite loops.
-   **Session-based RAG**
    -   Each session has its own ChromaDB collection.
    -   Research gathered during the session becomes reusable context.
    -   Follow-up queries can reuse previously collected evidence.
-   **Source relevance validation**
    -   Search results are treated as candidates, not automatically
        trusted sources.
    -   A relevance judge selects sources that actually match the query.
    -   Poor results can trigger query refinement.
-   **Token-conscious design**
    -   Limits conversation history.
    -   Uses short search snippets for relevance judging.
    -   Retrieves only top-k relevant chunks.
    -   Sends a compact evidence package to the final answer chain.
    -   Bounds self-healing retries.
-   **Observability**
    -   Logs research and recovery events.
    -   Tracks LLM calls, input/output tokens, latency, and errors.
-   **Streamlit UI**
    -   Interactive research interface.
    -   Deployable through Streamlit Community Cloud.

------------------------------------------------------------------------

## 🏗️ Architecture

``` text
User Query
    │
    ▼
Query / Intent Chain
    │
    ├── Ambiguous ──► Clarification
    │
    ▼
Agent 1: Research Agent
    │
    ├── Web Search
    ├── Relevance Check
    ├── Query Refinement
    └── Page Extraction + Fallback
    │
    ▼
Session ChromaDB
    │
    ▼
Semantic Retrieval
    │
    ▼
Agent 2: Evidence Analyst
    │
    ├── Sufficient ───────────────┐
    │                             │
    └── Insufficient              │
            │                     │
            ▼                     │
      Fresh Research              │
            │                     │
            ▼                     │
      Update ChromaDB ────────────┘
                                  │
                                  ▼
                         Final Answer Chain
                                  │
                                  ▼
                       Grounded Answer + Sources
```

------------------------------------------------------------------------

## 🔄 Query Flow

### 1. Query / Intent Chain

The first chain receives:

-   Current query
-   Recent conversation history
-   Last known topic

It returns a structured `QueryIntent`:

``` text
resolved_query
topic
needs_clarification
clarification_question
```

Example:

``` text
User: Python
```

The system can ask:

``` text
Do you mean Python the programming language or Python the snake?
```

If the conversation was:

``` text
User: Tell me about snakes.
User: Which snakes are venomous?
User: What about Python?
```

the session context can resolve `Python` as the snake.

------------------------------------------------------------------------

## 🔎 Agent 1 --- Research Agent

Agent 1 gathers evidence through:

``` text
Search
  ↓
Judge relevance
  ↓
Keep relevant sources
  ↓
Extract page content
  ↓
Validate extracted content
  ↓
Store usable sources
```

The system does **not** simply take five search results and scrape all
of them.

A relevance chain evaluates the candidates and can classify the search
as:

``` text
ok
too_broad
too_narrow
off_topic
```

If the results are poor, the query is refined and the search can be
retried.

------------------------------------------------------------------------

## 🩹 Self-Healing

The healing layer uses a reusable pattern:

``` text
EXECUTE
   ↓
VALIDATE
   ↓
Success?
  ├── Yes → Continue
  │
  └── No
       ↓
     Diagnose
       ↓
     Recover
       ↓
     Retry
```

Examples:

### Search recovery

``` text
Irrelevant results
      ↓
Diagnose relevance problem
      ↓
Refine query
      ↓
Search again
```

### Extraction recovery

``` text
Requests extraction fails
      ↓
Fallback extraction
      ↓
Validate extracted text
      ↓
Accept or discard source
```

### Evidence recovery

``` text
Chroma retrieval is weak
      ↓
Fresh research
      ↓
Add new sources
      ↓
Retrieve again
```

Retries are bounded by configuration so the application cannot endlessly
call the LLM or search service.

------------------------------------------------------------------------

## 🧠 Agent 2 --- Evidence Analyst

Agent 2 evaluates the evidence already gathered.

It returns an `EvidenceVerdict` containing:

``` text
sufficient
contradictions
missing
follow_up_query
evidence_summary
```

The decision loop is:

``` text
Retrieved Evidence
       ↓
Evidence Analyst
       ↓
Sufficient?
   ├── Yes → Final Answer
   │
   └── No
        ↓
   Follow-up Research
        ↓
   ChromaDB Update
        ↓
   Retrieve + Analyze Again
```

This makes the research process iterative rather than a single
search-and-answer operation.

------------------------------------------------------------------------

## 📚 Session-Based RAG

HealSearch does not require a PDF upload.

Instead, web research becomes the source of the RAG knowledge base:

``` text
Web Sources
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
Session ChromaDB
    ↓
Semantic Retrieval
```

Each session receives its own collection:

``` text
session_<session_id>
```

As research continues:

``` text
Query 1 → Research → ChromaDB
Query 2 → Reuse context + Research if needed
Query 3 → Reuse accumulated context + Research if needed
```

Duplicate chunks are identified using a hash of the chunk text so
repeated research does not unnecessarily add the same content.

------------------------------------------------------------------------

## ⚡ Token Optimization

The project is intentionally designed to be token-conscious.

### Techniques

**Limited history**

Only recent conversation turns are sent to the intent chain.

**Character limits**

Long messages and snippets are truncated before entering LLM prompts.

**Top-k retrieval**

Only the most relevant ChromaDB chunks are selected.

**Structured output**

Pydantic models constrain intermediate LLM responses.

**Compact evidence package**

The final answer chain receives selected evidence and the analyst's
summary rather than the complete internal process.

**Bounded retries**

Search, extraction, and evidence recovery have configurable retry
limits.

------------------------------------------------------------------------

## 📊 Observability

The system records pipeline events such as:

``` text
intent
search
relevance
extraction
source
retrieval
analyst
chroma
agent1
pipeline
```

Each event stores:

``` text
time
step
status
detail
```

LLM usage is also tracked:

``` text
input tokens
output tokens
latency
call count
errors
```

This makes it possible to inspect how much the system is searching,
healing, and using the LLM.

------------------------------------------------------------------------

## 📁 Project Structure

``` text
HealSearch/
│
├── app.py
│   └── Streamlit application entry point
│
├── pipeline.py
│   └── Coordinates the complete query workflow
│
├── agents.py
│   ├── Research Agent
│   ├── Evidence Analyst
│   ├── relevance checking
│   └── research recovery
│
├── chains.py
│   ├── Query / Intent Chain
│   ├── Evidence Verdict
│   └── Final Answer Chain
│
├── tools.py
│   ├── Web search
│   └── Targeted page extraction
│
├── memory.py
│   ├── Session management
│   ├── Conversation history
│   ├── ChromaDB storage
│   ├── Chunking
│   └── Retrieval
│
├── healing.py
│   ├── Self-healing execution
│   ├── Event logging
│   └── LLM usage tracking
│
├── config.py
│   └── Configuration and model setup
│
├── prompts.py
│   └── LLM prompts
│
├── requirements.txt
│   └── Python dependencies
│
└── .gitignore
```

------------------------------------------------------------------------

## 🛠️ Tech Stack

  Technology      Purpose
  --------------- --------------------------------------------
  Python          Core application
  Streamlit       UI and deployment
  LangChain       Chains, structured workflows and callbacks
  ChromaDB        Session-based vector storage
  Embeddings      Semantic representation of research chunks
  Tavily          Web search and fallback extraction
  BeautifulSoup   HTML cleaning
  Requests        Direct web-page retrieval
  Pydantic        Structured LLM outputs and validation

------------------------------------------------------------------------

## 🚀 Run Locally

### 1. Clone

``` bash
git clone https://github.com/aashi411/HealSearch.git
cd HealSearch
```

### 2. Create a virtual environment

Windows:

``` bash
python -m venv .venv
.venv\Scriptsctivate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file with the API credentials required by the configured
LLM and research provider.

Example:

``` env
TAVILY_API_KEY=your_tavily_key
YOUR_LLM_API_KEY=your_llm_key
```

Never commit `.env` or API keys to GitHub.

### 5. Run

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## ☁️ Deployment

HealSearch can be deployed through Streamlit Community Cloud.

``` text
GitHub
  ↓
Streamlit Community Cloud
  ↓
Select repository
  ↓
Select main branch
  ↓
Select app.py
  ↓
Configure deployment secrets
  ↓
Deploy
```

Secrets should be configured through the deployment platform rather than
hard-coded in the repository.

------------------------------------------------------------------------

## 🔐 Security

Do not commit:

``` text
.env
API keys
credentials
private tokens
session data
local ChromaDB data
```

Use environment variables locally and deployment secrets in hosted
environments.

------------------------------------------------------------------------

## 🎯 Design Goals

HealSearch is intentionally structured as a **learning project**.

The architecture prioritizes:

-   Clear separation of responsibilities
-   Simple and readable modules
-   Two explicit agents
-   Structured chains
-   Observable self-healing
-   Context-aware conversations
-   Incremental session RAG
-   Controlled token usage
-   Easy deployment

The goal is to make each part of an agentic research system
understandable, testable, and extensible rather than hiding the complete
workflow behind a large framework.

------------------------------------------------------------------------

## 🔮 Future Improvements

Possible extensions include:

-   Persistent production-grade vector storage
-   Source credibility scoring
-   Research-source diversity checks
-   Better contradiction detection
-   Parallel source extraction
-   Cross-session memory
-   Research quality benchmarking
-   Token/cost dashboards
-   Local or self-hosted LLM support
-   Additional self-healing strategies

------------------------------------------------------------------------

## 👩‍💻 Author

**Aashi**

GitHub: https://github.com/aashi411

------------------------------------------------------------------------

## 📌 Project Status

**Learning / Development Project**

HealSearch explores the design of context-aware, self-healing agentic
systems combined with web research, RAG, vector memory, and
evidence-based answer generation.
