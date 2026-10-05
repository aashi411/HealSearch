from langchain_core.prompts import ChatPromptTemplate

QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You analyze a user's query inside a research assistant session.

Resolve what the user actually means using, in priority order:
1. Explicit wording in the current query (always overrides everything else).
2. The recent conversation.
3. The last research topic.

Rules:
- Rewrite the query as a standalone, specific `resolved_query` (e.g. "What about python?"
  after a snakes conversation -> "Python snake: description, habitat and venom").
- Set needs_clarification=true ONLY if the query is genuinely ambiguous AND the
  conversation/topic does not settle it. Then write ONE short clarification_question.
- If the previous assistant message asked a clarification question, treat the current
  query as the answer to it and resolve it, do not ask again.
- `topic` = a short label for the subject area (e.g. "snakes", "python programming")."""),
    ("human", """Last research topic: {last_topic}

Recent conversation:
{history}

Current query: {query}"""),
])

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You judge search results for a research task.
Select only results that genuinely help answer the user's RESOLVED INTENT.
Judge by meaning, not keywords: a page about the Python snake is NOT relevant to
Python programming, even though it contains the word "Python".

Return:
- relevant_ids: ids of useful results (empty list if none)
- diagnosis: "ok" if at least one is relevant; otherwise why the search failed:
  "too_broad" (generic results), "too_narrow" (nothing found), or "off_topic" (wrong meaning/subject)
- note: one short sentence"""),
    ("human", """Resolved intent: {intent}
Search query used: {query}

Results:
{results}"""),
])

REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """A web search returned poor results. Write ONE better search query.
Fix the diagnosed problem: if too_broad add specifics (subject, year, terms);
if off_topic add words that pin the correct meaning; if too_narrow simplify.
Return only the query, under 12 words.  Do not add years or dates unless the user asked for them."""),
    ("human", """Resolved intent: {intent}
Previous query: {query}
Diagnosis: {diagnosis}
Titles that were rejected: {rejected}"""),
])

ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You analyze evidence chunks gathered for a research question.
Use ONLY the chunks. Do not add outside knowledge.

Return:
- sufficient: true if the chunks together answer the resolved question well
- contradictions: short statements of any conflicts between chunks (empty list if none)
- missing: what is still missing ("" if sufficient)
- follow_up_query: ONE specific web search query that would fill the gap ("" if sufficient).
  Do not add years or dates unless the user asked.
- evidence_summary: at most 5 short lines of key facts, each ending with its source numbers like [1][3]"""),
    ("human", """Resolved question: {intent}

Evidence chunks:
{evidence}"""),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You write grounded answers for a research assistant.
Use ONLY the evidence provided. Cite source numbers like [1] after the claims they support.Several chunks can share one number.
Do NOT write a "Sources" section; it is added automatically.
Start with a direct answer, then explain in simple language (about 150-250 words).
If contradictions are listed, mention them honestly.
If evidence status is INSUFFICIENT, answer what you can and clearly state what is missing.
Do not repeat the question. Do not invent sources."""),
    ("human", """Question: {query}
Resolved intent: {intent}
Evidence status: {status}

Key facts:
{summary}

Contradictions: {contradictions}

Evidence chunks:
{evidence}"""),
])