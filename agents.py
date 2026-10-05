from typing import Literal
from pydantic import BaseModel
from config import get_llm, MAX_SEARCH_RETRIES, MAX_SOURCES, MAX_EXTRACTION_RETRIES, MIN_CHUNKS, MAX_RESEARCH_ROUNDS
from prompts import RELEVANCE_PROMPT, REFINE_PROMPT, ANALYST_PROMPT
from chains import EvidenceVerdict, format_evidence
from memory import add_sources, retrieve_chunks
from tools import search_web, extract_page
from healing import run_with_healing, log_event, usage_summary


class RelevanceVerdict(BaseModel):
    relevant_ids: list[int]
    diagnosis: Literal["ok", "too_broad", "too_narrow", "off_topic"]
    note: str = ""


class RefinedQuery(BaseModel):
    query: str


#checks whether search results actually match the query.
def judge_relevance(intent: str, query: str, candidates: list[dict]) -> RelevanceVerdict:
    results = "\n".join(
        f"[{i}] {c['title']} | {c['url']} | {c['snippet']}" for i, c in enumerate(candidates)
    )
    chain = RELEVANCE_PROMPT | get_llm(0).with_structured_output(RelevanceVerdict)
    return chain.invoke({"intent": intent, "query": query, "results": results})


#improves a bad search query.
def refine_query(intent: str, query: str, diagnosis: str, rejected: list[str]) -> str:
    chain = REFINE_PROMPT | get_llm(0).with_structured_output(RefinedQuery)
    return chain.invoke({
        "intent": intent, "query": query, "diagnosis": diagnosis,
        "rejected": "; ".join(rejected[:5]) or "(none)",
    }).query


#coordinates searching + extraction.
def investigate(intent: str, heal_log: list[dict], first_query: str | None = None) -> list[dict]:
    """
    Agent 1: Search -> relevance check (self-heal by refining) -> extract (self-heal by fallback).
    Returns a list of {"title", "url", "text", "query"} ready for chunking in Stage 3.
    """

    def search_and_judge(query: str) -> dict:
        candidates = search_web(query)
        verdict = judge_relevance(intent, query, candidates)
        relevant = [candidates[i] for i in verdict.relevant_ids if 0 <= i < len(candidates)]
        rejected = [c["title"] for c in candidates if c not in relevant]
        log_event(heal_log, "relevance", "info",
                  f"query='{query}' | {len(candidates)} found, kept {len(relevant)}, "
                  f"diagnosis={verdict.diagnosis} | rejected: {rejected} | {verdict.note}")
        return {"query": query, "relevant": relevant, "rejected": rejected, "diagnosis": verdict.diagnosis}

    def has_relevant(result: dict) -> tuple[bool, str]:
        return bool(result["relevant"]), f"query '{result['query']}' gave no relevant sources ({result['diagnosis']})"

    def refine(query: str, problem: str, result: dict | None) -> str:
        if result is None:                       # search itself crashed: retry same query
            return query
        return refine_query(intent, query, result["diagnosis"], result["rejected"])

    search_result, ok = run_with_healing(
        "search", search_and_judge, has_relevant, refine,
        MAX_SEARCH_RETRIES, heal_log, first_query or intent,
    )
    if not ok or search_result is None:
        return []

    sources: list[dict] = []
    for cand in search_result["relevant"]:
        if len(sources) >= MAX_SOURCES:
            break
        page, ok = run_with_healing(
            "extraction",
            task=lambda arg: extract_page(*arg),                       # arg = (url, method)
            validate=lambda r: (r["ok"], r["error"]),
            recover=lambda arg, problem, r: (arg[0], "tavily") if arg[1] == "requests" else None,
            max_retries=MAX_EXTRACTION_RETRIES,
            heal_log=heal_log,
            arg=(cand["url"], "requests"),
        )
        if ok:
            sources.append({"title": cand["title"], "url": cand["url"],
                            "text": page["text"], "query": search_result["query"]})
            log_event(heal_log, "source", "ok", f"accepted: {cand['title']} ({len(page['text'])} chars)")
        else:
            log_event(heal_log, "source", "failed", f"discarded {cand['url']}, trying next relevant source")

    log_event(heal_log, "agent1", "info", f"finished with {len(sources)} usable source(s)")
    return sources



# Agent 2: Analyst / Synthesizer
def analyze_evidence(intent: str, docs: list) -> EvidenceVerdict:
    chain = ANALYST_PROMPT | get_llm(0).with_structured_output(EvidenceVerdict)
    return chain.invoke({"intent": intent, "evidence": format_evidence(docs)})


def build_evidence(sid: str, intent: str, heal_log: list[dict]) -> tuple[list, EvidenceVerdict | None, bool]:
    """
    Retrieve -> analyze -> (if weak/insufficient) fresh research -> repeat, bounded.
    Covers healing scenarios 4 (poor Chroma context) and 5 (low-confidence evidence).
    Returns (docs, verdict, sufficient).
    """

    def retrieve_and_analyze(_attempt) -> dict:
        scored = retrieve_chunks(sid, intent)
        docs = [d for d, _ in scored]
        log_event(heal_log, "retrieval", "info",
                  f"{len(docs)} relevant chunk(s) in session memory, scores={[s for _, s in scored]}")
        verdict = analyze_evidence(intent, docs) if len(docs) >= MIN_CHUNKS else None
        if verdict:
            log_event(heal_log, "analyst", "info",
                      f"sufficient={verdict.sufficient} | contradictions={len(verdict.contradictions)} | missing={verdict.missing}")
        return {"docs": docs, "verdict": verdict}

    def validate(r: dict) -> tuple[bool, str]:
        if r["verdict"] is None:
            return False, f"retrieval weak ({len(r['docs'])} relevant chunks, need {MIN_CHUNKS})"
        if not r["verdict"].sufficient:
            return False, f"evidence insufficient: {r['verdict'].missing}"
        return True, ""

    def recover(attempt, problem: str, r: dict | None):
        follow_up = r["verdict"].follow_up_query if r and r["verdict"] else ""
        sources = investigate(intent, heal_log, first_query=follow_up or intent)   # Agent 1 does fresh research
        if not sources:
            return None                                   # nothing new found: stop retrying
        n = add_sources(sid, sources)
        log_event(heal_log, "chroma", "info", f"added {n} chunk(s) to session memory")
        return follow_up or intent

    result, ok = run_with_healing("evidence", retrieve_and_analyze, validate, recover,
                                  MAX_RESEARCH_ROUNDS, heal_log, intent)
    return result["docs"], result["verdict"], ok


if __name__ == "__main__":
    from memory import new_session
    from chains import generate_answer

    sid = new_session()["id"]
    log: list[dict] = []
    tests = [
        "Python programming: how decorators work",            # empty memory -> research
        "Python programming: decorators that take arguments",  # same topic -> should reuse Chroma
        "Python programming: what are generators",             # new topic -> weak retrieval -> research
    ]
    for q in tests:
        start = len(log)
        print("\n" + "=" * 70 + f"\nQUERY: {q}")
        docs, verdict, ok = build_evidence(sid, q, log)
        for e in log[start:]:
            print(f" {e['time']} [{e['status']:<9}] {e['step']:<10} {e['detail'][:160]}")
        print("\nANSWER:\n", generate_answer(q, q, docs, verdict, ok))
    print("\nUSAGE:", usage_summary())

