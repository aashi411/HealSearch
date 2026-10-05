from pydantic import BaseModel
from config import get_llm
from prompts import QUERY_PROMPT
from pydantic import BaseModel
from config import get_llm
from prompts import QUERY_PROMPT, ANSWER_PROMPT


class QueryIntent(BaseModel):
    resolved_query: str
    topic: str
    needs_clarification: bool
    clarification_question: str = ""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def analyze_query(query: str, history: list[dict], last_topic: str = "") -> QueryIntent:
    """Query / Intent Chain: resolve meaning and detect ambiguity."""
    chain = QUERY_PROMPT | get_llm(temperature=0).with_structured_output(QueryIntent)
    return chain.invoke({
        "query": query,
        "history": _format_history(history),
        "last_topic": last_topic or "(none)",
    })

class EvidenceVerdict(BaseModel):
    """Agent 2's structured judgement of the retrieved evidence."""
    sufficient: bool
    contradictions: list[str] = []
    missing: str = ""
    follow_up_query: str = ""
    evidence_summary: str = ""


def source_numbers(docs: list) -> dict[str, int]:
    """url -> source number, in order of first appearance (best-scoring chunk first)."""
    numbers: dict[str, int] = {}
    for d in docs:
        numbers.setdefault(d.metadata["url"], len(numbers) + 1)
    return numbers


def format_evidence(docs: list) -> str:
    """Chunks labelled by SOURCE number, so citations match the final source list."""
    nums = source_numbers(docs)
    return "\n\n".join(
        f"[{nums[d.metadata['url']]}] ({d.metadata.get('title', 'source')[:60]})\n{d.page_content}"
        for d in docs
    )


def generate_answer(query: str, intent: str, docs: list, verdict: EvidenceVerdict | None,
                    sufficient: bool) -> str:
    """Final Answer Chain: compact evidence package -> grounded answer + numbered source list."""
    if not docs:
        return "I could not find reliable information for this after several research attempts."

    chain = ANSWER_PROMPT | get_llm(0.3, 2000)
    answer = chain.invoke({
        "query": query,
        "intent": intent,
        "status": "SUFFICIENT" if sufficient else "INSUFFICIENT (low confidence)",
        "summary": verdict.evidence_summary if verdict else "(none)",
        "contradictions": "; ".join(verdict.contradictions) if verdict and verdict.contradictions else "none",
        "evidence": format_evidence(docs),
    }).content

    titles = {d.metadata["url"]: d.metadata["title"] for d in docs}
    lines = [f"[{n}] {titles[url]}: {url}" for url, n in source_numbers(docs).items()]
    return answer + "\n\n**Sources**\n" + "\n".join(lines)

if __name__ == "__main__":
    def show(label, query, history, topic=""):
        r = analyze_query(query, history, topic)
        print(f"\n[{label}] {query!r}\n -> {r}")

    show("ambiguous", "Python", [])
    show("snake ctx", "What about python?",
         [{"role": "user", "content": "Tell me about snakes."},
          {"role": "user", "content": "Which ones are venomous?"}], "snakes")
    show("code ctx", "What about Python?",
         [{"role": "user", "content": "Explain Python decorators."},
          {"role": "user", "content": "What are generators?"},
          {"role": "user", "content": "How do virtual environments work?"}],
         "python programming")
    show("explicit override", "Tell me about the python snake",
         [{"role": "user", "content": "Explain Python decorators."}], "python programming")