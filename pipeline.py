from chains import analyze_query, generate_answer
from agents import build_evidence
from memory import add_message, recent_history
from healing import log_event


def handle_query(session: dict, query: str) -> dict:
    """Full flow for one user message. Returns {"answer", "heal_log", "clarification"}."""
    heal_log: list[dict] = []
    try:
        # 1. Query / Intent chain (uses history from BEFORE this message)
        intent = analyze_query(query, recent_history(session), session["last_topic"])
        add_message(session, "user", query)
        log_event(heal_log, "intent", "info",
                  f"resolved='{intent.resolved_query}' | topic='{intent.topic}' | clarify={intent.needs_clarification}")

        # 2. Ambiguous -> ask and stop
        if intent.needs_clarification:
            add_message(session, "assistant", intent.clarification_question)
            return {"answer": intent.clarification_question, "heal_log": heal_log, "clarification": True}

        # 3. Retrieve / research / analyze, with self-healing (Agent 1 + Agent 2)
        docs, verdict, ok = build_evidence(session["id"], intent.resolved_query, heal_log)

        # 4. Final Answer chain
        answer = generate_answer(query, intent.resolved_query, docs, verdict, ok)
        if not ok:
            log_event(heal_log, "pipeline", "info", "answered with LOW CONFIDENCE (retries exhausted)")

        session["last_topic"] = intent.topic
        add_message(session, "assistant", answer)          # also saves + refreshes last_active
        return {"answer": answer, "heal_log": heal_log, "clarification": False}

    except Exception as e:                                  # last safety net: never crash the UI
        log_event(heal_log, "pipeline", "failed", f"unhandled error: {e}")
        msg = "Something went wrong while researching. Please try again."
        add_message(session, "assistant", msg)
        return {"answer": msg, "heal_log": heal_log, "clarification": False}


if __name__ == "__main__":                                  # CLI test without the UI
    from memory import new_session
    from healing import usage_summary
    s = new_session()
    while True:
        q = input("\nYou (0 to exit): ")
        if q == "0":
            break
        r = handle_query(s, q)
        for e in r["heal_log"]:
            print(f" {e['time']} [{e['status']:<9}] {e['step']:<10} {e['detail'][:140]}")
        print("\nAI:", r["answer"])
    print(usage_summary())