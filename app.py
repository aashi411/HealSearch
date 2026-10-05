import time
import streamlit as st
from config import SESSION_TTL
from memory import new_session, load_session, cleanup_expired
from pipeline import handle_query
from healing import usage_summary

st.set_page_config(page_title="Self-Healing Research Assistant", page_icon="🔎", layout="wide")
st.title("🔎 Context-Aware Self-Healing Research Assistant")

ICONS = {"ok": "✅", "info": "ℹ️", "problem": "⚠️", "recovery": "🔧", "recovered": "🩹", "failed": "❌"}


# ---------- session recovery (survives refresh via ?sid=) ----------
cleanup_expired()                                   # delete sessions inactive > SESSION_TTL
sid = st.query_params.get("sid")
session = load_session(sid) if sid else None
if sid and session is None:
    st.session_state["expired_notice"] = True       # the URL had a session, but it expired
if session is None:
    session = new_session()
    st.query_params["sid"] = session["id"]

if st.session_state.pop("expired_notice", False):
    st.warning("Your previous session expired (inactive too long). Its research memory was deleted. Starting fresh.")

# ---------- chat history ----------
for m in session["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- new message ----------
query = st.chat_input("Ask a research question...")
if query:
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Understanding, researching, validating..."):
            result = handle_query(session, query)
        st.markdown(result["answer"])
    st.session_state["last_log"] = result["heal_log"]

# ---------- sidebar: session info + healing log ----------
with st.sidebar:
    remaining = max(0, int(SESSION_TTL - (time.time() - session["last_active"])))
    st.subheader("Session")
    st.caption(f"ID: `{session['id']}`")
    st.caption(f"Topic: {session['last_topic'] or '-'}")
    st.caption(f"Expires after {remaining // 60}m {remaining % 60}s of inactivity")
    if st.button("New session"):
        st.query_params.clear()
        st.rerun()

    st.subheader("Healing log (last query)")
    for e in st.session_state.get("last_log", []):
        st.markdown(f"{ICONS.get(e['status'], '•')} `{e['time']}` **{e['step']}**: {e['detail'][:220]}")

    st.subheader("LLM usage (this server run)")
    st.json(usage_summary())