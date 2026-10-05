import json
import os
import time
import uuid
import hashlib
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K, RETRIEVAL_FETCH_K,
                    MIN_RELEVANCE_SCORE, get_embeddings)
from config import SESSIONS_DIR, SESSION_TTL, HISTORY_TURNS, HISTORY_CHAR_LIMIT

os.makedirs(SESSIONS_DIR, exist_ok=True)


def _path(sid: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{sid}.json")


def new_session() -> dict:
    session = {"id": uuid.uuid4().hex[:12], "messages": [], "last_topic": "",
               "last_active": time.time()}
    save_session(session)
    return session


def save_session(session: dict) -> None:
    """Saving also refreshes last_active (inactivity-based TTL)."""
    session["last_active"] = time.time()
    with open(_path(session["id"]), "w", encoding="utf-8") as f:
        json.dump(session, f)


def load_session(sid: str) -> dict | None:
    """Return the session if it exists and has not expired, else None."""
    try:
        with open(_path(sid), encoding="utf-8") as f:
            session = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if time.time() - session["last_active"] > SESSION_TTL:
        delete_session(sid)
        return None
    return session


def delete_session(sid: str) -> None:
    """Delete the session file AND its Chroma collection."""
    if os.path.exists(_path(sid)):
        os.remove(_path(sid))
    try:
        _store(sid).delete_collection()
    except Exception:
        pass                      # collection may not exist yet

def cleanup_expired() -> None:
    for name in os.listdir(SESSIONS_DIR):
        if name.endswith(".json"):
            load_session(name[:-5])   # load_session deletes if expired


def add_message(session: dict, role: str, content: str) -> None:
    session["messages"].append({"role": role, "content": content})
    save_session(session)


def recent_history(session: dict) -> list[dict]:
    """Compact recent history for the intent chain (token saving)."""
    return [{"role": m["role"], "content": m["content"][:HISTORY_CHAR_LIMIT]}
            for m in session["messages"][-HISTORY_TURNS:]]

# Session Chroma (one collection per session) 
_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _store(sid: str) -> Chroma:
    return Chroma(
        collection_name=f"session_{sid}",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},   # so score = 1 - distance
    )


def add_sources(sid: str, sources: list[dict]) -> int:
    """Chunk -> embed -> store. Chunk id = hash of text, so duplicates are overwritten, not repeated."""
    unique: dict[str, Document] = {}
    for s in sources:
        for chunk in _splitter.split_text(s["text"]):
            unique[hashlib.md5(chunk.encode()).hexdigest()] = Document(
                page_content=chunk,
                metadata={"url": s["url"], "title": s["title"], "query": s["query"]},
            )
    if unique:
        _store(sid).add_documents(list(unique.values()), ids=list(unique.keys()))
    return len(unique)


def retrieve_chunks(sid: str, query: str) -> list[tuple[Document, float]]:
    """Top-k chunks of THIS session with similarity >= MIN_RELEVANCE_SCORE, best first."""
    results = _store(sid).similarity_search_with_score(query, k=RETRIEVAL_FETCH_K)
    scored = [(doc, round(1 - dist, 2)) for doc, dist in results]
    return [(d, s) for d, s in scored if s >= MIN_RELEVANCE_SCORE][:RETRIEVAL_K]