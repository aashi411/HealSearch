from functools import lru_cache
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ---- Models ----
MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- Session ----
SESSION_TTL = 15 * 60          # seconds of inactivity before a session expires
SESSIONS_DIR = "sessions"
CHROMA_DIR = "chroma_sessions"
HISTORY_TURNS = 6              # recent messages sent to the intent chain
HISTORY_CHAR_LIMIT = 200       # truncate each message (token saving)

# ---- Research ----
MAX_SEARCH_RETRIES = 2         # query refinements
MAX_SOURCES = 3                # sources that reach extraction
SEARCH_RESULTS = 5             # candidates per search
MAX_EXTRACTION_RETRIES = 1     # recovery attempts per source
MAX_RESEARCH_ROUNDS = 2        # fresh-research rounds (retrieval/analysis healing)
SNIPPET_CHARS = 200
EXTRACT_CHARS = 6000           # max text kept per page
MIN_TEXT_CHARS = 300           # extracted text shorter than this = failed extraction
REQUEST_TIMEOUT = 8

# ---- RAG (from StudyCat) ----
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
RETRIEVAL_FETCH_K = 10
MIN_CHUNKS = 2                 # fewer than this = insufficient context
MIN_RELEVANCE_SCORE = 0.35     # cosine similarity; chunks below this are ignored

# ----Logging ----
LOG_FILE = "logs/pipeline_log.jsonl"
BLOCKED_DOMAINS = ["youtube.com", "youtu.be", "tiktok.com"]


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.0, max_tokens: int = 1500) -> ChatGroq:
    """temperature=0 for judging/planning, higher for writing."""
    from healing import UsageLogger   # imported here to avoid a circular import (healing imports config)
    return ChatGroq(model=MODEL_NAME, temperature=temperature, max_tokens=max_tokens,
                    callbacks=[UsageLogger()])

@lru_cache(maxsize=None)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)