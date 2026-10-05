import os
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from config import SEARCH_RESULTS, SNIPPET_CHARS, EXTRACT_CHARS, MIN_TEXT_CHARS, REQUEST_TIMEOUT, BLOCKED_DOMAINS

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# Tool 1: Search 
def search_web(query: str) -> list[dict]:
    """Return candidate sources only (title, url, short snippet). No scraping here."""
    response = tavily.search(query=query, max_results=SEARCH_RESULTS, exclude_domains=BLOCKED_DOMAINS)
    return [
        {"title": r["title"], "url": r["url"], "snippet": r["content"][:SNIPPET_CHARS]}
        for r in response["results"]
    ]


# Tool 2: Targeted extraction
def _clean_html(html: str) -> str:
    """Keep paragraph/list/heading text only; drop menus, ads, scripts."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    blocks = [t.get_text(" ", strip=True) for t in soup.find_all(["h1", "h2", "h3", "p", "li"])]
    return "\n".join(b for b in blocks if len(b) > 40)


def extract_page(url: str, method: str = "requests") -> dict:
    """
    method="requests": fast direct download + BeautifulSoup cleaning.
    method="tavily":   fallback extractor (handles some pages requests cannot).
    Always returns {"ok", "url", "text", "error"}; never raises, never hides failure.
    """
    try:
        if method == "requests":
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            text = _clean_html(resp.text)
        else:
            data = tavily.extract(urls=[url])
            text = data["results"][0]["raw_content"] if data.get("results") else ""
        text = text[:EXTRACT_CHARS]
        if len(text) < MIN_TEXT_CHARS:
            return {"ok": False, "url": url, "text": "", "error": f"{method}: too little text ({len(text)} chars)"}
        return {"ok": True, "url": url, "text": text, "error": ""}
    except Exception as e:
        return {"ok": False, "url": url, "text": "", "error": f"{method}: {e}"}


if __name__ == "__main__":
    print(search_web("Python decorators explained")[0])
    print(extract_page("https://docs.python.org/3/glossary.html")["ok"])
    print(extract_page("https://this-site-does-not-exist-12345.com"))   # must fail cleanlys