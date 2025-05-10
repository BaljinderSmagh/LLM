from duckduckgo_search import DDGS

def search_duckduckgo(query: str, max_results: int = 3) -> str:
    """Perform a DuckDuckGo search and return top results."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            if not results:
                return "No results found."
            return "\n\n".join([f"{r['title']}:\n{r['href']}" for r in results])
    except Exception as e:
        return f"Search error: {str(e)}"
