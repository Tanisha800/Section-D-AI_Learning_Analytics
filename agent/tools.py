from app import run_full_pipeline
from duckduckgo_search import DDGS

def analyze_student(file_path):
    return run_full_pipeline(file_path)


def web_search(query):
    links = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            links.append(r["href"])
    return links