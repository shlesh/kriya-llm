import re, unicodedata
from typing import List

def normalize_text(s: str) -> str:
    s = s.replace('\\r\\n', '\\n').replace('\\r', '\\n')
    s = unicodedata.normalize('NFKC', s)
    # Fix common ligatures / stray unicode
    s = s.replace('\\u00a0', ' ').replace('\\t', ' ')
    # Collapse multiple spaces
    s = re.sub(r'[ \\u2000-\\u200b\\u202f\\u3000\\ufeff]+', ' ', s)
    # Remove trailing spaces on lines
    s = re.sub(r'[ \\t]+\\n', '\\n', s)
    return s.strip()

def remove_hyphenation(s: str) -> str:
    # join words split across line breaks with hyphens
    s = re.sub(r'(\\w+)-\\n(\\w+)', r'\\1\\2', s)
    # remove single newlines within paragraphs (but keep paragraph breaks)
    s = re.sub(r'([^\\n])\\n(?!\\n)', r'\\1 ', s)
    return s

def strip_headers_footers(pages: List[str], k: int = 3) -> List[str]:
    """
    Detect lines that repeat at top/bottom of pages (headers/footers) and drop them.
    k: how many lines from top/bottom to consider.
    """
    top_lines, bot_lines = {}, {}
    def first_k_lines(t): return [l for l in t.split('\\n')[:k] if l.strip()]
    def last_k_lines(t):  return [l for l in t.split('\\n')[-k:] if l.strip()]
    for p in pages:
        for l in first_k_lines(p):
            top_lines[l] = top_lines.get(l, 0) + 1
        for l in last_k_lines(p):
            bot_lines[l] = bot_lines.get(l, 0) + 1
    # Consider header/footer if appears on >= 30% of pages
    th = max(2, int(0.3 * len(pages)))
    headers = {l for l,c in top_lines.items() if c >= th}
    footers = {l for l,c in bot_lines.items() if c >= th}

    cleaned = []
    for p in pages:
        lines = p.split('\\n')
        lines = [l for l in lines if l not in headers and l not in footers]
        cleaned.append('\\n'.join(lines).strip())
    return cleaned
