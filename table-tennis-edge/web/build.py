"""Inline engine.js into page.html.

Writes two files:
  tt-edge.html             the page body, as published to claude.ai (the host adds the document wrapper)
  tt-edge-standalone.html  a complete document you can open straight from disk
"""
from pathlib import Path

here = Path(__file__).parent
page = (here / "page.html").read_text(encoding="utf-8")
engine = (here / "engine.js").read_text(encoding="utf-8")
assert "/*ENGINE*/" in page and "</script" not in engine
body = page.replace("/*ENGINE*/", engine)
(here / "tt-edge.html").write_text(body, encoding="utf-8")
standalone = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    "<style>body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n"
    + body + "\n</body>\n</html>\n"
)
(here / "tt-edge-standalone.html").write_text(standalone, encoding="utf-8")
print(f"tt-edge.html {len(body) / 1024:.0f} KB, tt-edge-standalone.html {len(standalone) / 1024:.0f} KB")
