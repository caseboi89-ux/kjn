"""Local web dashboard (standard library only)."""
from __future__ import annotations

import json
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from .app import Workspace, price_match
from .data import ODDS_FIELDS, RESULT_FIELDS
from .odds import parse_price

MAX_BODY = 64 * 1024 * 1024

SAMPLE_RESULTS = ",".join(RESULT_FIELDS) + "\n" + \
    "m1,2026-11-12T09:30,Czech Liga Pro,Player One,Player Two,11-7 9-11 11-5 11-8,5\n"
SAMPLE_ODDS = ",".join(ODDS_FIELDS) + "\n" + "\n".join([
    "m2,2026-11-13T10:00,TT Cup,Player Three,Player Four,BookA,ML,A,,1.72,current,5",
    "m2,2026-11-13T10:00,TT Cup,Player Three,Player Four,BookA,ML,B,,2.05,current,5",
    "m2,2026-11-13T10:00,TT Cup,Player Three,Player Four,BookA,GAME_HCP,A,-1.5,2.30,current,5",
    "m2,2026-11-13T10:00,TT Cup,Player Three,Player Four,BookA,GAME_HCP,B,1.5,1.55,current,5",
    "m2,2026-11-13T10:00,TT Cup,Player Three,Player Four,BookA,GAMES_TOTAL,OVER,3.5,1.40,current,5",
    "m2,2026-11-13T10:00,TT Cup,Player Three,Player Four,BookA,GAMES_TOTAL,UNDER,3.5,2.75,current,5",
]) + "\n"


def _index_html() -> bytes:
    return resources.files("ttedge").joinpath("static/index.html").read_bytes()


def make_handler(ws: Workspace):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ttedge"

        def log_message(self, fmt, *args):  # quieter console
            if not self.path.startswith("/api/"):
                return
            super().log_message(fmt, *args)

        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = 200) -> None:
            self._send(status, json.dumps(data, default=str).encode(), "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("upload too large (64 MB limit)")
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, _index_html(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self._json(ws.state())
                elif url.path == "/api/scan":
                    self._json(ws.scan_dict())
                elif url.path == "/api/event":
                    detail = ws.event_detail(q.get("id", [""])[0])
                    if detail is None:
                        self._error(404, "unknown match")
                    else:
                        self._json(detail)
                elif url.path == "/api/ratings":
                    self._json(ws.leaderboard(int(q.get("min_matches", ["1"])[0])))
                elif url.path == "/api/backtest":
                    self._json(ws.backtest())
                elif url.path == "/api/sample":
                    kind = q.get("kind", ["odds"])[0]
                    body = (SAMPLE_RESULTS if kind == "results" else SAMPLE_ODDS).encode()
                    self._send(200, body, "text/csv",
                               {"Content-Disposition": f'attachment; filename="{kind}-template.csv"'})
                else:
                    self._error(404, "not found")
            except Exception as exc:  # pragma: no cover - surfaced to the UI
                traceback.print_exc()
                self._error(500, str(exc))

        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            try:
                body = self._body()
                if url.path == "/api/price":
                    if body.get("p_point") not in (None, ""):
                        out = price_match(p_point=float(body["p_point"]), best_of=int(body.get("best_of", 5)),
                                          sigma=float(body.get("sigma", ws.config.game_sigma)))
                    else:
                        out = price_match(parse_price(body["price_a"]), parse_price(body["price_b"]),
                                          best_of=int(body.get("best_of", 5)),
                                          sigma=float(body.get("sigma", ws.config.game_sigma)),
                                          devig_method=body.get("devig_method", ws.config.devig_method))
                    self._json(out)
                elif url.path == "/api/config":
                    ws.update_config(body)
                    self._json(ws.state())
                elif url.path == "/api/upload":
                    kind = body.get("kind")
                    text = body.get("csv", "")
                    if kind == "results":
                        ws.load(results=text, source="uploaded CSV")
                    elif kind == "odds":
                        ws.load(odds=text, source="uploaded CSV")
                    else:
                        raise ValueError("kind must be 'results' or 'odds'")
                    self._json(ws.state())
                elif url.path == "/api/demo":
                    ws.load_demo(int(body.get("seed", 7)))
                    self._json(ws.state())
                elif url.path == "/api/clear":
                    ws.load(results="", odds="", source="empty")
                    self._json(ws.state())
                else:
                    self._error(404, "not found")
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._error(400, str(exc))
            except Exception as exc:  # pragma: no cover
                traceback.print_exc()
                self._error(500, str(exc))

    return Handler


def make_server(ws: Workspace, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(ws))


def serve(ws: Workspace, host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = make_server(ws, host, port)
    print(f"ttedge dashboard on http://{host}:{httpd.server_address[1]}  (Ctrl+C to stop)")
    print(f"data: {ws.source}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
