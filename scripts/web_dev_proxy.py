"""Local web/ with the PROD backend proxied behind it.

Serves the working-tree frontend while forwarding /health /papers /figures
/pages/* /query /demo/chat to the live deployment — full end-to-end testing
of uncommitted client code against the real model and corpus, no deploy.
"""

import functools
import http.server
import urllib.error
import urllib.request
from pathlib import Path

PROD = "https://spectrarag-ar6wxit42a-ew.a.run.app"
ROOT = str(Path(__file__).resolve().parents[1] / "web")


class H(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close-delimited responses, safe for SSE relay

    def _proxy(self) -> None:
        data = None
        if self.command == "POST":
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = urllib.request.Request(PROD + self.path, data=data, method=self.command)
        req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                self.send_response(r.status)
                self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
                self.end_headers()
                while True:
                    chunk = r.read(1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith(("/health", "/papers", "/figures", "/pages/")):
            return self._proxy()
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith(("/query", "/demo/chat")):
            return self._proxy()
        self.send_error(404)


http.server.ThreadingHTTPServer(
    ("127.0.0.1", 8765), functools.partial(H, directory=ROOT)
).serve_forever()
