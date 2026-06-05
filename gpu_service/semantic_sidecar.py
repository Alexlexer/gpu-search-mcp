"""Lightweight local-only Python semantic sidecar for sentence-transformers.

This exposes minimal endpoints: /health, /status, /embed, /index, /merge, /shutdown
and writes a small `.gpu-search-cache/sidecar.json` with connection info so Rust
can detect the sidecar and report semantic availability.

Run as: python gpu_service/semantic_sidecar.py --port 8770
"""
import argparse
import json
import os
import signal
import sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

SIDECAR_INFO = {
	"host": "127.0.0.1",
	"port": None,
	"token": None,
}


class SidecarHandler(BaseHTTPRequestHandler):
	def _send_json(self, obj, code=200):
		data = json.dumps(obj).encode("utf-8")
		self.send_response(code)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(data)))
		self.end_headers()
		self.wfile.write(data)

	def do_GET(self):
		if self.path == "/health":
			self._send_json({"ok": True, "version": "0.1-sidecar"})
			return
		if self.path == "/status":
			self._send_json({"ready": True, "message": "sidecar ready"})
			return
		self._send_json({"error": "not_found"}, 404)

	def do_POST(self):
		if self.path == "/shutdown":
			self._send_json({"ok": True})
			Thread(target=os._exit, args=(0,)).start()
			return
		if self.path == "/embed":
			length = int(self.headers.get("Content-Length", "0"))
			body = self.rfile.read(length).decode("utf-8")
			try:
				payload = json.loads(body)
				texts = payload.get("texts", [])
			except Exception:
				texts = []
			# Minimal fake embedding: return length of text as float vector.
			embeddings = [[float(len(t))] for t in texts]
			self._send_json({"embeddings": embeddings})
			return
		if self.path in ("/index", "/merge"):  # no-op for now
			self._send_json({"ok": True, "chunks": 0})
			return
		self._send_json({"error": "not_found"}, 404)


def write_sidecar_info(port: int, token: str | None = None):
	info = dict(SIDECAR_INFO)
	info["port"] = port
	info["token"] = token
	path = Path.cwd() / ".gpu-search-cache"
	path.mkdir(parents=True, exist_ok=True)
	(path / "sidecar.json").write_text(json.dumps(info), encoding="utf-8")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--port", type=int, default=8770)
	args = parser.parse_args()

	write_sidecar_info(args.port)

	server = ThreadingHTTPServer(("127.0.0.1", args.port), SidecarHandler)
	print(f"Semantic sidecar listening on http://127.0.0.1:{args.port}")
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass


if __name__ == "__main__":
	main()
