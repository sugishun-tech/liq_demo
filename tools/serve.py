#!/usr/bin/env python3
"""Serve only public site files on loopback; never expose training DBs or the repo root."""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json'}
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()
    def list_directory(self, path):
        self.send_error(403, 'Directory listing disabled'); return None

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=ROOT / 'docs')
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()
    directory = args.directory.resolve()
    if not (directory / 'index.html').is_file(): p.error('公開用index.htmlがありません。--directoryにindex.htmlがあるディレクトリを指定してください。')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), functools.partial(Handler, directory=str(directory)))
    print(f'http://localhost:{args.port}/  (Ctrl+C to stop)', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
if __name__ == '__main__': main()
