#!/usr/bin/env python3
"""On-demand supervisor for whisper.cpp's whisper-server.

whisper-server loads its model into VRAM at startup and keeps it resident for the
whole life of the process; it has no idle unload of its own.
On a shared GPU that pins several GB of VRAM even while nothing is being transcribed.

This supervisor fronts the real server on the public port and starts it lazily on
the first transcription request, then stops it again after a configurable idle
period (WHISPER_IDLE_TTL seconds), freeing all of its VRAM until the next request
transparently reloads it.
/health is answered here directly, so a health probe neither loads the model nor
keeps it warm.

Configuration (all via environment):
  HOST                  public bind address (default 0.0.0.0)
  PORT                  public port (default 8080)
  INTERNAL_PORT         private port the real server binds (default 8081)
  WHISPER_IDLE_TTL      seconds idle before the model is unloaded (default 300)
  WHISPER_START_TIMEOUT seconds to wait for the model to load (default 180)

The child command (the whisper-server binary and its flags) is passed as this
script's own arguments; the supervisor appends --host/--port so the child binds
the private internal port.
"""

import http.client
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
INTERNAL_HOST = "127.0.0.1"
INTERNAL_PORT = int(os.environ.get("INTERNAL_PORT", "8081"))
IDLE_TTL = float(os.environ.get("WHISPER_IDLE_TTL", "300"))
START_TIMEOUT = float(os.environ.get("WHISPER_START_TIMEOUT", "180"))

# Hop-by-hop headers must not be forwarded across a proxy hop (RFC 7230 6.1).
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}

# Everything after the script name is the child command, minus host/port which we
# add ourselves so the child only ever binds the private internal port.
CHILD_BASE = sys.argv[1:]
if not CHILD_BASE:
    print("[supervisor] no whisper-server command supplied", file=sys.stderr)
    sys.exit(2)


def log(msg):
    print(f"[supervisor] {msg}", flush=True)


class Manager:
    """Owns the whisper-server child process and its load/unload lifecycle."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._inflight = 0
        self._last_activity = time.monotonic()

    def _running(self):
        return self._proc is not None and self._proc.poll() is None

    def begin_request(self):
        # Hold the lock across the whole start so the idle monitor cannot unload
        # mid-startup; inflight is bumped only once the server is confirmed ready.
        with self._lock:
            self._last_activity = time.monotonic()
            self._ensure_started_locked()
            self._inflight += 1

    def end_request(self):
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._last_activity = time.monotonic()

    def _ensure_started_locked(self):
        if self._running():
            return
        cmd = CHILD_BASE + ["--host", INTERNAL_HOST, "--port", str(INTERNAL_PORT)]
        log(f"loading model: starting whisper-server on {INTERNAL_HOST}:{INTERNAL_PORT}")
        self._proc = subprocess.Popen(cmd)
        self._wait_ready()

    def _wait_ready(self):
        url = f"http://{INTERNAL_HOST}:{INTERNAL_PORT}/health"
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self._proc = None
                raise RuntimeError("whisper-server exited during startup")
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        log("model loaded, server ready")
                        return
            except Exception:
                pass  # not listening yet; keep polling until the deadline
            time.sleep(0.5)
        self._stop_locked()
        raise RuntimeError("whisper-server did not become ready in time")

    def maybe_unload(self):
        with self._lock:
            if not self._running() or self._inflight > 0:
                return
            if time.monotonic() - self._last_activity < IDLE_TTL:
                return
            log(f"idle for {IDLE_TTL:.0f}s, unloading model to free VRAM")
            self._stop_locked()

    def _stop_locked(self):
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def shutdown(self):
        with self._lock:
            self._stop_locked()


manager = Manager()


def idle_monitor():
    while True:
        time.sleep(5)
        try:
            manager.maybe_unload()
        except Exception as exc:
            log(f"idle monitor error: {exc}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # the supervisor logs its own lifecycle; skip the per-request noise

    def _serve_health(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        try:
            manager.begin_request()
        except Exception as exc:
            log(f"could not start whisper-server: {exc}")
            self.send_error(503, "transcription server unavailable")
            return
        try:
            conn = http.client.HTTPConnection(INTERNAL_HOST, INTERNAL_PORT, timeout=900)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_BY_HOP and k.lower() != "host"}
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            self.send_response(resp.status)
            for key, value in resp.getheaders():
                if key.lower() in HOP_BY_HOP or key.lower() == "content-length":
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            conn.close()
        except Exception as exc:
            log(f"proxy error: {exc}")
            try:
                self.send_error(502, "transcription server error")
            except Exception:
                pass  # response may already be partially written
        finally:
            manager.end_request()

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/health":
            self._serve_health()
        else:
            self._proxy()

    def do_POST(self):
        self._proxy()


def main():
    threading.Thread(target=idle_monitor, daemon=True).start()

    def on_signal(signum, frame):
        log("shutting down")
        manager.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log(f"listening on {HOST}:{PORT}; idle TTL {IDLE_TTL:.0f}s; model loads on first request")
    server.serve_forever()


if __name__ == "__main__":
    main()
