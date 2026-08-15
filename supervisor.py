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

The model files are not baked into the image. On startup the supervisor ensures
the transcription and VAD models exist in MODELS_DIR (a mountable directory) and
downloads them once via whisper.cpp's own scripts if they are missing, so the
model persists in the mounted volume across container rebuilds and restarts.

It also adapts the transcription request so a client written for faster-whisper
or the OpenAI API drives this whisper.cpp server unchanged: a few form fields use
different names between those APIs and whisper.cpp, so the supervisor renames them
on the way through (see FORM_FIELD_RENAMES). The rewrite is strictly defensive: on
anything unexpected it forwards the body untouched, so it can only improve
compatibility, never corrupt a request.

Configuration (all via environment):
  HOST                  public bind address (default 0.0.0.0)
  PORT                  public port (default 8080)
  INTERNAL_PORT         private port the real server binds (default 8081)
  WHISPER_IDLE_TTL      seconds idle before the model is unloaded (default 300)
  WHISPER_START_TIMEOUT seconds to wait for the model to load (default 180)
  WHISPER_START_RETRIES extra attempts if the server dies while loading (default 1)
  MODELS_DIR            directory the models live in / are downloaded to (default /models)
  WHISPER_MODEL         ggml model name to ensure present (default large-v2)
  VAD_MODEL             Silero VAD model name to ensure present (default silero-v5.1.2)
  WHISPER_CPP_DIR       whisper.cpp checkout holding the download scripts (default /app)

The child command (the whisper-server binary and its flags) is passed as this
script's own arguments; the supervisor appends --host/--port so the child binds
the private internal port. It also fills in the model paths (-m and, when --vad is
present, --vad-model) from WHISPER_MODEL / VAD_MODEL when the child command does not
already specify them, so the model NAME lives in one place (the env) and the
download target and the load path cannot drift apart. An explicit -m / --vad-model
in the child command is respected and left untouched.
"""

import collections
import http.client
import os
import re
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
# A start can fail for a transient reason (most often another process holding the VRAM this one
# needs on a shared GPU), so one retry turns a spurious 503 into a slightly slower success.
START_RETRIES = int(os.environ.get("WHISPER_START_RETRIES", "1"))
RETRY_DELAY = float(os.environ.get("WHISPER_START_RETRY_DELAY", "3"))
CHILD_LOG_TAIL = 40  # child output lines kept so a startup failure can quote what it last said

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v2")
VAD_MODEL = os.environ.get("VAD_MODEL", "silero-v5.1.2")
WHISPER_CPP_DIR = os.environ.get("WHISPER_CPP_DIR", "/app")


def _model_path(model_name):
    """The on-disk path of a ggml model in MODELS_DIR. The one place the ggml-<name>.bin naming lives, so the
    download target and the load path (-m / --vad-model) are derived from the same rule and cannot drift."""
    return os.path.join(MODELS_DIR, f"ggml-{model_name}.bin")

# Hop-by-hop headers must not be forwarded across a proxy hop (RFC 7230 6.1).
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}

# Transcription form fields whose NAME differs between the faster-whisper / OpenAI APIs and whisper.cpp's
# server, mapped onto whisper.cpp's own names so a client written for those APIs works here unchanged. Fields
# whisper.cpp does not recognise (e.g. compression_ratio_threshold, no_repeat_ngram_size,
# timestamp_granularities[]) are left as-is and simply ignored by the server; whisper.cpp's own defaults cover
# what they asked for (entropy_thold ~2.4 is its degenerate-output + temperature-fallback guard, and
# token_timestamps is on by default so verbose_json already carries per-word times).
FORM_FIELD_RENAMES = {
    b"vad_filter": b"vad",  # same boolean meaning (enable voice-activity detection)
}
# Boolean fields whose SENSE is inverted between the two APIs: rename and flip the value.
# faster-whisper condition_on_previous_text == NOT whisper.cpp no_context.
FORM_FIELD_INVERTED_RENAMES = {
    b"condition_on_previous_text": b"no_context",
}
_TRUEY = {b"true", b"1", b"yes", b"on"}


def _flip_bool(value):
    """Return the textual negation of a boolean form value (whatever isn't truthy becomes "true")."""
    return b"false" if value.strip().lower() in _TRUEY else b"true"


def _rewrite_multipart(body, boundary):
    """Rename the known faster-whisper/OpenAI form fields to whisper.cpp's names in a multipart body.

    Splits on the multipart boundary (which by definition never occurs inside any part's content, so the
    binary file part is preserved byte for byte) and rewrites only the small text parts whose name we
    recognise. Defensive by construction: any part we don't recognise is left exactly as-is, and the caller
    falls back to the original body if this raises, so a parse miss can never corrupt the request."""
    delim = b"--" + boundary
    segments = body.split(delim)
    changed = False
    for idx, seg in enumerate(segments):
        # A real part is "\r\n<headers>\r\n\r\n<content>\r\n"; the preamble ("") and closing ("--\r\n") are not.
        if not seg.startswith(b"\r\n"):
            continue
        head_end = seg.find(b"\r\n\r\n")
        if head_end == -1:
            continue
        headers = seg[2:head_end]
        match = re.search(rb'name="([^"]*)"', headers)
        if not match:
            continue
        name = match.group(1)
        if name in FORM_FIELD_RENAMES:
            new_headers = headers.replace(b'name="' + name + b'"', b'name="' + FORM_FIELD_RENAMES[name] + b'"', 1)
            segments[idx] = b"\r\n" + new_headers + seg[head_end:]
            changed = True
        elif name in FORM_FIELD_INVERTED_RENAMES:
            content_start = head_end + 4
            content_end = seg.rfind(b"\r\n")  # the trailing CRLF before the next delimiter
            if content_end <= content_start:
                continue
            new_name = FORM_FIELD_INVERTED_RENAMES[name]
            new_headers = headers.replace(b'name="' + name + b'"', b'name="' + new_name + b'"', 1)
            new_value = _flip_bool(seg[content_start:content_end])
            segments[idx] = b"\r\n" + new_headers + seg[head_end:content_start] + new_value + seg[content_end:]
            changed = True
    return delim.join(segments) if changed else body

# Everything after the script name is the child command, minus host/port which we
# add ourselves so the child only ever binds the private internal port.
CHILD_BASE = sys.argv[1:]
if not CHILD_BASE:
    print("[supervisor] no whisper-server command supplied", file=sys.stderr)
    sys.exit(2)


def _with_model_paths(argv):
    """Fill in the transcription (-m) and VAD (--vad-model) model paths from WHISPER_MODEL / VAD_MODEL when the
    child command doesn't already give them, so the model name has a single source of truth (the env). A path the
    caller specified explicitly is left as-is; --vad-model is only added when --vad is actually enabled."""
    args = list(argv)
    if not any(a in ("-m", "--model") for a in args):
        args += ["-m", _model_path(WHISPER_MODEL)]
    if "--vad" in args and not any(a in ("-vm", "--vad-model") for a in args):
        args += ["--vad-model", _model_path(VAD_MODEL)]
    return args


CHILD_BASE = _with_model_paths(CHILD_BASE)


def log(msg):
    print(f"[supervisor] {msg}", flush=True)


def _looks_like_ggml(path):
    """True if the file starts with a ggml/GGUF magic. A model left behind by an interrupted or
    redirected download (empty, truncated header, an HTML error page) is otherwise indistinguishable
    from a good one by size alone, and whisper-server then dies on every single request forever."""
    try:
        with open(path, "rb") as fh:
            # GGML_FILE_MAGIC is 0x67676d6c, written as a uint32, so it lands on disk as "lmgg" on a
            # little-endian host; accept the other byte order and GGUF too rather than risk a false
            # positive that throws away a perfectly good multi-GB model.
            return fh.read(4) in (b"lmgg", b"ggml", b"GGUF")
    except OSError:
        return False


def _ensure_model(script_name, model_name):
    """Download ggml-<model_name>.bin into MODELS_DIR via whisper.cpp's own script
    if it isn't already there. Both download-ggml-model.sh and download-vad-model.sh
    take the model name and an output directory and write ggml-<name>.bin into it."""
    target = _model_path(model_name)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        if _looks_like_ggml(target):
            log(f"model present: {target}")
            return
        # The download script skips a target that already exists, so the bad file has to go first.
        log(f"model {target} is not a ggml file (bad or interrupted download); re-downloading")
        os.remove(target)
    script = os.path.join(WHISPER_CPP_DIR, "models", script_name)
    log(f"model {model_name} missing from {MODELS_DIR}; downloading once via {script_name}")
    subprocess.run(["bash", script, model_name, MODELS_DIR], check=True)
    if not (os.path.exists(target) and os.path.getsize(target) > 0):
        raise RuntimeError(f"download of {model_name} did not produce {target}")
    if not _looks_like_ggml(target):
        raise RuntimeError(f"download of {model_name} produced a non-ggml file at {target}")
    log(f"model ready: {target}")


def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    _ensure_model("download-ggml-model.sh", WHISPER_MODEL)
    _ensure_model("download-vad-model.sh", VAD_MODEL)


class Manager:
    """Owns the whisper-server child process and its load/unload lifecycle."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._reader = None
        self._tail = collections.deque(maxlen=CHILD_LOG_TAIL)
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
        last_error = None
        for attempt in range(1, START_RETRIES + 2):
            try:
                self._start_once_locked()
                return
            except Exception as exc:
                last_error = exc
                self._stop_locked()  # reap the corpse (or kill a hung child) before trying again
                if attempt <= START_RETRIES:
                    log(f"start attempt {attempt} failed: {exc}; retrying in {RETRY_DELAY:.0f}s")
                    time.sleep(RETRY_DELAY)
        raise last_error

    def _start_once_locked(self):
        cmd = CHILD_BASE + ["--host", INTERNAL_HOST, "--port", str(INTERNAL_PORT)]
        log(f"loading model: starting whisper-server on {INTERNAL_HOST}:{INTERNAL_PORT}")
        self._tail.clear()
        # Pipe the child's stdout/stderr through us rather than letting it inherit ours: the reason a
        # start failed is only ever in the child's own output, and inherited output is interleaved with
        # everything else and gone by the time we raise.
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump_output, args=(self._proc,), daemon=True)
        self._reader.start()
        self._wait_ready()

    def _pump_output(self, proc):
        """Mirror the child's output into our log, keeping the last CHILD_LOG_TAIL lines so a failed
        start can quote what it said. Runs without the manager lock; deque.append is atomic."""
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                self._tail.append(line)
                print(f"[whisper-server] {line}", flush=True)
        except Exception:
            pass  # the pipe is torn down on stop; nothing here is worth failing a request over
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    def _exit_reason(self, rc):
        """Human-readable cause of a child exit: how it died, plus the last thing it printed."""
        if self._reader is not None:
            self._reader.join(timeout=2)  # let the final lines land before we quote them
        if rc < 0:
            # Compared by name, not by signal.SIGKILL etc: those constants are POSIX-only and this
            # module is also imported on the Windows dev box, where a negative rc never occurs anyway.
            try:
                name = signal.Signals(-rc).name
            except ValueError:
                name = f"signal {-rc}"
            how = f"killed by {name}"
            if name == "SIGKILL":
                how += " (usually the OOM killer: check host RAM and the container memory limit)"
            elif name in ("SIGABRT", "SIGSEGV"):
                how += " (typically a ggml/driver abort, e.g. a failed GPU allocation: check free VRAM)"
        else:
            how = f"exit code {rc}"
        tail = " | ".join(list(self._tail)[-8:])
        return f"{how}; last output: {tail}" if tail else f"{how}; it printed nothing"

    def _wait_ready(self):
        url = f"http://{INTERNAL_HOST}:{INTERNAL_PORT}/health"
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            rc = self._proc.poll()
            if rc is not None:
                raise RuntimeError(f"whisper-server exited during startup: {self._exit_reason(rc)}")
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        log("model loaded, server ready")
                        return
            except Exception:
                pass  # not listening yet; keep polling until the deadline
            time.sleep(0.5)
        # The caller stops the child; it only has to die here, not tell us why it was slow.
        raise RuntimeError(f"whisper-server did not become ready within {START_TIMEOUT:.0f}s")

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
        reader = self._reader
        self._proc = None
        self._reader = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if reader is not None:
            reader.join(timeout=5)  # drain the last of its output into the log before moving on

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
        # Adapt faster-whisper/OpenAI form-field names to whisper.cpp's before forwarding. Only multipart
        # POSTs (transcription requests) are touched; on any parse trouble we forward the body unchanged.
        ctype = self.headers.get("Content-Type", "")
        if body and self.command == "POST" and ctype.startswith("multipart/form-data"):
            bmatch = re.search(r'boundary=("?)([^";]+)\1', ctype)
            if bmatch:
                try:
                    body = _rewrite_multipart(body, bmatch.group(2).encode("latin-1"))
                except Exception as exc:
                    log(f"form-field adapter skipped, forwarding request unchanged: {exc}")
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
            # The rewrite can change the body length (renamed field, flipped value), so always send the real one.
            if body is not None:
                headers["Content-Length"] = str(len(body))
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
    # Fetch the models into the mounted dir before serving, if they aren't already
    # there. This downloads to disk only; the model is not loaded into VRAM until
    # the first transcription request, so on-demand loading is unaffected.
    ensure_models()

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
