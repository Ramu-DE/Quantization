"""
Unified server: runs both backend (FastAPI) and frontend (Next.js)
behind a single port for Cloud Run deployment.
"""
import subprocess
import os
import sys
import signal
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError
import threading
import json

PORT = int(os.environ.get("PORT", 8080))
BACKEND_PORT = 8000
FRONTEND_PORT = 3000


class ProxyHandler(BaseHTTPRequestHandler):
    """Routes /api/* to backend, everything else to frontend"""

    def do_request(self):
        if self.path.startswith("/api/") or self.path in ("/health", "/docs", "/openapi.json"):
            target = f"http://127.0.0.1:{BACKEND_PORT}{self.path}"
        else:
            target = f"http://127.0.0.1:{FRONTEND_PORT}{self.path}"

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else None

            req = Request(
                target,
                data=body,
                headers={k: v for k, v in self.headers.items() if k.lower() not in ("host",)},
                method=self.command,
            )

            resp = urlopen(req, timeout=600)
            self.send_response(resp.status)
            for key, value in resp.getheaders():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(resp.read())
        except URLError as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Service unavailable: {e}"}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        self.do_request()

    def do_POST(self):
        self.do_request()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logs


def start_backend():
    """Start FastAPI backend"""
    env = os.environ.copy()
    return subprocess.Popen(
        ["uv", "run", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
        cwd="/app/backend",
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def start_frontend():
    """Start Next.js frontend in production mode"""
    env = os.environ.copy()
    env["NEXT_PUBLIC_API_URL"] = ""  # Same-origin, proxy handles routing
    env["PORT"] = str(FRONTEND_PORT)
    return subprocess.Popen(
        ["npm", "run", "start"],
        cwd="/app/frontend",
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def wait_for_service(port, name, timeout=60):
    """Wait for a service to become healthy"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urlopen(f"http://127.0.0.1:{port}/", timeout=2)
            print(f"  {name} ready on port {port}")
            return True
        except Exception:
            time.sleep(1)
    print(f"  WARNING: {name} did not start within {timeout}s")
    return False


def main():
    print("=" * 50)
    print("  Quantization Visualizer - Cloud Run")
    print("=" * 50)

    # Start services
    print("\nStarting backend...")
    backend = start_backend()

    print("Starting frontend...")
    frontend = start_frontend()

    # Wait for services
    print("\nWaiting for services...")
    wait_for_service(BACKEND_PORT, "Backend")
    wait_for_service(FRONTEND_PORT, "Frontend")

    # Start proxy
    print(f"\nProxy listening on port {PORT}")
    print(f"  / -> frontend (:{FRONTEND_PORT})")
    print(f"  /api/* -> backend (:{BACKEND_PORT})")
    print("=" * 50)

    server = HTTPServer(("0.0.0.0", PORT), ProxyHandler)

    def shutdown(sig, frame):
        print("\nShutting down...")
        backend.terminate()
        frontend.terminate()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    server.serve_forever()


if __name__ == "__main__":
    main()
