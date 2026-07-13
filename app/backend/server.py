from __future__ import annotations

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "app" / "frontend"
INTERVAL_CONFIG_PATH = PROJECT_ROOT / "models" / "price_interval_config.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.predictor_service import (
    MODEL_PATH,
    compare_from_text,
    load_prediction_model,
    predict_from_raw,
    predict_from_text,
    sweep_from_raw,
)
from src.prediction import API_VERSION, load_interval_config


class LaptopPriceHandler(BaseHTTPRequestHandler):
    server_version = "LaptopPriceBackend/2.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self.write_json(
                {
                    "ok": True,
                    "api_version": API_VERSION,
                    "supports_price_range": True,
                    "supports_compare": True,
                    "supports_config_sweep": True,
                    "model_available": MODEL_PATH.exists(),
                    "model_path": str(MODEL_PATH),
                }
            )
            return
        if path == "/api/price-interval-config":
            self.write_json(load_interval_config(INTERVAL_CONFIG_PATH))
            return
        self.serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/predict", "/api/compare", "/api/config-sweep"}:
            self.write_json({"error": "Not found"}, status=404)
            return

        try:
            payload = self.read_json()
            if path == "/api/config-sweep":
                raw_features = payload.get("raw_features") or {}
                result = sweep_from_raw(raw_features)
            elif path == "/api/compare":
                description = str(payload.get("description") or "").strip()
                if not description:
                    raise ValueError("description is required for compare mode.")
                result = compare_from_text(description)
            else:
                mode = payload.get("mode", "text")
                task = payload.get("task", "predict")
                if task == "compare":
                    if mode != "text":
                        raise ValueError("Compare task is only available in Gemini text mode.")
                    description = str(payload.get("description") or "").strip()
                    if not description:
                        raise ValueError("description is required for compare mode.")
                    result = compare_from_text(description)
                elif mode == "manual":
                    result = predict_from_raw(payload.get("raw_features") or {})
                else:
                    description = str(payload.get("description") or "").strip()
                    if not description:
                        raise ValueError("description is required for text mode.")
                    result = predict_from_text(description)
        except Exception as exc:
            self.write_json({"error": str(exc)}, status=400)
            return

        self.write_json(result)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        if not raw_body:
            return {}
        parsed = json.loads(raw_body)
        if not isinstance(parsed, dict):
            raise ValueError("JSON body must be an object.")
        return parsed

    def write_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=self._json_default).encode("utf-8")
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            path = "/index.html"
        relative = unquote(path).lstrip("/")
        target = (FRONTEND_DIR / relative).resolve()

        if not str(target).startswith(str(FRONTEND_DIR.resolve())) or not target.is_file():
            self.write_json({"error": "Not found"}, status=404)
            return

        content = target.read_bytes()
        content_type, _ = mimetypes.guess_type(str(target))
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        if target.suffix.lower() in {".html", ".js", ".css"}:
            self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    @staticmethod
    def _json_default(value: object) -> float | int | str:
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    load_prediction_model()
    server = ThreadingHTTPServer((host, port), LaptopPriceHandler)
    print(f"Serving laptop price app at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
