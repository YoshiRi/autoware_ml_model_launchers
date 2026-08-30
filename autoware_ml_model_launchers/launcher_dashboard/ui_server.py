from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .bag_player import BagPlayerManager, browse_bag_path
from .process_manager import ProcessManager
from .registry import build_launch_command, default_registry_path, load_registry


STATIC_DIR = Path(__file__).with_name("static")


class LauncherDashboardHandler(BaseHTTPRequestHandler):
    server_version = "LauncherDashboard/0.1"
    manager: ProcessManager
    bag_manager: BagPlayerManager

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"", "/"}:
            self._send_file(STATIC_DIR / "index.html")
            return
        if path.startswith("/static/"):
            self._send_file(STATIC_DIR / path.removeprefix("/static/"))
            return
        if path == "/api/launchers":
            self._send_json(self.manager.registry.to_json())
            return
        if path == "/api/processes":
            self._send_json({"processes": self.manager.list_processes()})
            return
        if path == "/api/logs":
            self._handle_logs(parsed.query)
            return
        if path == "/api/bag/status":
            self._send_json({"player": self.bag_manager.status()})
            return
        if path == "/api/bag/logs":
            self._handle_bag_logs(parsed.query)
            return
        if path == "/api/bag/browse":
            self._handle_bag_browse(parsed.query)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        handlers = {
            "/api/preview": self._handle_preview,
            "/api/preview_comparison": self._handle_preview_comparison,
            "/api/start": self._handle_start,
            "/api/start_multi_yolox": self._handle_start_multi_yolox,
            "/api/start_comparison": self._handle_start_comparison,
            "/api/stop": self._handle_stop,
            "/api/stop_all": self._handle_stop_all,
            "/api/stop_group": self._handle_stop_group,
            "/api/close": self._handle_close,
            "/api/close_all": self._handle_close_all,
            "/api/close_group": self._handle_close_group,
            "/api/bag/start": self._handle_bag_start,
            "/api/bag/stop": self._handle_bag_stop,
            "/api/bag/pause": self._handle_bag_pause,
            "/api/bag/resume": self._handle_bag_resume,
            "/api/bag/set_rate": self._handle_bag_set_rate,
        }
        handler = handlers.get(parsed.path)
        if handler is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            handler(self._read_json())
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _handle_preview(self, payload: dict[str, Any]) -> None:
        spec = self.manager.registry.get(str(payload["launcher_id"]))
        args = payload.get("args", {})
        command = build_launch_command(spec, args)
        self._send_json({"command": command, "model_path": spec.resolve_model_path(args)})

    def _handle_preview_comparison(self, payload: dict[str, Any]) -> None:
        self._send_json(self.manager.preview_comparison(payload))

    def _handle_start(self, payload: dict[str, Any]) -> None:
        process = self.manager.start(str(payload["launcher_id"]), payload.get("args", {}))
        self._send_json({"process": process})

    def _handle_start_multi_yolox(self, payload: dict[str, Any]) -> None:
        cameras = [str(camera) for camera in payload.get("cameras", [])]
        processes = self.manager.start_multi_yolox(cameras, payload.get("args", {}))
        self._send_json({"processes": processes})

    def _handle_start_comparison(self, payload: dict[str, Any]) -> None:
        self._send_json(self.manager.start_comparison(payload))

    def _handle_stop(self, payload: dict[str, Any]) -> None:
        process = self.manager.stop(str(payload["id"]))
        self._send_json({"process": process})

    def _handle_stop_all(self, _payload: dict[str, Any]) -> None:
        self._send_json({"processes": self.manager.stop_all()})

    def _handle_stop_group(self, payload: dict[str, Any]) -> None:
        self._send_json({"processes": self.manager.stop_group(str(payload["group_id"]))})

    def _handle_close(self, payload: dict[str, Any]) -> None:
        process = self.manager.close(str(payload["id"]))
        self._send_json({"process": process})

    def _handle_close_all(self, _payload: dict[str, Any]) -> None:
        self._send_json({"processes": self.manager.close_all()})

    def _handle_close_group(self, payload: dict[str, Any]) -> None:
        self._send_json({"processes": self.manager.close_group(str(payload["group_id"]))})

    def _handle_bag_start(self, payload: dict[str, Any]) -> None:
        player = self.bag_manager.start(
            str(payload.get("bag_path", "")),
            rate=payload.get("rate", 1.0),
            loop=bool(payload.get("loop", False)),
            clock=bool(payload.get("clock", False)),
            start_paused=bool(payload.get("start_paused", False)),
        )
        self._send_json({"player": player})

    def _handle_bag_stop(self, _payload: dict[str, Any]) -> None:
        self._send_json({"player": self.bag_manager.stop()})

    def _handle_bag_pause(self, _payload: dict[str, Any]) -> None:
        self._send_json({"player": self.bag_manager.pause()})

    def _handle_bag_resume(self, _payload: dict[str, Any]) -> None:
        self._send_json({"player": self.bag_manager.resume()})

    def _handle_bag_set_rate(self, payload: dict[str, Any]) -> None:
        self._send_json({"player": self.bag_manager.set_rate(payload.get("rate", 1.0))})

    def _handle_logs(self, query: str) -> None:
        params = parse_qs(query)
        process_id = params.get("id", [""])[0]
        lines = int(params.get("lines", ["200"])[0])
        self._send_json({"id": process_id, "text": self.manager.tail_log(process_id, lines)})

    def _handle_bag_logs(self, query: str) -> None:
        params = parse_qs(query)
        lines = int(params.get("lines", ["200"])[0])
        self._send_json({"text": self.bag_manager.tail_log(lines)})

    def _handle_bag_browse(self, query: str) -> None:
        params = parse_qs(query)
        path = params.get("path", [None])[0]
        self._send_json(browse_bag_path(path))

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON payload must be an object")
        return payload

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ROS launcher dashboard UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/tmp/autoware_ml_model_launchers/launcher_dashboard"),
    )
    args = parser.parse_args()

    registry_path = args.registry or default_registry_path()
    registry = load_registry(registry_path)
    LauncherDashboardHandler.manager = ProcessManager(registry, args.log_dir)
    LauncherDashboardHandler.bag_manager = BagPlayerManager(args.log_dir)

    server = ThreadingHTTPServer((args.host, args.port), LauncherDashboardHandler)
    print(f"Serving launcher dashboard at http://{args.host}:{args.port}/")
    print(f"Registry: {registry_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        LauncherDashboardHandler.manager.stop_all()
        LauncherDashboardHandler.bag_manager.stop()
        server.server_close()


if __name__ == "__main__":
    main()
