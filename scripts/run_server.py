"""Start the MedDiag API server.

Examples
--------
    python scripts/run_server.py
    python scripts/run_server.py --reload --port 9000
    python scripts/run_server.py --host 127.0.0.1 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MedDiag API server.")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        print("uvicorn is not installed - run: pip install -r requirements.txt")
        return 1

    print(f"MedDiag {settings.version} starting on http://{args.host}:{args.port}")
    print(f"  dashboard : http://{args.host}:{args.port}/dashboard")
    print(f"  API docs  : http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=None if args.reload else args.workers,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
