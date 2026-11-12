#!/usr/bin/env python
"""Run the inspection API: python scripts/serve.py --db data/starter.sqlite [--port 8000]"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from catalog_onboard.api import create_app
from catalog_onboard.obs import configure_logging


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=1, help="uvicorn worker processes")
    args = ap.parse_args()
    configure_logging(logging.INFO)  # JSON access log (spec §8); uvicorn uses the root handler
    if args.workers > 1:
        # multi-process needs an import string; the factory reads the DB path from the environment
        import os

        os.environ["CATALOG_DB"] = args.db
        uvicorn.run("catalog_onboard.api.app:app_from_env", factory=True, host=args.host, port=args.port,
                    workers=args.workers, log_config=None, access_log=False)
    else:
        uvicorn.run(create_app(args.db), host=args.host, port=args.port, log_config=None, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
