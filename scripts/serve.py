#!/usr/bin/env python
"""Run the inspection API: python scripts/serve.py --db data/starter.sqlite [--port 8000]"""

from __future__ import annotations

import argparse

import uvicorn

from catalog_onboard.api import create_app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(create_app(args.db), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
