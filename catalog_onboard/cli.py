"""Command line entry point: python -m catalog_onboard ingest --db path.sqlite [...]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from catalog_onboard import obs
from catalog_onboard.config import Settings
from catalog_onboard.db.store import Store, open_readwrite
from catalog_onboard.pipeline.engine import Engine
from catalog_onboard.pipeline.llm import LLMClient
from catalog_onboard.pipeline.llm.fake import FakeLLM, ForbiddenLLM
from catalog_onboard.rules import load_rules
from catalog_onboard.taxonomy import Taxonomy, TaxonomyConfigError

EXIT_OK = 0
EXIT_CONFIG = 2


def _configure_logging(verbose: bool) -> None:
    # JSON on stderr with feed_item_id / run_id on every line; -v adds one INFO line per item.
    obs.configure_logging(logging.INFO if verbose else logging.WARNING)


def make_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "fake":
        return FakeLLM()
    from catalog_onboard.pipeline.llm.anthropic_client import AnthropicClient

    return AnthropicClient(model=settings.llm_model, max_tokens=settings.llm_max_tokens, effort=settings.llm_effort)


def _open(args: argparse.Namespace, *, llm_override: LLMClient | None = None) -> tuple[Engine, Settings] | int:
    """Shared setup. Returns (engine, settings) or an EXIT_CONFIG code (AC-3.6)."""
    settings = Settings.from_env()
    if getattr(args, "llm", None):
        settings = Settings(**{**settings.__dict__, "llm_provider": args.llm})
    if getattr(args, "concurrency", None):
        settings = Settings(**{**settings.__dict__, "llm_concurrency": args.concurrency})
    db = Path(args.db)
    if not db.exists():
        print(f"error: database not found: {db}", file=sys.stderr)
        return EXIT_CONFIG
    try:
        rules = load_rules(settings.rules_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: rules: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    store = Store(open_readwrite(db))
    if not store.has_starter_tables():
        print("error: database lacks canonical_categories / incoming_supplier_feeds", file=sys.stderr)
        return EXIT_CONFIG
    store.migrate()
    try:
        taxonomy = Taxonomy.from_store(store)
    except TaxonomyConfigError as exc:
        print(f"error: taxonomy: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    llm = llm_override or make_llm(settings)
    engine = Engine(store=store, taxonomy=taxonomy, rules=rules, llm=llm, settings=settings, dry_run=args.dry_run)
    return engine, settings


def _print_summary(rows: list[tuple[str, str]]) -> None:
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{k.ljust(width)}  {v}")


def cmd_ingest(args: argparse.Namespace) -> int:
    opened = _open(args)
    if isinstance(opened, int):
        return opened
    engine, settings = opened

    summary = engine.run_batch(supplier_id=args.supplier, limit=args.limit)
    rows = [
        ("run_id", summary.run_id),
        ("rules_version", engine.version),
        ("llm", settings.llm_provider),
        ("concurrency", str(engine.concurrency)),
        ("dry_run", str(args.dry_run)),
        ("processed", str(summary.processed)),
        ("draft", str(summary.draft)),
        ("quarantined", str(summary.quarantined)),
        ("errors", str(summary.errors)),
        ("skipped (already processed)", str(summary.skipped)),
    ]
    _print_summary(rows)
    return EXIT_OK


def cmd_revalidate(args: argparse.Namespace) -> int:
    """AC-2.4: re-decide stored outcomes under the current rules + taxonomy, with no LLM calls."""
    opened = _open(args, llm_override=ForbiddenLLM())
    if isinstance(opened, int):
        return opened
    engine, _ = opened
    summary = engine.revalidate_batch(limit=args.limit)
    _print_summary([
        ("run_id", summary.run_id),
        ("rules_version", engine.version),
        ("llm", "none (stored candidates)"),
        ("dry_run", str(args.dry_run)),
        ("revalidated", str(summary.processed)),
        ("draft", str(summary.draft)),
        ("quarantined", str(summary.quarantined)),
        ("errors", str(summary.errors)),
        ("skipped (already at this version)", str(summary.skipped)),
    ])
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="catalog_onboard")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="process incoming_supplier_feeds into draft_catalog / quarantine")
    ing.add_argument("--db", required=True)
    ing.add_argument("--limit", type=int, default=None)
    ing.add_argument("--supplier", default=None)
    ing.add_argument("--dry-run", action="store_true")
    ing.add_argument("--llm", choices=["anthropic", "fake"], default=None)
    ing.add_argument("--concurrency", type=int, default=None,
                     help="LLM worker threads (default 8; env LLM_CONCURRENCY)")
    ing.add_argument("-v", "--verbose", action="store_true")
    ing.set_defaults(func=cmd_ingest)
    rev = sub.add_parser("revalidate", help="re-run validate / comply / route from stored candidates (no LLM)")
    rev.add_argument("--db", required=True)
    rev.add_argument("--limit", type=int, default=None)
    rev.add_argument("--dry-run", action="store_true")
    rev.add_argument("-v", "--verbose", action="store_true")
    rev.set_defaults(func=cmd_revalidate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    return int(args.func(args))
