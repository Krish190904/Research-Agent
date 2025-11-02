"""Command-line interface for the Deep Researcher Agent.

Provides subcommands: ingest, index, query, export, stats
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from index.store import Indexer
from reasoner.reasoner import Reasoner
from export import Exporter
from pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent.cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--folder", required=True)
    p_ingest.add_argument("--recursive", action="store_true")

    p_index = sub.add_parser("index")
    p_index.add_argument("--rebuild", action="store_true")

    p_query = sub.add_parser("query")
    p_query.add_argument("--q", required=True)
    p_query.add_argument("--topk", type=int, default=5)
    p_query.add_argument("--mmr", action="store_true", help="Enable MMR reranking")
    p_query.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
        help="MMR lambda (relevance vs diversity)",
    )
    p_query.add_argument(
        "--candidate-multiplier",
        type=int,
        default=5,
        help="Candidate multiplier for retrieval",
    )

    p_export = sub.add_parser("export")
    p_export.add_argument("--session", default="last")
    p_export.add_argument("--format", choices=["md", "pdf"], default="md")

    sub.add_parser("stats")

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        pl = Pipeline()
        n = pl.run_folder(Path(args.folder), recursive=args.recursive)
        logger.info("Ingested and indexed %d chunks", n)
        return 0

    if args.cmd == "index":
        idx = Indexer()
        if args.rebuild:
            idx.rebuild()
        else:
            idx.load()
        return 0

    if args.cmd == "query":
        idx = Indexer()
        idx.load()
        reasoner = Reasoner(indexer=idx)
        result = reasoner.answer(
            args.q,
            top_k=args.topk,
            mmr_enabled=bool(args.mmr),
            mmr_lambda=float(args.mmr_lambda),
            candidate_multiplier=int(args.candidate_multiplier),
        )
        print(result["synthesis"])
        return 0

    if args.cmd == "export":
        exp = Exporter()
        exp.export_last(format=args.format)
        return 0

    if args.cmd == "stats":
        idx = Indexer()
        idx.load()
        print(idx.stats())
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
