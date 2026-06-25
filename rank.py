"""
rank.py — RAGnarok CLI entry point.

Runs the full ranking pipeline directly (no FastAPI) and writes submission.csv.

Usage:
    python rank.py [--input data/candidates.jsonl.gz] [--top-k 100] [--output output/submission.csv]

This is equivalent to POST /rank → POST /export/csv via the API, but runs
all pipeline modules in-process for offline / CI use.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
)
logger = logging.getLogger("rank")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAGnarok — offline candidate ranking CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=(
            config.CANDIDATES_JSONL_GZ if config.CANDIDATES_JSONL_GZ.exists()
            else config.CANDIDATES_JSONL if config.CANDIDATES_JSONL.exists()
            else config.SAMPLE_CANDIDATES_JSON
        ),
        help="Path to candidates file (.jsonl, .jsonl.gz, or .json array)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=config.SUBMISSION_TOP_K,
        help="Number of top candidates to return",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=config.DEFAULT_SUBMISSION_PATH,
        help="Output path for submission.csv",
    )
    parser.add_argument(
        "--jd",
        type=Path,
        default=None,
        help="Optional path to a job description file (defaults to parsed_job_description.json)",
    )
    return parser.parse_args()


import io
import orjson

logger = logging.getLogger(__name__)

def _load_candidates(input_path: Path) -> list:
    """Load and parse candidates efficiently using streaming and orjson."""
    logger.info("Loading candidates from %s", input_path)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    from pipeline.candidate_parser import CandidateParser
    parser = CandidateParser()
    candidates = []
    errors = 0

    # Open as stream — handles .gz without full decompression in RAM
    base_file = open(input_path, "rb")
    file_stream = gzip.open(base_file, "rb") if input_path.suffix == ".gz" else base_file

    try:
        # Peek at first non-empty byte to detect JSON array vs JSONL
        first_byte = b""
        for chunk in iter(lambda: file_stream.read(1), b""):
            if chunk.strip():
                first_byte = chunk
                break

        if first_byte == b"[":
            # JSON array
            try:
                full_bytes = first_byte + file_stream.read()
                arr = orjson.loads(full_bytes)
                for item in arr:
                    try:
                        candidates.append(parser.parse_candidate(item))
                    except Exception as e:
                        errors += 1
                        logger.debug("Parse error: %s", e)
            except orjson.JSONDecodeError as e:
                logger.error("JSON parse error: %s", e)
                sys.exit(1)
        else:
            # JSONL: stream line-by-line
            buffered_stream = io.BufferedReader(file_stream)
            for line_no, line in enumerate(buffered_stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidates.append(parser.parse_candidate(orjson.loads(line)))
                except Exception as e:
                    errors += 1
                    logger.debug("Line %d parse error: %s", line_no, e)

    finally:
        file_stream.close()
        if input_path.suffix == ".gz":
            base_file.close()

    logger.info("Loaded %d candidates (%d parse errors)", len(candidates), errors)
    return candidates



def _load_jd(jd_path: Path | None):
    """Load JD intent — from file if given, otherwise from pre-parsed JSON."""
    from pipeline.jd_parser import JDParser
    jd_parser = JDParser()
    if jd_path and jd_path.exists():
        logger.info("Parsing JD from %s", jd_path)
        return jd_parser.parse(str(jd_path))
    logger.info("Using pre-parsed JD from parsed_job_description.json")
    return jd_parser.load_parsed()


def _write_csv(ranked: list, output_path: Path) -> None:
    """Write ranked candidates to submission.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for rc in ranked:
            writer.writerow({
                "candidate_id": rc.candidate_id,
                "rank": rc.rank,
                "score": f"{rc.final_score:.6f}",
                "reasoning": rc.reasoning or "",
            })
    logger.info("Wrote %d rows to %s", len(ranked), output_path)


def _validate_submission(output_path: Path, top_k: int) -> bool:
    """Basic submission validation: row count, monotonic scores, rank range."""
    import re
    ok = True
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != top_k:
        logger.warning("⚠ Row count mismatch: expected %d, got %d", top_k, len(rows))
        ok = False

    id_pat = re.compile(config.SUBMISSION_CANDIDATE_ID_PATTERN)
    prev_score = float("inf")
    for i, row in enumerate(rows, start=1):
        cid = row.get("candidate_id", "")
        if not id_pat.match(cid):
            logger.warning("⚠ Row %d: invalid candidate_id %r", i, cid)
            ok = False
        rank = int(row.get("rank", 0))
        if rank != i:
            logger.warning("⚠ Row %d: expected rank=%d, got %d", i, i, rank)
            ok = False
        score = float(row.get("score", 0))
        if score > prev_score + 1e-9:
            logger.warning("⚠ Non-monotonic score at rank %d: %.6f > %.6f (prev)", i, score, prev_score)
            ok = False
        prev_score = score

    if ok:
        logger.info("✅ Submission validated: %d rows, monotonic scores, all IDs valid", len(rows))
    return ok


def main() -> None:
    args = _parse_args()
    t_start = time.perf_counter()

    candidates = _load_candidates(args.input)
    if not candidates:
        logger.error("No candidates loaded — aborting.")
        sys.exit(1)

    jd_intent = _load_jd(args.jd)

    logger.info("Starting pipeline (top_k=%d) …", args.top_k)
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(jd=jd_intent, candidates=candidates)
    ranked, timings = runner.run(top_k=args.top_k)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Pipeline complete: %d candidates ranked in %.1fs",
        len(ranked), elapsed,
    )
    for stage, ms in timings.items():
        logger.info("  %-25s %7.1f ms", stage, ms)

    _write_csv(ranked, args.output)
    _validate_submission(args.output, args.top_k)

    # Print top-10 summary
    print(f"\n{'-'*70}")
    print(f"  RAGnarok — Top {min(10, len(ranked))} Candidates")
    print(f"{'-'*70}")
    for rc in ranked[:10]:
        verdict = rc.trust.verdict if rc.trust else "—"
        print(
            f"  #{rc.rank:<3} {rc.candidate_id:<15} "
            f"score={rc.final_score:.4f}  verdict={verdict:<10}  "
            f"{rc.reasoning[:60] if rc.reasoning else ''}…"
        )
    print(f"{'-'*70}")
    print(f"  Output: {args.output}")
    print(f"  Total time: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
