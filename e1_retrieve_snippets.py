#!/usr/bin/env python3
"""E1 Phase 2: Retrieve corpus snippets for existing Phase 1 results.

Reads Phase 1 output JSON (with top_k_spans), retrieves snippets from
infini-gram API/local engine, and adds ExampleSnippets field.
Skips records that already have ExampleSnippets.

Usage:
    # OLMo 2 (API backend)
    python e1_retrieve_snippets.py \
        --model olmo2-1b

    # GPT-J (local engine)
    python e1_retrieve_snippets.py \
        --model gpt-j \
        --index_dir ./index

    # Custom parameters
    python e1_retrieve_snippets.py \
        --model olmo2-1b \
        --max_docs_per_span 10 \
        --max_disp_len 80
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

PKG_DIR = os.path.join(SCRIPT_DIR, "infini-gram", "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from transformers import AutoTokenizer

# Import from e1_verbatim_trace.py
from e1_verbatim_trace import (
    MODEL_CONFIGS,
    InfiniGramAPIEngine,
    init_engine,
    retrieve_snippets_for_span
)


def setup_logger(model_key: str, config: str = "standard"):
    log_dir = os.path.join("logs", model_key)
    os.makedirs(log_dir, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filepath = os.path.join(log_dir, f"e1_retrieve_snippets.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filepath, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging to file: %s", log_filepath)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="E1 Phase 2: Retrieve snippets for existing Phase 1 results"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model key (determines auto paths)")
    parser.add_argument("--config", type=str, default="standard",
                        help="HarmBench config name (default: standard)")
    parser.add_argument("--input", type=str, default=None,
                        help="Phase 1 results JSON. "
                             "Default: results/{model_dir}/e1_verbatim_{config}.json")
    parser.add_argument("--index_dir", type=str, default=None,
                        help="Path to local infini-gram index directory. "
                             "If omitted, use HTTP API.")
    parser.add_argument("--api_index", type=str,
                        default="v4_olmo-mix-1124_llama",
                        help="API index name (default: v4_olmo-mix-1124_llama)")
    parser.add_argument("--tokenizer_name", type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer matching the infini-gram index")
    parser.add_argument("--max_docs_per_span", type=int, default=10,
                        help="Max documents to retrieve per span (default: 10)")
    parser.add_argument("--max_disp_len", type=int, default=80,
                        help="Max tokens per document snippet (default: 80)")

    args = parser.parse_args()

    # Auto-generate input path if not specified
    model_dir = MODEL_CONFIGS[args.model]["out_dir"]
    if args.input is None:
        args.input = os.path.join(
            "results", model_dir,
            f"e1_verbatim_{args.config}.json"
        )

    return args


def main():
    args = parse_args()
    logger = setup_logger(args.model, args.config)

    logger.info("=== E1 Phase 2: Snippet Retrieval started at %s ===",
                datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("Arguments: %s", vars(args))
    start_time = time.time()

    # Load Phase 1 results
    logger.info("Loading Phase 1 results from %s ...", args.input)
    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)
    logger.info("Loaded %d records", len(results))

    # Initialize tokenizer
    logger.info("Loading tokenizer: %s", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        token=os.environ.get("HF_TOKEN"),
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
    )

    # Initialize engine
    engine, num_shards, corpus_size = init_engine(args, tokenizer, logger)

    # Process each record
    for rec_idx, rec in enumerate(results):
        rec_id = rec.get("id", rec_idx)
        e1 = rec.get("e1", {})

        # Skip if error or already has snippets
        if "error" in e1:
            logger.info("[%d/%d] Skipping record id=%s (has error)",
                        rec_idx + 1, len(results), rec_id)
            continue

        if "ExampleSnippets" in e1:
            logger.info("[%d/%d] Skipping record id=%s (snippets already exist)",
                        rec_idx + 1, len(results), rec_id)
            continue

        top_k_spans = e1.get("top_k_spans", [])
        if not top_k_spans:
            logger.info("[%d/%d] Skipping record id=%s (no top_k_spans)",
                        rec_idx + 1, len(results), rec_id)
            continue

        response_ids = tokenizer.encode(rec.get("response", ""))

        logger.info("=" * 70)
        logger.info("[%d/%d] Retrieving snippets for record id=%s (%d spans)",
                    rec_idx + 1, len(results), rec_id, len(top_k_spans))

        rec_start = time.time()
        snippets_by_span = []

        for span_idx, span in enumerate(top_k_spans):
            b = span["begin"]
            e_pos = span["end"]
            span_ids = response_ids[b:e_pos]

            snippets = retrieve_snippets_for_span(
                engine, span_ids,
                max_docs=args.max_docs_per_span,
                max_disp_len=args.max_disp_len,
                tokenizer=tokenizer,
            )

            snippets_by_span.append({
                "span_begin": b,
                "span_end": e_pos,
                "span_length": e_pos - b,
                "span_text": span.get("text", ""),
                "num_snippets": len(snippets),
                "snippets": snippets,
            })

            if (span_idx + 1) % 10 == 0:
                logger.info("  Span %d / %d done",
                            span_idx + 1, len(top_k_spans))

        e1["ExampleSnippets"] = snippets_by_span

        rec_elapsed = time.time() - rec_start
        total_snippets = sum(s["num_snippets"] for s in snippets_by_span)
        logger.info("  Done: %d spans, %d snippets, %.1fs",
                    len(snippets_by_span), total_snippets, rec_elapsed)

        # Incremental save after each record
        logger.info("  Saving to %s ...", args.input)
        with open(args.input, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    with_snippets = sum(1 for r in results
                        if "ExampleSnippets" in r.get("e1", {}))
    logger.info("  Records with snippets: %d / %d", with_snippets, len(results))

    elapsed_float = time.time() - start_time
    elapsed = int(elapsed_float)
    days = elapsed // 86400
    hours = (elapsed % 86400) // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    logger.info("=== Finished at %s ===",
                datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("=== Elapsed time: %d days %02d:%02d:%02d (total %.3f seconds) ===",
                days, hours, minutes, seconds, elapsed_float)


if __name__ == "__main__":
    main()