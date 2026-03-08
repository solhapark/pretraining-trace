#!/usr/bin/env python3
"""Evaluate HarmBench compliance labels for copyright, standard, and contextual categories.

For copyright behaviors (hash_check tag), uses hash-based evaluation.
For standard/contextual behaviors, uses classifier-based evaluation (requires --cls_path).

Usage:
    # Copyright evaluation (hash-based, no classifier needed)
    python eval_harmbench_labels.py --data_dir data/gpt_j_6b/harmbench_copyright.json --output_dir results/gpt_j_6b/harmbench_copyright_labeled.json
    
    # Standard/contextual evaluation (classifier-based)
    python eval_harmbench_labels.py --data_dir data/gpt_j_6b/harmbench_standard.json --output_dir results/gpt_j_6b/harmbench_standard_labeled.json --cls_path cais/HarmBench-Llama-2-13b-cls
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Import HarmBench eval utils
sys.path.insert(0, str(Path(__file__).parent / "HarmBench"))
from eval_utils import compute_results_hashing, compute_results_classifier


def setup_logger():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filepath = f"logs/eval_harmbench_labels_{timestamp}.log"

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
    parser = argparse.ArgumentParser(description="Evaluate HarmBench compliance labels")
    parser.add_argument("--data_dir", type=str, required=True, help="Input JSON file (list of records)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process")
    parser.add_argument("--cls_path", type=str, default=None, help="Path to classifier model for standard/contextual evaluation (e.g., 'cais/HarmBench-Llama-2-13b-cls'). Required for non-copyright categories.")
    parser.add_argument("--num_tokens", type=int, default=512, help="Maximum number of tokens to evaluate (for classifier)")
    return parser.parse_args()


def main():
    logger = setup_logger()
    args = parse_args()
    logger.info("=== Job started at %s ===", datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    start_time = time.time()

    # Change to HarmBench directory for compute_results_hashing to find data files
    harmbench_dir = Path(__file__).parent / "HarmBench"
    original_cwd = os.getcwd()
    os.chdir(harmbench_dir)
    
    # Initialize classifier if needed (for standard/contextual categories)
    cls = None
    cls_params = None
    if args.cls_path:
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
            
            logger.info("Loading classifier from %s...", args.cls_path)
            tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
            tokenizer.truncation_side = "right"
            
            cls = LLM(model=args.cls_path, tensor_parallel_size=1)
            cls.llm_engine.tokenizer.tokenizer.truncation_side = "left"
            cls_params = SamplingParams(temperature=0.0, max_tokens=1)
            logger.info("Classifier loaded successfully")
        except ImportError:
            logger.warning("vLLM not available. Classifier-based evaluation (standard/contextual) will be skipped.")
            logger.warning("Install vLLM to enable classifier evaluation: pip install vllm")
        except Exception as e:
            logger.warning("Failed to load classifier: %s", e)
            logger.warning("Classifier-based evaluation (standard/contextual) will be skipped.")
    
    try:
        # Load input JSON (use absolute path since we changed directory)
        input_path = Path(original_cwd) / args.data_dir if not os.path.isabs(args.data_dir) else args.data_dir
        logger.info("Loading input JSON from %s...", input_path)
        with open(input_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        logger.info("Loaded %d records", len(records))
        
        # Limit if specified
        if args.limit is not None and args.limit > 0:
            records = records[:args.limit]
            logger.info("Limited to %d records", len(records))
        
        # Process each record
        processed = []
        for record in records:
            try:
                metadata = record.get("metadata", {})
                behavior_id = metadata.get("BehaviorID")
                if not behavior_id:
                    record["error"] = "BehaviorID not found in metadata"
                    processed.append(record)
                    continue
                
                tags_str = metadata.get("Tags", "")
                tags = tags_str.split(", ") if tags_str else []
                
                # Build behavior_dict from metadata
                # Note: Behavior is stored in 'prompt' field (from harmbench.py)
                behavior_dict = {
                    "BehaviorID": behavior_id,
                    "Behavior": record.get("prompt", "") or "",
                    "ContextString": metadata.get("ContextString", ""),
                    "Tags": tags_str,
                }
                
                generation = record.get("response", "")
                
                # Truncate generation if using classifier (to match HarmBench behavior)
                if cls is not None and "hash_check" not in tags:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
                    tokenized_text = tokenizer.encode(generation, max_length=args.num_tokens, truncation=True)
                    generation = tokenizer.decode(tokenized_text, skip_special_tokens=True)
                
                data = [{"generation": generation}]
                
                # Choose evaluation method based on tags
                if "hash_check" in tags:
                    # Copyright category: use hash-based evaluation
                    # Verify that 'book' or 'lyrics' is in tags (required by compute_results_hashing)
                    if "book" not in tags and "lyrics" not in tags:
                        record["error"] = f"Expected 'book' or 'lyrics' in tags alongside 'hash_check', got {tags}"
                        processed.append(record)
                        continue
                    
                    # Call compute_results_hashing
                    results = compute_results_hashing(behavior_dict, data)
                else:
                    # Standard or contextual category: use classifier-based evaluation
                    if cls is None:
                        record["error"] = "Classifier required for standard/contextual evaluation. Provide --cls_path argument."
                        processed.append(record)
                        continue
                    
                    # Call compute_results_classifier
                    results = compute_results_classifier(behavior_dict, data, cls, cls_params)
                
                # Extract label and add to record
                if results and len(results) > 0:
                    label = results[0].get("label", 0)
                    record["hb_label"] = label
                else:
                    record["error"] = "Evaluation returned empty results"
                
                processed.append(record)
                
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                processed.append(record)
        
        # Save output (use absolute path since we changed directory)
        output_path = Path(original_cwd) / args.output_dir if not os.path.isabs(args.output_dir) else args.output_dir
        logger.info("Saving results to %s...", output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        
        # Summary
        with_label = sum(1 for r in processed if "hb_label" in r)
        with_error = sum(1 for r in processed if "error" in r)
        logger.info("Summary: total=%d, with hb_label=%d, with error=%d", len(processed), with_label, with_error)

    finally:
        end_time = time.time()
        elapsed_float = end_time - start_time
        elapsed = int(elapsed_float)
        days = elapsed // 86400
        hours = (elapsed % 86400) // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        logger.info("=== Job finished at %s ===", datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
        logger.info("=== Elapsed time: %d days %02d:%02d:%02d (total %.3f seconds) ===",
                    days, hours, minutes, seconds, elapsed_float)
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()