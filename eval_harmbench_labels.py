#!/usr/bin/env python3
"""Evaluate HarmBench compliance labels using hash_check method.

Usage:
    python eval_harmbench_labels.py --input data/gpt_j_6b/harmbench_copyright.json --output results/gpt_j_6b/harmbench_copyright_labeled.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import HarmBench eval utils
sys.path.insert(0, str(Path(__file__).parent / "HarmBench"))
from eval_utils import compute_results_hashing


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HarmBench compliance labels")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file (list of records)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Change to HarmBench directory for compute_results_hashing to find data files
    harmbench_dir = Path(__file__).parent / "HarmBench"
    original_cwd = os.getcwd()
    os.chdir(harmbench_dir)
    
    try:
        # Load input JSON (use absolute path since we changed directory)
        input_path = Path(original_cwd) / args.input if not os.path.isabs(args.input) else args.input
        print(f"Loading input JSON from {input_path}...")
        with open(input_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records")
        
        # Limit if specified
        if args.limit is not None and args.limit > 0:
            records = records[:args.limit]
            print(f"Limited to {len(records)} records")
        
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
                
                # Only process if hash_check is in tags
                if "hash_check" not in tags:
                    record["error"] = "hash_check not in tags"
                    processed.append(record)
                    continue
                
                # Build behavior_dict from metadata (only fields needed by compute_results_hashing)
                behavior_dict = {
                    "BehaviorID": behavior_id,
                    "Behavior": "",  # Not used by compute_results_hashing, but required
                    "ContextString": metadata.get("ContextString", ""),
                    "Tags": tags_str,
                }
                
                # Call compute_results_hashing
                generation = record.get("response", "")
                data = [{"generation": generation}]
                results = compute_results_hashing(behavior_dict, data)
                
                # Extract label and add to record
                if results and len(results) > 0:
                    label = results[0].get("label", 0)
                    record["hb_label"] = label
                else:
                    record["error"] = "compute_results_hashing returned empty results"
                
                processed.append(record)
                
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                processed.append(record)
        
        # Save output (use absolute path since we changed directory)
        output_path = Path(original_cwd) / args.output if not os.path.isabs(args.output) else args.output
        print(f"Saving results to {output_path}...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        
        # Print summary
        with_label = sum(1 for r in processed if "hb_label" in r)
        with_error = sum(1 for r in processed if "error" in r)
        print(f"\nSummary:")
        print(f"  Total records: {len(processed)}")
        print(f"  Records with hb_label: {with_label}")
        print(f"  Records with error: {with_error}")
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()