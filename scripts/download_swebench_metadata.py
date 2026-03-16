"""
Download SWE-bench instance metadata from HuggingFace.
Matches instance_ids from our trajectory files to get base_commit, test_patch, repo, etc.
"""

import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_trajectory_instance_ids(trajectory_dir: str) -> set:
    """Get all instance_ids from our trajectory files."""
    traj_dir = Path(trajectory_dir)
    ids = set()
    for f in traj_dir.glob("*.json"):
        # Instance ID is the filename without extension
        ids.add(f.stem)
    return ids


def download_metadata(trajectory_dir: str, output_path: str):
    """Download SWE-bench metadata for our trajectory instances."""
    from datasets import load_dataset

    # Get our instance IDs
    our_ids = get_trajectory_instance_ids(trajectory_dir)
    print(f"Found {len(our_ids)} trajectory files")

    metadata = {}
    total_matched = 0

    # Try nebius/SWE-bench-extra first (our trajectories come from here)
    print("\nLoading nebius/SWE-bench-extra from HuggingFace...")
    try:
        ds = load_dataset("nebius/SWE-bench-extra", split="train")
        print(f"  Loaded {len(ds)} instances")

        for row in ds:
            iid = row.get("instance_id", "")
            if iid in our_ids and iid not in metadata:
                metadata[iid] = {
                    "instance_id": iid,
                    "repo": row.get("repo", ""),
                    "base_commit": row.get("base_commit", ""),
                    "test_patch": row.get("test_patch", ""),
                    "patch": row.get("patch", ""),
                    "version": row.get("version", ""),
                    "FAIL_TO_PASS": row.get("FAIL_TO_PASS", []),
                    "PASS_TO_PASS": row.get("PASS_TO_PASS", []),
                    "environment_setup_commit": row.get("environment_setup_commit", ""),
                    "problem_statement": row.get("problem_statement", "")[:500],
                }
                total_matched += 1

        print(f"  Matched {total_matched} instances from SWE-bench-extra")
    except Exception as e:
        print(f"  Error loading SWE-bench-extra: {e}")

    # Also try princeton-nlp/SWE-bench_Lite for any remaining
    remaining = our_ids - set(metadata.keys())
    if remaining:
        print(f"\n{len(remaining)} instances not found in SWE-bench-extra, trying SWE-bench_Lite...")
        try:
            ds_lite = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
            print(f"  Loaded {len(ds_lite)} instances")

            for row in ds_lite:
                iid = row.get("instance_id", "")
                if iid in remaining and iid not in metadata:
                    metadata[iid] = {
                        "instance_id": iid,
                        "repo": row.get("repo", ""),
                        "base_commit": row.get("base_commit", ""),
                        "test_patch": row.get("test_patch", ""),
                        "patch": row.get("patch", ""),
                        "version": row.get("version", ""),
                        "FAIL_TO_PASS": row.get("FAIL_TO_PASS", []),
                        "PASS_TO_PASS": row.get("PASS_TO_PASS", []),
                        "environment_setup_commit": row.get("environment_setup_commit", ""),
                        "problem_statement": row.get("problem_statement", "")[:500],
                    }
                    total_matched += 1

            print(f"  Total matched after SWE-bench_Lite: {total_matched}")
        except Exception as e:
            print(f"  Error loading SWE-bench_Lite: {e}")

    # Also try full SWE-bench
    remaining = our_ids - set(metadata.keys())
    if remaining:
        print(f"\n{len(remaining)} instances still not found, trying full SWE-bench...")
        try:
            ds_full = load_dataset("princeton-nlp/SWE-bench", split="test")
            print(f"  Loaded {len(ds_full)} instances")

            for row in ds_full:
                iid = row.get("instance_id", "")
                if iid in remaining and iid not in metadata:
                    metadata[iid] = {
                        "instance_id": iid,
                        "repo": row.get("repo", ""),
                        "base_commit": row.get("base_commit", ""),
                        "test_patch": row.get("test_patch", ""),
                        "patch": row.get("patch", ""),
                        "version": row.get("version", ""),
                        "FAIL_TO_PASS": row.get("FAIL_TO_PASS", []),
                        "PASS_TO_PASS": row.get("PASS_TO_PASS", []),
                        "environment_setup_commit": row.get("environment_setup_commit", ""),
                        "problem_statement": row.get("problem_statement", "")[:500],
                    }
                    total_matched += 1

            print(f"  Total matched after full SWE-bench: {total_matched}")
        except Exception as e:
            print(f"  Error loading full SWE-bench: {e}")

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata for {len(metadata)} instances to {output}")

    # Report coverage
    remaining = our_ids - set(metadata.keys())
    if remaining:
        print(f"\n{len(remaining)} instances NOT found in any SWE-bench dataset:")
        for iid in sorted(remaining)[:10]:
            print(f"  - {iid}")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")
    else:
        print("\nAll trajectory instances matched!")

    return metadata


if __name__ == "__main__":
    trajectory_dir = "data/swebench/final_trajectories"
    output_path = "data/swebench/swebench_metadata.json"

    download_metadata(trajectory_dir, output_path)
