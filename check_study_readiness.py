"""
Check if system is ready for 1000-trajectory study
Verifies all prerequisites before starting
"""

import os
import sys
from pathlib import Path
import subprocess

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def check_ollama():
    """Check if Ollama is running and has correct model"""
    print("Checking Ollama...")

    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        output = result.stdout

        if 'qwen2.5-coder' in output:
            print("  ✅ Ollama is running")
            print("  ✅ qwen2.5-coder:7b model available")
            return True
        else:
            print("  ❌ qwen2.5-coder:7b model not found")
            print("  Run: ollama pull qwen2.5-coder:7b")
            return False

    except FileNotFoundError:
        print("  ❌ Ollama not found")
        print("  Install from: https://ollama.ai")
        return False


def check_trajectory_files(trajectory_dir: str):
    """Check if enough trajectory files exist"""
    print(f"\nChecking trajectory files in: {trajectory_dir}")

    if not os.path.exists(trajectory_dir):
        print(f"  ❌ Directory not found: {trajectory_dir}")
        return False

    trajectory_files = list(Path(trajectory_dir).glob("*.json"))
    count = len(trajectory_files)

    print(f"  Found {count} trajectory files")

    if count >= 1000:
        print(f"  ✅ Sufficient files for 1000-trajectory study")
        return True
    elif count >= 50:
        print(f"  ⚠️ Only {count} files available")
        print(f"  Can run {count // 50} batches ({count} trajectories)")
        print(f"  Need {1000 - count} more files for full 1000-trajectory study")
        return True  # Can still run partial study
    else:
        print(f"  ❌ Insufficient files (need at least 50 for one batch)")
        return False


def check_disk_space():
    """Check if enough disk space available"""
    print("\nChecking disk space...")

    # Estimate: 500MB per batch × 20 batches = 10GB
    required_gb = 10

    try:
        import shutil
        stat = shutil.disk_usage(os.getcwd())
        free_gb = stat.free / (1024**3)

        print(f"  Free space: {free_gb:.1f} GB")
        print(f"  Required: ~{required_gb} GB")

        if free_gb >= required_gb:
            print(f"  ✅ Sufficient disk space")
            return True
        else:
            print(f"  ⚠️ Low disk space ({free_gb:.1f} GB available)")
            print(f"  Recommendation: Free up at least {required_gb} GB")
            return True  # Warning, but not blocking

    except Exception as e:
        print(f"  ⚠️ Could not check disk space: {e}")
        return True


def check_python_packages():
    """Check if required packages installed"""
    print("\nChecking Python packages...")

    required = [
        'ollama',
        'asyncio',
        'pathlib',
        'json',
        'datetime'
    ]

    all_ok = True

    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} not installed")
            if package == 'ollama':
                print(f"     Run: pip install ollama")
            all_ok = False

    return all_ok


def check_scripts_exist():
    """Check if all required scripts exist"""
    print("\nChecking required scripts...")

    required_scripts = [
        'run_daily_batch.py',
        'validate_daily_batch.py',
        'aggregate_1000_results.py',
        'run_complete_pipeline.py'
    ]

    all_ok = True

    for script in required_scripts:
        if os.path.exists(script):
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} not found")
            all_ok = False

    return all_ok


def estimate_timeline(trajectory_count: int):
    """Estimate study timeline"""
    print("\n" + "=" * 80)
    print("TIMELINE ESTIMATE")
    print("=" * 80)

    trajectories_per_batch = 50
    hours_per_batch = 3.5  # Average

    total_batches = trajectory_count // trajectories_per_batch
    total_hours = total_batches * hours_per_batch
    total_days = total_batches  # Assuming 1 batch per day

    print(f"Total trajectories: {trajectory_count}")
    print(f"Total batches: {total_batches}")
    print(f"Estimated compute time: {total_hours:.1f} hours")
    print(f"Estimated calendar time: {total_days} days")
    print(f"Expected completion: ~{total_days + 5} days (with buffer)")
    print()


def main():
    """Run all checks"""
    print("=" * 80)
    print("1000-TRAJECTORY STUDY READINESS CHECK")
    print("=" * 80)
    print()

    # Get trajectory directory
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir",
                       default="C:\\Users\\amita\\Downloads\\mp\\data\\swebench\\final_trajectories",
                       help="Directory with trajectory files")
    args = parser.parse_args()

    # Run checks
    checks = {
        'Ollama': check_ollama(),
        'Trajectory files': check_trajectory_files(args.trajectory_dir),
        'Disk space': check_disk_space(),
        'Python packages': check_python_packages(),
        'Scripts': check_scripts_exist()
    }

    # Count trajectory files for estimate
    trajectory_files = list(Path(args.trajectory_dir).glob("*.json"))
    trajectory_count = len(trajectory_files)

    # Summary
    print("\n" + "=" * 80)
    print("READINESS SUMMARY")
    print("=" * 80)

    all_passed = all(checks.values())

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")

    print()

    if all_passed:
        print("🎉 SYSTEM IS READY!")
        print()
        print("You can start the study now:")
        print()
        print(f'  python run_daily_batch.py --trajectory-dir "{args.trajectory_dir}"')
        print()

        # Show timeline estimate
        estimate_timeline(min(trajectory_count, 1000))

        print("Next steps:")
        print("1. Read: START_1000_TRAJECTORY_STUDY.md")
        print("2. Run first batch (above command)")
        print("3. Validate: python validate_daily_batch.py --batch 1")
        print("4. Repeat daily for 20 days")
        print()
    else:
        print("⚠️ SYSTEM NOT READY")
        print()
        print("Please fix the issues marked with ❌ above.")
        print()
        print("Common fixes:")
        print("  - Install Ollama: https://ollama.ai")
        print("  - Pull model: ollama pull qwen2.5-coder:7b")
        print("  - Install Python packages: pip install ollama")
        print("  - Download more trajectory files if needed")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
