"""
Daily Batch Processing Script for 1000-Trajectory Study
Runs 50 trajectories per day with automatic validation and progress tracking
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_complete_pipeline import MasterPipeline, setup_llm


class DailyBatchRunner:
    """
    Manages daily batch runs for large-scale experiments
    """

    def __init__(self, base_dir: str = "results_1000_study"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress_file = self.base_dir / "progress.json"
        self.load_progress()

    def load_progress(self):
        """Load progress from previous runs"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                'total_trajectories_target': 1000,
                'trajectories_per_day': 50,
                'completed_batches': [],
                'total_trajectories_completed': 0,
                'start_date': None,
                'last_run_date': None
            }
            self.save_progress()

    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def get_next_batch_number(self):
        """Get the next batch number to run"""
        return len(self.progress['completed_batches']) + 1

    def get_trajectory_range(self, batch_number: int):
        """Get trajectory indices for this batch"""
        start_idx = (batch_number - 1) * self.progress['trajectories_per_day']
        end_idx = start_idx + self.progress['trajectories_per_day']
        return start_idx, end_idx

    async def run_daily_batch(self, trajectory_dir: str, llm):
        """
        Run today's batch of 50 trajectories
        """
        batch_number = self.get_next_batch_number()
        start_idx, end_idx = self.get_trajectory_range(batch_number)

        print("=" * 80)
        print(f"DAILY BATCH RUN - Day {batch_number}")
        print("=" * 80)
        print(f"Batch number: {batch_number}/20")
        print(f"Trajectories: {start_idx+1}-{end_idx} (out of 1000)")
        print(f"Already completed: {self.progress['total_trajectories_completed']}")
        print(f"After this run: {self.progress['total_trajectories_completed'] + 50}")
        print(f"Progress: {(self.progress['total_trajectories_completed'] / 1000) * 100:.1f}% → {((self.progress['total_trajectories_completed'] + 50) / 1000) * 100:.1f}%")
        print()

        # Create batch-specific output directory
        batch_output_dir = self.base_dir / f"batch_{batch_number:02d}"
        batch_output_dir.mkdir(parents=True, exist_ok=True)

        # Run pipeline
        print(f"Starting pipeline for batch {batch_number}...")
        start_time = datetime.now()

        pipeline = MasterPipeline(llm, output_base_dir=str(batch_output_dir))

        try:
            results = await pipeline.run_experiments(
                trajectory_dir=trajectory_dir,
                max_trajectories=50,  # Only process 50 trajectories
                start_index=start_idx  # Start from correct position for this batch
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Save batch results
            batch_info = {
                'batch_number': batch_number,
                'trajectory_range': [start_idx + 1, end_idx],
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'duration_hours': duration / 3600,
                'results': results,
                'status': 'completed'
            }

            batch_info_file = batch_output_dir / "batch_info.json"
            with open(batch_info_file, 'w') as f:
                json.dump(batch_info, f, indent=2, default=str)

            # Update progress
            if self.progress['start_date'] is None:
                self.progress['start_date'] = start_time.isoformat()

            self.progress['completed_batches'].append({
                'batch_number': batch_number,
                'date': start_time.isoformat(),
                'duration_hours': duration / 3600,
                'trajectory_count': 50
            })
            self.progress['total_trajectories_completed'] += 50
            self.progress['last_run_date'] = end_time.isoformat()
            self.save_progress()

            print()
            print("=" * 80)
            print(f"BATCH {batch_number} COMPLETE!")
            print("=" * 80)
            print(f"Duration: {duration / 3600:.2f} hours")
            print(f"Total completed: {self.progress['total_trajectories_completed']}/1000")
            print(f"Overall progress: {(self.progress['total_trajectories_completed'] / 1000) * 100:.1f}%")
            print(f"Batches remaining: {20 - len(self.progress['completed_batches'])}")
            print()

            return batch_info

        except Exception as e:
            print(f"\n❌ Batch {batch_number} failed: {e}")
            import traceback
            traceback.print_exc()

            # Save failure info
            batch_info = {
                'batch_number': batch_number,
                'trajectory_range': [start_idx + 1, end_idx],
                'start_time': start_time.isoformat(),
                'error': str(e),
                'status': 'failed'
            }

            batch_info_file = batch_output_dir / "batch_info.json"
            with open(batch_info_file, 'w') as f:
                json.dump(batch_info, f, indent=2)

            return None

    def reset_for_rerun(self):
        """Reset progress for a complete re-run (e.g., after pipeline changes)"""
        import shutil

        backup_name = f"{self.base_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir = self.base_dir.parent / backup_name

        print(f"\nBacking up existing results to: {backup_dir}")
        shutil.copytree(self.base_dir, backup_dir)

        # Remove individual result files and aggregates so they get re-processed
        for batch_dir in sorted(self.base_dir.glob("batch_*")):
            for run_dir in batch_dir.glob("run_*"):
                individual_dir = run_dir / "experiments" / "individual"
                if individual_dir.exists():
                    shutil.rmtree(individual_dir)
                agg_file = run_dir / "experiments" / "aggregate_statistics.json"
                if agg_file.exists():
                    agg_file.unlink()
                all_results_file = run_dir / "experiments" / "all_results.json"
                if all_results_file.exists():
                    all_results_file.unlink()
                # Remove figures too (will be regenerated)
                figures_dir = run_dir / "figures"
                if figures_dir.exists():
                    shutil.rmtree(figures_dir)

        # Reset progress counter
        self.progress = {
            'total_trajectories_target': 1000,
            'trajectories_per_day': 50,
            'completed_batches': [],
            'total_trajectories_completed': 0,
            'start_date': None,
            'last_run_date': None
        }
        self.save_progress()
        print("Progress reset to 0. Ready for re-run with new pipeline.")

    def print_overall_progress(self):
        """Print overall study progress"""
        print("\n" + "=" * 80)
        print("1000-TRAJECTORY STUDY PROGRESS")
        print("=" * 80)

        completed = self.progress['total_trajectories_completed']
        target = self.progress['total_trajectories_target']
        pct = (completed / target) * 100

        print(f"Total progress: {completed}/{target} trajectories ({pct:.1f}%)")
        print(f"Batches completed: {len(self.progress['completed_batches'])}/20")
        print(f"Batches remaining: {20 - len(self.progress['completed_batches'])}")

        if self.progress['start_date']:
            print(f"\nStart date: {self.progress['start_date']}")
            print(f"Last run: {self.progress['last_run_date']}")

        if len(self.progress['completed_batches']) > 0:
            total_hours = sum(b['duration_hours'] for b in self.progress['completed_batches'])
            avg_hours = total_hours / len(self.progress['completed_batches'])
            remaining_hours = avg_hours * (20 - len(self.progress['completed_batches']))

            print(f"\nAverage batch duration: {avg_hours:.2f} hours")
            print(f"Estimated remaining time: {remaining_hours:.2f} hours ({remaining_hours/24:.1f} days)")

        print("\nProgress bar:")
        bar_width = 50
        filled = int(bar_width * completed / target)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"[{bar}] {pct:.1f}%")
        print("=" * 80)


async def main():
    """
    Main entry point for daily batch processing
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run daily batch of 50 trajectories")
    parser.add_argument("--trajectory-dir", required=True,
                       help="Directory with all trajectory files")
    parser.add_argument("--provider", choices=["groq", "openai", "anthropic", "ollama"],
                       default="ollama", help="LLM provider")
    parser.add_argument("--base-dir", default="results_1000_study",
                       help="Base directory for all batches")
    parser.add_argument("--rerun", action="store_true",
                       help="Re-run from batch 1 (backs up old results). Use after pipeline changes.")

    args = parser.parse_args()

    # Setup LLM
    print("\n" + "=" * 80)
    print("SETTING UP LLM")
    print("=" * 80)
    print(f"Provider: {args.provider}")

    llm = setup_llm(provider=args.provider)
    print(f"✓ LLM initialized: {args.provider}")

    # Create batch runner
    runner = DailyBatchRunner(base_dir=args.base_dir)

    # Handle --rerun flag
    if args.rerun:
        runner.reset_for_rerun()

    # Show current progress
    runner.print_overall_progress()

    # Check if study is complete
    if runner.progress['total_trajectories_completed'] >= 1000:
        print("\n🎉 1000-trajectory study is COMPLETE!")
        print("Run aggregate_1000_results.py to generate final analysis.")
        return

    # Run today's batch
    print(f"\n▶ Starting today's batch...")
    batch_info = await runner.run_daily_batch(args.trajectory_dir, llm)

    if batch_info:
        print("\n✅ Daily batch completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Run: python validate_daily_batch.py --batch {batch_info['batch_number']}")
        print(f"2. Review results in: {args.base_dir}/batch_{batch_info['batch_number']:02d}")
        print(f"3. Run this script again tomorrow for the next batch")

        if runner.progress['total_trajectories_completed'] >= 1000:
            print(f"\n🎉 STUDY COMPLETE! Run aggregate_1000_results.py for final analysis.")
    else:
        print("\n❌ Daily batch failed. Check errors above and retry.")


if __name__ == "__main__":
    asyncio.run(main())
