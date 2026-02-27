"""
Master Script - Complete Code Domain Extension Pipeline
Runs the entire AgentDebug extension from start to finish
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded API keys from .env file")
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector.swebench_integration import SWEBenchLoader, create_sample_trajectory
from detector.code_phase1_detector import CodePhase1Detector
from detector.code_phase2_detector import CodePhase2Detector
from experiments.run_code_experiments import CodeExperimentRunner
from analysis.cross_domain_analysis import CrossDomainAnalyzer, ResultsVisualizer


class MasterPipeline:
    """
    Complete pipeline for code domain extension experiments
    """

    def __init__(self, llm, output_base_dir: str = "results"):
        """
        Initialize master pipeline

        Args:
            llm: Language model (Groq, OpenAI, Claude, or local)
            output_base_dir: Base directory for all outputs
        """
        self.llm = llm
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing incomplete run to resume, otherwise create new
        existing_runs = sorted(self.output_base_dir.glob("run_*"))
        incomplete_run = None
        for run_dir in reversed(existing_runs):  # Check latest first
            agg_file = run_dir / "experiments" / "aggregate_statistics.json"
            individual_dir = run_dir / "experiments" / "individual"
            if individual_dir.exists() and not agg_file.exists():
                # Has individual results but no final aggregate = incomplete
                incomplete_run = run_dir
                break

        if incomplete_run:
            self.run_dir = incomplete_run
            self.run_timestamp = incomplete_run.name.replace("run_", "")
            print(f"\n🔄 RESUMING INCOMPLETE RUN: {self.run_dir}")
        else:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.output_base_dir / f"run_{self.run_timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"MASTER PIPELINE INITIALIZED")
        print(f"{'='*80}")
        print(f"Run directory: {self.run_dir}")
        print(f"Timestamp: {self.run_timestamp}")

    async def run_demo(self):
        """
        Run quick demo with sample trajectory
        """
        print(f"\n{'='*80}")
        print(f"RUNNING DEMO MODE (Sample Trajectory)")
        print(f"{'='*80}")

        # Create sample trajectory
        print("\n[1/4] Creating sample trajectory...")
        trajectory = create_sample_trajectory()
        print(f"✓ Sample trajectory created: {trajectory['instance_id']}")

        # Phase 1 Analysis
        print("\n[2/4] Running Phase 1 analysis...")
        phase1_detector = CodePhase1Detector(self.llm, use_automatic_detection=True)
        phase1_results = await phase1_detector.analyze_trajectory(trajectory)
        print(f"✓ Phase 1 complete:")
        print(f"  - Errors detected: {phase1_results['summary']['total_errors']}")
        print(f"  - Automatic detection rate: {phase1_results['summary']['automatic_detection_rate']:.1f}%")

        # Phase 2 Analysis
        print("\n[3/4] Running Phase 2 critical error identification...")
        phase2_detector = CodePhase2Detector(self.llm)
        phase2_results = await phase2_detector.analyze_with_phase2(phase1_results, trajectory)

        if phase2_results['critical_error']:
            critical = phase2_results['critical_error']
            print(f"✓ Phase 2 complete:")
            print(f"  - Critical error: {critical['error_type']}")
            print(f"  - Step: {critical['step_number']}")
            print(f"  - Module: {critical['module']}")

        # Save results
        print("\n[4/4] Saving demo results...")
        demo_results = {
            'phase1': phase1_results,
            'phase2': phase2_results,
            'timestamp': datetime.now().isoformat()
        }

        demo_file = self.run_dir / "demo_results.json"
        with open(demo_file, 'w', encoding='utf-8') as f:
            json.dump(demo_results, f, indent=2)

        print(f"✓ Demo results saved to: {demo_file}")
        print(f"\n{'='*80}")
        print(f"DEMO COMPLETE!")
        print(f"{'='*80}")

        return demo_results

    async def run_experiments(self, trajectory_dir: str, max_trajectories: int = 100, start_index: int = 0):
        """
        Run full experiments on SWE-bench trajectories

        Args:
            trajectory_dir: Directory containing trajectory files
            max_trajectories: Maximum trajectories to process
            start_index: Index to start loading from (for batch processing)
        """
        print(f"\n{'='*80}")
        print(f"RUNNING FULL EXPERIMENTS")
        print(f"{'='*80}")
        print(f"Trajectory directory: {trajectory_dir}")
        print(f"Max trajectories: {max_trajectories}")
        print(f"Start index: {start_index}")

        # Setup experiment runner
        exp_output_dir = self.run_dir / "experiments"
        runner = CodeExperimentRunner(self.llm, output_dir=str(exp_output_dir))

        # Run batch experiments
        print("\n[1/3] Running batch experiments...")
        aggregate_stats = await runner.run_batch_experiments(
            trajectory_dir, max_trajectories=max_trajectories, start_index=start_index
        )

        # Print summary
        print("\n[2/3] Generating summary report...")
        runner.print_summary_report(aggregate_stats)

        # Generate visualizations
        print("\n[3/3] Generating visualizations...")
        visualizer = ResultsVisualizer(
            aggregate_stats,
            output_dir=str(self.run_dir / "figures")
        )
        visualizer.generate_all_figures()

        print(f"\n{'='*80}")
        print(f"EXPERIMENTS COMPLETE!")
        print(f"{'='*80}")
        print(f"Results saved to: {exp_output_dir}")
        print(f"Figures saved to: {self.run_dir / 'figures'}")

        return aggregate_stats

    async def run_cross_domain_analysis(self, embodied_results_file: str = None):
        """
        Run cross-domain comparison analysis

        Args:
            embodied_results_file: Path to embodied domain results (optional)
        """
        print(f"\n{'='*80}")
        print(f"RUNNING CROSS-DOMAIN ANALYSIS")
        print(f"{'='*80}")

        # Find code domain results
        code_results_file = self.run_dir / "experiments" / "aggregate_statistics.json"

        if not code_results_file.exists():
            print(f"⚠️ No code domain results found. Run experiments first.")
            return None

        # Load code results
        with open(code_results_file, 'r') as f:
            code_results = json.load(f)

        if embodied_results_file and Path(embodied_results_file).exists():
            # Full cross-domain analysis
            print("\n[1/3] Loading embodied domain results...")
            analyzer = CrossDomainAnalyzer(
                code_results_file=str(code_results_file),
                embodied_results_file=embodied_results_file
            )

            print("\n[2/3] Computing cross-domain comparison...")
            comparison = analyzer.compare_error_distributions()

            print("\n[3/3] Generating comparison report...")
            report_file = self.run_dir / "cross_domain_report.txt"
            analyzer.generate_comparison_report(output_file=str(report_file))

            # Generate comparison visualizations
            visualizer = ResultsVisualizer(
                code_results,
                output_dir=str(self.run_dir / "figures")
            )
            visualizer.plot_error_distribution(comparison=comparison)

        else:
            # Code-only analysis
            print("\n[1/2] Analyzing code domain results...")
            print("(No embodied results provided - generating code-only visualizations)")

            print("\n[2/2] Generating visualizations...")
            visualizer = ResultsVisualizer(
                code_results,
                output_dir=str(self.run_dir / "figures")
            )
            visualizer.generate_all_figures()

            comparison = None

        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE!")
        print(f"{'='*80}")

        return comparison

    async def run_complete_pipeline(self, mode: str = "demo",
                                   trajectory_dir: str = None,
                                   max_trajectories: int = 100,
                                   embodied_results: str = None):
        """
        Run complete pipeline from start to finish

        Args:
            mode: 'demo' or 'full'
            trajectory_dir: Directory with trajectories (for full mode)
            max_trajectories: Max trajectories to process
            embodied_results: Path to embodied results for comparison
        """
        start_time = datetime.now()

        print(f"\n{'='*80}")
        print(f"STARTING COMPLETE PIPELINE - Mode: {mode.upper()}")
        print(f"{'='*80}")

        results = {
            'mode': mode,
            'start_time': start_time.isoformat(),
            'demo_results': None,
            'experiment_results': None,
            'analysis_results': None
        }

        try:
            if mode == "demo":
                # Run demo mode
                demo_results = await self.run_demo()
                results['demo_results'] = demo_results

            elif mode == "full":
                # Run full experiments
                if not trajectory_dir:
                    print("⚠️ Error: trajectory_dir required for full mode")
                    return None

                exp_results = await self.run_experiments(trajectory_dir, max_trajectories)
                results['experiment_results'] = exp_results

                # Run cross-domain analysis
                analysis_results = await self.run_cross_domain_analysis(embodied_results)
                results['analysis_results'] = analysis_results

            else:
                print(f"⚠️ Error: Unknown mode '{mode}'. Use 'demo' or 'full'")
                return None

        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Save final results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = duration

        final_results_file = self.run_dir / "pipeline_results.json"
        with open(final_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE!")
        print(f"{'='*80}")
        print(f"Mode: {mode}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Results directory: {self.run_dir}")
        print(f"Final results: {final_results_file}")
        print(f"\n✅ All outputs saved successfully!")

        return results


def setup_llm(provider: str = "groq", api_key: str = None):
    """
    Setup LLM based on provider

    Args:
        provider: 'groq', 'openai', 'anthropic', or 'ollama'
        api_key: API key (not needed for ollama)

    Returns:
        LLM instance
    """
    if provider == "groq":
        from groq import Groq
        class GroqLLM:
            def __init__(self, api_key):
                self.client = Groq(api_key=api_key)
                self.model = "llama-3.3-70b-versatile"

            def invoke(self, prompt):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": str(prompt)}],
                    temperature=0.0,
                    max_tokens=1000
                )
                return response.choices[0].message.content

        return GroqLLM(api_key=api_key or os.getenv("GROQ_API_KEY"))

    elif provider == "openai":
        from openai import OpenAI
        class OpenAILLM:
            def __init__(self, api_key):
                self.client = OpenAI(api_key=api_key)
                self.model = "gpt-4-turbo-preview"

            def invoke(self, prompt):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": str(prompt)}],
                    temperature=0.0,
                    max_tokens=1000
                )
                return response.choices[0].message.content

        return OpenAILLM(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    elif provider == "anthropic":
        from anthropic import Anthropic
        class AnthropicLLM:
            def __init__(self, api_key):
                self.client = Anthropic(api_key=api_key)
                self.model = "claude-3-5-sonnet-20241022"

            def invoke(self, prompt):
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": str(prompt)}],
                    temperature=0.0,
                    max_tokens=1000
                )
                return response.content[0].text

        return AnthropicLLM(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    elif provider == "ollama":
        # Real Ollama integration using local models
        import ollama
        class OllamaLLM:
            def __init__(self):
                self.model = "qwen2.5-coder:7b"  # Fast code-specific model
                self.max_prompt_length = 8000  # Truncate for 7B model context limits
                print(f"✓ Ollama LLM initialized with model: {self.model}")

            def invoke(self, prompt, timeout=600):
                prompt_str = str(prompt)
                if len(prompt_str) > self.max_prompt_length:
                    prompt_str = prompt_str[:self.max_prompt_length] + "\n\n[TRUNCATED - analyze what is shown above]\n"
                    prompt = prompt_str
                print(f"🔄 Calling Ollama (prompt length: {len(str(prompt))} chars)...")
                import time
                import concurrent.futures
                start = time.time()

                def _call_ollama():
                    return ollama.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": str(prompt)}],
                        options={
                            "temperature": 0.0,
                            "num_predict": 1000  # max tokens
                        }
                    )

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(_call_ollama)
                    try:
                        response = future.result(timeout=timeout)
                        elapsed = time.time() - start
                        print(f"✓ Ollama responded in {elapsed:.1f}s")
                        return response['message']['content']
                    except concurrent.futures.TimeoutError:
                        elapsed = time.time() - start
                        print(f"⚠ Ollama timed out after {elapsed:.0f}s, skipping...")
                        return "TIMEOUT: Unable to analyze this step"

        return OllamaLLM()

    else:
        raise ValueError(f"Unknown provider: {provider}")


async def main():
    """
    Main entry point
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run complete code domain extension pipeline")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo",
                       help="Run mode: demo (sample) or full (experiments)")
    parser.add_argument("--provider", choices=["groq", "openai", "anthropic", "ollama"],
                       default="ollama", help="LLM provider")
    parser.add_argument("--api-key", help="API key for LLM provider")
    parser.add_argument("--trajectory-dir", help="Directory with SWE-bench trajectories (full mode)")
    parser.add_argument("--max-trajectories", type=int, default=100,
                       help="Maximum trajectories to process")
    parser.add_argument("--embodied-results", help="Path to embodied domain results for comparison")
    parser.add_argument("--output-dir", default="results", help="Base output directory")

    args = parser.parse_args()

    # Setup LLM
    print("\n" + "="*80)
    print("SETTING UP LLM")
    print("="*80)
    print(f"Provider: {args.provider}")

    llm = setup_llm(provider=args.provider, api_key=args.api_key)
    print(f"✓ LLM initialized: {args.provider}")

    # Create pipeline
    pipeline = MasterPipeline(llm, output_base_dir=args.output_dir)

    # Run pipeline
    results = await pipeline.run_complete_pipeline(
        mode=args.mode,
        trajectory_dir=args.trajectory_dir,
        max_trajectories=args.max_trajectories,
        embodied_results=args.embodied_results
    )

    if results:
        print("\n✅ SUCCESS! Pipeline completed successfully.")
        print(f"\nNext steps:")
        print(f"1. Review results in: {pipeline.run_dir}")
        print(f"2. Check figures in: {pipeline.run_dir / 'figures'}")
        print(f"3. Use results for paper writing")
    else:
        print("\n❌ Pipeline failed. Check errors above.")

    return results


if __name__ == "__main__":
    # Run pipeline
    results = asyncio.run(main())
