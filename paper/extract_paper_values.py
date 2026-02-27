"""
Extract values from aggregate_1000_trajectories.json for paper
Run this after completing all 20 batches
"""

import json
import sys
from pathlib import Path

def extract_values():
    """Extract all values needed for paper"""

    # Load aggregate results
    results_file = Path("../results_1000_study/aggregate_1000_trajectories.json")

    if not results_file.exists():
        print(f"Error: {results_file} not found!")
        print("Run: python aggregate_1000_results.py")
        sys.exit(1)

    with open(results_file, 'r') as f:
        agg = json.load(f)

    print("=" * 80)
    print("VALUES FOR AGENTDEBUG PAPER")
    print("=" * 80)
    print()

    # Section 1: Abstract values
    print("ABSTRACT (update lines 24-35):")
    print(f"  Total trajectories: {agg['study_info']['total_trajectories']}")
    print(f"  Total steps: {agg['study_info']['total_steps_analyzed']}")
    print(f"  Total errors: {agg['total_errors_detected']}")
    print(f"  Automatic detection: {agg['automatic_detection_rate']:.1f}%")
    print(f"  Total cost: ${agg['total_cost']:.2f}")
    print()

    # Section 2: Dataset statistics
    print("DATASET STATISTICS (Section 4.1, lines ~270-280):")
    print(f"  Total trajectories: {agg['study_info']['total_trajectories']}")
    print(f"  Total steps: {agg['study_info']['total_steps_analyzed']}")
    print(f"  Avg steps/trajectory: {agg['average_steps_per_trajectory']:.1f}")
    print()

    # Section 3: Table 1 - Overall Performance
    print("TABLE 1: Overall Performance (lines ~340-355):")
    print(f"  Total trajectories: {agg['study_info']['total_trajectories']}")
    print(f"  Total steps analyzed: {agg['study_info']['total_steps_analyzed']}")
    print(f"  Total errors detected: {agg['total_errors_detected']}")
    print(f"  Average errors per trajectory: {agg['average_errors_per_trajectory']:.1f}")
    print(f"  Automatic detection rate: {agg['automatic_detection_rate']:.1f}%")
    print(f"  Total cost: ${agg['total_cost']:.2f}")
    print(f"  Processing time: {agg['study_info']['total_duration_hours']:.0f} hours")
    print()

    # Section 4: Table 2 - Errors by Module
    print("TABLE 2: Error Distribution by Module (lines ~360-375):")
    total_errors = agg['total_errors_detected']
    for module, count in sorted(agg['errors_by_module'].items(),
                                 key=lambda x: -x[1]):
        pct = (count / total_errors) * 100
        print(f"  {module:12s} & {count:,} & {pct:.1f}\\% \\\\")
    print(f"  Total: {total_errors}")
    print()

    # Section 5: Table 3 - Top 10 Error Types
    print("TABLE 3: Top 10 Error Types (lines ~390-410):")
    sorted_types = sorted(agg['errors_by_type'].items(), key=lambda x: -x[1])
    for i, (error_type, count) in enumerate(sorted_types[:10], 1):
        pct = (count / total_errors) * 100
        print(f"  {i}. {error_type:30s} & {count:,} & {pct:.1f}\\% \\\\")
    print()

    # Section 6: Table 4 - Critical Errors by Module
    print("TABLE 4: Critical Errors by Module (lines ~430-445):")
    total_critical = sum(agg['critical_errors_by_module'].values())
    for module, count in sorted(agg['critical_errors_by_module'].items(),
                                key=lambda x: -x[1]):
        pct = (count / total_critical) * 100 if total_critical > 0 else 0
        print(f"  {module:12s} & {count:,} & {pct:.0f}\\% \\\\")
    print(f"  Total: {total_critical}")
    print()

    # Section 7: Table 5 - Top Critical Error Types
    print("TABLE 5: Top 10 Critical Error Types (lines ~450-470):")
    sorted_critical = sorted(agg['critical_errors_by_type'].items(),
                             key=lambda x: -x[1])
    for error_type, count in sorted_critical[:10]:
        print(f"  {error_type:30s} & {count:,} \\\\")
    print()

    # Section 8: Key percentages for text
    print("KEY PERCENTAGES FOR TEXT:")
    action_pct = (agg['errors_by_module'].get('action', 0) / total_errors) * 100
    memory_pct = (agg['errors_by_module'].get('memory', 0) / total_errors) * 100
    reflection_pct = (agg['errors_by_module'].get('reflection', 0) / total_errors) * 100

    print(f"  Action module: {action_pct:.1f}%")
    print(f"  Memory module: {memory_pct:.1f}%")
    print(f"  Reflection module: {reflection_pct:.1f}%")
    print()

    # Top error types
    syntax_count = agg['errors_by_type'].get('syntax_error', 0)
    indent_count = agg['errors_by_type'].get('indentation_error', 0)
    dependency_count = agg['errors_by_type'].get('dependency_omission', 0)
    dismissal_count = agg['errors_by_type'].get('error_dismissal', 0)

    syntax_pct = (syntax_count / total_errors) * 100
    indent_pct = (indent_count / total_errors) * 100
    dependency_pct = (dependency_count / total_errors) * 100
    dismissal_pct = (dismissal_count / total_errors) * 100

    print(f"  Syntax errors: {syntax_pct:.1f}%")
    print(f"  Indentation errors: {indent_pct:.1f}%")
    print(f"  Dependency omission: {dependency_pct:.1f}%")
    print(f"  Error dismissal: {dismissal_pct:.1f}%")
    print()

    # Preventable errors calculation
    preventable_syntax_indent = syntax_pct + indent_pct
    preventable_dependency = dependency_pct
    preventable_dismissal = dismissal_pct
    total_preventable = preventable_syntax_indent + preventable_dependency + preventable_dismissal

    print("PREVENTABLE ERRORS (Section 6):")
    print(f"  Code pre-validation (syntax+indent): {preventable_syntax_indent:.1f}%")
    print(f"  Dependency tracking: {preventable_dependency:.1f}%")
    print(f"  Stricter reflection: {preventable_dismissal:.1f}%")
    print(f"  TOTAL PREVENTABLE: {total_preventable:.1f}%")
    print()

    # Batch consistency
    print("BATCH CONSISTENCY:")
    auto_rates = [b['automatic_rate'] for b in agg['batch_summaries']]
    costs = [b['cost'] for b in agg['batch_summaries']]
    durations = [b['duration_hours'] for b in agg['batch_summaries']]

    print(f"  Automatic detection range: {min(auto_rates):.1f}% - {max(auto_rates):.1f}%")
    print(f"  Cost range: ${min(costs):.2f} - ${max(costs):.2f}")
    print(f"  Duration range: {min(durations):.1f}h - {max(durations):.1f}h")
    print(f"  Average duration: {sum(durations)/len(durations):.1f}h per batch")
    print()

    print("=" * 80)
    print("COPY THESE VALUES TO agentdebug_paper.tex")
    print("=" * 80)

    # Generate LaTeX-ready output
    print("\n\nLATEX-READY SNIPPETS:\n")

    print("% Abstract snippet:")
    print(f"% {agg['total_errors_detected']:,} errors spanning 5 cognitive modules")
    print(f"% {agg['study_info']['total_trajectories']:,} trajectories")
    print(f"% {agg['study_info']['total_steps_analyzed']:,} steps")
    print()

    print("% Table 2 snippet (errors by module):")
    for module, count in sorted(agg['errors_by_module'].items(), key=lambda x: -x[1]):
        pct = (count / total_errors) * 100
        print(f"{module.capitalize()} & {count:,} & {pct:.1f}\\% \\\\")
    print()

    # Save to file
    output_file = Path("paper_values.txt")
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AGENTDEBUG PAPER VALUES\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total trajectories: {agg['study_info']['total_trajectories']}\n")
        f.write(f"Total steps: {agg['study_info']['total_steps_analyzed']}\n")
        f.write(f"Total errors: {agg['total_errors_detected']}\n")
        f.write(f"Automatic detection: {agg['automatic_detection_rate']:.1f}%\n")
        f.write(f"Total cost: ${agg['total_cost']:.2f}\n")
        f.write(f"Processing time: {agg['study_info']['total_duration_hours']:.0f} hours\n\n")

        f.write("Errors by Module:\n")
        for module, count in sorted(agg['errors_by_module'].items(), key=lambda x: -x[1]):
            pct = (count / total_errors) * 100
            f.write(f"  {module}: {count} ({pct:.1f}%)\n")

        f.write("\nTop 10 Error Types:\n")
        for i, (error_type, count) in enumerate(sorted_types[:10], 1):
            pct = (count / total_errors) * 100
            f.write(f"  {i}. {error_type}: {count} ({pct:.1f}%)\n")

    print(f"\n✓ Values saved to: {output_file}")


if __name__ == "__main__":
    extract_values()
