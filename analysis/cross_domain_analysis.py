"""
Cross-Domain Analysis and Results Visualization
Compares error distributions across domains and generates figures.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


class CrossDomainAnalyzer:
    """
    Compares error distributions across code and embodied domains.
    """

    def __init__(self, code_results_file: str, embodied_results_file: str = None):
        self.code_results_file = code_results_file
        self.embodied_results_file = embodied_results_file

        with open(code_results_file, 'r') as f:
            self.code_results = json.load(f)

        self.embodied_results = None
        if embodied_results_file and Path(embodied_results_file).exists():
            with open(embodied_results_file, 'r') as f:
                self.embodied_results = json.load(f)

    def compare_error_distributions(self) -> Dict[str, Any]:
        """Compare error distributions between code and embodied domains"""
        comparison = {
            'code_domain': {
                'total_errors': self.code_results.get('total_errors_detected', 0),
                'errors_by_module': self.code_results.get('errors_by_module', {}),
                'errors_by_type': self.code_results.get('errors_by_type', {}),
            }
        }

        if self.embodied_results:
            comparison['embodied_domain'] = {
                'total_errors': self.embodied_results.get('total_errors_detected', 0),
                'errors_by_module': self.embodied_results.get('errors_by_module', {}),
                'errors_by_type': self.embodied_results.get('errors_by_type', {}),
            }

        return comparison

    def generate_comparison_report(self, output_file: str = None) -> str:
        """Generate a text report comparing the two domains"""
        comparison = self.compare_error_distributions()
        lines = []
        lines.append("=" * 80)
        lines.append("CROSS-DOMAIN COMPARISON REPORT")
        lines.append("=" * 80)

        code = comparison['code_domain']
        lines.append(f"\nCode Domain: {code['total_errors']} total errors")
        lines.append("  Errors by module:")
        for mod, count in sorted(code['errors_by_module'].items(), key=lambda x: -x[1]):
            lines.append(f"    {mod}: {count}")

        if 'embodied_domain' in comparison:
            emb = comparison['embodied_domain']
            lines.append(f"\nEmbodied Domain: {emb['total_errors']} total errors")
            lines.append("  Errors by module:")
            for mod, count in sorted(emb['errors_by_module'].items(), key=lambda x: -x[1]):
                lines.append(f"    {mod}: {count}")

        report = "\n".join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)

        return report


class ResultsVisualizer:
    """
    Generates figures from aggregate experiment results.
    """

    def __init__(self, aggregate_stats: Dict[str, Any], output_dir: str = "figures"):
        self.stats = aggregate_stats
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_figures(self):
        """Generate all standard figures"""
        try:
            self._plot_error_distribution_by_module()
            self._plot_error_types_bar()
            self._plot_critical_errors()
            print(f"  Figures saved to: {self.output_dir}")
        except ImportError:
            print("  matplotlib not available, skipping figure generation")
        except Exception as e:
            print(f"  Figure generation error: {e}")

    def plot_error_distribution(self, comparison: Dict = None):
        """Plot error distribution, optionally with cross-domain comparison"""
        try:
            if comparison:
                self._plot_comparison(comparison)
            else:
                self._plot_error_distribution_by_module()
        except ImportError:
            print("  matplotlib not available, skipping figure generation")
        except Exception as e:
            print(f"  Figure generation error: {e}")

    def _plot_error_distribution_by_module(self):
        """Plot error distribution by module"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            errors_by_module = self.stats.get('errors_by_module', {})
            if not errors_by_module:
                return

            modules = list(errors_by_module.keys())
            counts = list(errors_by_module.values())

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(modules, counts, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0'])
            ax.set_xlabel('Module')
            ax.set_ylabel('Error Count')
            ax.set_title('Error Distribution by Module')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'error_distribution_code.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"  Error plotting module distribution: {e}")

    def _plot_error_types_bar(self):
        """Plot top error types"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            errors_by_type = self.stats.get('errors_by_type', {})
            if not errors_by_type:
                return

            # Top 10
            sorted_types = sorted(errors_by_type.items(), key=lambda x: -x[1])[:10]
            types = [t[0] for t in sorted_types]
            counts = [t[1] for t in sorted_types]

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(types[::-1], counts[::-1], color='#2196F3')
            ax.set_xlabel('Count')
            ax.set_title('Top Error Types')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'top_error_types.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"  Error plotting error types: {e}")

    def _plot_critical_errors(self):
        """Plot critical error distribution"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            critical_by_module = self.stats.get('critical_errors_by_module', {})
            if not critical_by_module:
                return

            modules = list(critical_by_module.keys())
            counts = list(critical_by_module.values())

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(counts, labels=modules, autopct='%1.1f%%',
                   colors=['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0'])
            ax.set_title('Critical Errors by Module')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'critical_errors_analysis.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"  Error plotting critical errors: {e}")

    def _plot_comparison(self, comparison: Dict):
        """Plot cross-domain comparison"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            code_modules = comparison.get('code_domain', {}).get('errors_by_module', {})

            if 'embodied_domain' in comparison:
                emb_modules = comparison['embodied_domain']['errors_by_module']
                all_modules = sorted(set(list(code_modules.keys()) + list(emb_modules.keys())))

                fig, ax = plt.subplots(figsize=(12, 6))
                x = range(len(all_modules))
                width = 0.35

                code_vals = [code_modules.get(m, 0) for m in all_modules]
                emb_vals = [emb_modules.get(m, 0) for m in all_modules]

                ax.bar([i - width/2 for i in x], code_vals, width, label='Code Domain')
                ax.bar([i + width/2 for i in x], emb_vals, width, label='Embodied Domain')

                ax.set_xlabel('Module')
                ax.set_ylabel('Error Count')
                ax.set_title('Cross-Domain Error Distribution')
                ax.set_xticks(list(x))
                ax.set_xticklabels(all_modules)
                ax.legend()
                plt.tight_layout()
                plt.savefig(self.output_dir / 'cross_domain_comparison.png', dpi=150)
                plt.close()
            else:
                self._plot_error_distribution_by_module()
        except Exception as e:
            print(f"  Error plotting comparison: {e}")
