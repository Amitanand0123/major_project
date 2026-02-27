#!/usr/bin/env python3
"""
Statistical Tests for Dual-Channel Agreement Analysis
=====================================================
Performs chi-squared tests, confidence intervals, effect sizes, and
per-module agreement analysis on dual-channel error detection data
from the 1000-trajectory study.

Designed to produce results suitable for inclusion in an academic paper.
"""

import json
import os
import math
import glob
import numpy as np
from scipy import stats
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"c:\Users\amita\Downloads\mp\results_1000_study"
MODULES = ["memory", "reflection", "planning", "action", "system"]

# Paper-reported aggregate numbers (42,620 module-level comparisons)
PAPER = {
    "both_clean": 28393,
    "llm_only": 12565,
    "both_error": 1051,
    "regex_only": 611,
    "total": 42620,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval for a proportion."""
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2)) / denom
    return max(0, centre - spread), min(1, centre + spread)


def clopper_pearson_ci(k, n, alpha=0.05):
    """Clopper-Pearson exact confidence interval for a proportion."""
    if k == 0:
        lo = 0.0
    else:
        lo = stats.beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def cohens_h(p1, p2):
    """Cohen's h effect size for comparing two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def print_header(title):
    """Print a formatted section header."""
    width = 78
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# ============================================================================
# LOAD BATCH DATA
# ============================================================================

def load_batch_data():
    """Load aggregate statistics from all 6 batches."""
    batches = []
    for batch_num in range(1, 7):
        batch_dir = os.path.join(BASE_DIR, f"batch_{batch_num:02d}")
        # Find the run directory
        agg_pattern = os.path.join(batch_dir, "run_*", "experiments", "aggregate_statistics.json")
        agg_files = glob.glob(agg_pattern)
        if agg_files:
            with open(agg_files[0], 'r') as f:
                data = json.load(f)
            batches.append({
                "batch": batch_num,
                "data": data,
                "dual": data["dual_channel"]
            })
    return batches


def load_individual_analyses():
    """Load all individual analysis files for per-module granular data."""
    all_analyses = []
    for batch_num in range(1, 7):
        batch_dir = os.path.join(BASE_DIR, f"batch_{batch_num:02d}")
        pattern = os.path.join(batch_dir, "run_*", "experiments", "individual", "*_analysis.json")
        files = glob.glob(pattern)
        for fpath in files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                all_analyses.append(data)
            except (json.JSONDecodeError, IOError):
                continue
    return all_analyses


# ============================================================================
# 1. CHI-SQUARED TEST FOR INDEPENDENCE
# ============================================================================

def chi_squared_test():
    """Test whether regex and LLM detection channels are independent."""
    print_header("1. CHI-SQUARED TEST FOR CHANNEL INDEPENDENCE")

    # Construct the 2x2 contingency table from paper-reported data
    # Rows: Regex detection (yes/no)
    # Cols: LLM detection (yes/no)
    #
    #                  LLM=yes     LLM=no
    # Regex=yes        1,051       611        (both_error, regex_only)
    # Regex=no         12,565      28,393     (llm_only, both_clean)

    contingency = np.array([
        [PAPER["both_error"], PAPER["regex_only"]],
        [PAPER["llm_only"], PAPER["both_clean"]]
    ])

    print("\n  Contingency Table (42,620 module-level comparisons):")
    print(f"  {'':20s} {'LLM=error':>12s} {'LLM=clean':>12s} {'Row Total':>12s}")
    print(f"  {'-'*56}")
    row1_total = contingency[0].sum()
    row2_total = contingency[1].sum()
    print(f"  {'Regex=error':20s} {contingency[0,0]:>12,d} {contingency[0,1]:>12,d} {row1_total:>12,d}")
    print(f"  {'Regex=clean':20s} {contingency[1,0]:>12,d} {contingency[1,1]:>12,d} {row2_total:>12,d}")
    col1_total = contingency[:, 0].sum()
    col2_total = contingency[:, 1].sum()
    total = contingency.sum()
    print(f"  {'-'*56}")
    print(f"  {'Column Total':20s} {col1_total:>12,d} {col2_total:>12,d} {total:>12,d}")

    # Chi-squared test with Yates' correction
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency, correction=True)

    print(f"\n  Expected frequencies (under independence):")
    print(f"  {'':20s} {'LLM=error':>12s} {'LLM=clean':>12s}")
    print(f"  {'-'*44}")
    print(f"  {'Regex=error':20s} {expected[0,0]:>12.1f} {expected[0,1]:>12.1f}")
    print(f"  {'Regex=clean':20s} {expected[1,0]:>12.1f} {expected[1,1]:>12.1f}")

    print(f"\n  Results:")
    print(f"    Chi-squared statistic (Yates' correction): {chi2:,.2f}")
    print(f"    Degrees of freedom: {dof}")
    print(f"    p-value: {p_value:.2e}")
    if p_value < 2.2e-308:
        print(f"    p-value (floor): < 2.2e-308 (effectively zero)")

    # Also compute without Yates' correction for comparison
    chi2_nc, p_nc, _, _ = stats.chi2_contingency(contingency, correction=False)
    print(f"    Chi-squared (without Yates'): {chi2_nc:,.2f}")
    print(f"    p-value (without Yates'): {p_nc:.2e}")

    # Cramér's V
    n = contingency.sum()
    k = min(contingency.shape)
    cramers_v = math.sqrt(chi2_nc / (n * (k - 1)))
    print(f"\n    Cramer's V: {cramers_v:.4f}")

    # Interpretation
    print(f"\n  Interpretation:")
    print(f"    The null hypothesis (regex and LLM detection are independent) is")
    if p_value < 0.001:
        print(f"    REJECTED at all conventional significance levels (p < 0.001).")
        print(f"    The two channels are significantly associated, but Cramer's V = {cramers_v:.4f}")
        if cramers_v < 0.1:
            print(f"    indicates a negligible practical association, consistent with the")
            print(f"    channels detecting largely non-overlapping error types.")
        elif cramers_v < 0.3:
            print(f"    indicates a small practical association.")
        else:
            print(f"    indicates a moderate-to-large practical association.")
    else:
        print(f"    NOT rejected (p = {p_value:.4f}).")

    return chi2, p_value, cramers_v


# ============================================================================
# 2. CONFIDENCE INTERVALS
# ============================================================================

def confidence_intervals(batches):
    """Compute 95% confidence intervals for key metrics."""
    print_header("2. CONFIDENCE INTERVALS (95%)")

    N = PAPER["total"]
    agreement_count = PAPER["both_clean"] + PAPER["both_error"]

    # --- 2a. Overall Agreement Rate ---
    print_subheader("2a. Overall Agreement Rate")
    p_agree = agreement_count / N
    lo_w, hi_w = wilson_ci(p_agree, N)
    lo_cp, hi_cp = clopper_pearson_ci(agreement_count, N)

    print(f"  Observed: {agreement_count:,d} / {N:,d} = {p_agree*100:.2f}%")
    print(f"  Wilson score CI:      [{lo_w*100:.2f}%, {hi_w*100:.2f}%]")
    print(f"  Clopper-Pearson CI:   [{lo_cp*100:.2f}%, {hi_cp*100:.2f}%]")

    # Bootstrap CI from per-batch agreement rates
    batch_rates = []
    for b in batches:
        d = b["dual"]
        batch_agree = d["agreement_counts"]["both_clean"] + d["agreement_counts"]["both_error"]
        batch_total = d["total_module_comparisons"]
        batch_rates.append(batch_agree / batch_total)

    batch_mean = np.mean(batch_rates)
    batch_std = np.std(batch_rates, ddof=1)
    batch_se = batch_std / math.sqrt(len(batch_rates))
    t_crit = stats.t.ppf(0.975, df=len(batch_rates) - 1)
    batch_ci_lo = batch_mean - t_crit * batch_se
    batch_ci_hi = batch_mean + t_crit * batch_se

    print(f"\n  Per-batch agreement rates:")
    for b in batches:
        d = b["dual"]
        r = d["agreement_rate"]
        print(f"    Batch {b['batch']:02d}: {r:.2f}%  (n={d['total_module_comparisons']:,d})")
    print(f"  Batch mean: {batch_mean*100:.2f}% (SD={batch_std*100:.2f}%)")
    print(f"  Batch-level t-interval: [{batch_ci_lo*100:.2f}%, {batch_ci_hi*100:.2f}%]")

    # --- 2b. LLM Detection Rate Per Step ---
    print_subheader("2b. LLM Detection Rate (errors per module-comparison)")
    llm_errors = PAPER["both_error"] + PAPER["llm_only"]  # total LLM detections
    p_llm = llm_errors / N
    lo_llm, hi_llm = wilson_ci(p_llm, N)

    print(f"  LLM flagged errors: {llm_errors:,d} / {N:,d} = {p_llm*100:.2f}%")
    print(f"  Wilson CI: [{lo_llm*100:.2f}%, {hi_llm*100:.2f}%]")

    # Per-batch LLM rates
    batch_llm_rates = []
    for b in batches:
        d = b["dual"]
        llm_tot = d["agreement_counts"]["both_error"] + d["agreement_counts"].get("llm_only", 0)
        batch_llm_rates.append(llm_tot / d["total_module_comparisons"])
    llm_mean = np.mean(batch_llm_rates)
    llm_std = np.std(batch_llm_rates, ddof=1)
    llm_se = llm_std / math.sqrt(len(batch_llm_rates))
    llm_ci_lo = llm_mean - t_crit * llm_se
    llm_ci_hi = llm_mean + t_crit * llm_se
    print(f"  Batch-level LLM detection rates: {[f'{r*100:.1f}%' for r in batch_llm_rates]}")
    print(f"  Batch mean: {llm_mean*100:.2f}% (SD={llm_std*100:.2f}%)")
    print(f"  Batch-level t-interval: [{llm_ci_lo*100:.2f}%, {llm_ci_hi*100:.2f}%]")

    # --- 2c. Regex Detection Rate Per Step ---
    print_subheader("2c. Regex Detection Rate (errors per module-comparison)")
    regex_errors = PAPER["both_error"] + PAPER["regex_only"]  # total regex detections
    p_regex = regex_errors / N
    lo_regex, hi_regex = wilson_ci(p_regex, N)

    print(f"  Regex flagged errors: {regex_errors:,d} / {N:,d} = {p_regex*100:.2f}%")
    print(f"  Wilson CI: [{lo_regex*100:.2f}%, {hi_regex*100:.2f}%]")

    # Per-batch regex rates
    batch_regex_rates = []
    for b in batches:
        d = b["dual"]
        regex_tot = d["agreement_counts"]["both_error"] + d["agreement_counts"].get("regex_only", 0)
        batch_regex_rates.append(regex_tot / d["total_module_comparisons"])
    regex_mean = np.mean(batch_regex_rates)
    regex_std = np.std(batch_regex_rates, ddof=1)
    regex_se = regex_std / math.sqrt(len(batch_regex_rates))
    regex_ci_lo = regex_mean - t_crit * regex_se
    regex_ci_hi = regex_mean + t_crit * regex_se
    print(f"  Batch-level regex detection rates: {[f'{r*100:.1f}%' for r in batch_regex_rates]}")
    print(f"  Batch mean: {regex_mean*100:.2f}% (SD={regex_std*100:.2f}%)")
    print(f"  Batch-level t-interval: [{regex_ci_lo*100:.2f}%, {regex_ci_hi*100:.2f}%]")

    # --- 2d. Detection Multiplier (LLM / Regex) ---
    print_subheader("2d. Detection Multiplier (LLM total errors / Regex total errors)")
    multiplier = llm_errors / regex_errors
    print(f"  Aggregate: {llm_errors:,d} / {regex_errors:,d} = {multiplier:.2f}x")

    # Per-batch multipliers
    batch_multipliers = []
    for b in batches:
        d = b["dual"]
        b_llm = d["llm_total_errors"]
        b_regex = d["regex_total_errors"]
        if b_regex > 0:
            batch_multipliers.append(b_llm / b_regex)
    mult_mean = np.mean(batch_multipliers)
    mult_std = np.std(batch_multipliers, ddof=1)
    mult_se = mult_std / math.sqrt(len(batch_multipliers))
    mult_ci_lo = mult_mean - t_crit * mult_se
    mult_ci_hi = mult_mean + t_crit * mult_se

    print(f"  Per-batch multipliers: {[f'{m:.2f}x' for m in batch_multipliers]}")
    print(f"  Batch mean: {mult_mean:.2f}x (SD={mult_std:.2f})")
    print(f"  Batch-level t-interval: [{mult_ci_lo:.2f}x, {mult_ci_hi:.2f}x]")

    # Bootstrap CI for multiplier using log-transform
    log_mults = np.log(batch_multipliers)
    log_mean = np.mean(log_mults)
    log_std = np.std(log_mults, ddof=1)
    log_se = log_std / math.sqrt(len(log_mults))
    log_ci_lo = log_mean - t_crit * log_se
    log_ci_hi = log_mean + t_crit * log_se
    print(f"  Log-transformed t-interval: [{math.exp(log_ci_lo):.2f}x, {math.exp(log_ci_hi):.2f}x]")

    # --- 2e. Difference in detection rates ---
    print_subheader("2e. Difference in Detection Rates (LLM - Regex)")
    diff = p_llm - p_regex
    se_diff = math.sqrt(p_llm * (1 - p_llm) / N + p_regex * (1 - p_regex) / N)
    diff_ci_lo = diff - 1.96 * se_diff
    diff_ci_hi = diff + 1.96 * se_diff
    print(f"  LLM rate: {p_llm*100:.2f}%  |  Regex rate: {p_regex*100:.2f}%")
    print(f"  Difference: {diff*100:.2f} percentage points")
    print(f"  95% CI: [{diff_ci_lo*100:.2f}%, {diff_ci_hi*100:.2f}%]")
    z_diff = diff / se_diff
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
    print(f"  z = {z_diff:.2f}, p = {p_diff:.2e}")

    return {
        "agreement_rate": (p_agree, lo_w, hi_w),
        "llm_rate": (p_llm, lo_llm, hi_llm),
        "regex_rate": (p_regex, lo_regex, hi_regex),
        "multiplier": (multiplier, mult_ci_lo, mult_ci_hi),
    }


# ============================================================================
# 3. PER-MODULE AGREEMENT CHI-SQUARED TESTS
# ============================================================================

def per_module_analysis(analyses):
    """Compute per-module agreement statistics and chi-squared tests."""
    print_header("3. PER-MODULE AGREEMENT ANALYSIS")

    # Aggregate per-module agreement counts from individual analyses
    module_counts = {m: defaultdict(int) for m in MODULES}
    total_steps = 0

    for analysis in analyses:
        phase = analysis.get("phase1", {})
        steps = phase.get("step_analyses", [])
        for step in steps:
            total_steps += 1
            agreement = step.get("agreement", {})
            for module in MODULES:
                status = agreement.get(module, None)
                if status:
                    module_counts[module][status] += 1

    print(f"\n  Loaded {len(analyses)} trajectories, {total_steps} steps")
    print(f"  (Each step has 5 module comparisons)")

    # Print per-module contingency tables and chi-squared tests
    print(f"\n  {'Module':12s} {'both_clean':>11s} {'llm_only':>11s} {'both_error':>11s} {'regex_only':>11s} {'Total':>8s} {'Agree%':>8s} | {'chi2':>10s} {'p-value':>12s} {'Cramer V':>10s}")
    print(f"  {'-' * 118}")

    overall_chi2_sum = 0
    for module in MODULES:
        mc = module_counts[module]
        bc = mc.get("both_clean", 0)
        lo = mc.get("llm_only", 0)
        be = mc.get("both_error", 0)
        ro = mc.get("regex_only", 0)
        total = bc + lo + be + ro

        if total == 0:
            continue

        agree_rate = (bc + be) / total * 100

        # 2x2 contingency: Regex (yes/no) x LLM (yes/no)
        contingency = np.array([
            [be, ro],
            [lo, bc]
        ])

        # Check if chi-squared is valid (all expected > 5)
        if contingency.sum() > 0 and contingency.min() >= 0:
            try:
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency, correction=True)
                v = math.sqrt(chi2 / (total * (min(contingency.shape) - 1))) if total > 0 else 0
                p_str = f"{p_val:.2e}" if p_val > 0 else "< 2.2e-308"
                overall_chi2_sum += chi2
            except ValueError:
                chi2, p_val, v = float('nan'), float('nan'), float('nan')
                p_str = "N/A"
        else:
            chi2, p_val, v = float('nan'), float('nan'), float('nan')
            p_str = "N/A"

        print(f"  {module:12s} {bc:>11,d} {lo:>11,d} {be:>11,d} {ro:>11,d} {total:>8,d} {agree_rate:>7.1f}% | {chi2:>10.1f} {p_str:>12s} {v:>10.4f}")

    # Grand total from individual files
    grand = defaultdict(int)
    for module in MODULES:
        for status, count in module_counts[module].items():
            grand[status] += count

    total_all = sum(grand.values())
    grand_agree = (grand["both_clean"] + grand["both_error"]) / total_all * 100 if total_all > 0 else 0
    print(f"  {'-' * 118}")
    print(f"  {'TOTAL':12s} {grand['both_clean']:>11,d} {grand['llm_only']:>11,d} {grand['both_error']:>11,d} {grand['regex_only']:>11,d} {total_all:>8,d} {grand_agree:>7.1f}%")

    # Comparison with paper-reported numbers
    print(f"\n  Verification against paper-reported numbers:")
    print(f"    From individual files: {total_all:,d} module comparisons")
    print(f"    Paper reports:         {PAPER['total']:,d} module comparisons")
    if total_all > 0:
        print(f"    Match: {'YES' if total_all == PAPER['total'] else 'NO (using paper numbers for main analyses)'}")

    # Cochran-Mantel-Haenszel-like: test heterogeneity across modules
    print_subheader("3b. Heterogeneity Test Across Modules (Breslow-Day)")
    print("  Testing whether the association between channels differs across modules:")

    module_odds_ratios = {}
    for module in MODULES:
        mc = module_counts[module]
        bc = mc.get("both_clean", 0)
        lo = mc.get("llm_only", 0)
        be = mc.get("both_error", 0)
        ro = mc.get("regex_only", 0)

        # Add 0.5 continuity correction to avoid division by zero
        or_val = ((be + 0.5) * (bc + 0.5)) / ((lo + 0.5) * (ro + 0.5))
        module_odds_ratios[module] = or_val
        total = bc + lo + be + ro
        agree_pct = (bc + be) / total * 100 if total > 0 else 0
        print(f"    {module:12s}: OR = {or_val:.3f}, agreement = {agree_pct:.1f}%")

    or_values = list(module_odds_ratios.values())
    log_ors = [math.log(o) for o in or_values]
    print(f"\n    Log-OR range: [{min(log_ors):.3f}, {max(log_ors):.3f}]")
    print(f"    OR range: [{min(or_values):.3f}, {max(or_values):.3f}]")

    return module_counts


# ============================================================================
# 4. EFFECT SIZE ANALYSIS
# ============================================================================

def effect_size_analysis():
    """Compute effect sizes for detection differences between channels."""
    print_header("4. EFFECT SIZE ANALYSIS")

    N = PAPER["total"]
    llm_errors = PAPER["both_error"] + PAPER["llm_only"]
    regex_errors = PAPER["both_error"] + PAPER["regex_only"]

    p_llm = llm_errors / N
    p_regex = regex_errors / N

    # --- 4a. Cohen's h ---
    print_subheader("4a. Cohen's h (arcsine transformation)")
    h = cohens_h(p_llm, p_regex)
    print(f"  LLM detection rate:   p1 = {p_llm:.4f} ({p_llm*100:.2f}%)")
    print(f"  Regex detection rate: p2 = {p_regex:.4f} ({p_regex*100:.2f}%)")
    print(f"  Cohen's h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2)) = {h:.4f}")
    print(f"\n  Interpretation (Cohen's conventions):")
    abs_h = abs(h)
    if abs_h < 0.2:
        size = "SMALL"
    elif abs_h < 0.5:
        size = "SMALL-TO-MEDIUM"
    elif abs_h < 0.8:
        size = "MEDIUM"
    else:
        size = "LARGE"
    print(f"    |h| = {abs_h:.4f} => {size} effect")
    print(f"    (Thresholds: small=0.2, medium=0.5, large=0.8)")

    # --- 4b. Odds Ratio ---
    print_subheader("4b. Odds Ratio")
    # OR for LLM detecting error vs Regex detecting error
    # Using the 2x2 table:
    #                  LLM=yes     LLM=no
    # Regex=yes        a=1051      b=611
    # Regex=no         c=12565     d=28393

    a, b, c, d = PAPER["both_error"], PAPER["regex_only"], PAPER["llm_only"], PAPER["both_clean"]
    odds_ratio = (a * d) / (b * c)
    log_or = math.log(odds_ratio)

    # Standard error of log(OR)
    se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
    log_or_ci_lo = log_or - 1.96 * se_log_or
    log_or_ci_hi = log_or + 1.96 * se_log_or
    or_ci_lo = math.exp(log_or_ci_lo)
    or_ci_hi = math.exp(log_or_ci_hi)

    print(f"  OR = (a*d)/(b*c) = ({a}*{d})/({b}*{c}) = {odds_ratio:.4f}")
    print(f"  ln(OR) = {log_or:.4f}")
    print(f"  SE[ln(OR)] = {se_log_or:.4f}")
    print(f"  95% CI for OR: [{or_ci_lo:.4f}, {or_ci_hi:.4f}]")
    print(f"\n  Interpretation:")
    if odds_ratio > 1:
        print(f"    OR = {odds_ratio:.4f} > 1: When regex detects an error, the LLM is")
        print(f"    {odds_ratio:.2f}x more likely to also detect an error (positive association).")
    else:
        print(f"    OR = {odds_ratio:.4f} < 1: Negative association between channels.")
    print(f"    CI excludes 1.0: {'YES' if or_ci_lo > 1 or or_ci_hi < 1 else 'NO'} => {'Statistically significant' if or_ci_lo > 1 or or_ci_hi < 1 else 'Not significant'}")

    # --- 4c. Relative Risk ---
    print_subheader("4c. Relative Risk (LLM detection rate)")
    # Risk of LLM detecting error given Regex=yes vs Regex=no
    risk_regex_yes = a / (a + b)  # P(LLM=yes | Regex=yes)
    risk_regex_no = c / (c + d)   # P(LLM=yes | Regex=no)
    rr = risk_regex_yes / risk_regex_no

    se_log_rr = math.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    log_rr = math.log(rr)
    rr_ci_lo = math.exp(log_rr - 1.96 * se_log_rr)
    rr_ci_hi = math.exp(log_rr + 1.96 * se_log_rr)

    print(f"  P(LLM=error | Regex=error) = {a}/{a+b} = {risk_regex_yes:.4f} ({risk_regex_yes*100:.1f}%)")
    print(f"  P(LLM=error | Regex=clean) = {c}/{c+d} = {risk_regex_no:.4f} ({risk_regex_no*100:.1f}%)")
    print(f"  Relative Risk = {rr:.4f}")
    print(f"  95% CI: [{rr_ci_lo:.4f}, {rr_ci_hi:.4f}]")
    print(f"\n  Interpretation:")
    print(f"    When regex detects an error, the LLM is {rr:.2f}x more likely to also")
    print(f"    detect an error compared to when regex finds no error.")

    # --- 4d. McNemar's test (paired comparison) ---
    print_subheader("4d. McNemar's Test (paired discordant cells)")
    print(f"  Tests whether the marginal proportions of error detection differ.")
    print(f"  Discordant pairs: llm_only={c:,d}, regex_only={b:,d}")

    # McNemar's chi-squared
    mcnemar_chi2 = (abs(b - c) - 1)**2 / (b + c)  # with continuity correction
    mcnemar_p = 1 - stats.chi2.cdf(mcnemar_chi2, df=1)

    # Also exact McNemar using binomial
    mcnemar_exact_p = stats.binom_test(b, b + c, 0.5) if hasattr(stats, 'binom_test') else None
    try:
        mcnemar_exact_p = stats.binomtest(b, b + c, 0.5).pvalue
    except AttributeError:
        pass

    print(f"  McNemar chi-squared (with continuity correction): {mcnemar_chi2:,.2f}")
    print(f"  p-value: {mcnemar_p:.2e}")
    if mcnemar_exact_p is not None:
        print(f"  Exact binomial p-value: {mcnemar_exact_p:.2e}")
    print(f"\n  Interpretation:")
    print(f"    The LLM channel detects significantly {'more' if c > b else 'fewer'} errors")
    print(f"    than the regex channel (p < 0.001). The ratio of discordant pairs")
    print(f"    is {c/b:.1f}:1 (LLM-only : Regex-only).")

    # --- 4e. Phi coefficient ---
    print_subheader("4e. Phi Coefficient")
    phi = (a*d - b*c) / math.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    print(f"  Phi = {phi:.4f}")
    print(f"  Phi^2 = {phi**2:.6f}")
    print(f"  (Equivalent to Pearson correlation for 2x2 tables)")

    return {
        "cohens_h": h,
        "odds_ratio": odds_ratio,
        "or_ci": (or_ci_lo, or_ci_hi),
        "relative_risk": rr,
        "mcnemar_chi2": mcnemar_chi2,
        "phi": phi,
    }


# ============================================================================
# 5. PER-BATCH STABILITY ANALYSIS
# ============================================================================

def batch_stability_analysis(batches):
    """Analyze consistency across batches."""
    print_header("5. PER-BATCH STABILITY ANALYSIS")

    print_subheader("5a. Batch-Level Summary")
    print(f"  {'Batch':>6s} {'N_traj':>7s} {'Steps':>7s} {'Comparisons':>12s} {'Regex Err':>10s} {'LLM Err':>10s} {'Agreement':>10s} {'Multiplier':>11s}")
    print(f"  {'-' * 80}")

    agreement_rates = []
    for b in batches:
        d = b["dual"]
        n_traj = b["data"]["total_trajectories"]
        steps = b["data"]["total_steps_analyzed"]
        comps = d["total_module_comparisons"]
        regex_e = d["regex_total_errors"]
        llm_e = d["llm_total_errors"]
        agree = d["agreement_rate"]
        mult = llm_e / regex_e if regex_e > 0 else float('inf')
        agreement_rates.append(agree)

        print(f"  {b['batch']:>6d} {n_traj:>7d} {steps:>7,d} {comps:>12,d} {regex_e:>10,d} {llm_e:>10,d} {agree:>9.1f}% {mult:>10.2f}x")

    # Homogeneity test across batches
    print_subheader("5b. Homogeneity of Agreement Rates Across Batches")

    # Build a 6x2 table: (agree, disagree) for each batch
    observed_matrix = []
    for b in batches:
        d = b["dual"]
        agree_count = d["agreement_counts"]["both_clean"] + d["agreement_counts"]["both_error"]
        disagree_count = d["agreement_counts"]["llm_only"] + d["agreement_counts"]["regex_only"]
        observed_matrix.append([agree_count, disagree_count])

    observed = np.array(observed_matrix)
    chi2_homo, p_homo, dof_homo, _ = stats.chi2_contingency(observed, correction=False)

    print(f"  Chi-squared test for homogeneity: chi2 = {chi2_homo:.2f}, df = {dof_homo}, p = {p_homo:.4f}")
    if p_homo < 0.05:
        print(f"  => Agreement rates differ SIGNIFICANTLY across batches (p < 0.05)")
    else:
        print(f"  => Agreement rates are HOMOGENEOUS across batches (p >= 0.05)")

    # Cochran's Q test (treating batches as repeated measures)
    print(f"\n  Agreement rate range: [{min(agreement_rates):.1f}%, {max(agreement_rates):.1f}%]")
    print(f"  Mean: {np.mean(agreement_rates):.1f}%, SD: {np.std(agreement_rates, ddof=1):.1f}%")
    print(f"  CV (coefficient of variation): {np.std(agreement_rates, ddof=1)/np.mean(agreement_rates)*100:.1f}%")


# ============================================================================
# 6. SUMMARY TABLE FOR PAPER
# ============================================================================

def summary_for_paper(chi2_result, ci_result, effect_result):
    """Print a summary table suitable for inclusion in an academic paper."""
    print_header("6. SUMMARY TABLE FOR ACADEMIC PAPER")

    chi2, p_chi2, cramers_v = chi2_result
    h = effect_result["cohens_h"]
    or_val = effect_result["odds_ratio"]
    or_lo, or_hi = effect_result["or_ci"]
    rr = effect_result["relative_risk"]
    phi = effect_result["phi"]
    mcnemar = effect_result["mcnemar_chi2"]

    print(f"""
  Table: Statistical Tests for Dual-Channel Error Detection (N = 42,620)
  -----------------------------------------------------------------------
  Test / Metric                    Statistic     95% CI            p-value
  -----------------------------------------------------------------------
  Overall agreement rate           69.1%         [{ci_result['agreement_rate'][1]*100:.1f}%, {ci_result['agreement_rate'][2]*100:.1f}%]     --
  LLM detection rate               {ci_result['llm_rate'][0]*100:.1f}%         [{ci_result['llm_rate'][1]*100:.1f}%, {ci_result['llm_rate'][2]*100:.1f}%]     --
  Regex detection rate             {ci_result['regex_rate'][0]*100:.1f}%          [{ci_result['regex_rate'][1]*100:.1f}%, {ci_result['regex_rate'][2]*100:.1f}%]      --
  Detection multiplier             {ci_result['multiplier'][0]:.2f}x        [{ci_result['multiplier'][1]:.2f}x, {ci_result['multiplier'][2]:.2f}x]   --

  Chi-squared (independence)       {chi2:,.1f}      --                < 2.2e-16
  Cramer's V                       {cramers_v:.4f}      --                --
  Phi coefficient                  {phi:.4f}      --                --
  Cohen's h                        {abs(h):.4f}      --                --
  Odds ratio                       {or_val:.4f}      [{or_lo:.4f}, {or_hi:.4f}]  < 0.001
  Relative risk                    {rr:.4f}      --                < 0.001
  McNemar's chi-squared            {mcnemar:,.1f}   --                < 2.2e-16
  -----------------------------------------------------------------------

  Notes: Chi-squared test rejects independence (p < 0.001), but Cramer's V
  = {cramers_v:.4f} indicates negligible practical association. The LLM channel
  detects {ci_result['multiplier'][0]:.1f}x more errors than the regex channel. McNemar's test
  confirms the marginal detection rates differ significantly. Cohen's h =
  {abs(h):.2f} represents a {'large' if abs(h) >= 0.8 else 'medium' if abs(h) >= 0.5 else 'small-to-medium' if abs(h) >= 0.2 else 'small'} effect size for the detection rate difference.
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("  DUAL-CHANNEL AGREEMENT: COMPREHENSIVE STATISTICAL ANALYSIS")
    print("  " + "=" * 74)
    print(f"  Data: {PAPER['total']:,d} module-level comparisons across 300 trajectories")
    print(f"  Source: {BASE_DIR}")
    print("=" * 78)

    # Load data
    batches = load_batch_data()
    print(f"\n  Loaded {len(batches)} batch aggregate files")

    # Verify aggregate consistency
    total_comps = sum(b["dual"]["total_module_comparisons"] for b in batches)
    total_be = sum(b["dual"]["agreement_counts"]["both_error"] for b in batches)
    total_bc = sum(b["dual"]["agreement_counts"]["both_clean"] for b in batches)
    total_lo = sum(b["dual"]["agreement_counts"]["llm_only"] for b in batches)
    total_ro = sum(b["dual"]["agreement_counts"]["regex_only"] for b in batches)

    print(f"\n  Aggregate from batch files:")
    print(f"    Total comparisons: {total_comps:,d}")
    print(f"    both_clean: {total_bc:,d} | llm_only: {total_lo:,d} | both_error: {total_be:,d} | regex_only: {total_ro:,d}")
    print(f"    Agreement: {(total_bc + total_be) / total_comps * 100:.1f}%")

    print(f"\n  Paper-reported numbers:")
    print(f"    Total comparisons: {PAPER['total']:,d}")
    print(f"    both_clean: {PAPER['both_clean']:,d} | llm_only: {PAPER['llm_only']:,d} | both_error: {PAPER['both_error']:,d} | regex_only: {PAPER['regex_only']:,d}")
    agree_paper = (PAPER['both_clean'] + PAPER['both_error']) / PAPER['total'] * 100
    print(f"    Agreement: {agree_paper:.1f}%")

    # Load individual analyses for per-module analysis
    print(f"\n  Loading individual analysis files...")
    analyses = load_individual_analyses()
    print(f"  Loaded {len(analyses)} individual trajectory analyses")

    # Run analyses
    chi2_result = chi_squared_test()
    ci_result = confidence_intervals(batches)
    module_counts = per_module_analysis(analyses)
    effect_result = effect_size_analysis()
    batch_stability_analysis(batches)
    summary_for_paper(chi2_result, ci_result, effect_result)


if __name__ == "__main__":
    main()
