# AgentDebug: Dual-Channel Error Detection for LLM Agents on Software Engineering Tasks

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.ai)

> **Paper:** *AgentDebug: Dual-Channel Error Detection for LLM Agents on Software Engineering Tasks Using Hybrid Pattern-LLM Architecture* (IEEE Access, 2026)

A hybrid framework that runs **two independent detection channels** — 95 hand-crafted regex patterns (Channel A) and a locally hosted LLM (Channel B) — over every step of an LLM agent trajectory to detect, classify, and compare errors across cognitive modules. The entire pipeline runs locally on consumer hardware at **zero API cost**.

---

## Key Results (300 SWE-bench Trajectories)

| Metric | Value |
|--------|-------|
| Trajectories analyzed | 300 |
| Total steps processed | 8,524 |
| Module-level comparisons | 42,620 |
| Regex errors detected | 1,662 |
| LLM errors detected | 13,616 |
| Detection multiplier (LLM / Regex) | **8.2x** |
| Module-level agreement rate | **69.1%** |
| Error-type agreement (when both flag same step) | 0.0% |
| Total API cost | **$0.00** |
| Total inference time | ~70.8 hours |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AgentDebug Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Trajectory JSON ──► Phase 1: Dual-Channel Detection        │
│                      ┌──────────────────────────────┐       │
│                      │  Channel A: Regex (95 rules)  │       │
│                      │  Channel B: LLM (Qwen 7B)    │       │
│                      └──────────┬───────────────────┘       │
│                                 │                           │
│                      Per step-module agreement:             │
│                      both_error │ both_clean                │
│                      regex_only │ llm_only                  │
│                                 │                           │
│                      ──► Phase 2: Root-Cause ID             │
│                      ┌──────────────────────────────┐       │
│                      │  Counterfactual reasoning     │       │
│                      │  3-layer taxonomy validation  │       │
│                      └──────────────────────────────┘       │
│                                 │                           │
│                      ──► Aggregate Statistics + Figures      │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1: Dual-Channel Detection

Every trajectory step passes through two channels simultaneously:

- **Channel A (Regex):** 95 hand-crafted patterns organized by cognitive module detect syntactic errors — syntax errors, import failures, indentation issues, timeouts, tool execution failures, etc.
- **Channel B (LLM):** Qwen2.5-Coder:7B via Ollama reads the full step context and classifies errors across all 5 cognitive modules in a single inference call.

For each `(step, module)` pair, the system records one of four agreement states:
| State | Meaning |
|-------|---------|
| `both_error` | Both channels detected an error (high confidence) |
| `both_clean` | Neither channel detected an error |
| `regex_only` | Only the regex channel flagged an error |
| `llm_only` | Only the LLM channel flagged an error |

### Phase 2: Root-Cause Identification

A second LLM pass performs **counterfactual reasoning** over Phase 1 results — asking *"which single error, if removed, would most likely have let the agent succeed?"*

Three-layer validation ensures taxonomy compliance:
1. **Module inference** from error type
2. **Error type correction** against valid taxonomy entries
3. **Fallback selection** from Phase 1 detections if the LLM output is invalid

---

## Error Taxonomy

5 cognitive modules, 20 error types (19 predefined + 1 emergent):

| Module | Error Types |
|--------|-------------|
| **Memory** (5) | `dependency_omission`, `file_location_forgetting`, `over_simplification`, `hallucination`, `retrieval_failure` |
| **Reflection** (4) | `progress_misjudge`, `outcome_misinterpretation`, `error_dismissal`, `repetition_blindness` |
| **Planning** (4) | `constraint_ignorance`, `impossible_action`, `api_hallucination`, `scope_violation` |
| **Action** (4) | `syntax_error`, `indentation_error`, `parameter_error`, `wrong_file_edit`* |
| **System** (3) | `tool_execution_error`, `environment_error`, `test_timeout` |

\* `wrong_file_edit` is an **emergent type** surfaced by the LLM channel during analysis — it was not part of the original predefined taxonomy.

---

## Project Structure

```
AgentDebug/
├── detector/                           # Core detection modules
│   ├── code_error_taxonomy.py          # 20 error types across 5 cognitive modules
│   ├── automatic_error_detection.py    # 95-pattern regex engine (Channel A)
│   ├── code_phase1_detector.py         # Dual-channel Phase 1 detection
│   ├── code_phase2_detector.py         # Counterfactual root-cause identification (Phase 2)
│   └── swebench_integration.py         # SWE-bench trajectory loading & parsing
│
├── experiments/
│   └── run_code_experiments.py         # Experiment orchestrator (Phase 1 + Phase 2)
│
├── analysis/
│   ├── __init__.py
│   └── cross_domain_analysis.py        # Cross-domain comparison utilities
│
├── paper/
│   ├── agentdebug_paper.tex            # Full paper (IEEE Access format)
│   ├── extract_paper_values.py         # Extract statistics from results for paper
│   └── figures/                        # Generated figures used in the paper
│       ├── error_distribution_code.png
│       ├── top_error_types.png
│       └── critical_errors_analysis.png
│
├── run_complete_pipeline.py            # Master pipeline with resume logic
├── run_daily_batch.py                  # Batch processing (50 trajectories/batch)
├── validate_daily_batch.py             # Post-batch validation checks
├── validation_sample_100.py            # Manual validation sampling script
├── aggregate_1000_results.py           # Cross-batch result aggregation
├── statistical_tests.py                # Statistical significance tests
├── error_outcome_analysis.py           # Error-outcome correlation analysis
├── analyze_cross_model_comparison.py   # Cross-model comparison analysis
├── analyze_error_outcome_correlation.py # Error-outcome correlation deep dive
├── check_study_readiness.py            # Pre-run environment verification
│
├── data/
│   └── swebench/
│       └── final_trajectories/         # 1000 SWE-bench trajectory JSONs (not in repo)
│
├── .env.example                        # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.9+**
- **[Ollama](https://ollama.ai)** installed and running
- **~5 GB disk space** for the Qwen2.5-Coder:7B model
- **16 GB RAM** recommended (works with partial GPU offload)

### Installation

```bash
# Clone the repository
git clone https://github.com/Amitanand0123/major_project.git
cd major_project

# Install Python dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull qwen2.5-coder:7b
```

### Download Trajectory Data

The SWE-bench trajectory data is sourced from the [Nebius SWE-agent trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) dataset on HuggingFace.

```bash
# Download and prepare trajectory files
python download_nebius_fixed.py
```

This places 1,000 JSON trajectory files into `data/swebench/final_trajectories/`.

### Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys (optional — only needed for Groq/Google providers)
# The default Ollama setup requires NO API keys
```

---

## Usage

### Run a Single Batch (50 Trajectories)

```bash
python run_daily_batch.py \
  --trajectory-dir data/swebench/final_trajectories \
  --provider ollama \
  --base-dir results_1000_study
```

Each batch processes 50 trajectories through the full dual-channel pipeline. Progress is tracked automatically in `results_1000_study/progress.json`.

### Run the Complete Pipeline on Custom Data

```bash
python run_complete_pipeline.py \
  --trajectory-dir data/swebench/final_trajectories \
  --provider ollama \
  --max-trajectories 10
```

### Validate a Completed Batch

```bash
python validate_daily_batch.py --batch 1
```

### Aggregate Results Across All Batches

```bash
python aggregate_1000_results.py
```

### Run Statistical Tests

```bash
python statistical_tests.py
```

---

## Supported LLM Providers

| Provider | Model | Cost | Setup |
|----------|-------|------|-------|
| **Ollama** (default) | Qwen2.5-Coder:7B | Free | `ollama pull qwen2.5-coder:7b` |
| Groq | LLaMA-based models | Free tier (14,400 req/day) | Set `GROQ_API_KEY` in `.env` |
| OpenAI | GPT models | Paid | Set `OPENAI_API_KEY` in `.env` |
| Anthropic | Claude models | Paid | Set `ANTHROPIC_API_KEY` in `.env` |

---

## Computing Environment

All experiments in the paper ran on:
- **OS:** Windows 11 Home (16 GB RAM)
- **LLM:** Qwen2.5-Coder:7B (4.7 GB quantized) via Ollama with partial GPU offload
- **Total inference time:** ~70.8 hours across 300 trajectories (6 batches of 50)
- **API cost:** $0.00 — all inference is local

---

## Citation

If you use AgentDebug in your research, please cite:

```bibtex
@article{anand2026agentdebug,
  title     = {AgentDebug: Dual-Channel Error Detection for LLM Agents on
               Software Engineering Tasks Using Hybrid Pattern-LLM Architecture},
  author    = {Anand, Amit and Aggarwal, Yash and Kumar, Krishn Kant and Kachhava, Rajendra},
  journal   = {IEEE Access},
  year      = {2026},
  publisher = {IEEE}
}
```

---

## Authors

- **Amit Anand** — Indian Institute of Information Technology, Kota
- **Yash Aggarwal** — Indian Institute of Information Technology, Kota
- **Krishn Kant Kumar** — Indian Institute of Information Technology, Kota
- **Rajendra Kachhava** (Advisor) — Indian Institute of Information Technology, Kota

---

## License

This project is licensed under the MIT License.
