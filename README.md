# AgentDebug: Dual-Channel Error Detection and Root-Cause Diagnosis for LLM Agents

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.ai)
[![SWE-bench](https://img.shields.io/badge/Dataset-SWE--bench-purple.svg)](https://www.swebench.com/)

> **Paper:** *AgentDebug: Dual-Channel Error Detection and Root-Cause Diagnosis for LLM Agents on Software Engineering Tasks* (IEEE Access, 2026)

AgentDebug is a framework for **detecting**, **diagnosing**, and **categorizing** errors in LLM coding-agent trajectories. It runs two independent detection channels — 95 hand-crafted regex patterns and a locally hosted LLM — over every step of an agent trajectory, then applies counterfactual reasoning to pinpoint root causes. The entire pipeline runs on consumer hardware at **$0.00 API cost**.

---

## Key Results (300 SWE-bench Trajectories)

| Metric | Value |
|--------|-------|
| Trajectories analyzed | 300 |
| Total steps processed | 8,524 |
| Module-level comparisons | 42,620 |
| Regex errors detected | 1,662 |
| LLM errors detected | 13,616 |
| Detection multiplier (LLM / Regex) | **8.2×** |
| Module-level agreement rate | **69.1%** |
| Root causes identified | 293 / 300 |
| Total API cost | **$0.00** |
| Total inference time | ~70.8 hours |

### Key Findings

- **Module-level asymmetry:** Regex over-represents Action + System errors (62.4%) while under-detecting cognitive failures; the LLM channel captures Memory, Reflection, and Planning errors at 58.7%
- **Frequency ≠ Criticality:** Rare System errors cause 2.0× more root-cause failures than their frequency predicts, while common Action errors are less likely to be the actual root cause
- **Complementary channels:** The two channels agree 69.1% of the time at the module level but characterize errors at different abstraction levels — regex sees syntactic symptoms, the LLM names semantic causes

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph Input
        T["📄 SWE-bench Trajectory JSON<br/>(observation-action pairs)"]
    end

    subgraph Phase1["Phase 1: Dual-Channel Detection"]
        direction LR
        A["🔍 Channel A<br/>95 Regex Patterns<br/>(deterministic)"]
        B["🤖 Channel B<br/>Qwen2.5-Coder:7B<br/>(semantic)"]
    end

    subgraph Agreement["Agreement Computation"]
        AG["Per (step, module) pair:<br/>both_error | both_clean<br/>regex_only | llm_only"]
    end

    subgraph Phase2["Phase 2: Root-Cause Identification"]
        CF["Counterfactual Reasoning<br/>'If this error hadn't occurred,<br/>would the agent have succeeded?'"]
        VAL["3-Layer Taxonomy Validation<br/>1. Prompt engineering (94%)<br/>2. Post-parsing correction<br/>3. Fallback mapping"]
    end

    subgraph Phase3["Phase 3: Iterative Debugging (Framework)"]
        FB["Corrective Feedback<br/>Generation"]
        PATCH["Patch Generation<br/>(unified diff)"]
        VERIFY["Real Test-Suite<br/>Verification"]
    end

    subgraph Output["Output"]
        STATS["📊 Aggregate Statistics<br/>& Figures"]
        ROOT["🎯 Root-Cause Error<br/>per Trajectory"]
    end

    T --> A
    T --> B
    A --> AG
    B --> AG
    AG --> CF
    CF --> VAL
    VAL --> ROOT
    ROOT --> STATS
    ROOT --> FB
    FB --> PATCH
    PATCH --> VERIFY
```

---

## Pipeline Execution Flow

```mermaid
flowchart LR
    subgraph Data["1. Data"]
        DL["Download 1000<br/>SWE-bench trajectories"]
    end

    subgraph Process["2. Processing"]
        BATCH["run_daily_batch.py<br/>(50 per batch)"]
        P1["Phase 1<br/>Dual-Channel"]
        P2["Phase 2<br/>Root-Cause"]
    end

    subgraph Validate["3. Validation"]
        VAL["validate_daily_batch.py<br/>Check integrity"]
    end

    subgraph Aggregate["4. Aggregation"]
        AGG["aggregate_1000_results.py<br/>Combine all batches"]
        STAT["statistical_tests.py<br/>Significance tests"]
    end

    DL --> BATCH --> P1 --> P2 --> VAL --> AGG --> STAT
```

---

## Phase 1: Dual-Channel Detection

Every trajectory step passes through two independent channels simultaneously:

```mermaid
flowchart TB
    STEP["Step sᵢ = (observation, action)"]

    subgraph ChannelA["Channel A: Regex (Deterministic)"]
        R1["95 hand-crafted patterns"]
        R2["Organized by 5 cognitive modules"]
        R3["Confidence = 1.0 on match"]
        R1 --> R2 --> R3
    end

    subgraph ChannelB["Channel B: LLM (Semantic)"]
        L1["Qwen2.5-Coder:7B via Ollama"]
        L2["3-step context window"]
        L3["Classifies all 5 modules per call"]
        L1 --> L2 --> L3
    end

    STEP --> ChannelA
    STEP --> ChannelB

    ChannelA --> AGREE
    ChannelB --> AGREE

    AGREE{"Agreement<br/>Label"}
    AGREE -->|"Both flag error"| BE["both_error"]
    AGREE -->|"Neither flags"| BC["both_clean"]
    AGREE -->|"Only regex"| RO["regex_only"]
    AGREE -->|"Only LLM"| LO["llm_only"]
```

### Regex Pattern Distribution (95 patterns)

| Module | Patterns | Target Error Types |
|--------|----------|--------------------|
| **Action** | 28 | `syntax_error`, `indentation_error`, `parameter_error` |
| **Planning** | 22 | `api_hallucination`, `scope_violation`, `constraint_ignorance` |
| **Memory** | 18 | `dependency_omission`, `hallucination`, `retrieval_failure` |
| **Reflection** | 15 | `error_dismissal`, `repetition_blindness`, `outcome_misinterpretation` |
| **System** | 12 | `tool_execution_error`, `environment_error`, `test_timeout` |

---

## Phase 2: Root-Cause Identification

```mermaid
flowchart TB
    ERRORS["Combined errors from<br/>Channel A ∪ Channel B"]

    CF["Counterfactual Reasoning<br/>(single LLM call)"]

    subgraph Validation["3-Layer Taxonomy Validation"]
        L1["Layer 1: Prompt Engineering<br/>Valid modules + types in prompt<br/>(catches ~94% of issues)"]
        L2["Layer 2: Post-Parsing<br/>Programmatic check + auto-correct<br/>(catches ~80% of remaining)"]
        L3["Layer 3: Fallback<br/>String similarity mapping<br/>(catches the rest)"]
    end

    ROOT["Root-Cause Error e*<br/>(module, type, step, evidence)"]

    ERRORS --> CF --> L1 --> L2 --> L3 --> ROOT
```

---

## Phase 3: Iterative Debugging with Real Verification (Framework)

Phase 3 closes the loop from detection to repair. Given the root-cause error from Phase 2:

```mermaid
flowchart TB
    ROOT["Root-Cause Error e*<br/>from Phase 2"]

    subgraph Loop["Iterative Loop (K=3 max)"]
        FB["Generate corrective<br/>feedback"]
        PATCH["Generate unified-diff<br/>patch"]
        EVAL["Evaluate quality<br/>(specificity, actionability)"]
        PRED["Predict success<br/>(confidence score)"]

        FB --> PATCH --> EVAL --> PRED
        PRED -->|"Low confidence"| FB
    end

    subgraph Verify["Real Verification"]
        CLONE["Clone repo at<br/>correct commit"]
        VENV["Create isolated<br/>venv"]
        APPLY["git apply<br/>patch"]
        TEST["Run FAIL_TO_PASS<br/>tests (600s timeout)"]

        CLONE --> VENV --> APPLY --> TEST
    end

    subgraph Baseline["Gold-Patch Baseline"]
        GOLD["Verify known-correct<br/>SWE-bench patch first"]
        TIER["Four-Tier Comparison:<br/>both_pass | gold_only<br/>corrective_only | neither"]
    end

    ROOT --> Loop
    PRED -->|"High confidence + quality"| Verify
    Verify --> Baseline
```

---

## Error Taxonomy

5 cognitive modules, 20 error types (19 predefined + 1 emergent):

```mermaid
mindmap
  root((AgentError<br/>Taxonomy))
    Memory
      dependency_omission
      file_location_forgetting
      over_simplification
      hallucination
      retrieval_failure
    Reflection
      progress_misjudge
      outcome_misinterpretation
      error_dismissal
      repetition_blindness
    Planning
      constraint_ignorance
      impossible_action
      api_hallucination
      scope_violation
    Action
      syntax_error
      indentation_error
      parameter_error
      wrong_file_edit†
    System
      tool_execution_error
      environment_error
      test_timeout
```

> † `wrong_file_edit` is an **emergent type** — not part of the original taxonomy, discovered by the LLM channel during analysis.

---

## Project Structure

```
AgentDebug/
├── detector/                              # Core detection engine
│   ├── code_error_taxonomy.py             # 20 error types across 5 cognitive modules
│   ├── automatic_error_detection.py       # Channel A: 95-pattern regex engine
│   ├── code_phase1_detector.py            # Phase 1: Dual-channel orchestrator
│   ├── code_phase2_detector.py            # Phase 2: Counterfactual root-cause ID
│   ├── code_phase3_debugger.py            # Phase 3: Iterative debugging framework
│   ├── patch_verifier.py                  # Real test-suite verification
│   └── swebench_integration.py            # SWE-bench trajectory loader & parser
│
├── experiments/
│   └── run_code_experiments.py            # Experiment orchestrator (Phase 1 + 2)
│
├── analysis/
│   ├── __init__.py
│   └── cross_domain_analysis.py           # Cross-domain comparison utilities
│
├── paper/
│   ├── agentdebug_paper.tex               # Full paper (IEEE Access format)
│   ├── extract_paper_values.py            # Extract statistics for paper tables
│   └── figures/                           # Generated visualizations
│       ├── error_distribution_code.png
│       ├── top_error_types.png
│       └── critical_errors_analysis.png
│
├── scripts/
│   └── download_swebench_metadata.py      # Download SWE-bench instance metadata
│
├── data/
│   └── swebench/
│       └── final_trajectories/            # 1000 trajectory JSONs (not in repo)
│
├── run_complete_pipeline.py               # Master pipeline with resume logic
├── run_daily_batch.py                     # Batch processing (50 trajectories/batch)
├── validate_daily_batch.py                # Post-batch validation checks
├── aggregate_1000_results.py              # Cross-batch result aggregation
├── statistical_tests.py                   # Statistical significance tests
├── error_outcome_analysis.py              # Error-outcome correlation analysis
├── analyze_cross_model_comparison.py      # Cross-model comparison
├── analyze_error_outcome_correlation.py   # Error-outcome deep dive
├── check_study_readiness.py               # Pre-run environment verification
├── validation_sample_100.py               # Manual validation sampling
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.9+**
- **[Ollama](https://ollama.ai)** installed and running
- **~5 GB disk space** for the Qwen2.5-Coder:7B model
- **16 GB RAM** recommended

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
python scripts/download_swebench_metadata.py
```

This places 1,000 JSON trajectory files into `data/swebench/final_trajectories/`.

### Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env if using API providers (optional)
# Default Ollama setup requires NO API keys
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

Progress is tracked automatically in `results_1000_study/progress.json`.

### Run the Complete Pipeline

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

## How It Works — Step by Step

```mermaid
sequenceDiagram
    participant U as User
    participant B as run_daily_batch.py
    participant P1 as Phase 1 Detector
    participant RA as Channel A (Regex)
    participant RB as Channel B (LLM)
    participant P2 as Phase 2 Detector
    participant O as Output JSON

    U->>B: Start batch (50 trajectories)
    loop For each trajectory
        B->>P1: Send trajectory steps
        loop For each step
            P1->>RA: Run 95 regex patterns
            RA-->>P1: Syntactic errors
            P1->>RB: Send to Qwen2.5-Coder:7B
            RB-->>P1: Semantic errors (all 5 modules)
            P1->>P1: Compute agreement labels
        end
        P1-->>B: Phase 1 results
        B->>P2: Send combined errors + trajectory
        P2->>P2: Counterfactual reasoning
        P2->>P2: 3-layer taxonomy validation
        P2-->>B: Root-cause error
        B->>O: Save trajectory analysis JSON
    end
    B->>O: Save batch statistics + figures
```

---

## Supported LLM Providers

| Provider | Model | Cost | Setup |
|----------|-------|------|-------|
| **Ollama** (default) | Qwen2.5-Coder:7B | Free | `ollama pull qwen2.5-coder:7b` |
| Groq | LLaMA-based models | Free tier | Set `GROQ_API_KEY` in `.env` |
| OpenAI | GPT models | Paid | Set `OPENAI_API_KEY` in `.env` |
| Anthropic | Claude models | Paid | Set `ANTHROPIC_API_KEY` in `.env` |

---

## Computing Environment

All experiments in the paper ran on:

| Component | Specification |
|-----------|---------------|
| **OS** | Windows 11 Home |
| **RAM** | 16 GB |
| **LLM** | Qwen2.5-Coder:7B (4.7 GB quantized) via Ollama |
| **Processing** | 6 batches × 50 trajectories |
| **Total inference time** | ~70.8 hours (~14.2 min/trajectory) |
| **LLM calls** | ~8,800 |
| **Timeout rate** | < 0.3% |
| **API cost** | $0.00 |

---

## Research Questions

| RQ | Question | Key Finding |
|----|----------|-------------|
| **RQ1** | What error types and frequencies do LLM agents exhibit? | LLM detects 8.2× more errors; Action errors dominate regex (53.1%) while LLM captures more cognitive errors |
| **RQ2** | How do the two detection channels agree? | 69.1% module-level agreement; channels characterize errors at different abstraction levels |
| **RQ3** | Which errors actually cause failures? | Frequency ≠ criticality; rare System errors are 2.0× more likely to be root causes |
| **RQ4** | What changes would reduce agent failures? | Syntax pre-validation + scope-aware editing would address >60% of detected errors |

---

## Citation

```bibtex
@article{anand2026agentdebug,
  title   = {AgentDebug: Dual-Channel Error Detection and Root-Cause Diagnosis
             for LLM Agents on Software Engineering Tasks},
  author  = {Anand, Amit and Aggarwal, Yash and Kumar, Krishn Kant and Kachhava, Rajendra},
  journal = {IEEE Access},
  year    = {2026},
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
