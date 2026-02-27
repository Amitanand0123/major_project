# AgentDebug: Dual-Channel Error Detection for LLM Agents

A hybrid framework that runs two independent detection channels (regex + local LLM) over every step of an LLM agent trajectory to detect, classify, and compare errors across cognitive modules.

## Key Results (300 SWE-bench Trajectories)

| Metric | Value |
|--------|-------|
| Trajectories analyzed | 300 |
| Total steps | 8,524 |
| Regex errors detected | 1,662 |
| LLM errors detected | 13,616 |
| Detection multiplier | 8.2x |
| Module-level comparisons | 42,620 |
| Agreement rate | 69.1% |
| Type-level agreement | 0.0% |
| API cost | $0.00 |

## How It Works

**Phase 1 (Dual-Channel Detection):** Every trajectory step passes through two channels simultaneously:
- **Channel A (Regex):** 95+ hand-crafted patterns catch syntactic errors (syntax errors, import failures, timeouts)
- **Channel B (LLM):** Qwen2.5-Coder:7B via Ollama detects semantic errors (wrong file edits, scope violations, progress misjudgment)

For each step-module pair, the system records whether both channels found an error, neither did, or only one did.

**Phase 2 (Root-Cause Identification):** Counterfactual reasoning identifies the single error most likely responsible for trajectory failure. Three-layer validation ensures taxonomy compliance.

## Error Taxonomy

5 cognitive modules, 19 error types:

- **Memory** (5): dependency\_omission, file\_location\_forgetting, over\_simplification, hallucination, retrieval\_failure
- **Reflection** (4): progress\_misjudge, outcome\_misinterpretation, error\_dismissal, repetition\_blindness
- **Planning** (4): constraint\_ignorance, impossible\_action, api\_hallucination, scope\_violation
- **Action** (3): syntax\_error, indentation\_error, parameter\_error
- **System** (3): tool\_execution\_error, environment\_error, test\_timeout

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) with `qwen2.5-coder:7b` pulled
- Trajectory data from [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)

### Install

```bash
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
```

### Download Trajectories

```bash
python download_nebius_fixed.py
```

This places JSON trajectories into `data/swebench/final_trajectories/`.

### Run Batch Processing

```bash
python run_daily_batch.py --trajectory-dir data/swebench/final_trajectories --base-dir results_1000_study
```

Each batch processes 50 trajectories through the dual-channel pipeline.

## Project Structure

```
mp/
├── detector/                    # Core detection modules
│   ├── code_phase1_detector.py  # Fine-grained regex detection (95+ patterns)
│   ├── code_phase2_detector.py  # LLM-based critical error identification
│   ├── code_error_taxonomy.py   # Error taxonomy definitions
│   ├── swebench_integration.py  # Trajectory loading and parsing
│   └── ...
├── paper/                       # Research paper (IEEE Access format)
│   ├── agentdebug_paper.tex
│   └── figures/
├── results_1000_study/          # Batch results (6 batches, 300 trajectories)
│   ├── batch_01/ ... batch_06/
│   └── progress.json
├── run_daily_batch.py           # Batch processing script
├── run_complete_pipeline.py     # Master pipeline with resume logic
└── requirements.txt
```

## Computing Environment

All experiments ran on a consumer-grade Windows 11 laptop (16 GB RAM, partial GPU offload). Total LLM inference: ~70.8 hours across 300 trajectories. Zero API cost.

## Citation

```bibtex
@article{anand2026agentdebug,
  title={AgentDebug: Dual-Channel Error Detection for LLM Agents on Software Engineering Tasks Using Hybrid Pattern-LLM Architecture},
  author={Anand, Amit and Aggarwal, Yash and Kumar, Krishn Kant and Kachhava, Rajendra},
  year={2026}
}
```

## License

MIT
