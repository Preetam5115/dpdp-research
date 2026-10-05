# Adversarial stress-testing of a multi-agent LLM pipeline for DPDP Act compliance checking

Code, data and results for the paper *"Adversarial Stress-Testing of a Multi-Agent LLM Pipeline for DPDP Act
Compliance Checking"* (P. M. Wagh and V. P, VIT Chennai).

The pipeline checks short organisational texts for violations of India's Digital Personal Data Protection Act,
2023 and DPDP Rules, 2025. An attacker model rewrites each test case before it reaches the system under test:

- **evasion** – violating texts are rewritten to hide the violation;
- **smear** – compliant texts are rewritten to sound suspicious without adding a violation.

The system under test (a retrieval-augmented Compliance Agent followed by a Judge Agent, built with LangGraph)
never sees the ground-truth label. Three configurations are compared:

| Config | What is evaluated |
|---|---|
| A | Compliance Agent on the original text |
| B | Compliance Agent on the adversarial rewrite |
| C | Judge Agent on the same rewrite, given the Compliance Agent's verdict |

## Repository layout

```
data/        test_cases.json (200 cases: 102 violating, 98 compliant) and the knowledge-base PDFs
src/         pipeline and experiment code
  prompts.py          agent prompts and temperatures
  llm.py              Ollama / Groq client and output parsing
  knowledge_base.py   ChromaDB index over the DPDP Act, Rules and Rules summary
  pipeline.py         agents and LangGraph wiring
  run_experiment.py   runs configurations A, B and C (resumable)
  report.py           metrics and per-case output files
  analysis.py         McNemar tests and bootstrap confidence intervals
notebooks/   Kaggle notebook used for the reported runs
results/     outputs reported in the paper
archive/     first version of the pipeline (superseded, see archive/README.md)
```

## Setup

Python 3.10+ and [Ollama](https://ollama.com).

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull gemma2:9b
```

## Running

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=2 ollama serve &

python src/run_experiment.py --model llama3.1:8b --attacker-model qwen2.5:7b \
    --temps 0.5 0.7 0.9 --runs 3 --workers 4 --out results/run/llama3.1-8b
```

Useful options: `--sample 10` for a quick check on 10 cases, `--exclude-ids ...` to drop cases,
`--backend groq` to use the Groq API (set `GROQ_API_KEY`). Every finished case is appended to
`checkpoint.jsonl`, so an interrupted run continues where it stopped when the same command is run again.

Each output directory contains `summary.txt`, `summary.json`, `per_case_results.csv` (every verdict, rewrite and
reason) and `checkpoint.jsonl` (raw records). Significance tests:

```bash
cd results && python ../src/analysis.py llama3.1-8b qwen2.5-7b gemma2-9b
```

The reported runs were done on Kaggle (2× T4) with `notebooks/kaggle_run.ipynb`, using Ollama's default 4-bit
builds of each model. Outputs depend on hardware and model builds, so re-runs will not match per case exactly.

## Results

Pooled over attacker temperatures 0.5, 0.7 and 0.9 (Llama: 3 runs per temperature; Qwen and Gemma: 1 run).
The attacker is Qwen 2.5 7B in all experiments.

| Model | Config | Precision | Recall | FPR | F1 (95% CI) |
|---|---|---|---|---|---|
| Llama 3.1 8B | A | 0.810 | 0.990 | 0.241 | 0.891 (0.85–0.93) |
| | B | 0.718 | 0.925 | 0.379 | 0.808 (0.76–0.85) |
| | C | 0.525 | 0.093 | 0.087 | 0.157 (0.11–0.21) |
| Qwen 2.5 7B | A | 0.924 | 0.951 | 0.082 | 0.937 (0.90–0.97) |
| | B | 0.899 | 0.758 | 0.088 | 0.823 (0.78–0.86) |
| | C | 0.853 | 0.588 | 0.105 | 0.696 (0.64–0.75) |
| Gemma 2 9B | A | 0.981 | 0.990 | 0.020 | 0.985 (0.97–1.00) |
| | B | 0.890 | 0.768 | 0.099 | 0.825 (0.78–0.87) |
| | C | 0.687 | 0.849 | 0.405 | 0.760 (0.71–0.81) |

All A→B and B→C differences are significant (exact McNemar, p < 10⁻⁵; see `results/significance.json`).

Other files in `results/`:

- `<model>/summary_no_pilot.*` – the same metrics without the 10 pilot cases used when the Judge prompt was revised;
- `label_audit.csv` – manual check of 50 Llama-run rewrites (43 kept their label, 4 changed, 3 uncertain);
- `replication_2026-10-02/` – an earlier, independent run of the full design (aggregate results only).

## Citation

```
P. M. Wagh and V. P, "Adversarial Stress-Testing of a Multi-Agent LLM Pipeline for DPDP Act Compliance Checking."
```
