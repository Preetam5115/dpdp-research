# Archive: first version (superseded)

`v1/` is the first implementation of the pipeline, kept for provenance only. Do not use its results.

- In `v1/src/agents.py` the Judge Agent was given the ground-truth label (it chose its prompt from
  `true_label` and its prediction depended on it), so Configuration C in this version did not measure detection.
- Configurations A and B in `v1/src/run_ablation.py` used slightly different prompts from the full pipeline.
- `v1/results/reported_experimental_results.csv` holds aggregate figures from this version. They are not
  supported by retained per-case outputs and should not be cited.
- `v1/historical-artifacts/` contains per-case outputs from an earlier 100-case version of the dataset.

The current pipeline (`src/`) runs the Judge label-blind, uses a separate attacker model, and stores every
per-case output; its results are in `results/`.
