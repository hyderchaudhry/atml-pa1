# Task 3 — Domain generalization

Notebook: [task3.ipynb](task3.ipynb). This experiment compares the exact Task 2 Source-only ERM checkpoint with DAN-DG and SAM using the three labeled PACS source domains. Sketch is opened only in the final-evaluation section. See Section 3 of the supplied ATML-PA1 manual.

## Prepare and run

Run [Task 2](../task2/README.md) first so `../common/splits/pacs_sketch_seed6304.json` and `../task2/results/source_only.pt` exist. Install [`requirements.txt`](../requirements.txt) and use the same PACS directory layout. Review the first cell's `CONFIG` before running: seed `6304`, 30 epochs, patience 5, AdamW `1e-4` learning rate and weight decay, eight images per source per update, DAN-DG weight `1`, SAM radius `0.05`, and the fixed SAM radius study `{0.01, 0.05, 0.1}`.

Run source data setup, ERM reuse, DAN-DG/SAM training, and source diagnostics first. Only then run the final Sketch section. Neither Sketch images nor Task 2 Sketch results should influence Task 3 settings or model selection.

## Implementation and outputs

The notebook reads Task 2's saved source split and uses the same ResNet-18 architecture, ImageNet weights, preprocessing, domain-balanced batches, and frozen BatchNorm running statistics. It loads ERM from `source_only.pt`; it does not retrain that baseline. DAN-DG adds the average three-kernel RBF MMD over the Photo–Art, Photo–Cartoon, and Art–Cartoon feature pairs, using only observed sources. SAM makes a normalized ascent perturbation and a second forward/backward pass before an AdamW update. Checkpoints use mean source-validation macro-F1.

Source diagnostics include per-domain accuracy, mean/worst source accuracy, a three-way logistic-regression source-domain separability probe, and a common radius-0.05 sharpness proxy on a fixed validation batch. The final section computes Sketch aggregate metrics and a preconfigured SAM-radius study. Outputs are `results/source_diagnostics.csv`, `results/final_metrics.csv`, and `results/sam_strength_study.csv`; `print_metrics` displays the final table. The current notebook does not yet save per-class Sketch changes, confusion cases, or training-curve plots required by the manual, so those evidence items need implementation before the task is considered complete.

## Method references

- Long et al., [*Deep Adaptation Networks*](https://proceedings.mlr.press/v37/long15.html), 2015 (the MMD mechanism adapted here to source-source alignment).
- Foret et al., [*Sharpness-Aware Minimization for Efficiently Improving Generalization*](https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/), 2021 (SAM).

The notebook implements the methods directly; it does not copy external method code.
