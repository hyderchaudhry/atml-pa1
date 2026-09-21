# Task 2 — Unsupervised domain adaptation

Notebook: [task2.ipynb](task2.ipynb). The experiment compares Source-only ERM, DAN, DANN, and CDAN on PACS. Photo, Art Painting, and Cartoon are labeled sources; Sketch images are available without labels during adaptation. See Section 2 of the supplied ATML-PA1 manual.

## Prepare and run

Install [`requirements.txt`](../requirements.txt), then place PACS in `/content/drive/MyDrive/ATML_PA1/data/PACS/<domain>/<class>/` using the notebook's lowercase domain names: `photo`, `art_painting`, `cartoon`, and `sketch`. The notebook mounts Drive in Colab. Check the first cell's `CONFIG` before running. Its defaults specify seed `6304`, 30 epochs, patience 5, AdamW learning rate `1e-4`, weight decay `1e-4`, eight examples from each source per update, and 24 unlabeled target examples for adaptation methods. `plot` and `print_metrics` control inline output.

Run the notebook from top to bottom. It creates `../common/splits/pacs_sketch_seed6304.json`, which Task 3 reuses. Keep Task 3's design fixed independently of the final Sketch results here. The final evaluation section is where Sketch labels are used for metrics.

## Implementation and outputs

Within each source domain and class, the code creates an 80/20 train/validation split. Training uses a 256 × 256 resize, random 224 × 224 crop, and horizontal flip; evaluation uses a center crop. Every method fine-tunes ImageNet ResNet-18 with a seven-class head. BatchNorm running means and variances remain at their pretrained values while affine parameters stay trainable. The 512-dimensional feature before the classifier is used for alignment and diagnostics.

Source-only optimizes source cross-entropy and does not iterate the target loader. DAN adds a three-bandwidth RBF MMD penalty between pooled source and target features. DANN applies a scheduled gradient-reversal signal from a binary domain discriminator. CDAN feeds the discriminator the outer product of features and soft class probabilities. Each source domain contributes eight examples per update; shorter loaders cycle until the longest source loader has been covered. Checkpoints are chosen by the mean of the three source-validation macro-F1 values, without consulting target labels.

The selected Source-only ERM checkpoint persists at `/content/drive/MyDrive/ATML_PA1/checkpoints/task2/source_only.pt` for Task 3. The final section writes repository files `results/final_metrics.csv` with source/target aggregate metrics and source-vs-target domain separability, `results/target_per_class.csv` and `results/target_confusions.csv` for class-level analysis, and `results/mmd_strength_study.csv` for the fixed `{0.1, 1, 10}` DAN study. Per-method `results/<method>_history.csv` files record classification and alignment/domain losses. With `print_metrics=True`, the notebook displays the final, per-class, dominant-confusion, study, and final-epoch tables. With `plot=True`, it displays both loss curves inline and saves `results/training_losses.png`.

## Method references

- Ganin et al., [*Domain-Adversarial Training of Neural Networks*](https://jmlr.org/papers/v17/15-239.html), 2016 (DANN and gradient reversal).
- Long et al., [*Deep Adaptation Networks*](https://proceedings.mlr.press/v37/long15.html), 2015 (MMD alignment).
- Long et al., [*Conditional Adversarial Domain Adaptation*](https://proceedings.neurips.cc/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html), 2018 (CDAN conditioning).

The notebook implements these objectives directly; no method source file from another repository is copied here.
