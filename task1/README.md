# Task 1 — Inductive biases and representations

Notebook: [task1.ipynb](task1.ipynb). This notebook studies how frozen ResNet-50, ViT-B/16, and OpenCLIP ViT-B-32 respond to changes in color, shape and texture cues, object position, and spatial arrangement on STL-10. The assignment requirements are in Section 1 of the supplied ATML-PA1 manual.

## Prepare and run

1. Install the packages in [`requirements.txt`](../requirements.txt) and put extracted STL-10 data at `/content/drive/MyDrive/ATML_PA1/data/stl10_binary/`. The notebook mounts Drive in Colab; `download_data` defaults to `False`.
2. Review the first cell's `CONFIG`: seed `6304`, selected class pairs, AdaIN strength, rejection rule, and `plot`/`print_metrics` switches. These choices should be fixed before looking at results.
3. Run the existing experiment sections in notebook order. Before the cue-conflict section, install the cited AdaIN code and its `decoder.pth` and `vgg_normalised.pth` weights under `/content/drive/MyDrive/ATML_PA1/external/pytorch-AdaIN/`.
4. Call `generate_conflicts()`, then use `show_conflict_candidates(start=0, count=12)` to inspect pages inline. Fill `accepted` and `rejection_reason` in `results/cue_conflict_manifest.json` using the stated visual rule only. Then call `evaluate_cue_conflicts()` and run the representation section. At least 200 accepted images are required. Recalling `generate_conflicts()` preserves an existing reviewed manifest.

No notebook cells have been executed in this repository. The cue-conflict review is an intentional human step; merely running the notebook top-to-bottom cannot complete that review.

## Implementation and outputs

The official STL-10 training set is split by class into 80% training and 20% validation with seed `6304`. A saved manifest selects 50 official test images per class. Images first become common 224 × 224 RGB inputs; each backbone then receives its own normalization. The backbones stay frozen, and an AdamW-trained linear head is selected by validation accuracy. CLIP also uses the single fixed prompt `a photo of a {class}.` for zero-shot scores.

The notebook compares clean, grayscale, fixed hue rotation, 4 × 4 patch shuffle, and four-direction translations at 0/8/16/32 pixels. It reports accuracy, macro-F1, confidence where requested, clean-relative accuracy changes, and prediction consistency. AdaIN candidates use five unordered class pairs in both directions; accepted examples produce shape, texture, and other counts plus shape bias and coverage. Paired clean/transformed features produce cosine stability and t-SNE plots for grayscale, accepted cue conflicts, translation, and patch shuffle.

The notebook writes small JSON artifacts to `results/`: `test_subset_manifest.json`, `condition_metrics.json`, `translation_metrics.json`, `cue_conflict_manifest.json`, `cue_conflict_acceptance_counts.json`, `cue_conflict_metrics.json`, and `representation_stability.json`. `cue_conflict_predictions.csv` records each accepted image's shape/texture/other decision for example selection. Values display inline when `print_metrics` is true; final graphs and image grids display inline and save as PNGs to `results/` when `plot` is true. Generated conflict images persist separately at `/content/drive/MyDrive/ATML_PA1/artifacts/task1/cue_conflicts/`. Commit the small reviewed manifest before a Colab runtime reset.

## External resources

- [torchvision](https://pytorch.org/vision/stable/) provides STL-10, image transforms, and the specified ResNet-50 and ViT weights.
- [OpenCLIP](https://github.com/mlfoundations/open_clip) provides the pretrained CLIP image/text encoders.
- [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) supplies the external style-transfer implementation and weights. Method: Huang and Belongie, [*Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*](https://arxiv.org/abs/1703.06868), 2017. Its source is not copied into this repository.
