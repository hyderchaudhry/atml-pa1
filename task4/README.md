# Task 4 — Open-set recognition

Notebook: [task4.ipynb](task4.ipynb). CIFAR-10 supplies all ten known classes; the fixed near and far unknown groups come only from the CIFAR-100 test partition. See Section 4 of the supplied ATML-PA1 manual.

## Prepare and run

Install [`requirements.txt`](../requirements.txt) and put extracted `cifar-10-batches-py/` and `cifar-100-python/` under `/content/drive/MyDrive/ATML_PA1/data/`, or set `CONFIG["download_data"]` if you want torchvision to download them there when running later. The notebook mounts Drive in Colab. Review the first cell's seed, optimizer settings, placeholder settings, and `plot`/`print_metrics` switches before executing. CIFAR-100 must remain outside all training, checkpoint-selection, score-design, and threshold-selection decisions.

Run the notebook in order: fixed known-class split, Vanilla and GCSC training, frozen-model score definitions, PROSER training, then the final CIFAR-100 evaluation. Training and checkpoint selection use only CIFAR-10. The optional reciprocal-point extension is not included.

## Implementation and outputs

The code makes a class-stratified 90/10 split of official CIFAR-10 training images using seed `6304`. Its CIFAR ResNet-18 replaces the usual 7 × 7 stride-2 stem with a 3 × 3 stride-1 convolution and removes max pooling. Vanilla uses the prescribed crop/flip and SGD/cosine schedule. GCSC adds `RandAugment(num_ops=2, magnitude=9)` after crop/flip. Both select their best checkpoint by known validation accuracy.

The frozen Vanilla outputs supply four unknownness scores: one minus maximum softmax probability (MSP), negative maximum logit (MLS), negative log-sum-exp Energy, and a diagonal shared-covariance Mahalanobis distance fitted from unaugmented known training features. PROSER starts from the selected Vanilla weights, adds five dummy logits, and uses classifier placeholders plus between-class feature mixing after ResNet layer2. Its final evaluation includes known-logit MLS and a dummy-vs-known placeholder score.

For each score, the final section sets a threshold at the 95th percentile of known validation unknownness. It evaluates near, far, and combined unknowns, produces a three-panel score-distribution plot, and records accepted near/far failures under the Vanilla MLS threshold. `print_metrics=True` displays the OSR and failure tables; `plot=True` displays the score distributions and failure-image grid inline and saves PNGs under repository `results/`. The repository receives `results/cifar10_split.json`, `results/osr_metrics.csv`, and `results/vanilla_mls_accepted_failures.csv`; Vanilla, GCSC, and PROSER `.pt` checkpoints persist only under `/content/drive/MyDrive/ATML_PA1/checkpoints/task4/`. The PROSER loss uses the strongest of five dummy logits as the paper's aggregated unknown response; the placeholder score compares that response with the strongest known logit, with its rejection threshold calibrated from known validation examples only.

## Method references

- Vaze et al., [*Open-Set Recognition: A Good Closed-Set Classifier is All You Need?*](https://arxiv.org/abs/2110.06207), 2022 (strong closed-set classification and MLS).
- Zhou et al., [*Learning Placeholders for Open-Set Recognition*](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Learning_Placeholders_for_Open-Set_Recognition_CVPR_2021_paper.html), 2021 (PROSER).

The notebook uses torchvision and implements its own training/evaluation blocks; no external method implementation is copied.
