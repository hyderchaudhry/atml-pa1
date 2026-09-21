# ATML PA1 - Beyond IID Learning

This repository contains one readable, unexecuted Jupyter notebook for each assignment task:

- `task1/task1.ipynb` - controlled representation and inductive-bias experiments on STL-10.
- `task2/task2.ipynb` - PACS unsupervised domain adaptation (Source-only, DAN, DANN, CDAN).
- `task3/task3.ipynb` - PACS domain generalization (reused ERM, DAN-DG, SAM).
- `task4/task4.ipynb` - CIFAR-10/CIFAR-100 open-set recognition (Vanilla, GCSC, PROSER).

The notebooks are designed to run top-to-bottom after dependencies and datasets are prepared locally. They intentionally have no executed outputs. Each has `plot` and `print_metrics` configuration switches and saves small result tables/manifests under its task-specific `results/` directory. Raw datasets, caches, and checkpoints are excluded from Git.

## Setup and data

Use Python 3.10+ and install the listed packages with `pip install -r requirements.txt`. Place datasets under `data/` (which is ignored): STL-10 for Task 1; PACS with `photo`, `art_painting`, `cartoon`, and `sketch` folders for Tasks 2–3; and CIFAR-10/CIFAR-100 for Task 4. Dataset downloading is disabled by default in the notebook configuration.

Run Task 2 before Task 3: Task 3 loads the selected Source-only checkpoint from Task 2, as required by the manual. Do not run any Task 3 Sketch code until all source-side configurations are locked. Likewise, leave Task 4's CIFAR-100 loading in its final-evaluation section only.

## External resources and attribution

The experiment code is original notebook code. It uses standard library APIs from [PyTorch/torchvision](https://pytorch.org/vision/stable/), [OpenCLIP](https://github.com/mlfoundations/open_clip), and [scikit-learn](https://scikit-learn.org/stable/).

Task 1 is designed to call the public [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) implementation and its published weights after the user installs it separately; it is not copied into this repository. The method is Huang and Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (ICCV 2017), [arXiv:1703.06868](https://arxiv.org/abs/1703.06868).

The methods implemented from their papers are: Ganin et al., *Domain-Adversarial Training of Neural Networks* (JMLR 2016); Long et al., *Conditional Adversarial Domain Adaptation* (NeurIPS 2018); Foret et al., *Sharpness-Aware Minimization* (ICLR 2021); Vaze et al., *Open-Set Recognition: A Good Closed-Set Classifier is All You Need?* (ICLR 2022); and Zhou et al., *Learning Placeholders for Open-Set Recognition* (CVPR 2021). Task 4's PROSER section follows the loss construction described in Zhou et al.; consult the paper and its authors' reference code when comparing implementation details.

No raw dataset, external implementation source, or pretrained checkpoint is committed here.
