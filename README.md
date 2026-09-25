# ATML PA1 - Beyond IID Learning

This repository contains one readable, unexecuted Jupyter notebook for each assignment task:

- [Task 1](task1/README.md): controlled representation and inductive-bias experiments on STL-10.
- [Task 2](task2/README.md): PACS unsupervised domain adaptation (Source-only, DAN, DANN, CDAN).
- [Task 3](task3/README.md): PACS domain generalization (reused ERM, DAN-DG, SAM).
- [Task 4](task4/README.md): CIFAR-10/CIFAR-100 open-set recognition (Vanilla, GCSC, PROSER).

The notebooks are organized in execution order after dependencies and datasets are prepared. Task 1 pauses for a visual review of generated cue conflicts before its final analysis. They intentionally have no executed outputs. `print_metrics=True` displays numerical summaries; `plot=True` displays final figures inline and saves them as PNGs under the relevant `taskN/results/`. Each notebook creates that directory when its configuration cell runs and writes its small CSV/JSON/PNG outputs there. These files do not appear until execution. Raw datasets, large generated collections, external assets, and checkpoints stay outside Git.

## Setup and data

Use Python 3.10+ and install the listed packages with `pip install -r requirements.txt`. In Colab, clone this Git repository under `/content/` and change the working directory to the clone root or the notebook's task folder. The notebooks find the repository by its tracked files, so the clone name does not matter. They mount Google Drive at `/content/drive` and create `/content/drive/MyDrive/ATML_PA1/` subdirectories automatically:

```text
ATML_PA1/
├── data/                 # STL-10, PACS, CIFAR-10, CIFAR-100
├── checkpoints/task1/ ... task4/
├── artifacts/task1/ ... task4/
└── external/             # cited AdaIN implementation and weights
```

Place datasets under `ATML_PA1/data/`: the extracted STL-10 `stl10_binary/` folder, `PACS/<domain>/<class>/` with lowercase `photo`, `art_painting`, `cartoon`, and `sketch`, and torchvision's extracted `cifar-10-batches-py/` and `cifar-100-python/` folders. Dataset downloading remains disabled by default. The configuration raises a clear error if a required Drive dataset or checkpoint is missing.

Run Task 2 before Task 3: Task 3 loads the selected Source-only checkpoint from Task 2, as required by the manual. Do not run any Task 3 Sketch code until all source-side configurations are locked. Likewise, leave Task 4's CIFAR-100 loading in its final-evaluation section only.

The Git clone is not stored in Drive. Checkpoints, AdaIN-generated images, external assets, and pretrained-weight caches persist in Drive; final result tables and figures are written directly to the clone's `taskN/results/`. After a run, commit and push those small outputs from the Colab clone (`git status`, `git add .`, `git commit`, `git push`) before its temporary runtime resets. Task 1's small review manifest is also in `task1/results/`; commit it after the visual review to preserve acceptance decisions. Do not add datasets or model files to Git.

## External resources and attribution

The experiment code is original notebook code. It uses standard library APIs from [PyTorch/torchvision](https://pytorch.org/vision/stable/), [OpenCLIP](https://github.com/mlfoundations/open_clip), and [scikit-learn](https://scikit-learn.org/stable/).

Task 1 is designed to call the public [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) implementation and its published weights after the user installs it separately; it is not copied into this repository. The method is Huang and Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (ICCV 2017), [arXiv:1703.06868](https://arxiv.org/abs/1703.06868).

The methods implemented from their papers are: Ganin et al., *Domain-Adversarial Training of Neural Networks* (JMLR 2016); Long et al., *Conditional Adversarial Domain Adaptation* (NeurIPS 2018); Foret et al., *Sharpness-Aware Minimization* (ICLR 2021); Vaze et al., *Open-Set Recognition: A Good Closed-Set Classifier is All You Need?* (ICLR 2022); and Zhou et al., *Learning Placeholders for Open-Set Recognition* (CVPR 2021). Task 4's PROSER section follows the loss construction described in Zhou et al.; consult the paper and its authors' reference code when comparing implementation details.

