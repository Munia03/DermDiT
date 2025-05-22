## [Prompting Medical Vision-Language Models to Mitigate Diagnosis Bias by Generating Realistic Dermoscopic Images](https://ieeexplore.ieee.org/abstract/document/10980892) 

### [arXiv](https://arxiv.org/abs/2504.01838)

## Acknowledgments

This work builds upon the repository DiT(https://github.com/facebookresearch/DiT) by Facebook Research. We extend their implementation for our DermDiT model.
The setup and training workflow are also adapted from the original repository.

## Setup

First, download and set up the repo:

```bash
git clone https://github.com/Munia03/DermDiT.git
cd DermDiT
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment.

```bash
conda env create -f environment.yml
conda activate DiT
```


## Training DermDiT

We provide a training script for DiT in [`train_text_to_image.py`](train_text_to_image.py). This script can be used to train text-conditional DermDiT model.

To launch DiT-L/4 (256x256) training with `N` GPUs on one node:

```bash
torchrun --nnodes=1 --nproc_per_node=N train_text_to_image.py --model DiT-L/4 --data-path /path/to/imagenet/train
```

## Generating Dermsocopic images

We include a [`sample_text2img.py`](sample_text2img.py) script which samples a large number of images from a DiT model in parallel. This script
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM&#39;s TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from a trained model over `N` GPUs, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_text2img.py --model DiT-L/4 --image-size 256 --num-fid-samples 50000 --ckpt /path/to/model.pt
```

## BibTeX

```bibtex
@inproceedings{munia2025prompting,
  title={Prompting Medical Vision-Language Models to Mitigate Diagnosis Bias by Generating Realistic Dermoscopic Images},
  author={Munia, Nusrat and Imran, Abdullah Al Zubaer},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},
  pages={1--4},
  year={2025},
  organization={IEEE}
}
```
