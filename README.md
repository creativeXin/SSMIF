# SSMIF: Enhanced Spatial-Spectral Mamba Interactive Fusion Network for Hyperspectral Change Detection

SSMIF is a hyperspectral change detection framework that integrates the powerful Mamba architecture with spatial-spectral interactive fusion mechanisms. It aims to accurately identify and classify change regions in bitemporal hyperspectral images, leveraging both spectral semantics and spatial structures.

## 🚀 Getting Started

### 📦 Installation

To set up the environment, please run the following commands:

```bash
conda create -n mamba python=3.10
conda activate mamba

# Install PyTorch with CUDA 11.7 support
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install required Python packages
pip install packaging==23.2
pip install triton==2.0.0
pip install mamba-ssm==1.2.0
pip install spectral
pip install scikit-learn==1.4.2
pip install calflops
``` 

## 📚 Citation

If you find this project helpful for your research, please kindly consider citing our paper and giving this repo a ⭐:

```bibtex
@INPROCEEDINGS{10868119,
  author    = {Wu, Lanxin and Peng, Jiangtao and Yang, Bing and Sun, Weiwei and Ye, Zhijing},
  booktitle = {2024 IEEE International Conference on Signal, Information and Data Processing (ICSIDP)}, 
  title     = {SSMIF: Enhanced Spatial-Spectral Mamba Interactive Fusion Network for Hyperspectral Change Detection}, 
  year      = {2024},
  volume    = {},
  number    = {},
  pages     = {1-5},
  keywords  = {Accuracy;Source coding;Noise;Termination of employment;Interference;Feature extraction;Transformers;Data models;Long short term memory;Hyperspectral imaging;Hyperspectral image;change detection;mamba;state space model},
  doi       = {10.1109/ICSIDP62679.2024.10868119}
}
