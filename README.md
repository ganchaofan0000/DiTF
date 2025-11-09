# Diffusin Transformer Features (DiTF)
In this paper, we investigate how pretrained **Diffusion Transformers (DiTs)** can be effectively employed as **universal feature extractors**. Importantly, we reveal and analyze the impact of **AdaLN** in DiTs on **representation learning**.

Finally, we propose a training-free **AdaLN-based** framework that extracts semantically discriminative features from DiTs.

### [Project Page](https://arxiv.org/pdf/2505.18584?) | [Paper](https://arxiv.org/pdf/2505.18584?)

## Links
You are welcomed to check a series of works from our group on Diffusion Transformers as listed below:
- 🚀 <u>[DG](https://arxiv.org/pdf/2510.11538?)</u> [ARXIV'25]: Detail Guidance for DiTs! [[Paper](https://arxiv.org/pdf/2510.11538?)] [[Project](https://ganchaofan0000.github.io/project_dg/)]

## Prerequisites
Python environment:
```
conda env create -f environment.yml
conda activate DiTF
pip install -e ".[all]"
```
Download Model Weights (e.g. Flux-dev):
```
black-forest-labs/FLUX.1-dev
google/t5-v1_1-xxl
openai/clip-vit-large-patch14
```


## Semantic correspondence (SPair-71k)
First, download SPair-71k data:
```
wget https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar -xzvf SPair-71k.tar.gz
```
Run the following script to get PCK (both per point and per img) of Flux on SPair-71k:
```
python eval_spair.py \
--dataset spair \
--dataset_path /mnt/nvme0n1/chaofan/dataset/SPair-71k \
--save_path Features/spair_flux \
--dit_model flux \
--img_size 640 640 \
--t 260 \
--k 28 \
--ensemble_size 8 \
--cd
```
Here're the explanation for each argument:
- `dataset_path`: path to the input image file.
- `save_path`: path to save the output features as torch tensor.
- `dit_model`: name of DiT model.
- `img_size`: the width and height of the resized image before fed into diffusion model.
- `t`: time step for diffusion, choose from range [0, 1000], must be an integer. `t=260` by default for semantic correspondence.
- `k`: the index of the dit block to extract the feature map, `k=28` by default for semantic correspondence.
- `ensemble_size`: the number of repeated images in each batch used to get features. `ensemble_size=8` by default. You can reduce this value if encountering memory issue.
- `cd`: adopt channel discard for the massive activations.

Ablation study of different part for our DiTF_flux model.
| Model | Acc|
|--|--|
| original| 29.9|
| +AdaLN|65.3|
| +Channel discard| 67.1|


### Temporal Correspondence (DAVIS)

We follow the evaluation protocal as in DINO's [implementation](https://github.com/facebookresearch/dino#evaluation-davis-2017-video-object-segmentation).

First, prepare DAVIS 2017 data and evaluation tools:
```
cd $HOME
git clone https://github.com/davisvideochallenge/davis-2017 && cd davis-2017
./data/get_davis.sh
cd $HOME
git clone https://github.com/davisvideochallenge/davis2017-evaluation
```

Then, get segmentation results using DiTF_flux:
```
CUDA_VISIBLE_DEVICES=7 \
python eval_davis.py \
--img_size 960 \
--k 24 \
--t 90 \
--cd \
--ensemble_size 8 \
--output_dir ./results/ft_24_t_90_960_cd_final/
```

Finally, evaluate the results:
```
python $HOME/davis2017-evaluation/evaluation_method.py \
    --task semi-supervised \
    --results_path ./results/ft_24_t_90_960_cd_final \
    --davis_path /mnt/nvme0n1/chaofan/dataset/davis17/
```

## Citation
If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:
```
@inproceedings{
    gan2025unleashing,
    title={Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations},
    author={Chaofan Gan and Yuanpeng Tu and Xi Chen and Tieyuan Chen and Yuxi Li and Mehrtash Harandi and Weiyao Lin},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=s3MwCBuqav}
}
```

## Acknowledgement
- We thank all authors of SD-based Diffusion feature extractor [DIFT](https://arxiv.org/pdf/2505.18584?) for presenting such an excellent work.
