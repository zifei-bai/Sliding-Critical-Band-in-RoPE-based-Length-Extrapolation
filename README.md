# Sliding Critical Band in RoPE-based Length Extrapolation

*Zifei Bai & Zhiwei Xu*, [2nd DeLTa Workshop @ ICLR 2026](https://delta-workshop.github.io/DeLTa2026/)

This Repo contains the code for "Sliding Critical Band in RoPE-based Length Extrapolation", [paper link](https://openreview.net/forum?id=y2IeQSTxmc)

## Dependencies

```
torch==2.10.0+cu128
pandas==2.2.2
numpy==2.0.2
matplotlib==3.10.0
wandb==0.25.1
tqdm==4.67.3
transformers==5.0.0
datasets==4.0.0
```

## Experiments

### Get Raw Results

Download train and test data from: [Google Drive](https://drive.google.com/drive/folders/1YGlSPncM8ZF91vwQwvd71Cd9E8ACLC9w?usp=sharing)

Create `/data/` folder to store the data

Create `/results/` and `/graphs/` folders to store raw experiment results and figures

```
!bash scripts/find_ubs.sh
```

```
!bash scripts/find_lbs.sh
```

### Draw Figure 1 (Visializing $d_\text{upper}$)

```
!bash scripts/ub_graph.sh
```

### Draw Figure 2 (Visializing $d_\text{lower}$)

```
!bash scripts/lb_graph.sh
```

### Draw Figure 3 and 4 (Visializing Critical Band)

```
!bash scripts/draw_scb.sh
```

### Draw Figure 5 (Comparing Attention Pattern of Vanilla RoPE and Interpolating Sliding Critical Band)

```
!bash scripts/attn.sh
```

### Draw Figure 6 (Real Model's Sliding Critical Band)

Find $d_\text{upper}$ and $d_\text{lower}$

```
!bash scripts/real_ppl.sh
```

Plotting Sliding Critical Band

```
!bash scripts/plot_real_scb.sh
```


#### Figure 13 (PPL changes on dimension $d$)

```
!bash scripts/real_plot.sh
```

### Pretraining Model From Scratch

Pretraining model from scratch and run the above code to find the Sliding Critical Band

```
!bash scripts/train.sh
```

### Citation

```
@inproceedings{
bai2026sliding,
title={Sliding Critical Band in Ro{PE}-based Length Extrapolation},
author={Zifei Bai and Zhiwei Xu},
booktitle={ICLR 2026 2nd Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy},
year={2026},
url={https://openreview.net/forum?id=y2IeQSTxmc}
}
```