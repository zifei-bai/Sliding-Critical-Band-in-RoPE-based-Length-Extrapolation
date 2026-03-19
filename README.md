# Sliding Critical Band in RoPE-based Length Extrapolation

*Zifei Bai & Zhiwei Xu*,2nd DeLTa Workshop @ ICLR 2026

This Repo contains the code for "Sliding Critical Band in RoPE-based Length Extrapolation"

## File Structures

## Experiments

### Get Raw Results

Download data from: TODO

Create `/data/` folder to store the data

Create `/results/` and `/graphs/` folders

```
!bash find_ubs.sh
```

```
!bash find_lbs.sh
```

### Draw Figure 1 (Visializing $d_\text{upper}$)

```
!bash ub_graph.sh
```

### Draw Figure 2 (Visializing $d_\text{lower}$)

```
!bash lb_graph.sh
```

### Draw Figure 3 (Visializing Critical Band)

```
!bash draw_scb.sh
```

### Draw Figure 5 (Comparing Attention Pattern of Vanilla RoPE and Interpolating Sliding Critical Band)