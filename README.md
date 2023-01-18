# Learning Causal Representations of Single Cells via Sparse Mechanism Shift Modeling

This repository contains an implementation of the sparse VAE framework applied to single-cell perturbation data, as descibed in ["Learning Causal Representations of Single Cells via Sparse Mechanism Shift Modeling"](https://arxiv.org/abs/2211.03553). 


[![Stars](https://img.shields.io/github/stars/Genentech/sVAE?logo=GitHub&color=yellow)](https://github.com/Genentech/sVAE/stargazers)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

<center>
    <img src="svae+.png?raw=true" width="750">
</center>
Overview of the sparse VAE framework applied to single-cell perturbation data. (A) Input data are gene expression profiles of cells under different genetic or chemical perturbations (colors), as well as the intervention label. (B) A schematic of the generative model, and the causal semantics of the sparse VAE (C) Three method outputs. (i) identification of target latent variables, encoded as a causal graph between the interventions and latent variables; (ii) a disentangled latent model for which individual latent variables are more likely to be interpreted as the activity of a relevant biological process; and (iii) the generalization of the generative model to unseen interventions (e.g., for latent target identification).

## User guide


### Installation
Download or clone this repository. Then from inside the folder simply run:
```
pip install -e . 
```

### Example
An example script for the sandbox can be found in ``` entry_points/demo.py```.
The code for reproducing the real data analysis can be found in ``` entry_points/run_real_data_replogle_wandb.py```.

## References

```
@article{svae+,
  title={Learning Causal Representations of Single Cells via Sparse Mechanism Shift Modeling},
  author={Lopez, Romain and Tagasovska, Natasa and Ra, Stephen and Cho, Kyunghyun and Pritchard, Jonathan K. and Regev, Aviv },
  journal={Conference on Causal Learning and Reasoning},
  year={2023},
}
```
