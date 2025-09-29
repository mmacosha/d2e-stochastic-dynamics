# Data-to-Energy Stochastic Dynamics

<p align="center">
    🌐 <a href="https:" target="_blank">Website</a> | 📃 <a href="https://arxiv.org/abs/" target="_blank">Paper</a>  <br>
</p>

<p align="center">
  <img src="assets/d2e_cifar10_main.png" alt="Project Screenshot/Logo" width="700"/>
</p>

---
> **Data-to-Energy Stochastic Dynamics**<br>
> Kirill Tamogashev & Nikolay Malkin<br><br>
>**Abstract:** The Schrödinger bridge problem is concerned with finding a stochastic dynamical system bridging two marginal distributions that minimises a certain transportation cost. This problem, which represents a generalisation of optimal transport to the stochastic case, has received attention due to its connections to diffusion models and flow matching, as well as its applications in the natural sciences. However, all existing algorithms allow to infer such dynamics only for cases where samples from both distributions are available.  In this paper, we propose the first general method for modelling Schrödinger bridges when one (or both) distributions are given by their unnormalised densities, with no access to data samples. Our algorithm relies on a generalisation of the iterative proportional fitting (IPF) procedure to the data-free case, inspired by recent developments in off-policy reinforcement learning for training of diffusion samplers. We demonstrate the efficacy of the proposed *data-to-energy IPF* on synthetic problems, finding that it can successfully learn transports between multimodal distributions.  As a secondary consequence of our reinforcement learning formulation, which assumes a fixed time discretisation scheme for the dynamics, we find that existing data-to-data Schrödinger bridge algorithms can be substantially improved by learning the diffusion coefficient of the dynamics. Finally, we apply the newly developed algorithm to the problem of sampling posterior distributions in latent spaces of generative models, thus creating a data-free image-to-image translation method.

## Table of Contents

* [Project structure](#️-installation)
* [Installation](#️-installation)
* [2D experiments](#-2d-experiments)
* [Image experiments](#-image-experiments)
* [Citation](#-citation)
* [Acknowledgements](#-acknowledgements)
---

## Features

* **Feature One:** A short description of a key capability.
* **Feature Two:** Explain what makes this feature useful.
* **Cross-Platform:** Runs on Windows, macOS, and Linux.

---

## Installation

Here's how you can install this repository and reproduce the experiments

* Python 3.11+
* We use [uv](https://docs.astral.sh/uv/) package manager

```bash
# This examples assumes that uv is installed. 
# If not, follow the link above to install it or use a package mangaer of your choice.
uv venv -n sb --python 3.11
source sb/bin/activate
pip install -e .
```

## 2D Experiments
To reproduce 2D experiments you should just 


## Image experiments


## Citation
Please, cite this work as follows
```
@misc{tamogashev@d2edynamics,
    author    = {Kirill, Tamogashev and Nikolai, Malking},
    title     = {Data-to-Energy Stochastic dynamics},
    year      = {2025},
    notes     = {Submitted to ICLR 2026.}
}
```
