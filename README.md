# Domain-Agnostic Fourier Neural Operator (DAFNO) – PyTorch Lightning Version

This repository provides a modular and scalable **[PyTorch Lightning](https://www.pytorchlightning.ai/)** implementation of the **[DAFNO](https://github.com/ningliu-iga/DAFNO?tab=readme-ov-file)**, originally proposed by Liu *et al.* in [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/940a7634dab556b67af15bacd337f7db-Abstract-Conference.html). It extends the standard Fourier Neural Operator (FNO) framework to support **irregular geometries**, **evolving domains**, and **topology changes**, while maintaining the efficiency of FFT-based operations.

> ✅ This work builds upon the original implementation by **Ning Liu**, published in the paper:  
> 📄 *"Fourier Neural Operator with Geometry-aware Input Mapping"*  
> [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/940a7634dab556b67af15bacd337f7db-Abstract-Conference.html)
>
---

## 🧠 What is DAFNO?

DAFNO (Domain-Agnostic Fourier Neural Operator) is a neural operator that solves PDEs on **non-rectangular, time-evolving, and topologically dynamic domains**. It achieves this by introducing a **smoothed characteristic function** into the Fourier operator, which enables FFT-based convolution even in geometrically irregular settings.

### Key Features
- ✔️ Domain-agnostic surrogate modeling with **explicit geometric encoding**
- ✔️ Compatible with **irregular grids**, **voids**, and **fractures**
- ✔️ High generalizability to **unseen geometries and load conditions**
- ✔️ Efficient via **FFT acceleration** and **integral kernel structure**

---

## ✨ What's New in This Implementation?

This PyTorch Lightning-based version includes:

- 🧩 Clean and modular training loop
- 📊 Integrated logging and checkpointing
- 🎯 Support for **custom loss functions** (e.g., masked MAE)
- 🧱 Ready for **structured and unstructured** grid inputs
- 🧪 Includes **elasticity** dataset pipeline and demo

---

## 🚀 Getting Started

### 📦 Requirements

- `pytorch >= 1.8.0`
- `pytorch-lightning >= 2.0.0`
- `h5py`, `numpy`, `matplotlib`, `scipy`

<!---
```bash
pip install -r requirements.txt
-->
