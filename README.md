# Domain-Agnostic Fourier Neural Operator (DAFNO) – PyTorch Lightning Version

This repository provides a modular and scalable **PyTorch Lightning** implementation of the **Domain-Agnostic Fourier Neural Operator (DAFNO)**, originally proposed by Liu *et al.* in NeurIPS 2023. It extends the standard Fourier Neural Operator (FNO) framework to support **irregular geometries**, **evolving domains**, and **topology changes**, while maintaining the efficiency of FFT-based operations.

> ✅ This reimplementation refactors the official DAFNO codebase using PyTorch Lightning for better modularity, readability, and experiment tracking.

---

## 🧠 What is DAFNO?

DAFNO (Domain-Agnostic Fourier Neural Operator) is a neural operator that solves PDEs on **non-rectangular, time-evolving, and topologically dynamic domains**. It achieves this by introducing a **smoothed characteristic function** into the Fourier operator, which enables FFT-based convolution even in geometrically irregular settings.

### Key Features
- ✔️ Domain-agnostic surrogate modeling with **explicit geometric encoding**
- ✔️ Compatible with **irregular grids**, **voids**, and **fractures**
- ✔️ High generalizability to **unseen geometries and load conditions**
- ✔️ Efficient via **FFT acceleration** and **integral kernel structure**

---

## 📄 Reference Paper

**Domain Agnostic Fourier Neural Operators**  
*Ning Liu, Siavash Jafarzadeh, Yue Yu*  
[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/940a7634dab556b67af15bacd337f7db-Abstract-Conference.html)

> *"To lift the restriction of FFT to rectangular grids, DAFNO incorporates a smoothed characteristic function to model PDEs on arbitrary and evolving domains."*

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
- `h5py`
- `numpy`
- `matplotlib`
- `scipy`

```bash
pip install -r requirements.txt
