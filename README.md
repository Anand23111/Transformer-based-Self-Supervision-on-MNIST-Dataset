# Transformer-based Self-Supervision on MNIST

This repository demonstrates several self-supervised and semi-supervised Vision Transformer (ViT) methods on a subset of the MNIST dataset (digits 0, 1, and 2). We showcase four main parts:

1. **Part 1**: Masked Autoencoder (MAE) for patch-based reconstruction.  
2. **Part 2**: InfoNCE contrastive learning on the CLS token.  
3. **Part 3**: A combined MAE + InfoNCE model for joint self-supervised pretraining.  
4. **Part 4**: A “Video MAE” approach, simulating multi-frame data by repeating MNIST images as frames.

---

## Table of Contents
- [Introduction](#introduction)
- [Method Overview](#method-overview)
- [Installation and Requirements](#installation-and-requirements)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction

Self-supervised learning has become a powerful paradigm for leveraging unlabeled data to learn rich, transferable representations. By applying Vision Transformers (ViT) and techniques such as Masked Autoencoding (MAE) and contrastive InfoNCE, we can obtain feature encoders that are then fine-tuned on smaller labeled subsets with better performance and data efficiency.

This repository uses MNIST digit images (specifically digits `[0, 1, 2]`) to illustrate:
1. **MAE**: The model learns to reconstruct masked patches of an image.
2. **InfoNCE**: The model learns discriminative representations by contrasting each image embedding against others.
3. **Combined MAE+InfoNCE**: Both reconstruction and contrastive objectives run simultaneously.
4. **Video MAE**: Demonstrates how to extend the MAE concept to multiple frames.

---

## Method Overview

### 1. **ViT Architecture**
- **Patch Embedding**: Images are split into patches (using `nn.Conv2d`).
- **Transformer Blocks**: Each block includes Multi-Head Self-Attention and a FeedForward MLP with LayerNorm and residual connections.
- **CLS Token**: In InfoNCE or combined tasks, a special CLS token is added to aggregate global image information.

### 2. **Pretraining Objectives**
1. **MAE**: Mask a portion of input patches and reconstruct them.  
2. **InfoNCE**: Enforce similarity of each sample’s CLS token with itself while pushing away other samples’ embeddings.  
3. **Combined**: Use a weighted sum of MAE and InfoNCE losses.  
4. **Video MAE**: Extend MAE to a (mock) multi-frame scenario by repeating each MNIST image 3 times.

### 3. **Fine-Tuning**
After pretraining, we attach a classification head on top of the ViT encoder (specifically on the CLS token’s embedding) and train with a standard cross-entropy loss for digit classification.

---

## Installation and Requirements

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/transformer-mnist-selfsupervision.git
   cd transformer-mnist-selfsupervision
