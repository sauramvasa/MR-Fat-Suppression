Git repository for MR fat suppression failure detection
# MR Fat Suppression

A PyTorch-based computer vision pipeline for detecting fat contamination in magnetic resonance imaging (MRI) slices.

The project creates synthetically corrupted MRI images by adding spatially varying fat signal to non-fat MRI images. A convolutional neural network (CNN) is then trained to classify whether an image has low or high levels of fat contamination.

> **Project status:** Research prototype  
> This repository contains experimental code for MRI preprocessing, synthetic fat-signal generation, CNN training, model evaluation, and feature visualization.

---

## Overview

Fat suppression is an important MRI technique used to reduce signal from fatty tissue and improve the visibility of anatomical structures.

This project investigates whether a neural network can identify residual fat signal in MRI images. The pipeline:

1. Loads MRI DICOM series.
2. Identifies corresponding fat and non-fat image series.
3. Generates spatially varying synthetic fat contamination.
4. Produces corrupted MRI slices with known corruption factors.
5. Trains a CNN to classify image corruption.
6. Evaluates model predictions using accuracy, confusion matrices, and visualizations.
7. Provides tools for visualizing convolutional filters and intermediate feature maps.

The primary implemented task is binary classification:

- `0`: low or negligible fat contamination
- `1`: significant fat contamination

The code also contains experimental support for continuous corruption-factor prediction.

---

## Pipeline

```text
DICOM MRI Series
        │
        ▼
Load and pair fat/non-fat series
        │
        ▼
Generate spatially smooth fat-corruption masks
        │
        ▼
Create synthetically corrupted MRI slices
        │
        ▼
Resize and normalize images
        │
        ▼
Train convolutional neural network
        │
        ▼
Evaluate predictions and visualize model behavior