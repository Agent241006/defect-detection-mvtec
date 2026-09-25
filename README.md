# MVTec AD Defect Detection System

A deep learning-based industrial defect detection system built using the **MVTec Anomaly Detection (MVTec AD)** dataset.

The project combines:

- **ResNet50 Transfer Learning** for binary defect classification
- **U-Net** for pixel-level defect segmentation
- **Grad-CAM** for visual explanation of classification decisions

The system is designed to determine whether an industrial product image is defective, localize the defective region, and provide a visual explanation of the classification model.

---

## Features

### 1. Defect Classification

A pretrained **ResNet50** model is fine-tuned for binary classification:

- `Good`
- `Defect`

The classification model predicts whether the input product image contains a defect and provides a confidence score.

### 2. Defect Segmentation

A **U-Net** architecture is trained to identify the defective region at pixel level.

The segmentation output indicates:

- Where the defect is located
- The approximate shape/area of the defect

### 3. Grad-CAM Explainability

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to visualize which regions of the input image influenced the classification decision.

This helps make the classification model more interpretable.

### 4. MVTec AD Dataset

The project is based on the **MVTec Anomaly Detection dataset**, which contains industrial inspection images from multiple object and texture categories.

The pipeline supports category-wise processing and model organization.

---

## System Architecture

```text
                 MVTec AD Dataset
                        |
                        v
              Dataset Preprocessing
                        |
                        v
              +-------------------+
              |   Input Image     |
              +-------------------+
                        |
             +----------+----------+
             |                     |
             v                     v
      ResNet50 Classifier       U-Net
             |                     |
             v                     v
      Good / Defect          Defect Mask
             |                     |
             +----------+----------+
                        |
                        v
                   Prediction
                        |
                        v
                  Grad-CAM
                        |
                        v
              Visual Explanation
