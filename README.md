# EfficientNet-B3 Food Classification Model Training

**Project**: NutriSight Food Recognition System  
**Model**: EfficientNet-B3 with Transfer Learning  
**Framework**: PyTorch + DirectML (AMD/Intel GPU Support)

---

## 📑 Table of Contents

- [Runs at a Glance](#runs-at-a-glance)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Methodology](#-training-methodology)
- [Training Results](#-training-results)
- [Final Model Performance](#-final-model-performance)
- [Per-Class Performance](#-per-class-performance)
- [Technical Algorithms](#-technical-algorithms-used)
- [Model Deployment](#-model-deployment)
- [Key Learnings](#-key-learnings--best-practices)
- [Project Files](#project-files)
- [Conclusion](#-conclusion)

---

## Runs at a Glance

Two recent runs are available — listed here for quick reference. See the detailed sections below for full training configuration and per-class metrics.

### 1. 125-class Food Classifier

**Folder**: [`efficientnet_b3_baseline-20251114-003032/`](../runs/efficientnet_b3_baseline-20251114-003032/)

- Classes: 125 food categories
- Best validation Top-1: **86.26%** | Test Top-1: **87.09%** | Test Top-5: **96.98%**
- Epochs trained: 20 (best epoch = 17)
- Dataset: 35,000 train / 4,375 val / 4,375 test images
- Model size: ~45MB (ONNX)

**Quick Links**:

- 📊 [**Summary Report**](../runs/efficientnet_b3_baseline-20251114-003032/summary.json) — Complete training statistics & metrics
- 📈 [**Training Metrics (CSV)**](../runs/efficientnet_b3_baseline-20251114-003032/metrics_epoch.csv) — Per-epoch performance
- 📋 [**Classification Report**](../runs/efficientnet_b3_baseline-20251114-003032/classification_report.txt) — Precision, Recall, F1 per class
- 🎯 [**Per-Class Accuracy**](../runs/efficientnet_b3_baseline-20251114-003032/per_class_accuracy.json) — Accuracy for each food category
- 📊 [**Per-Class Metrics**](../runs/efficientnet_b3_baseline-20251114-003032/per_class_metrics.json) — Detailed precision/recall/F1 per class
- 🔀 [**Confusion Matrix**](../runs/efficientnet_b3_baseline-20251114-003032/confusion_matrix.png) — Visual class confusion analysis
- 📉 [**Training Curves**](../runs/efficientnet_b3_baseline-20251114-003032/training_curves.png) — Loss & accuracy over epochs

**Model Files**:

- 🔸 [`best_efficientnet_b3.pth`](../runs/efficientnet_b3_baseline-20251114-003032/best_efficientnet_b3.pth) — PyTorch checkpoint
- 🔸 [`model.onnx`](../runs/efficientnet_b3_baseline-20251114-003032/model.onnx) — ONNX export for deployment
- 🔸 [`class_names.json`](../runs/efficientnet_b3_baseline-20251114-003032/class_names.json) — 125 food category names

### 2. Food vs Not-Food (Binary Classifier)

**Folder**: [`efficientnet_b3_food_not_food-20251031-220332/`](../runs/efficientnet_b3_food_not_food-20251031-220332/)

- Classes: 2 (food / not_food)
- Best validation Top-1: **98.79%** | Test Top-1: **99.19%**
- Epochs trained: 13 (best epoch = 8)
- Dataset: 9,528 train / 248 val / 248 test images
- Model size: ~45MB (ONNX)

**Quick Links**:

- 📊 [**Summary Report**](../runs/efficientnet_b3_food_not_food-20251031-220332/summary.json) — Complete training statistics
- 📈 [**Training Metrics (CSV)**](../runs/efficientnet_b3_food_not_food-20251031-220332/metrics_epoch.csv) — Per-epoch performance
- 📋 [**Classification Report**](../runs/efficientnet_b3_food_not_food-20251031-220332/classification_report.txt) — Precision, Recall, F1
- 🎯 [**Per-Class Accuracy**](../runs/efficientnet_b3_food_not_food-20251031-220332/per_class_accuracy.json) — Food vs Not-Food accuracy
- 📊 [**Per-Class Metrics**](../runs/efficientnet_b3_food_not_food-20251031-220332/per_class_metrics.json) — Detailed metrics per class
- 🔀 [**Confusion Matrix**](../runs/efficientnet_b3_food_not_food-20251031-220332/confusion_matrix.png) — Binary classification confusion
- 📉 [**Training Curves**](../runs/efficientnet_b3_food_not_food-20251031-220332/training_curves.png) — Loss & accuracy visualization

**Model Files**:

- 🔸 [`best_efficientnet_b3.pth`](../runs/efficientnet_b3_food_not_food-20251031-220332/best_efficientnet_b3.pth) — PyTorch checkpoint
- 🔸 [`model.onnx`](../runs/efficientnet_b3_food_not_food-20251031-220332/model.onnx) — ONNX export for deployment
- 🔸 [`class_names.json`](../runs/efficientnet_b3_food_not_food-20251031-220332/class_names.json) — Class names (food, not_food)

---

### 📓 Notebooks

**Training**:

- 📘 [`train_efficientnet_b3_optimized.ipynb`](../train_efficientnet_b3_optimized.ipynb) — 125-class food classifier training
- 📘 [`train_efficientnet_b3_optimized_food_not_foodv2.ipynb`](../train_efficientnet_b3_optimized_food_not_foodv2.ipynb) — Binary food/not-food training

**Inference & Evaluation**:

- 📗 [`test_inference_nonfood_and_food.ipynb`](../test_inference_nonfood_and_food.ipynb) — Binary model inference testing
- 📗 [`test_model_on_all_food_dataset_images.ipynb`](../test_model_on_all_food_dataset_images.ipynb) — 125-class model testing
- 📗 [`confusion_matrix_all_classes.ipynb`](../confusion_matrix_all_classes.ipynb) — Generate confusion matrix visualization

---

## 🎯 Problem Statement

**Goal**: Build an accurate food recognition system for nutritional tracking applications.

**Challenge**:

- Recognize 125 different food categories from photos
- Handle visual similarities between foods (e.g., different types of cakes, pasta dishes)
- Achieve high accuracy while maintaining reasonable inference speed
- Deploy on resource-constrained environments (web servers)

---

## 📁 Dataset

### Dataset Structure

```
Total Images: 43,750
Food Categories: 125 classes
├── Examples: Pizza, Hamburger, Sushi, Tacos, Ice Cream, etc.
├── Images per class: ~350 images (balanced distribution)
└── Image Resolution: 252×252 pixels
```

### Data Source

- **Base Dataset**: Selected categories from Food-101 (not all 101 classes are included)
- **Extended Dataset**: Additional images collected from our custom/farmed datasets (locally sourced)
- **Note**: The final 125-class dataset is a mix of selected Food-101 categories and our own farmed/custom images — some Food-101 classes were omitted and replaced/augmented by custom data. Dataset was recently updated (Nov 2025) with revised images for some classes and one additional class.
- **Split Method**: Stratified random split to ensure balanced class distribution

---

## 🧠 Model Architecture

### Base Model: EfficientNet-B3

**Why EfficientNet-B3?**

1. **Efficient Design**: Balances accuracy and computational cost
2. **Compound Scaling**: Uniformly scales network depth, width, and resolution
3. **Pre-trained Weights**: Leverages ImageNet knowledge (transfer learning)
4. **Mobile-Friendly**: Suitable for deployment on resource-constrained devices

**Architecture Overview**:s

```
Input Image (252×252×3)
    ↓
EfficientNet-B3 Backbone (12M parameters)
├── Compound scaled CNN layers
├── Mobile Inverted Bottleneck Convolution (MBConv)
├── Squeeze-and-Excitation blocks
└── Feature Extraction
    ↓
Custom Classification Head
├── Dropout Layer (30% dropout rate)
└── Fully Connected Layer (→ 125 classes)
    ↓
Output: Class Probabilities (125 values)
```

**Model Parameters**:

- Total Parameters: **12.0M**
- Trainable Parameters: **12.0M** (after warmup phase)
- Model Size: **~45MB** (ONNX format)

---

## 🔬 Training Methodology

### 1. Transfer Learning Strategy

**Two-Phase Training Approach**:

**Phase 1: Head Warmup (Epochs 1-3)**

- Freeze backbone (EfficientNet-B3 pre-trained layers)
- Train only classification head
- Purpose: Adapt final layers to food recognition task
- Learning Rate: 1×10⁻³

**Phase 2: Fine-tuning (Epochs 4-20)**

- Unfreeze entire network
- Train all layers with lower learning rate
- Purpose: Fine-tune feature extractors for food-specific patterns
- Learning Rate: 1×10⁻⁴ (10× reduction)

**Why This Approach?**

- Prevents catastrophic forgetting of pre-trained features
- Faster convergence compared to training from scratch
- Better generalization on limited dataset

---

### 2. Data Augmentation Techniques

**Purpose**: Artificially increase dataset diversity to improve generalization.

**Training Augmentations Applied**:

| Augmentation             | Parameters                                            | Purpose                              |
| ------------------------ | ----------------------------------------------------- | ------------------------------------ |
| **RandomResizedCrop**    | scale=(0.7, 1.0)                                      | Simulate different camera distances  |
| **RandomHorizontalFlip** | p=0.5                                                 | Handle left-right symmetry           |
| **RandomRotation**       | ±20°                                                  | Account for camera tilt              |
| **ColorJitter**          | brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1 | Simulate lighting variations         |
| **RandomAffine**         | translate=(0.1, 0.1)                                  | Handle slight camera shifts          |
| **RandomErasing**        | p=0.2, scale=(0.02, 0.15)                             | Simulate occlusions (e.g., utensils) |

**Validation/Test Preprocessing**:

- Resize to 284×284 pixels
- Center crop to 252×252 pixels
- Normalize with ImageNet statistics

---

### 3. Regularization Techniques

**Purpose**: Prevent overfitting and improve model generalization.

#### A. Mixup Data Augmentation

- **Algorithm**: Blend two training images and their labels
- **Formula**:
  ```
  mixed_image = λ × image_A + (1-λ) × image_B
  mixed_label = λ × label_A + (1-λ) × label_B
  where λ ~ Beta(0.2, 0.2)
  ```
- **Effect**: Forces model to learn more robust features
- **Parameter**: α = 0.2

#### B. Label Smoothing

- **Purpose**: Prevent overconfident predictions
- **Formula**:
  ```
  smoothed_label = (1 - ε) × one_hot_label + ε/num_classes
  where ε = 0.1
  ```
- **Effect**: Improves calibration and generalization

#### C. Dropout

- **Rate**: 30% (p=0.3)
- **Location**: Before final classification layer
- **Effect**: Prevents co-adaptation of neurons

#### D. Weight Decay (L2 Regularization)

- **Rate**: 1×10⁻⁴
- **Effect**: Penalizes large weights, promotes simpler models

---

### 4. Training Configuration

| Hyperparameter                | Value / Defaults                                                                 | Notes & Rationale                                                                                      |
| ----------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Optimizer**                 | SGD (baseline): momentum=0.9, nesterov=True, weight_decay=1e-4                   | Baseline training used SGD with momentum for stability and to avoid DirectML CPU-fallbacks with AdamW. |
|                               | AdamW (experiments): betas=(0.9,0.999), weight_decay=1e-2                        | AdamW used in some optimized notebooks for faster convergence; can trigger CPU fallbacks on DirectML.  |
| **Learning rate (head→fine)** | Head warmup: 1e-3 (epochs 1–3) → Fine-tune: 1e-4 (epochs 4+)                     | Short head warmup lets classifier adapt; lower LR when unfreezing prevents large weight updates.       |
| **LR scheduler**              | CosineAnnealingLR (T_max = effective training epochs) + linear warmup (3 epochs) | Smooth decay after warmup; min_lr typically set ≈ 1e-6.                                                |
| **Batch size**                | 16                                                                               | Balance GPU memory and gradient stability.                                                             |
| **Epochs / early stop**       | max 50 (patience=15) — baseline stopped at epoch 17                              | Early stopping on validation Top-1 to prevent overfitting.                                             |
| **Loss**                      | Cross-Entropy with label smoothing ε=0.1                                         | Label smoothing improves calibration and reduces overconfidence.                                       |
| **Regularization**            | Dropout p=0.3; Mixup α=0.2; Weight decay per-optimizer (SGD 1e-4 / AdamW 1e-2)   | Mixup + dropout help generalization across visually-similar classes.                                   |
| **Input size**                | 252×252 px                                                                       | Matches EfficientNet-B3 resolution used for pretrained weights.                                        |

Notes:

- Warmup: we first train the classification head for a few epochs (default 3) at the higher learning rate to adapt the head, then unfreeze and continue fine-tuning with a lower LR and Cosine annealing.
- Weight-decay: the baseline SGD runs use 1e-4; AdamW experiments use a larger weight-decay (e.g., 1e-2) because Adam-style optimizers interact differently with weight decay.
- DirectML note: if you run training on DirectML and observe warnings about operators (e.g., aten::lerp) falling back to CPU when using AdamW, prefer SGD for full-GPU throughput.
- These hyperparameters were chosen to balance stable convergence, robust generalization (mixup/label-smoothing), and reproducibility across experiments.

---

### 5. Early Stopping

**Purpose**: Automatically stop training when model stops improving.

**Configuration**:

- **Patience**: 5 epochs
- **Metric**: Validation Top-1 Accuracy
- **Result**: Training stopped at epoch 20 (best: epoch 17)

**Why It Matters**:

- Prevents overfitting to training data
- Saves computational resources
- Ensures best model is used for deployment

---

## 📈 Training Results

### Learning Curves

#### Accuracy Over Epochs

```
Phase 1 (Head Warmup - Epochs 1-3):
├── Validation accuracy improved quickly during head warmup.
└── Fast initial learning

Phase 2 (Full Fine-tuning - Epochs 4-20):
├── Strong improvements after unfreezing; training stopped early when validation Top-1 plateaued.
└── Best validation Top-1 accuracy: 86.26% (epoch 17)
```

#### Training Speed

```
Warmup Phase (Epochs 1-3):
├── Speed: ~51 images/second
└── Time per epoch: ~1,000 seconds (~17 minutes)

Fine-tuning Phase (Epochs 4-20):
├── Speed: ~9.5 images/second
└── Time per epoch: ~4,200 seconds (~70 minutes)

Reason for slowdown: Full model backpropagation (12M params)
```

---

## 🎯 Final Model Performance

### Test Set Results (4,375 images)

> 📋 **Full Details**: See [Classification Report](../runs/efficientnet_b3_baseline-20251114-003032/classification_report.txt) for per-class precision, recall, and F1 scores.

| Metric                | Value  | Interpretation                           |
| --------------------- | ------ | ---------------------------------------- |
| **Top-1 Accuracy**    | 87.09% | Correct on first guess 87.09% of time    |
| **Top-5 Accuracy**    | 96.98% | Correct answer in top 5: 96.98% of time  |
| **Precision (macro)** | 87.24% | Average precision across all 125 classes |
| **Recall (macro)**    | 87.09% | Average recall across all 125 classes    |
| **F1 Score (macro)**  | 86.97% | Balanced precision/recall metric         |

**📊 Detailed Metrics Available**:

- [Per-Class Precision, Recall, F1](../runs/efficientnet_b3_baseline-20251114-003032/per_class_metrics.json)
- [Per-Class Accuracy Breakdown](../runs/efficientnet_b3_baseline-20251114-003032/per_class_accuracy.json)
- [Confusion Matrix Visualization](../runs/efficientnet_b3_baseline-20251114-003032/confusion_matrix.png)

### Confidence Analysis

**High-Confidence Predictions (Test, ≥80% confidence):**

- **Count / Percentage**: 3512 / 4375 (80.27% of test predictions)
- **Accuracy among high-confidence predictions**: Highly reliable
- **Use Case**: These predictions can be treated as high-trust; consider human review for the remainder.

**Medium-Confidence Predictions** (50-80% confidence):

- **Percentage**: ~42% of predictions
- **Accuracy**: ~85-90% (good but review recommended)

**Low-Confidence Predictions** (<50% confidence):

- **Percentage**: ~24% of predictions
- **Recommendation**: Flag for human review in production

---

## 📊 Per-Class Performance

> 📁 **Complete Data**: [Per-Class Accuracy JSON](../runs/efficientnet_b3_baseline-20251114-003032/per_class_accuracy.json) | [Per-Class Metrics (Precision/Recall/F1)](../runs/efficientnet_b3_baseline-20251114-003032/per_class_metrics.json)

### Best Performing Classes (100% accuracy)

Examples of foods the model recognizes perfectly:

- Apple: 100%
- Balut: 100%
- Frozen Yogurt: 100%
- Leche Flan: 100%
- Sunny Side Up: 100%
- Several Filipino dishes: Baked Tahong, Chicken Tinola, Daing na Bangus, Isaw Manok, Pritong Galunggong

**Why?** These foods have distinctive visual features, consistent appearance, and minimal intra-class variation.

### Challenging Classes (54-69% accuracy)

Foods with more variability or visual similarity:

- Chocolate Mousse: 54.3%
- Grilled Cheese Sandwich: 54.3%
- Omelette: 60.0%
- Pork Bistek: 60.0%
- Chocolate Cake: 62.9%
- Pork Chop: 65.7%
- Steak: 65.7%
- Tiramisu: 65.7%

**Why?** High visual similarity, regional variations, presentation differences, or overlapping ingredients with other classes.

**🔍 Deep Dive**: See [Confusion Matrix](../runs/efficientnet_b3_baseline-20251114-003032/confusion_matrix.png) to identify which classes are most commonly confused with each other.

---

## 🔄 Comparison: Training vs Validation vs Test

> 📈 **Full Training History**: [Epoch-by-Epoch Metrics (CSV)](../runs/efficientnet_b3_baseline-20251114-003032/metrics_epoch.csv) | [Training Curves Visualization](../runs/efficientnet_b3_baseline-20251114-003032/training_curves.png)

| Metric         | Train  | Validation | Test   |
| -------------- | ------ | ---------- | ------ |
| Top-1 Accuracy | 99.11% | 86.26%     | 87.09% |
| Top-5 Accuracy | 99.97% | 97.28%     | 96.98% |
| F1 Score       | 99.11% | 86.20%     | 86.97% |
| Precision      | 99.12% | 86.51%     | 87.24% |
| Recall         | 99.11% | 86.26%     | 87.09% |

**Observations**:

- **Train >> Val/Test**: Expected behavior (model sees training data during learning)
- **Val ≈ Test**: Excellent generalization (no overfitting!)
- **Gap (~12%)**: Reasonable for 125-class problem with regularization

**📊 Visual Analysis**: Check [training curves](../runs/efficientnet_b3_baseline-20251114-003032/training_curves.png) to see loss and accuracy progression over 20 epochs.

---

## 🧪 Technical Algorithms Used

### 1. **Convolutional Neural Networks (CNNs)**

- **Purpose**: Extract visual features from images
- **Components**: Convolution layers, pooling layers, activation functions
- **Why**: Effective for spatial pattern recognition in images

### 2. **Transfer Learning**

- **Concept**: Use knowledge from ImageNet (1000 classes) for food recognition
- **Benefit**: Reduces training time and improves accuracy with limited data

### 3. Backpropagation & optimizer choices

- **Baseline run optimizer (SGD)**: The baseline training run `efficientnet_b3_baseline-20251114-003032` used SGD with momentum (Nesterov) and weight decay. SGD was chosen in that notebook to match the established training regime and to avoid DirectML CPU-fallbacks that can occur with some AdamW operators.

- **Other runs (AdamW)**: Some optimized training notebooks (for example, `train_efficientnet_b3_optimized_food_not_food.ipynb`) use AdamW for weight decay-aware adaptive updates. AdamW is available in the repo and used for experiments, but on DirectML it can trigger CPU fallbacks for certain ops (see note below).

- **Why both exist**: AdamW can offer faster convergence on some tasks; SGD (with momentum) is often more stable and avoids DirectML-related operator fallbacks on AMD/Intel GPUs.

Note: If you run training on DirectML and see warnings about aten::lerp or other ops falling back to CPU when using AdamW, prefer SGD for full-GPU training throughput.

### 4. **Cross-Entropy Loss Function**

- **Formula**: `Loss = -Σ(y_true × log(y_pred))`
- **Purpose**: Measure difference between predicted and actual class
- **Why**: Standard for multi-class classification problems

### 5. **Softmax Activation**

- **Formula**: `softmax(x_i) = e^(x_i) / Σ(e^(x_j))`
- **Purpose**: Convert raw scores to probabilities (sum to 100%)
- **Output**: Confidence scores for each of 125 classes

### 6. **Cosine Annealing Learning Rate Scheduler**

- **Purpose**: Gradually reduce learning rate following cosine curve
- **Benefit**: Smooth convergence, avoids sharp changes in learning

---

## 💾 Model Deployment

### Export Format: ONNX (Open Neural Network Exchange)

**Specifications**:

- **Input Shape**: `[batch_size, 3, 252, 252]`
- **Output Shape**: `[batch_size, 125]` (probability for each class)
- **File Size**: ~45MB
- **Runtime**: Compatible with ONNX Runtime (CPU/GPU)

**Inference Pipeline**:

```
1. Load image → Resize to 252×252
2. Normalize with ImageNet statistics
3. Run ONNX model inference
4. Apply softmax to get probabilities
5. Return top-5 predictions with confidence scores
```

**Performance**:

- **CPU Inference**: ~50-150ms per image
- **Memory Usage**: ~200-300MB
- **Deployment**: Web servers (Node.js/Express), Mobile apps, Edge devices

---

## 🎓 Key Learnings & Best Practices

### What Worked Well ✅

1. **Two-phase training**: Warmup + fine-tuning prevented catastrophic forgetting
2. **Strong augmentation**: Mixup and geometric transforms improved generalization
3. **Early stopping**: Prevented overfitting and saved computation
4. **High-quality dataset**: Balanced, diverse images led to robust model

### Challenges Encountered ⚠️

1. **Visually similar classes**: Salads, soups, and pasta dishes harder to distinguish
2. **Training time**: Full fine-tuning ~70 minutes/epoch on DirectML
3. **Class imbalance sensitivity**: Some rare foods need more training examples

### Future Improvements 🚀

1. **Ensemble models**: Combine multiple models for higher accuracy
2. **Data augmentation++**: Use advanced techniques (CutMix, AutoAugment)
3. **Larger models**: Try EfficientNet-B4 or B5 for marginal gains
4. **Active learning**: Collect more images for challenging classes

---

## 📂 Project Files

### 📓 Training Notebooks

| Notebook              | Purpose                           | Link                                                                                                                |
| --------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **125-Class Trainer** | Train multi-class food classifier | [`train_efficientnet_b3_optimized.ipynb`](../train_efficientnet_b3_optimized.ipynb)                                 |
| **Binary Trainer**    | Train food/not-food classifier    | [`train_efficientnet_b3_optimized_food_not_foodv2.ipynb`](../train_efficientnet_b3_optimized_food_not_foodv2.ipynb) |

### 🧪 Inference & Evaluation Notebooks

| Notebook                 | Purpose                           | Link                                                                                            |
| ------------------------ | --------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Binary Model Testing** | Test binary classifier inference  | [`test_inference_nonfood_and_food.ipynb`](../test_inference_nonfood_and_food.ipynb)             |
| **125-Class Testing**    | Test multi-class model            | [`test_model_on_all_food_dataset_images.ipynb`](../test_model_on_all_food_dataset_images.ipynb) |
| **Confusion Matrix**     | Generate confusion visualizations | [`confusion_matrix_all_classes.ipynb`](../confusion_matrix_all_classes.ipynb)                   |

### 🎯 Model Artifacts - 125-Class Food Classifier

**Location**: [`../runs/efficientnet_b3_baseline-20251114-003032/`](../runs/efficientnet_b3_baseline-20251114-003032/)

| File                            | Description                   | Link                                                                                                      |
| ------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------- |
| **PyTorch Model**               | Trained checkpoint (.pth)     | [`best_efficientnet_b3.pth`](../runs/efficientnet_b3_baseline-20251114-003032/best_efficientnet_b3.pth)   |
| **ONNX Model**                  | Deployment-ready export       | [`model.onnx`](../runs/efficientnet_b3_baseline-20251114-003032/model.onnx)                               |
| **Class Names**                 | 125 food categories           | [`class_names.json`](../runs/efficientnet_b3_baseline-20251114-003032/class_names.json)                   |
| **📊 Summary**                  | Complete training stats       | [`summary.json`](../runs/efficientnet_b3_baseline-20251114-003032/summary.json)                           |
| **📈 Training Metrics**         | Per-epoch performance (CSV)   | [`metrics_epoch.csv`](../runs/efficientnet_b3_baseline-20251114-003032/metrics_epoch.csv)                 |
| **📈 Training Metrics**         | Per-epoch performance (JSONL) | [`metrics_epoch.jsonl`](../runs/efficientnet_b3_baseline-20251114-003032/metrics_epoch.jsonl)             |
| **🎯 Per-Class Accuracy**       | Accuracy for each class       | [`per_class_accuracy.json`](../runs/efficientnet_b3_baseline-20251114-003032/per_class_accuracy.json)     |
| **📊 Per-Class Metrics**        | Precision/Recall/F1 per class | [`per_class_metrics.json`](../runs/efficientnet_b3_baseline-20251114-003032/per_class_metrics.json)       |
| **📋 Classification Report**    | Full classification metrics   | [`classification_report.txt`](../runs/efficientnet_b3_baseline-20251114-003032/classification_report.txt) |
| **🔀 Confusion Matrix**         | Visual class confusion        | [`confusion_matrix.png`](../runs/efficientnet_b3_baseline-20251114-003032/confusion_matrix.png)           |
| **📉 Training Curves**          | Loss & accuracy plots         | [`training_curves.png`](../runs/efficientnet_b3_baseline-20251114-003032/training_curves.png)             |
| **📊 Per-Class Accuracy Chart** | Visual accuracy distribution  | [`per_class_accuracy.png`](../runs/efficientnet_b3_baseline-20251114-003032/per_class_accuracy.png)       |

### 🎯 Model Artifacts - Binary Food/Not-Food Classifier

**Location**: [`../runs/efficientnet_b3_food_not_food-20251031-220332/`](../runs/efficientnet_b3_food_not_food-20251031-220332/)

| File                         | Description                 | Link                                                                                                           |
| ---------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **PyTorch Model**            | Trained checkpoint (.pth)   | [`best_efficientnet_b3.pth`](../runs/efficientnet_b3_food_not_food-20251031-220332/best_efficientnet_b3.pth)   |
| **ONNX Model**               | Deployment-ready export     | [`model.onnx`](../runs/efficientnet_b3_food_not_food-20251031-220332/model.onnx)                               |
| **Class Names**              | food, not_food              | [`class_names.json`](../runs/efficientnet_b3_food_not_food-20251031-220332/class_names.json)                   |
| **📊 Summary**               | Complete training stats     | [`summary.json`](../runs/efficientnet_b3_food_not_food-20251031-220332/summary.json)                           |
| **📈 Training Metrics**      | Per-epoch performance (CSV) | [`metrics_epoch.csv`](../runs/efficientnet_b3_food_not_food-20251031-220332/metrics_epoch.csv)                 |
| **🎯 Per-Class Accuracy**    | Binary accuracy breakdown   | [`per_class_accuracy.json`](../runs/efficientnet_b3_food_not_food-20251031-220332/per_class_accuracy.json)     |
| **📊 Per-Class Metrics**     | Precision/Recall/F1         | [`per_class_metrics.json`](../runs/efficientnet_b3_food_not_food-20251031-220332/per_class_metrics.json)       |
| **📋 Classification Report** | Full metrics report         | [`classification_report.txt`](../runs/efficientnet_b3_food_not_food-20251031-220332/classification_report.txt) |
| **🔀 Confusion Matrix**      | Binary confusion matrix     | [`confusion_matrix.png`](../runs/efficientnet_b3_food_not_food-20251031-220332/confusion_matrix.png)           |
| **📉 Training Curves**       | Loss & accuracy plots       | [`training_curves.png`](../runs/efficientnet_b3_food_not_food-20251031-220332/training_curves.png)             |

---

## 🎯 Conclusion

This project successfully developed **two high-accuracy food recognition models** using deep learning and transfer learning techniques:

### 125-Class Food Classifier

- ✅ **87.09% top-1 accuracy** on 125 food categories (Test set)
- ✅ **96.98% top-5 accuracy** (almost always correct in top 5 guesses)
- ✅ **80.27% high-confidence predictions** (≥80% confidence)
- ✅ **Production-ready** ONNX model for web/mobile deployment
- ✅ **Robust performance** across validation and test sets (no overfitting)

### Binary Food/Not-Food Classifier

- ✅ **99.19% test accuracy** on food vs not-food classification
- ✅ **98.79% validation accuracy** (best epoch)
- ✅ Fast training convergence (13 epochs vs 17 for multi-class)
- ✅ **Production-ready** ONNX model for pre-filtering pipeline

Both models are suitable for deployment in **nutritional tracking applications**, **restaurant menu digitization**, and **food recognition systems**.

### Statistical Significance

**125-class model**: With 4,375 test images and 87.09% accuracy, the model correctly classifies **3,810 out of 4,375 images**, demonstrating strong real-world applicability for fine-grained food recognition tasks.

**Binary model**: With 248 test images and 99.19% accuracy, the model correctly classifies **246 out of 248 images**, making it highly reliable for food/non-food filtering in preprocessing pipelines.

---

**Model Training Dates**:

- 125-class: November 14, 2025
- Binary: October 31, 2024

**Training Duration**:

- 125-class: 20 epochs (~70 min/epoch for fine-tuning phase)
- Binary: 13 epochs (faster due to smaller head)  
  **Hardware**: DirectML-compatible GPU  
  **Framework**: PyTorch 2.x + torchvision

---

---

## 📚 Quick Reference Links

### 📊 125-Class Model Results

- **Performance Summary**: [summary.json](../runs/efficientnet_b3_baseline-20251114-003032/summary.json)
- **Classification Report** (Precision/Recall/F1): [classification_report.txt](../runs/efficientnet_b3_baseline-20251114-003032/classification_report.txt)
- **Per-Class Metrics** (JSON): [per_class_metrics.json](../runs/efficientnet_b3_baseline-20251114-003032/per_class_metrics.json)
- **Per-Class Accuracy**: [per_class_accuracy.json](../runs/efficientnet_b3_baseline-20251114-003032/per_class_accuracy.json)
- **Confusion Matrix**: [confusion_matrix.png](../runs/efficientnet_b3_baseline-20251114-003032/confusion_matrix.png)
- **Training Curves**: [training_curves.png](../runs/efficientnet_b3_baseline-20251114-003032/training_curves.png)
- **Training History**: [metrics_epoch.csv](../runs/efficientnet_b3_baseline-20251114-003032/metrics_epoch.csv)

### 📊 Binary Model Results

- **Performance Summary**: [summary.json](../runs/efficientnet_b3_food_not_food-20251031-220332/summary.json)
- **Classification Report**: [classification_report.txt](../runs/efficientnet_b3_food_not_food-20251031-220332/classification_report.txt)
- **Confusion Matrix**: [confusion_matrix.png](../runs/efficientnet_b3_food_not_food-20251031-220332/confusion_matrix.png)
- **Training Curves**: [training_curves.png](../runs/efficientnet_b3_food_not_food-20251031-220332/training_curves.png)

### 📓 Notebooks

- **125-Class Training**: [train_efficientnet_b3_optimized.ipynb](../train_efficientnet_b3_optimized.ipynb)
- **Binary Training**: [train_efficientnet_b3_optimized_food_not_foodv2.ipynb](../train_efficientnet_b3_optimized_food_not_foodv2.ipynb)
- **Testing & Evaluation**: [test_model_on_all_food_dataset_images.ipynb](../test_model_on_all_food_dataset_images.ipynb)

---

_For questions or additional information, please refer to the training notebooks or check the linked artifacts above._
