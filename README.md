# Scene Classification Project: Classical ML and CNN Approaches

This project implements scene recognition to classify images into three categories: **Library**, **Museum**, and **Shopping Mall**, using classical machine learning techniques and convolutional neural networks (CNNs).

---

## Project Overview

The project is divided into two phases:

* **Phase 1: Classical Machine Learning (ML)** – Feature engineering (HOG, LBP, Color Histograms, and Bag-of-Visual-Words) combined with ML algorithms (Random Forest, SVM, Decision Trees, semi-supervised learning).
* **Phase 2: Deep Learning (CNNs)** – CNN architectures (Custom CNN, Compact VGG, Hybrid model) trained with extensive data augmentation and hyperparameter tuning.

---

## Dataset

The dataset used is a balanced subset from the MIT Places dataset:

* **Classes:** Library indoor, Museum indoor, Shopping Mall indoor
* **Training set:** 5,000 images per class
* **Validation set:** 500 images per class (auto-split from training set)
* **Test set:** 100 images per class

**Download and organize the dataset:**

* [MIT Places Official Site](http://places.csail.mit.edu/downloadData.html)

**Folder Structure:**

```
data/
├── Training/
│   ├── library-indoor/
│   ├── museum-indoor/
│   └── shopping_mall-indoor/
└── Test/
    ├── library-indoor/
    ├── museum-indoor/
    └── shopping_mall-indoor/
```

---

## Requirements

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib torchvision torch opencv-python-headless scikit-image
```

---

## Phase 1: Classical Machine Learning

### Feature Extraction and Training

Run these notebooks in order to reproduce results:

1. **Feature Extraction (BoVW & LBP):**

   ```
   BoVW_+_LBP_Feature_Extraction.ipynb
   ```

2. **Feature Fusion and Model Training:**

   ```
   Project_Applied_AI.ipynb
   ```

### Classical ML Models:

* **Random Forest**
* **Linear and RBF SVM**
* **Decision Trees (semi-supervised)**

### Pre-trained Classical ML Models:

Models available in the `models/` folder:

* `best_rf.joblib`
* `best_svm_linear.joblib`
* `best_rbf_svm.joblib`
* `semisup_tree.joblib`

### Single-Image Prediction Demo (Phase 1):

Use the notebook to load a model and predict on a single image:

```
Load_and_predict_on_a_single_image.ipynb
```

---

## Phase 2: Convolutional Neural Networks (CNN)

### CNN Architectures Implemented:

* **Custom CNN (CNN4x4)** – Lightweight architecture (1.15M params)
* **Compact VGG** – Deeper, VGG-style CNN (1.17M params)
* **Hybrid Model** – Balanced depth and performance (0.40M params)

### CNN Training Procedure:

* **Optimizer:** Adam (initial lr = 1e-3, weight decay = 1e-4)
* **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3 epochs)
* **Loss:** Cross-Entropy with label smoothing (ε=0.05)
* **Batch size:** 64
* **Epochs:** 40 (early stopping at patience=6)

### Notebooks for CNN training:

* **CNN Training, Hyperparameter Tuning, and Evaluation:**

```
Scene Classification with CNNs.ipynb
```

### CNN Results & Trained Models:

Final models in the `models/` folder:

* `CNN_bs64_lr3e-04.pth` (baseline CNN)
* `CompactVGG_bs64_lr3e-04.pth`
* `HybridModel_bs64_lr3e-04.pth`
* `CNN_bs64_lr3e-04 best_from_grid_search.pth` (tuned CNN)

### Single-Image CNN Prediction Demo (Phase 2):

Quickly load and run inference on any single image:

```
CNN-Load and predict on a single image.ipynb
```

---

## Project Structure

The repository is organized clearly for reproducibility:

```
Scene-Classification-Project/
├── BoVW_+_LBP_Feature_Extraction.ipynb
├── Project_Applied_AI.ipynb (Phase 1 Feature Fusion and Model Training)
├── Load_and_predict_on_a_single_image.ipynb (Phase 1 ML Demo)
├── CNN-Load and predict on a single image.ipynb (Phase 2 CNN Demo)
├── Scene Classification with CNNs.ipynb (CNN training & tuning)
├── vocab_kmeans_600.joblib
├── models/
│   ├── best_rf.joblib
│   ├── best_svm_linear.joblib
│   ├── best_rbf_svm.joblib
│   ├── semisup_tree.joblib
│   ├── CNN_bs64_lr3e-04.pth
│   ├── CompactVGG_bs64_lr3e-04.pth
│   ├── HybridModel_bs64_lr3e-04.pth
│   └── CNN_bs64_lr3e-04 best_from_grid_search.pth
├── README.md
└── requirements.txt
```

---

## Performance Summary

**Classical ML vs. CNN Performance (Test Set Accuracy):**

| Model                   | Train Acc (%) | Val Acc (%) | Test Acc (%) |
| ----------------------- | ------------- | ----------- | ------------ |
| **RBF SVM**             | 98.8          | 73.5        | 75.3         |
| **Linear SVM**          | 86.8          | 67.5        | 66.0         |
| **Random Forest**       | 88.8          | 66.4        | 71.7         |
| **DT (adaptive conf.)** | 61.9          | 59.4        | 61.0         |
| **DT (fixed conf.)**    | 63.1          | 55.5        | 60.0         |
| **CNN4x4 (Baseline)**   | 95.2          | 86.1        | **87.3**     |
| **CompactVGG**          | 95.0          | 86.1        | 86.3         |
| **HybridModel**         | 91.1          | 84.6        | 86.0         |
| **CNN4x4 (Tuned)**      | 93.9          | 85.8        | **88.7**     |

CNN models demonstrate significantly improved performance over classical ML approaches, due to their capability of automatic and hierarchical feature extraction, providing robustness to variability in the scene images.
---

Feel free to explore the notebooks and trained models to replicate our results and further extend the work!

