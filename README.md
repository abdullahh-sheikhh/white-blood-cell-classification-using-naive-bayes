# Blood Cell Classification using Gaussian Naive Bayes

University course project (M1 BIMA, Sorbonne) -- classifying white blood cells into 4 types using handcrafted image features and a Gaussian Naive Bayes classifier.

## Dataset

[Blood Cells Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) by Paul Mooney (sourced from [BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset)).

The dataset contains **12,500 augmented JPEG images** of blood cell microscopy, organized into 4 classes of white blood cells:

| Class | Train | Test | Total |
|---|---|---|---|
| Eosinophil | 2,497 | 623 | 3,120 |
| Lymphocyte | 2,483 | 620 | 3,103 |
| Monocyte | 2,478 | 620 | 3,098 |
| Neutrophil | 2,499 | 624 | 3,123 |
| **Total** | **9,957** | **2,487** | **12,444** |

Each image shows a stained blood smear containing a central white blood cell surrounded by red blood cells. The original dataset had only 410 images (with XML bounding box annotations); the augmented version used here (`dataset2-master`) provides balanced classes with ~3,000 images each.

## Approach

### 1. WBC Segmentation

White blood cells are isolated from background red blood cells using HSV color thresholding + morphological operations. This mask focuses feature extraction on the cell of interest.

### 2. Feature Extraction

Several feature types were explored, both on the full image and on the segmented WBC region:

- **Color histograms** -- RGB and HSV distributions at various bin sizes
- **Texture** -- Local Binary Patterns (LBP)
- **Higher-order color statistics** -- mean, std, skewness of HSV channels
- **Cell morphology** -- nucleus lobe count, circularity, solidity, granule count, nucleus-to-cytoplasm ratio, cell size

### 3. Classification

All features are standardized (StandardScaler) and fed into a **Gaussian Naive Bayes** classifier. Evaluated with train/test split and stratified K-fold cross-validation.

## Results

### Individual Features

| Feature | Train Acc | Test Acc |
|---|---|---|
| Mean RGB | 30.37% | 25.69% |
| RGB Histogram (48 bins) | 50.17% | 42.74% |
| HSV Histogram (64 bins) | 68.64% | 55.73% |
| LBP Texture (10 bins) | 35.49% | 31.44% |
| HSV Histogram Masked (64 bins) | 71.03% | 60.27% |
| LBP Texture Masked | 50.87% | 49.42% |
| Color Stats Masked | 57.07% | 57.14% |
| Cell Morphology | 41.51% | 44.95% |

### Feature Combinations

| Combination | Train Acc | Test Acc |
|---|---|---|
| Masked HSV + Cell Morphology | 71.52% | 60.92% |
| Masked HSV + Masked LBP | 73.39% | 65.10% |
| **All Combined** (LBP + HSV + Stats + Morphology) | **73.37%** | **65.26%** |

### K-Fold Cross-Validation (All Combined Features)

| K | Train Acc | Validation Acc |
|---|---|---|
| 5 | 72.59% (+/-0.39%) | **72.21% (+/-0.73%)** |
| 10 | 72.55% (+/-0.45%) | **72.11% (+/-0.91%)** |

### Key Takeaways

- **WBC segmentation helps**: masking consistently improves accuracy (e.g., HSV histogram: 55.73% -> 60.27%)
- **HSV > RGB**: HSV color space captures cell staining differences better than RGB
- **Feature combination matters**: combining color, texture, and morphology yields the best results
- **K-fold validation** shows stable ~72% accuracy with low variance, indicating the model generalizes well for a simple classifier

## Tech Stack

Python, NumPy, OpenCV, scikit-learn, scikit-image, Matplotlib, Seaborn

## How to Run

Open `IMA_Project.ipynb` in Google Colab or Jupyter. The dataset is downloaded automatically via `kagglehub`.
