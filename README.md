# Medical Visual Question Answering on Gastrointestinal Tract

## Introduction

This project presents a comprehensive pipeline for Medical Visual Question Answering (MedVQA) on gastrointestinal (GI) tract images. The solution leverages an advanced Attention-on-Attention (AoA) model, enhanced with machine learning (ML) ensemble techniques, to achieve improved performance in answering medical questions related to endoscopic images. The pipeline covers data preprocessing, the core PAoAT model implementation, and a sophisticated ensemble strategy using traditional ML models.

## Pipeline Overview

The entire pipeline is structured into three main phases:

1.  **Data Preprocessing**: Cleaning and transforming raw endoscopic images to remove artifacts and extract relevant regions.
2.  **PAoAT Model Execution**: Training and evaluating the core Attention-on-Attention (PAoAT) model for VQA.
3.  **Performance Enhancement with ML Models**: Combining predictions from multiple PAoAT model variants using an intelligent ensemble strategy with ML classifiers to boost overall accuracy.

## Data Acquisition

The datasets used in this project, particularly the raw image data and preprocessed images, are hosted on Google Drive due to their size.

To request access to the data, please send an email to:
**22520968@gm.uit.edu.vn** and **22520224@gm.uit.edu.vn**

## Setup

To set up the environment and run the code, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd Code-MedVQA
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Required Libraries:**
    All necessary Python libraries and their exact versions are listed in `requirements.txt`. You can install them by running:

    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have a compatible CUDA installation if you intend to use GPU for training. The notebooks include checks for GPU availability.*

## Step-by-Step Execution Guide

The pipeline is designed to be executed sequentially through the provided Jupyter notebooks.

### 1. Data Preprocessing (`image_preprocessing.ipynb`)

This notebook is responsible for preparing the raw endoscopic images by performing various cleaning and masking operations.

*   **Purpose**: To remove visual artifacts like highlights, surgical instruments, text boxes, and black frames, which can interfere with the VQA model's performance.
*   **Key Operations**:
    *   Mounting Google Drive to access raw image data and pre-trained models.
    *   Installing `paddleocr` for text detection and `segmentation_models` for instrument detection.
    *   Implementing `EndoscopyImageProcessor` to:
        *   Create masks for highlights (bright regions), instruments, and text.
        *   Utilize `simple-lama-inpainting` and OpenCV's `inpaint` for effective removal of masked areas.
        *   Detect and remove black frames and other border artifacts.
    *   Saving intermediate and final processed images.
*   **Execution**:
    *   Open `image_preprocessing.ipynb` in a Jupyter environment.
    *   Run all cells sequentially. This will download necessary data (if not already present), preprocess the images, and save the outputs to designated folders (e.g., `processed_images`, `masks`).

### 2. PAoAT Model Execution (`BM_PAoAT(best).ipynb`)

This notebook implements, trains, and evaluates the core Attention-on-Attention (PAoAT) model for medical visual question answering.

*   **Purpose**: To answer questions based on the visual content of processed endoscopic images by integrating information from both text (questions) and images.
*   **Key Components**:
    *   **Data Preparation**: Loading question-answer pairs and image metadata, performing answer type filtering, and splitting data into training, validation, and test sets.
    *   **Model Architecture**: The `MultimodalVQAModel2` class integrates a pre-trained text encoder (e.g., BioBERT) and a pre-trained image encoder (e.g., BeiT). It uses custom `SAoA` (Self-Attention-on-Attention) and `GAoA` (Gated-Attention-on-Attention) modules for effective multimodal feature fusion.
    *   **Multimodal Collator**: Handles tokenization of questions and featurization of images for batch processing during training.
    *   **Training**: Configures training arguments (`TrainingArguments` from Hugging Face Transformers) and uses the `Trainer` class for model training, including logging and evaluation.
    *   **Evaluation**: Computes accuracy, precision, recall, and F1-score on validation and test sets.
*   **Execution**:
    *   Ensure that the preprocessing step (`image_preprocessing.ipynb`) has been completed and processed images are available.
    *   Open `BM_PAoAT(best).ipynb`.
    *   Run all cells sequentially. This will download datasets (if not already present), prepare the data, initialize and train the PAoAT model, and save model checkpoints. The final cells will evaluate the model's performance.

### 3. Performance Enhancement with ML Models (`PAoAT_ML.ipynb`)

This notebook demonstrates how to further enhance the performance of the PAoAT model by ensembling predictions from different variants of the model using traditional machine learning classifiers.

*   **Purpose**: To combine the strengths of multiple PAoAT models (trained on different preprocessed image variants) to achieve higher overall VQA performance.
*   **Key Strategies**:
    *   **Loading Variant Predictions**: Loads pre-computed probability predictions from various PAoAT model runs (e.g., trained on `original`, `blackmask`, `highlight`, and `final` processed images).
    *   **Advanced Variant Selector**: The `AdvancedVariantSelector` class trains a `RandomForestClassifier` for each output label. This classifier learns to predict which PAoAT variant provides the best prediction for a given sample, based on "meta-features" extracted from the variants' predictions (e.g., confidence, entropy, agreement).
    *   **Ultra-Enhanced Meta-Stacking**: The `UltraEnhancedMetaStacking` class orchestrates the ensemble. It trains variant-specific XGBoost classifiers on the PAoAT model's output probabilities. It also uses Optuna for hyperparameter tuning and optimizes an adaptive ensemble strategy with confidence-based weighting and label-specific thresholds.
*   **Execution**:
    *   Ensure that the PAoAT model execution step (`BM_PAoAT(best).ipynb`) has been completed and that predictions from different variants (if applicable, these might be saved as `val_preds.csv` and `test_preds.csv` in specific directories) are available.
    *   Open `PAoAT_ML.ipynb`.
    *   Run all cells sequentially. This will load the predictions, train the ensemble models, and output the final enhanced performance metrics.

## Results and Evaluation

The performance of the models is primarily evaluated using standard multi-label classification metrics:
*   **Accuracy**
*   **Precision (samples average)**
*   **Recall (samples average)**
*   **F1-score (samples average)**

The `BM_PAoAT(best).ipynb` and `PAoAT_ML.ipynb` notebooks provide detailed evaluations and visualizations of these metrics.

## Contact

For any questions regarding the dataset or the project, please reach out to:
**22520968@gm.uit.edu.vn** or **22520224@gm.uit.edu.vn**
