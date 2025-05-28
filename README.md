# Glomeruli Segmentation Project

This project focuses on the segmentation of glomeruli in kidney tissue images using deep learning models. It includes scripts for data preprocessing, model training (U-Net and Mask R-CNN), and evaluation.

## Project Structure

The project is organized as follows:

```text
.
├── check_env_versions_1.py       # Script to check Python and library versions
├── data_preprocess/              # Scripts and raw data for preprocessing
│   ├── 256x256-glo-patching.ipynb
│   ├── data_select.ipynb
│   ├── hubmap-kidney-segmentation/ # Example raw dataset source
│   ├── mask.zip
│   ├── masks/
│   ├── train/
│   └── train.zip
├── datasets/                     # Processed and structured datasets
│   ├── test/
│   ├── train/
│   └── val/
├── evaluation/                   # Notebooks for model evaluation
│   ├── evaluation_maskrcnn.ipynb
│   └── evaluation_unet.ipynb
├── maskrcnn_train/               # Scripts and outputs for Mask R-CNN training
│   ├── outputs/
│   └── scripts/
├── outputs_models/               # Saved trained model weights
└── unet_train/                   # Notebook for U-Net model training
    └── unet_train.ipynb

## Overview

The project is organized into several key components:

* **Data Preprocessing (`data_preprocess/`):** Contains notebooks and potentially scripts for preparing the raw image data and masks into a usable format. This might include patching, resizing, and organizing data.
* **Datasets (`datasets/`):** This directory holds the final, structured datasets (images and corresponding masks) split into training, validation, and test sets, ready to be consumed by the model training scripts.
* **Model Training:**
    * **U-Net (`unet_train/unet_train.ipynb`):** Contains the Jupyter Notebook for training the U-Net segmentation model.
    * **Mask R-CNN (`maskrcnn_train/`):** Contains scripts and resources related to training a Mask R-CNN model (likely for instance segmentation).
* **Model Evaluation (`evaluation/`):** Jupyter Notebooks for evaluating the performance of the trained U-Net and Mask R-CNN models on the test set. This includes calculating metrics like IoU and Dice score, and visualizing predictions.
* **Saved Models (`outputs_models/`):** This directory is intended to store the weights of the trained models.
* **Environment Check (`check_env_versions_1.py`):** A utility script to verify the versions of Python and critical libraries in the current environment.

## Getting Started (General Steps)

1.  **Setup Environment:** Ensure you have a Python environment with all necessary libraries installed (PyTorch, Torchvision, OpenCV, Albumentations, etc.). You can use `check_env_versions_1.py` to help verify.
2.  **Data Preparation:**
    * Place raw datasets as needed (e.g., in `data_preprocess/hubmap-kidney-segmentation/`).
    * Run preprocessing scripts/notebooks from `data_preprocess/` to generate the structured `datasets/` directory.
3.  **Model Training:**
    * Navigate to `unet_train/` and run `unet_train.ipynb` to train the U-Net model.
    * (Similarly for Mask R-CNN if applicable).
    * Ensure model weights are saved to `outputs_models/`.
4.  **Model Evaluation:**
    * Navigate to `evaluation/` and run `evaluation_unet.ipynb` or `evaluation_maskrcnn.ipynb` to assess model performance, ensuring the `MODEL_PATH` in these notebooks points to your saved models.

---

### Model Training:

* **U-Net:**
    To train the U-Net model, navigate to the `unet_train/` directory. Open and run the cells within the `unet_train.ipynb` Jupyter Notebook. You may need to configure paths and GPU settings directly inside the notebook.

* **Mask R-CNN:**
    To train the Mask R-CNN model, first change your directory to `maskrcnn_train/`. The training can then be initiated using a command similar to the example below. **Important:** You must modify the Conda environment path (`-p /tempory/21410437/conda_envs/maskrcnn_pth` in the example) to point to **your specific Conda environment location**. The `nohup` command is used here to run the process in the background and save output to `nohup.out`; its use is optional and depends on your requirements.

    ```bash
    cd maskrcnn_train/
    # Example command (modify environment path and nohup usage as needed):
    nohup conda run -p /tempory/21410437/conda_envs/maskrcnn_pth \
        --no-capture-output \
        python -u -m scripts.train \
        > nohup.out 2>&1 &
    ```
    Ensure that any necessary configurations for the Mask R-CNN training script (e.g., dataset paths, hyperparameters) are correctly set within the `scripts.train` module or its associated configuration files.

After training, ensure your model weights are saved, typically to the `outputs_models/` directory.

This README provides a basic outline. You can expand it with more details on specific preprocessing steps, model architectures, training parameters, and exact commands as your project develops.