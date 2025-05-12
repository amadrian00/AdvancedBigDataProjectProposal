# Advanced Big Data Analysis Project

This repository contains the implementation of a deep learning pipeline for molecular property prediction using graph neural networks.

## 📁 Project Structure

- `main.py`:  
  Main execution script. Handles loop logic for running all seeds and 5-fold splits.  
  ⚠️ **Do not modify** the test partition or the final metric reporting sections.


- `model.py`:  
  Contains the definition of the base graph neural network model.


- `train_eval.py`:  
  Implements training and evaluation functions. It also computes all required metrics for performance tracking.

## 📦 Required Libraries

Ensure the following Python packages are installed:

- [`torch`](https://pytorch.org/get-started/locally/)
- [`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [`ogb`](https://ogb.stanford.edu/docs/home/)
- [`tqdm`](https://tqdm.github.io/)
- [`torcheval`](https://docs.pytorch.org/torcheval/stable/)
- [`scipy`](https://scipy.org/install/)
- [`scikit-learn`](https://scikit-learn.org/stable/install.html)


## 🚀 Execution
You can execute the baseline code as follows (replace `[device_id]` with your GPU ID, e.g., `0`). Use `nvidia-smi` to check available devices:

```bash
CUDA_VISIBLE_DEVICES=[device_id] python3 main.py
```

## 📊 Baselines — 5-Fold Average + Paired t-test
### 5-Fold Cross-Validation Results + Paired t-test

| Metric      | Mean BACE | Std BACE | Mean BBBP | Std BBBP | % Inc. | p-value  |
|-------------|-----------|----------|-----------|----------|--------|----------|
| Accuracy    | 0.5508    | 0.0733   | 0.5637    | 0.0607   |  2.34  | 0.5217   |
| Sensitivity | 0.2919    | 0.2877   | 0.2756    | 0.1862   | -5.58  | 0.8245   |
| Specificity | 0.8462    | 0.1937   | 0.8924    | 0.0893   |  5.46  | 0.3154   |
| F1 Score    | 0.3370    | 0.2571   | 0.3724    | 0.1712   | 10.49  | 0.5887   |
| ROC AUC     | 0.6115    | 0.0908   | 0.6637    | 0.0462   |  8.54  | **0.0284** |

> **Note:** *% Inc.* refers to the relative change from BACE to BBBP.  
> Statistically significant results (p < 0.05) are highlighted in **bold**.


## 👤 Contact

If you have any questions or would like to get in touch, feel free to reach out:

📧 **Email:** [amadrian@korea.ac.kr](mailto:amadrian@korea.ac.kr)