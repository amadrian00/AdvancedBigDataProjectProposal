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

You can execute the baseline code as follows (replace `[device_id]` with your GPU ID, e.g., `0`). Use `nvidia-smi` to check available devices:

```bash
CUDA_VISIBLE_DEVICES=[device_id] python3 main.py
```

## Baselines — 5-Fold Average + Paired t-test

| Metric      | Mean BACE | Std BACE | Mean BBBP | Std BBBP | % Inc. | p-value |
|-------------|-----------|----------|-----------|----------|--------|---------|
| Accuracy    | 0.5482    | 0.0740   | 0.5118    | 0.0773   | -6.63  | 0.0603  |
| Sensitivity | 0.3481    | 0.3646   | 0.3689    | 0.1907   |  5.96  | 0.7971  |
| Specificity | 0.7763    | 0.2865   | 0.6749    | 0.2055   | -13.06 | 0.1767  |
| F1 Score    | 0.3480    | 0.2946   | 0.4221    | 0.1493   | 21.27  | 0.2372  |
| ROC AUC     | 0.5923    | 0.1213   | 0.5140    | 0.0942   | -13.22 | **0.0078**  |

> **Note:** *% Inc.* refers to the relative change from BACE to BBBP.  
> Statistically significant results (p < 0.05) are highlighted in **bold**.


## Contact

If you have any questions or would like to get in touch, feel free to reach out:

📧 **Email:** [amadrian@korea.ac.kr](mailto:amadrian@korea.ac.kr)