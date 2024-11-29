# Hyperparameter Optimization with Optuna

![GitHub Marketplace](https://img.shields.io/badge/marketplace-hyperparameter--optimization-blue?logo=github)

Perform hyperparameter optimization for Python models using Optuna. This action supports both classification and regression tasks, dynamically selects hyperparameters, and saves the optimized model as an artifact.

---

## Features

- Supports classification and regression models.
- Automatically tunes hyperparameters using [Optuna](https://optuna.org/).
- Dynamically fetches datasets and models via user-specified methods.
- Saves the optimized model as an artifact for further use.
- Customizable input parameters to suit your workflow.

---

## Inputs

| Name              | Required | Default                | Description                                              |
|-------------------|----------|------------------------|----------------------------------------------------------|
| `script_path`     | ✅       | -                      | Path to the script containing `get_data` and `get_model` methods. |
| `get_data_method` | ❌       | `get_data`             | Name of the method in the script to fetch the dataset.   |
| `get_model_method`| ❌       | `get_model`            | Name of the method in the script to fetch the model.     |
| `n_trials`        | ❌       | `20`                   | Number of trials for hyperparameter optimization.        |
| `n_jobs`          | ❌       | `-1`                   | Number of parallel jobs for Optuna optimization.         |
| `artifact_name`   | ❌       | `optimized_model.pkl`  | Name of the output artifact containing the optimized model. |

---

## Outputs

| Name             | Description                                               |
|------------------|-----------------------------------------------------------|
| `best_params`    | Best hyperparameters found during optimization.           |
| `best_accuracy`  | Best accuracy (or equivalent score) achieved during optimization. |

---

## Example Usage

Here’s an example workflow to use this action:

```yaml
name: Hyperparameter Optimization

on:
  push:
    branches:
      - main

jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v3

    - name: Run Hyperparameter Optimization
      uses: your-username/your-repo-name@v1.0
      with:
        script_path: "scripts/my_model.py"
        get_data_method: "get_data"
        get_model_method: "get_model"
        n_trials: 50
        n_jobs: 4
        artifact_name: "final_model.pkl"

    - name: Upload Optimized Model Artifact
      uses: actions/upload-artifact@v3
      with:
        name: optimized_model
        path: final_model.pkl
```
## Requirements

- **Python 3.9 or later**
- **Dependencies**:
  - `optuna`
  - `scikit-learn`
  - `joblib`
  - `numpy`
  - `pandas`

---

## How It Works

1. **Input Script**: Provide a Python script containing:
   - A `get_data` method to return the dataset as `(X, y)`.
   - A `get_model` method to return an uninitialized model (e.g., `RandomForestClassifier`).

2. **Optimization**: The action dynamically loads the script, retrieves the data and model, and performs hyperparameter optimization using Optuna.

3. **Artifact Output**: After optimization, the best model is trained on the entire dataset and saved as an artifact.

---

## About

This action simplifies machine learning workflows by automating the hyperparameter optimization process. It is ideal for both classification and regression tasks in Python.

For more details, see the [GitHub Actions Documentation](https://docs.github.com/en/actions).
