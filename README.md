# 🏡 NeXus Estate — House Price Prediction

A machine learning project that predicts median house values using the **Boston Housing Dataset**, built with scikit-learn. The model is trained, evaluated with cross-validation, and exported as a reusable `.joblib` file.

---

## 📌 Project Overview

NeXus Estate uses a **Random Forest Regressor** to predict the median value of owner-occupied homes (`MEDV`) based on 13 socioeconomic and geographic features. The project covers the full ML pipeline: data loading → EDA → preprocessing → model training → evaluation → saving.

---

## 📂 Project Structure

```
NeXus-Estate/
├── ML_proj_1.ipynb       # Main Jupyter Notebook
├── housing.data          # Boston Housing Dataset
├── Nexus_Estate.joblib   # Saved trained model
└── README.md
```

---

## 📊 Dataset

**Boston Housing Dataset** — 506 samples, 13 features + 1 target.

| Feature | Description |
|--------|-------------|
| `CRIM` | Per capita crime rate by town |
| `ZN` | Proportion of residential land zoned for large lots |
| `INDUS` | Proportion of non-retail business acres |
| `CHAS` | Charles River dummy variable (1 if bounds river) |
| `NOX` | Nitric oxide concentration |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of owner-occupied units built before 1940 |
| `DIS` | Distances to employment centres |
| `RAD` | Index of accessibility to radial highways |
| `TAX` | Property tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio |
| `B` | Proportion of Black residents by town |
| `LSTAT` | % lower status of the population |
| **`MEDV`** | **Target — Median value of homes in $1000s** |

---

## ⚙️ ML Pipeline

1. **Data Loading & Cleaning** — Parsed whitespace-delimited `.data` file with custom headers
2. **Exploratory Data Analysis** — Correlation matrix, scatter matrix, feature plots
3. **Feature Engineering** — Created `TAXRM` (Tax-to-Room ratio)
4. **Stratified Train/Test Split** — 70/30 split stratified on `CHAS`
5. **Preprocessing Pipeline** — Median imputation + StandardScaler via `sklearn.pipeline`
6. **Model Training** — `RandomForestRegressor` (also experimented with Linear Regression and Decision Tree)
7. **Evaluation** — RMSE + 10-fold Cross Validation
8. **Model Export** — Saved with `joblib`

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn joblib
```

### Run the Notebook

```bash
git clone https://github.com/your-username/NeXus-Estate.git
cd NeXus-Estate
jupyter notebook ML_proj_1.ipynb
```

### Load the Saved Model

```python
from joblib import load
import numpy as np

model = load('Nexus_Estate.joblib')
# Pass a preprocessed sample to predict
# prediction = model.predict(prepared_data)
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Model | Random Forest Regressor |
| Evaluation | 10-Fold Cross Validation |
| Metric | RMSE (Root Mean Squared Error) |

> Final RMSE on test set printed at the end of the notebook.

---

## 🛠️ Tech Stack

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 📚 References

- Tutorial: [Machine Learning with Python – Sadguru / YouTube](https://youtu.be/iIkJrwVUl1c)
- Dataset: [Boston Housing Dataset (UCI / StatLib)](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
