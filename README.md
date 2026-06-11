# 🏡 NeXus Estate — House Price Prediction

A machine learning project that predicts median house values using the **Boston Housing Dataset**, built with scikit-learn. Deployed as a live web app with a FastAPI backend and a custom HTML/CSS/JS frontend.

🔗 **Live Demo:** [nexus-estate-cf27.onrender.com](https://nexus-estate-cf27.onrender.com)

---

## 📸 Preview

> Fill in the form with property details and get an instant price estimate.

<img width="745" height="791" alt="image" src="https://github.com/user-attachments/assets/5714692e-6ab7-4acb-bdf4-1f3800f40380" />


**Sample prediction** — First row of the Boston Housing Dataset:

| Field | Value | Field | Value |
|-------|-------|-------|-------|
| CRIM | 0.00632 | RAD | 1 |
| ZN | 18.0 | TAX | 296 |
| INDUS | 2.31 | PTRATIO | 15.3 |
| CHAS | 0 | B | 396.9 |
| NOX | 0.538 | LSTAT | 4.98 |
| RM | 6.575 | AGE | 65.2 |
| DIS | 4.09 | **Predicted MEDV** | **~$29,000** |

---

## 📌 Project Overview

NeXus Estate uses a **Random Forest Regressor** to predict the median value of owner-occupied homes (`MEDV`) based on 13 socioeconomic and geographic features. The project covers the full ML pipeline: data loading → EDA → preprocessing → model training → evaluation → deployment.

---

## 📂 Project Structure

```
NeXus-Estate/
├── app.py                # FastAPI backend
├── index.html            # Frontend UI
├── ML_proj_1.ipynb       # Jupyter Notebook (training)
├── housing.data          # Boston Housing Dataset
├── Nexus_Estate.joblib   # Saved trained model
├── requirements.txt
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

## 📈 Results

| Metric | Value |
|--------|-------|
| Model | Random Forest Regressor |
| Evaluation | 10-Fold Cross Validation |
| Metric | RMSE (Root Mean Squared Error) |

---

## 🌐 Deployment

- **Backend:** FastAPI served via Uvicorn
- **Frontend:** Vanilla HTML/CSS/JS
- **Hosting:** [Render.com](https://render.com) (free tier)

### Run locally

```bash
git clone https://github.com/OmKadam989/NeXus-Estate.git
cd NeXus-Estate
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000`

---

## 🛠️ Tech Stack

- Python 3.x · Pandas · NumPy · Matplotlib
- Scikit-learn · Joblib
- FastAPI · Uvicorn
- HTML / CSS / JavaScript
- Render (deployment)

---

## 📚 References

- Tutorial: [Machine Learning with Python – Sadguru / YouTube](https://youtu.be/iIkJrwVUl1c)
- Dataset: [Boston Housing Dataset (UCI / StatLib)](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
