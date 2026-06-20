# 🏠 Housing Price Prediction — EDA & Regression Models

Two end-to-end data science projects on classic housing datasets:
- **Boston Housing** — In-depth Exploratory Data Analysis (EDA)
- **California Housing** — Machine Learning Regression (R² = 0.79)

---

## 📁 Project 1: Boston Housing — EDA

### Dataset
- 506 records, 14 features
- Target: `medv` (Median house value in $1000s)

### What's covered
- Data loading, shape, info, describe
- Missing value detection & handling (fillna, replace)
- Sorting, indexing (loc/iloc), groupby, pivot_table, crosstab
- Feature engineering with `apply()` and `map()`
- Automated EDA reports using **ydata-profiling** and **sweetviz**

---

## 📁 Project 2: California Housing — Regression Models

### Dataset
- 20,640 records, 8 features (loaded from `sklearn.datasets`)
- Target: Median house value in $100,000s

### ML Pipeline
1. **EDA** — Correlation heatmap, feature distributions, skewness/kurtosis
2. **Outlier Detection** — IQR method
3. **Preprocessing** — IQR capping + RobustScaler
4. **Train/Test Split** — 80/20

### Model Results

| Model | R² Score | MSE | MAE |
|-------|----------|-----|-----|
| **RandomForestRegressor** | **0.79** | **0.27** | **0.34** |
| ExtraTreesRegressor | 0.78 | 0.29 | 0.36 |
| GradientBoostingRegressor | 0.77 | 0.30 | 0.38 |
| LinearRegression | 0.64 | 0.47 | 0.51 |
| LinearSVR | 0.63 | 0.48 | 0.50 |

**Best model**: RandomForestRegressor with **R² = 0.79**

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Missingno |
| Auto EDA | ydata-profiling, Sweetviz |
| ML Models | Scikit-learn (RandomForest, GradientBoosting, ExtraTrees, LinearSVR) |
| Preprocessing | RobustScaler, IQR Outlier Capping |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn missingno ydata-profiling sweetviz
```

Run notebooks in order:
1. `EDA1_BostonHousing_24Sept.ipynb` — EDA
2. `california_housing_random_regression_week5_2.ipynb` — ML Modeling

---

## 📁 Project Structure

```
housing-price-prediction/
│
├── EDA1_BostonHousing_24Sept.ipynb                    # Boston EDA
├── california_housing_random_regression_week5_2.ipynb # California ML
├── BostonHousing.csv                                  # Boston dataset
└── README.md
```

> California Housing dataset is loaded directly from `sklearn.datasets` — no download needed.

---

## 👩‍💻 Author

**Angela** — Data Scientist | AI • ML • GenAI • RAG  
📍 Kuala Lumpur, Malaysia  
🔗 [GitHub](https://github.com/angelaadida)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
