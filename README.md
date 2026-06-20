# 🏠 Housing Price Prediction — EDA & Regression Models

Two end-to-end data science projects on classic housing datasets:
- **Boston Housing** — Full ML Pipeline with Feature Importance Analysis (R² = 0.80)
- **California Housing** — Multi-model Comparison (R² = 0.79)

---

## 📁 Project 1: Boston Housing — Full ML Pipeline

### Dataset
- 506 records, 13 features
- Target: `medv` (Median house value in $1000s)
- Source: BostonHousing.csv

### ML Pipeline
1. **EDA** — ydata-profiling auto report, correlation heatmap, distributions
2. **Preprocessing** — Median imputation (missing `rm`), IQR outlier capping, RobustScaler
3. **Train/Test Split** — 80/20 (404 train, 102 test)
4. **Model**: RandomForestRegressor

### Results

| Metric | Score |
|--------|-------|
| **R² Score** | **0.80** |
| MSE | 14.68 |
| MAE | 2.46 |

### 🔑 Feature Importance

| Feature | Importance | Description |
|---------|-----------|-------------|
| `rm` | **46%** | Average number of rooms |
| `lstat` | **36%** | % lower status population |
| `dis` | 6% | Distance to employment centers |
| `crim` | 4% | Crime rate |
| Others | 8% | Remaining features |

**Key insight**: Room count and socioeconomic status explain **82%** of house price variation.

---

## 📁 Project 2: California Housing — Multi-model Comparison

### Dataset
- 20,640 records, 8 features
- Target: Median house value in $100,000s
- Source: `sklearn.datasets.fetch_california_housing()`

### ML Pipeline
1. **EDA** — Correlation heatmap, feature distributions, skewness/kurtosis
2. **Preprocessing** — IQR outlier capping, RobustScaler
3. **Train/Test Split** — 80/20 (16,512 train, 4,128 test)

### Model Comparison

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
| Visualization | Matplotlib, Seaborn |
| Auto EDA | ydata-profiling, Sweetviz |
| ML Models | Scikit-learn (RandomForest, GradientBoosting, ExtraTrees, LinearSVR) |
| Tuning | GridSearchCV, RandomizedSearchCV |
| Preprocessing | SimpleImputer, RobustScaler, IQR Outlier Capping |

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn missingno ydata-profiling sweetviz
```

Run notebooks:
1. `BostonHousing_V3_submite.ipynb` — Boston full pipeline
2. `california_housing_random_regression_week5_2.ipynb` — California multi-model

---

## 📁 Project Structure

```
housing-price-prediction/
│
├── BostonHousing_V3_submite.ipynb                     # Boston — R²=0.80
├── california_housing_random_regression_week5_2.ipynb # California — R²=0.79
├── BostonHousing.csv                                  # Boston dataset
└── README.md
```

---

## 👩‍💻 Author

**Angela** — Data Scientist | AI • ML • GenAI • RAG  
📍 Kuala Lumpur, Malaysia  
🔗 [GitHub](https://github.com/angelaadida)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
