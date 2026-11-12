## Overview

Autism Spectrum Disorder (ASD) Prediction is a machine learning project that estimates the likelihood of ASD traits from demographic details, clinical history and the 10 behavioural screening questions (AQ-10). The trained model is served through **Streamlit** web app.


**Live demo:** https://autism-prediction-system-machine-learning-cefdcyocusjcaux7ioy4.streamlit.app/

---

## Features

- Dark-theme Streamlit interface with three sections: **Demographics**, **Clinical History** and **AQ-10 behavioural questions**
- Requires every question to be answered before the **Predict** button gives a result
- Shows the prediction and the model's probability of ASD traits
- Leak-free training pipeline (SMOTE applied only inside cross-validation folds)
- Model chosen by cross-validated F1 on the ASD class, with ROC-AUC as tie-breaker

---

## Project Structure

```
Autism Correction Machine Learning Project/
│
├── app.py                          ← Streamlit app
├── requirements.txt                ← pinned library versions
├── best_model.pkl                  ← trained XGBoost model
├── encoders.pkl                    ← label encoders
├── model_config.json               ← threshold + feature order
├── README.md                       ← project description
│
├── .streamlit/
│   └── config.toml                 ← dark theme
│
├── notebook/
│   └── Autism_Prediction_fixed.ipynb
│
└── data/
    └── train.csv
```
---
## Dataset
 - ML Olympiad autism prediction dataset conatining (800 rows, 22 columns) taken from Kaggle

---

## Methodology

1. **Cleaning:** removed `ID` and `age_desc`, fixed country names, merged unknown or duplicate ethnicity and relation labels, stripped stray spaces
2. **EDA:** distributions, box plots, count plots, correlation heatmap
3. **Preprocessing:** label encoding of categorical columns (encoders saved), stratified 80/20 train/test split, `age` outliers handled with IQR bounds learned from the training set only
4. **Imbalance handling:** SMOTE inside an `imblearn` Pipeline so synthetic samples never leak into validation folds
5. **Models compared:** Decision Tree, Random Forest, XGBoost, tuned with `RandomizedSearchCV` (20 iterations, stratified 5-fold CV, optimising F1)
6. **Selection:** highest cross-validated F1, ROC-AUC as tie-breaker
7. **Decision threshold:** tuned on out-of-fold training predictions (0.45), not on the test set
8. **Evaluation:** the test set was used once, at the end

---

## Results

**Cross-validation (tuned models, training data)**

| Model | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| Decision Tree | 0.640 | 0.837 | 0.520 | 0.852 |
| Random Forest | 0.679 | 0.868 | 0.562 | 0.911 |
| **XGBoost** | **0.679** | **0.868** | **0.562** | **0.913** |

Random Forest and XGBoost tie on F1, recall and precision, so XGBoost was selected on ROC-AUC. The difference between the two is very small.

**test set (160 records, XGBoost, threshold 0.45)**

| Metric | Value |
|---|---|
| Accuracy | 0.82 |
| ASD recall | 0.88 (28 of 32 cases found) |
| ASD precision | 0.53 |
| ASD F1 | 0.66 |
| ROC-AUC | 0.906 |

The model is tuned to catch most ASD cases (high recall), which suits a screening tool, at the cost of some false alarms (moderate precision).

---

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/autism-prediction-app.git
cd autism-prediction-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

To retrain the model, open the notebook in `notebook/`, run all cells, and copy the generated `best_model.pkl`, `encoders.pkl` and `model_config.json` next to `app.py`.

---



## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Streamlit

---

## Limitations

- Trained on only 800 records, so results can change noticeably with a different split
- Precision is moderate: about half of the people flagged will not have ASD
- The AQ-10 is a screening questionnaire, not a diagnostic instrument
- Some categories (for example certain countries and ethnicities) have very few examples

---

## Author

_Khushi Singh_ · [GitHub](https://github.com/khushi802/composio) · [LinkedIn](www.linkedin.com/in/khushisingh2353)
