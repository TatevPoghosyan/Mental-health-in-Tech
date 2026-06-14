# 🧠 Mental Health in Tech — Prediction Project

A machine learning project to predict whether a tech industry worker has sought treatment for a mental health condition, based on workplace and demographic factors.

---

## 📌 Project Overview

**Type:** Binary Classification  
**Dataset:** OSMI Mental Health in Tech Survey — 2014 & 2016 (merged)  
**Target Variable:** `treatment` — Has the person sought mental health treatment? (1 = Yes, 0 = No)  
**Total Samples:** 2,681 (after cleaning and merging)  
**Features:** 25

---

## 📁 Project Structure

```
mental-health-in-tech/
├── data/
│   ├── survey_2014.csv          # Raw 2014 survey data (1,259 rows, 27 features)
│   ├── survey_2016.csv          # Raw 2016 survey data (1,433 rows, 63 features)
│   ├── merged_clean.csv         # Cleaned & merged (human-readable)
│   └── merged_encoded.csv       # Encoded & model-ready
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb   # Data cleaning, merging, encoding
│   └── 03_Modeling.ipynb        # Model training, evaluation, comparison
└── README.md
```

---

## 📊 Dataset Description

| Feature | Description |
|---------|-------------|
| `Age` | Respondent's age |
| `Gender` | Standardized: Male / Female / Other |
| `Country` | Country of residence |
| `self_employed` | Whether self-employed |
| `family_history` | Family history of mental illness |
| `work_interfere` | Does mental health affect work? (Often / Sometimes / Rarely / Never) |
| `no_employees` | Company size |
| `remote_work` | Works remotely? |
| `tech_company` | Employer is a tech company? |
| `benefits` | Employer provides mental health benefits? |
| `care_options` | Aware of mental health care options at work? |
| `wellness_program` | Employer has a wellness program? |
| `anonymity` | Anonymity protected when using mental health resources? |
| `coworkers` | Comfortable discussing mental health with coworkers? |
| `supervisor` | Comfortable discussing mental health with supervisor? |
| `mental_vs_physical` | Employer takes mental health as seriously as physical health? |
| `obs_consequence` | Observed negative consequences for others who disclosed? |
| `year` | Survey year (2014 or 2016) |

---

## 🔍 EDA Key Findings

- **Treatment rate increased** from 50.6% (2014) to 58.5% (2016), suggesting growing awareness
- **Family history** is the strongest single predictor of treatment-seeking (correlation = 0.38)
- Workers who report mental health issues **"Often" interfering** with work seek treatment at much higher rates
- Companies that provide **mental health benefits and care options** show higher treatment rates among employees
- **Female respondents** seek treatment more often than male respondents
- Significant **age outliers** were found in the raw data (values like -1726 and 99999999999), requiring cleaning
- **Gender column** had 49+ unique raw values — standardized to Male / Female / Other

---

## ⚙️ Preprocessing Steps

1. **Dropped** irrelevant columns: `Timestamp`, `state`, `comments`
2. **Age cleaning** — removed values < 15 and > 80
3. **Gender standardization** — 49 raw values → Male / Female / Other
4. **Missing value imputation** — mode for categorical, median for numerical
5. **2016 column mapping** — renamed 63 long-form questions to match 2014 column names
6. **Value standardization** — aligned 2016 response formats to 2014 (e.g. Always/Sometimes/Never → Yes/No)
7. **Merged** 2014 + 2016 into single dataset (2,681 rows)
8. **Feature engineering** — created `age_group` (15-25 / 26-35 / 36-45 / 46+)
9. **Label Encoding** — all categorical features encoded to integers
10. **Train/Test split** — 80% train, 20% test, stratified by target

---

## 🤖 Models & Results

Four classification models were trained and compared:

| Model | Accuracy | ROC-AUC | F1 Score |
|-------|----------|---------|----------|
| **Random Forest**  | **0.784** | **0.842** | **0.803** |
| XGBoost | 0.743 | 0.821 | 0.768 |
| SVM (RBF) | 0.754 | 0.800 | 0.778 |
| Logistic Regression | 0.715 | 0.778 | 0.723 |

*Evaluation: 5-fold Stratified Cross-Validation + held-out test set (20%)*

---

## 🔑 Top Predictive Features (Random Forest)

1. `work_interfere` — Whether mental health interferes with work
2. `Age` — Younger workers show higher treatment-seeking rates
3. `family_history` — Strong genetic and environmental signal
4. `Country` — Cultural differences in stigma and access
5. `care_options` — Awareness of employer-provided resources

---

## 💡 Why Each Model Performed the Way It Did

**Random Forest (best — ROC-AUC 0.842)**  
Handles non-linear relationships and mixed feature types naturally. Does not require feature scaling. The ensemble of trees reduces overfitting and captures complex patterns between categorical workplace factors.

**XGBoost (second — ROC-AUC 0.821)**  
Sequential boosting corrects errors from previous trees, making it powerful. Slightly lower than Random Forest here, possibly because the dataset is not large enough to fully leverage boosting's strength.

**SVM (third — ROC-AUC 0.800)**  
Works well after scaling, but the RBF kernel may not be optimal for this heavily categorical dataset. Sensitive to the choice of hyperparameters.

**Logistic Regression (baseline — ROC-AUC 0.778)**  
As expected for a linear model, it serves as a strong interpretable baseline but cannot capture non-linear interactions between features.

---

## 📌 Conclusions

- **Workplace factors matter** — Companies that provide mental health benefits, care options, and wellness programs have employees who are more likely to seek treatment.
- **Awareness drives treatment** — The most important predictor is `work_interfere`: workers who acknowledge that mental health affects their performance are far more likely to seek help.
- **Family history is a strong signal** — Genetic predisposition and early exposure to mental health conversations in the family significantly increases treatment-seeking.
- **There is a positive trend** — Treatment-seeking rates increased from 2014 to 2016, suggesting that the tech industry is gradually reducing mental health stigma.

---

## 🛠️ Tech Stack

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost
- Jupyter Notebook

---

## 📂 Data Sources

- [OSMI Mental Health in Tech Survey 2014](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- [OSMI Mental Health in Tech Survey 2016](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016)
