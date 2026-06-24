# 🏥 AI-Powered Maternal Health Risk Prediction System

> An end-to-end Machine Learning project that predicts maternal health risk levels (**Low / Mid / High**) from clinical vitals, explains its predictions using Explainable AI, and recommends a course of action — built as an internship capstone project.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Internship Details

| Field | Detail |
|---|---|
| **Internship Title** | Machine Learning Internship |
| **Organization** | Coding Block School of Technology |
| **Duration** | 01/June/2026 – 15/July/2026 |
| **Mentor / Supervisor** | Aryesh Rai |
| **Internship Type** | Remote  |

## 👩‍💻 Candidate Details

| Field | Detail |
|---|---|
| **Name** | Kanishka Pal |
| **Institution** | Hi-Tech Institute of Engineering and Technology Ghaziabad |
| **Course / Branch** | Computer Science and Engineering |
| **Roll No. / Registration No.** | 2402200100024 |
| **Email** | kksnehapal@gmail.com |
| **LinkedIn** | https://www.linkedin.com/in/kanishkapal2005/ |

---

## 📌 Project Name

**AI-Powered Maternal Health Risk Prediction System**

---

## 📖 Project Description

Maternal health risk prediction is a multi-class classification problem that uses
six routinely-measured clinical vitals — **Age, Systolic Blood Pressure, Diastolic
Blood Pressure, Blood Sugar, Body Temperature, and Heart Rate** — to estimate
whether a pregnant patient is at **Low, Mid, or High** risk of pregnancy-related
complications.

This project implements a **complete, production-style ML pipeline** in a single
Google Colab notebook:

- Cleans and explores the **UCI Maternal Health Risk Dataset**.
- Engineers two additional clinical features (**Pulse Pressure**, **Mean Arterial
  Pressure**).
- Trains and benchmarks **7 classification algorithms**.
- Explains predictions using **Feature Importance + SHAP** (Explainable AI).
- Wraps everything into an **interactive prediction tool**, a **rule-based
  healthcare recommendation engine**, and a **patient report generator**.
- Persists the best model with **Pickle** and **Joblib** for reuse/deployment.
- Adds internship-grade rigor: **cross-validation, GridSearchCV tuning, ROC-AUC,
  learning curves, and feature selection analysis**.

---

## ❓ Problem Statement

> **Given a set of clinical vital signs recorded for a pregnant woman, predict
> whether she is at Low, Mid, or High risk of maternal health complications, and
> recommend an appropriate course of action.**

**Why it matters:**
- Maternal risk depends on multiple interacting vitals — no single threshold rule
  (e.g. "BP > 140 = risky") captures the full picture.
- Early, consistent risk flagging can help clinicians intervene before
  complications escalate, which is especially valuable in low-resource settings
  with limited continuous monitoring.
- A model that also *explains itself* is far more useful — and trustworthy — to
  healthcare staff than a black-box prediction.

**ML formulation:** Supervised, multi-class classification (3 classes — Low / Mid
/ High Risk).

---

## 🗂️ Dataset

**Source:** [Maternal Health Risk Data Set — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)
Originally collected from hospitals, community clinics, and maternal healthcare
centers via an IoT-based risk monitoring system.

| Feature | Description | Type |
|---|---|---|
| `Age` | Age of the patient (years) | Numeric |
| `SystolicBP` | Upper blood pressure value (mmHg) | Numeric |
| `DiastolicBP` | Lower blood pressure value (mmHg) | Numeric |
| `BS` | Blood glucose level (mmol/L) | Numeric |
| `BodyTemp` | Body temperature (°F) | Numeric |
| `HeartRate` | Resting heart rate (bpm) | Numeric |
| `RiskLevel` | **Target** — Low / Mid / High Risk | Categorical |

**File path used in the notebook:**
```
/content/Maternal Health Risk Data Set.csv
```

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.10 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Explainable AI | SHAP |
| Interactivity | ipywidgets |
| Model Persistence | Pickle, Joblib |
| Environment | Google Colab / Jupyter Notebook |

---

## 🧭 Project Pipeline (High-Level Flow)

```
Raw CSV → Data Cleaning → EDA → Feature Engineering → Preprocessing
   → Train 7 Models → Evaluate & Compare → Explainable AI (SHAP)
   → Risk Prediction System → Recommendation Engine → Patient Report
   → Save Best Model → Advanced Tuning & Validation → Conclusion
```

---

## 🔍 Step-by-Step Explanation of the Notebook (Section by Section)

The notebook is organized into 19 sections. Below is what every section's code
actually does, and why.

### 1. Project Introduction, Problem Statement & Dataset Overview
Pure markdown context cells — explain the motivation, the exact prediction task,
and document every column in the dataset before any code runs. Setting this
context first makes the notebook readable as a standalone report, not just code.

### 2. Import Libraries
- A `!pip install -q xgboost shap joblib` cell guarantees the environment has the
  packages Colab doesn't pre-install.
- All other imports (Pandas, NumPy, Matplotlib/Seaborn/Plotly, scikit-learn
  modules, XGBoost, SHAP, pickle/joblib) are grouped in one cell so dependencies
  are visible at a glance instead of scattered through the notebook.
- `warnings.filterwarnings('ignore')` and Seaborn/Matplotlib style settings keep
  the notebook's output clean and presentation-ready.

### 3. Data Loading
```python
DATA_PATH = "/content/Maternal Health Risk Data Set.csv"
df = pd.read_csv(DATA_PATH)
```
Loads the CSV from Colab's local filesystem into a DataFrame and immediately
prints its shape so you instantly know how many records/columns you're working
with.

### 4. Data Exploration
A series of small, single-purpose cells check:
- `df.shape` → row/column count.
- `df.dtypes` / `df.info()` → confirms every feature is numeric except the
  target.
- `df.isnull().sum()` → a missing-value report (with percentages).
- `df.duplicated().sum()` → counts exact duplicate rows.
- `df.describe()` → mean/median/std/min/max for every numeric column.
- `df['RiskLevel'].value_counts()` → class balance of the target variable.

Each check answers a specific question a data scientist must answer before
modeling: *Is the data complete? Is it balanced? Are there structural surprises?*

### 5. Data Cleaning
```python
df_clean = df.drop_duplicates().reset_index(drop=True)
df_clean['RiskLevel'] = df_clean['RiskLevel'].str.strip().str.lower()
```
- Removes exact duplicate rows so they don't bias statistics or let the model
  "memorize" repeated examples.
- Normalizes the text in `RiskLevel` (trims whitespace, lowercases) so that
  `"High Risk"`, `" high risk"`, and `"HIGH RISK"` are all treated identically
  during encoding later.
- Re-asserts that no missing values remain, then resets the index.

### 6. Exploratory Data Analysis (EDA)
This is the largest section, broken into focused sub-cells:
- **Class distribution** — bar chart + pie chart (Seaborn) and an interactive
  Plotly bar chart of how many patients fall into each risk category.
- **Correlation heatmap** — `df[numeric_cols].corr()` visualized with
  `sns.heatmap`, showing which vitals move together (e.g. Systolic & Diastolic
  BP are usually strongly correlated).
- **Feature distribution plots** — histograms with KDE overlays for all 6 raw
  vitals, to see skewness/spread.
- **Boxplots by risk level** — for every feature, a boxplot split by
  `RiskLevel`, used to visually spot which vitals separate the risk classes and
  to eyeball outliers.
- **IQR-based outlier counts** — a small function (`count_outliers_iqr`)
  quantifies outliers per feature instead of relying on eyeballing alone.
- **Pairplot** — `sns.pairplot` colored by `RiskLevel`, showing pairwise feature
  relationships and class separability at a glance.
- **Risk-level comparison bar charts** — mean value of every feature grouped by
  `RiskLevel`, in both static (Seaborn) and interactive (Plotly grouped bar)
  form, to directly compare "what does a typical High Risk patient look like
  versus a Low Risk one?"

### 7. Feature Engineering
```python
data['PulsePressure'] = data['SystolicBP'] - data['DiastolicBP']
data['MAP'] = data['DiastolicBP'] + (data['SystolicBP'] - data['DiastolicBP']) / 3
```
Wrapped in a reusable `engineer_features()` function (important: this exact
function is reused later inside the live prediction system, so training and
inference always compute features identically):
- **Pulse Pressure** — a known clinical proxy for arterial stiffness/cardiac
  risk.
- **Mean Arterial Pressure (MAP)** — average pressure in the arteries over one
  heartbeat cycle; a standard clinical risk indicator.

Both come "for free" from vitals already collected, adding signal without
requiring new measurements.

### 8. Data Preprocessing
- **Label encoding (explicit, ordinal):** `{'low risk': 0, 'mid risk': 1, 'high risk': 2}`
  — done with an explicit dictionary instead of `LabelEncoder` so the natural
  risk ordering (Low < Mid < High) is preserved rather than alphabetically
  scrambled.
- **Feature/target split:** `X` = the 6 raw vitals + 2 engineered features (8
  total), `y` = the encoded `RiskLevel`.
- **Train-Test Split (80:20):** `train_test_split(..., stratify=y)` — stratifying
  ensures both the train and test sets keep the same class proportions as the
  full dataset, which matters a lot with an already-imbalanced 3-class target.
- **Feature Scaling:** `StandardScaler` is **fit only on `X_train`**, then used
  to `.transform()` both train and test — fitting on test data would leak
  information from the test set into training, silently inflating accuracy.

### 9. Model Training
Seven models are defined in a single dictionary and trained in one loop, so
every model sees exactly the same training data and random seed:

| Model | Why it was included |
|---|---|
| **Logistic Regression** | Simple, fast, interpretable linear baseline — useful to know how much a non-linear model actually buys you. |
| **Decision Tree** | Captures non-linear splits and feature interactions; easy to visualize/explain. |
| **Random Forest** | Ensemble of decision trees — usually robust to noise/outliers and handles non-linear, mixed-scale clinical data well. Also the model literature most consistently recommends for this exact dataset. |
| **Gradient Boosting** | Sequentially corrects previous trees' errors — often squeezes out extra accuracy over a single Random Forest. |
| **XGBoost** | Optimized, regularized gradient boosting — typically among the top performers on tabular health datasets in published studies on this same dataset. |
| **K-Nearest Neighbors** | A simple, non-parametric baseline that classifies by "similar patients" — useful sanity check against the more complex models. |
| **Support Vector Machine (SVM)** | Effective in higher-dimensional, scaled feature spaces; `probability=True` lets it produce the same probability outputs as the other models. |

Comparing all seven (rather than picking one model up front) is exactly what a
rigorous internship-level project should do — it lets the **data** decide the
best model instead of an assumption.

### 10. Model Evaluation
For every trained model:
- **Accuracy, Precision, Recall, F1-Score** (weighted average, to be fair to
  minority classes) are computed and assembled into one comparison table
  (`results_df`), sorted by accuracy.
- A grouped bar chart visualizes all four metrics side-by-side across all 7
  models.
- A **3×3 grid of confusion matrices** (one per model) shows exactly which risk
  classes each model confuses most (in this dataset, "Mid Risk" is typically the
  hardest class to separate from "Low Risk").
- A full **`classification_report`** (per-class precision/recall/F1/support) is
  printed for every model.
- The **best model** is selected programmatically:
  `results_df.sort_values(['F1 Score','Accuracy'], ascending=False).iloc[0]` —
  i.e., highest F1-score, accuracy as the tiebreaker. F1 is preferred over raw
  accuracy as the primary criterion because it balances precision and recall,
  which matters more than accuracy in healthcare contexts where missing a
  genuinely "High Risk" patient is costlier than a false alarm.

### 11. Explainable AI
- **Feature Importance (Random Forest):** `rf_model.feature_importances_`
  ranked and plotted — a fast, built-in measure of which features the trees
  split on most.
- **SHAP (`TreeExplainer`):** computes Shapley values for every test sample,
  specifically for the **"High Risk"** class (index 2), since that's the most
  clinically actionable class to understand.
  - `shap.summary_plot(..., show=True)` — shows each feature's spread of impact
    across all patients (color = feature value, position = pull toward/away
    from High Risk).
  - `shap.summary_plot(..., plot_type='bar')` — collapses that into a simple
    mean-absolute-impact ranking.
  - `shap.waterfall_plot(...)` — explains **one single patient's** prediction,
    showing exactly which features pushed their risk score up or down — this is
    the kind of explanation a doctor could actually read.

> Random Forest is used specifically for the explainability plots (separate from
> whichever model wins the accuracy/F1 comparison) because tree ensembles pair
> with the fast, exact SHAP `TreeExplainer`, and Random Forest is broadly
> interpretable by design.

### 12. AI-Powered Risk Prediction System
- `predict_maternal_risk()` — a single reusable function that takes the 6 raw
  vitals, calls the **same** `engineer_features()` used in training, scales the
  inputs with the **same fitted `scaler`** (never re-fit at inference time), and
  returns the predicted class + probability for all 3 risk levels.
- An **`ipywidgets`** form (sliders for Age, BP, Blood Sugar, Temperature, Heart
  Rate + a "Predict" button) wraps that function into an interactive,
  point-and-click tool that runs directly inside the Colab/Jupyter notebook —
  no separate web app needed.

### 13. Healthcare Recommendation Engine
`get_healthcare_recommendations(risk_level)` is a simple rule-based lookup
(High → "immediate consultation, monitor BP/BS daily…", Mid → "weekly
monitoring, follow physician guidance…", Low → "routine prenatal care…"). It's
intentionally simple and transparent — the goal is to demonstrate how a model's
output can be translated into an actionable next step, not to replace medical
guidelines.

### 14. Patient Health Report Generator
`generate_patient_report()` ties everything together: it calls
`predict_maternal_risk()`, then `get_healthcare_recommendations()`, then prints
a clean, formatted report containing the **Patient Summary**, **Predicted Risk
Level**, **Probability Scores**, **Medical Recommendations**, and a plain-English
**Risk Interpretation** — essentially a mini clinical decision-support printout.

### 15. Model Saving
- Saves the best model and the fitted `scaler` with **both** `pickle.dump()`
  and `joblib.dump()` (Joblib is generally preferred for scikit-learn objects,
  but Pickle is shown too since the project spec explicitly asked for it).
- Immediately **reloads** both files (`joblib.load(...)`) and re-runs a
  prediction with the reloaded objects to prove they produce identical results
  to the original in-memory model — confirming the saved files are genuinely
  deployment-ready.

### 16. Advanced Enhancements
- **Cross-Validation:** 5-fold `StratifiedKFold` accuracy for every model,
  reported as mean ± standard deviation — a more reliable performance estimate
  than a single train/test split.
- **GridSearchCV:** exhaustively searches a Random Forest hyperparameter grid
  (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`) using
  5-fold CV optimizing weighted F1-score, then evaluates the tuned model on the
  held-out test set.
- **ROC-AUC (One-vs-Rest):** binarizes the 3-class labels and plots an ROC curve
  per class for the best model, plus an overall weighted ROC-AUC score — useful
  because accuracy alone can hide poor ranking/probability quality.
- **Learning Curves:** plots training vs. cross-validation accuracy as training
  set size grows, to visually diagnose under/overfitting and whether more data
  would likely help.
- **Feature Selection (ANOVA F-test):** `SelectKBest(f_classif)` ranks every
  feature by statistical association with the target — a second, independent
  check on feature importance alongside the SHAP/Random-Forest results.

### 17. Conclusion & Future Scope
Markdown summary of what was built, what was learned, and where the project
could go next (larger/more diverse datasets, deep learning, real-time IoT
integration, time-series tracking across pregnancy, clinical validation, and
REST API / web deployment).

---

## 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 0.6813 | 0.6608 | 0.6813 | 0.6393 |
| Decision Tree | 0.6484 | 0.6514 | 0.6484 | 0.6494 |
| Random Forest | 0.6593 | 0.6134 | 0.6593 | 0.6299 |
| Gradient Boosting | 0.6374 | 0.5983 | 0.6374 | 0.6143 |
| XGBoost | 0.6374 | 0.5954 | 0.6374 | 0.6123 |
| K-Nearest Neighbors | 0.7033 | 0.6782 | 0.7033 | 0.6815 |
| Support Vector Machine | **0.7253** | **0.6855** | **0.7253** | 0.6765 |

## 🏆 Best Performing Model

Based on the evaluation metrics, **Support Vector Machine (SVM)** achieved the highest overall performance:

- **Accuracy:** 72.53%
- **Precision:** 68.55%
- **Recall:** 72.53%

Although **K-Nearest Neighbors (KNN)** achieved the highest **F1-Score (68.15%)**, SVM demonstrated the best balance across all major evaluation metrics and was selected as the final model for this project.

### Reference Benchmarks from Published Literature
For context, independent studies on this same UCI dataset have reported the
following (your results may differ based on preprocessing choices and random
seeds):

| Study / Source | Best Model | Reported Accuracy |
|---|---|---|
| [GitHub – sias01](https://github.com/sias01/maternal_health_risk) | Random Forest | ~90.2% |
| [Journal of Neonatal Surgery](https://jneonatalsurg.com/index.php/jns/article/view/7929) | Random Forest | ~86.7% |
| [Scientific Reports](https://www.nature.com/articles/s41598-024-71934-x) | LightGBM / Random Forest | 86–88% |
| [PMC review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11657143/) | Random Forest | ~90% |
| [arXiv – IyaCare](https://arxiv.org/pdf/2512.07333) | XGBoost | ~85.2% |

Across the literature, **Random Forest and XGBoost consistently rank among the
top performers** on this dataset, and **Blood Sugar and Age** are repeatedly
flagged as the most influential features — consistent with what this notebook's
own SHAP analysis should reveal on the real data.

---

## ▶️ How to Run

1. Open the notebook in **Google Colab**.
2. Upload `Maternal Health Risk Data Set.csv` to the `/content/` folder (drag &
   drop into the Colab file browser, or mount Google Drive and adjust
   `DATA_PATH`).
3. Run all cells in order: **Runtime → Run all**.
4. The first code cell installs `xgboost`, `shap`, and `joblib` automatically —
   no manual setup required.
5. Scroll to the **Risk Prediction System** section to use the interactive
   `ipywidgets` form, or call `predict_maternal_risk(...)` /
   `generate_patient_report(...)` directly with your own values.

---

## 💡 Reason for Choosing This Project

Maternal health is a high-stakes, real-world domain where early risk detection
genuinely saves lives, making it a meaningful and motivating problem to apply
machine learning to. It was also chosen because it touches **every stage** of a
realistic ML workflow — messy real-world tabular data, multi-class imbalance,
model comparison, explainability, and deployment-style packaging — making it an
ideal project to demonstrate end-to-end ML competency for an internship
submission.

## 📚 What I Learned

- How to structure a complete ML project from raw data to a deployable artifact.
- Practical EDA techniques (correlation analysis, outlier detection, pairplots)
  and how to read them to inform modeling decisions.
- Why **stratified splits** and **fit-on-train-only scaling** matter for
  preventing data leakage.
- How to train, compare, and fairly evaluate **multiple classification
  algorithms** instead of defaulting to one.
- Why **Explainable AI (SHAP)** is essential in sensitive domains like
  healthcare — a black-box prediction isn't enough.
- How to translate a model's output into **actionable recommendations**, not
  just a label.
- How to **persist and reload models** (Pickle/Joblib) so they're actually
  reusable outside the training notebook.
- The value of **cross-validation, hyperparameter tuning, and learning curves**
  in validating that a model will generalize, not just memorize.

## ✅ Conclusion

This project successfully delivers a complete, explainable, and interactive
machine learning system for maternal health risk prediction — covering the full
pipeline from raw data to a usable decision-support prototype. By comparing
seven algorithms, applying explainable AI, and validating with cross-validation
and hyperparameter tuning, the project goes beyond a basic classroom exercise
and reflects an industry-style applied ML workflow suitable for a portfolio,
GitHub showcase, or technical interview discussion.

## ⚠️ Special Notes

- **Not a medical device:** This project is for educational/portfolio purposes
  only. It must **not** be used for real clinical decision-making without proper
  clinical validation, regulatory approval, and oversight by licensed healthcare
  professionals.
- **Dataset path:** The notebook expects the CSV at exactly
  `/content/Maternal Health Risk Data Set.csv`. Update `DATA_PATH` in the Data
  Loading cell if your file is named or located differently.
- **Reproducibility:** All models use `random_state=42` so results are
  reproducible on the same data/environment.
- **SHAP plots** can take a little longer to render the first time `shap` is
  imported in a fresh Colab runtime — this is expected.
- **ipywidgets:** the interactive prediction form requires running the notebook
  in an environment that supports Jupyter widgets (Colab and standard Jupyter
  both work).
- Remember to replace every `[bracketed placeholder]` in this README (Internship
  Details, Candidate Details, Model Results table) with your actual information
  before submission.

---

## 📄 License

This project is released under the **MIT License** — feel free to use, modify,
and build upon it with attribution.

## 🙏 Acknowledgments

- **Dataset:** Ahmed, M., Kashem, M.A. — *Maternal Health Risk Data Set*, UCI
  Machine Learning Repository (2020).
- Built using open-source tools: Python, Pandas, NumPy, Scikit-learn, XGBoost,
  SHAP, Matplotlib, Seaborn, and Plotly.

---

<div align="center">
<div align="center">

# 🚀 Kanishka Pal

### 🏆 National Technical Paper Writing Winner

### 💻 MERN Stack Developer | 🤖 AI/ML Enthusiast | 🌐 Full-Stack Learner

<p>
  <a href="mailto:kksnehapal@gmail.com">📧 Email</a> •
  <a href="https://www.linkedin.com/in/kanishkapal2005/">💼 LinkedIn</a> •
</p>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&pause=1000&center=true&width=700&lines=Building+AI-Powered+Applications;MERN+Stack+Developer;Machine+Learning+Enthusiast;Always+Learning+New+Technologies" />

</div>

*Made with care during my internship — building AI for better maternal healthcare outcomes.*

</div>
