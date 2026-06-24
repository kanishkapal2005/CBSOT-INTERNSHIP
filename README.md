# 🚀 Sonar: Mine vs Rock Prediction using Logistic Regression

## 📌 Project Overview

This project is a Machine Learning classification model that predicts whether an object detected by sonar signals is a **Mine** or a **Rock**. The model is built using **Logistic Regression** and trained on the Sonar Dataset containing signal measurements collected from sonar returns.

The project was developed as part of my **AI & Machine Learning Internship at Coding Block School of Technology** to gain practical experience in data preprocessing, model training, and predictive analytics.

---

## 🎯 Objective

The primary objective of this project is to build a binary classification model capable of identifying underwater objects based on sonar signal data and accurately classifying them as:

* **Rock (R)**
* **Mine (M)**

---

## 🛠️ Technologies Used

* Python
* Google Colab
* NumPy
* Pandas
* Scikit-learn (Sklearn)

---

## 📂 Repository Structure

```text
├── Sonar_Mine_vs_Rock_Prediction.ipynb
├── sonar_data.csv
├── README.md
```

* **Sonar_Mine_vs_Rock_Prediction.ipynb** → Complete project implementation.
* **sonar_data.csv** → Dataset used for training and testing.
* **README.md** → Project documentation.

---

## 📊 Dataset Information

The Sonar Dataset contains sonar signals bounced off various surfaces.

### Features

* 60 numerical attributes representing sonar signal energy values.
* 1 target attribute indicating:

  * **R** → Rock
  * **M** → Mine

### Dataset Characteristics

* Binary Classification Problem
* Numerical Features
* No Missing Values

---

## ⚙️ Project Workflow

### 1. Data Collection

* Loaded the Sonar Dataset using Pandas.
* Explored dataset shape, columns, and statistical information.

### 2. Data Preprocessing

* Separated features and target labels.
* Converted data into training and testing sets.

### 3. Model Training

* Applied Logistic Regression from Scikit-learn.
* Trained the model using the training dataset.

### 4. Model Evaluation

* Evaluated model performance using accuracy score.
* Compared predictions on training and testing data.

### 5. Prediction System

* Implemented a predictive system that accepts new sonar signal inputs and classifies them as:

  * Mine
  * Rock

---

## 📈 Model Performance

The Logistic Regression model achieved a satisfactory classification accuracy on both training and testing datasets, demonstrating its effectiveness in distinguishing between mines and rocks based on sonar signal patterns.

---

## ▶️ How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/kanishkapal2005/CBSOT-INTERNSHIP.git
```

### Navigate to the Project Folder

```bash
cd CBSOT-INTERNSHIP
```

### Open Google Colab

1. Upload the notebook file.
2. Upload the dataset file (`sonar_data.csv`).
3. Run all cells sequentially.

---

## 🎓 Internship Information

**Organization:** Coding Block School of Technology
**Domain:** Artificial Intelligence & Machine Learning
**Project:** Sonar: Mine vs Rock Prediction using Logistic Regression

This project was completed as part of my internship training to strengthen my understanding of:

* Machine Learning Fundamentals
* Data Preprocessing
* Classification Algorithms
* Model Evaluation
* Predictive Analytics

---

## 🔮 Future Improvements

* Experiment with advanced classification algorithms.
* Perform hyperparameter tuning.
* Deploy the model using Flask or Streamlit.
* Create an interactive web application for real-time predictions.

---

## 👨‍💻 Author

**Kanishka Pal**
AI/ML Intern | MERN Stack Developer | Technical Content Writer

If you found this project useful, feel free to ⭐ the repository and connect with me on LinkedIn.
