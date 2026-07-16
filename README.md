# AutoML Analytics Platform

## Overview

AutoML Analytics Platform is a Python-based web application that enables users to upload datasets, perform automatic data validation, visualize data through interactive dashboards, create custom charts, and train Machine Learning models without writing code.

The application is built using **Streamlit** for the frontend and backend, **Pandas** for data processing, **Plotly** for interactive visualizations, and **Scikit-learn** for Machine Learning.

---

# Features

## 📂 Data Upload

* Upload CSV files
* Upload Excel (.xlsx) files
* Automatic dataset preview
* Session-based dataset storage

---

## ✅ Data Validation

Automatically performs:

* Missing value detection
* Duplicate record detection
* Data type identification
* Statistical summary
* Dataset information
* Dataset preview

---

## 📊 Interactive Dashboard

Automatically generates:

* KPI Cards
* Histogram
* Bar Chart
* Line Chart
* Pie Chart
* Correlation Heatmap
* Box Plot
* Statistical Summary
* Dataset Preview

---

## 📈 Custom Chart Builder

Supports dynamic chart creation.

Available charts:

* Bar Chart
* Line Chart
* Scatter Plot
* Histogram
* Pie Chart
* Box Plot
* Violin Plot
* Area Chart

Features:

* X-axis selection
* Y-axis selection
* Color grouping
* Dataset filtering
* HTML chart download

---

## 🤖 Machine Learning

Supported Models

### Regression

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

### Classification

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### Clustering

* K-Means Clustering

The application automatically detects whether the selected target column represents a regression or classification problem.

---

## 📊 Model Evaluation

Regression Metrics

* RMSE
* MAE
* R² Score

Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Clustering Metrics

* Silhouette Score

---

## 📋 Prediction Results

Displays:

* Prediction table
* Feature importance (Tree models)
* Model coefficients (Linear / Logistic Regression)

---

## Project Structure

```text
AutoMLAnalytics/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   └── style.css
│
├── config/
│   ├── config.py
│   └── logger.py
│
├── pages/
│   ├── upload.py
│   ├── dashboard.py
│   ├── custom_chart.py
│   └── ml_analysis.py
│
├── services/
│   ├── data_loader.py
│   ├── validator.py
│   ├── visualization.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── ml_models.py
│
├── utils/
│   ├── helpers.py
│   └── session.py
│
├── uploads/
│
└── logs/
```

---

# Technology Stack

| Layer            | Technology     |
| ---------------- | -------------- |
| Frontend         | Streamlit      |
| Backend          | Python         |
| Data Processing  | Pandas, NumPy  |
| Visualization    | Plotly         |
| Machine Learning | Scikit-learn   |
| File Handling    | OpenPyXL       |
| Logging          | Python Logging |

---

# Installation

## Prerequisites

* Python 3.11 or 3.12 (recommended)
* pip
* Visual Studio Code (optional)

---

## Clone Repository

```bash
git clone <repository-url>

cd AutoMLAnalytics
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# Workflow

1. Upload Dataset
2. Review Data Validation
3. Explore Dashboard
4. Create Custom Charts
5. Select Target Column
6. Choose Machine Learning Model
7. Train Model
8. Review Results

---

# Supported File Formats

* CSV
* XLSX

---

# Machine Learning Workflow

```text
Upload Dataset
        │
        ▼
Data Validation
        │
        ▼
Preprocessing
        │
        ▼
Task Detection
        │
        ├── Regression
        ├── Classification
        └── Clustering
                │
                ▼
Model Training
                │
                ▼
Evaluation
                │
                ▼
Predictions
```

---

# Current Features

* Upload CSV and Excel datasets
* Automatic validation
* Interactive dashboard
* Custom chart builder
* Automatic preprocessing
* Machine learning model training
* Interactive metrics
* Prediction results
* Logging support

---

# Planned Enhancements

* SHAP Explainability
* Hyperparameter Tuning
* GridSearchCV
* Cross Validation
* Model Comparison Dashboard
* AutoML Recommendations
* Download Trained Model
* Download Predictions
* PDF Report Generation
* Dark Theme
* User Authentication
* Database Integration
* Azure Deployment
* Docker Support
* REST API

---

# Logging

Application logs are stored in:

```text
logs/application.log
```

---

# Troubleshooting

### ModuleNotFoundError

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Streamlit Not Found

```bash
pip install streamlit
```

---

### OpenPyXL Error

```bash
pip install openpyxl
```

---

### Scikit-learn Error

```bash
pip install scikit-learn
```

---

### Clear Streamlit Cache

```bash
streamlit cache clear
```

---

# Future Roadmap

* Authentication
* User Management
* AutoML Pipeline
* Deep Learning Models
* Time Series Forecasting
* NLP Analysis
* AI Insights
* Azure Deployment
* Docker Deployment
* CI/CD Pipeline
* GitHub Actions

---

# License

This project is intended for educational, research, and business analytics purposes. Choose and add an appropriate open-source license (such as MIT or Apache 2.0) if you plan to distribute it publicly.

---

# Author

**Ramesh**

Senior Business Analyst | Python Enthusiast | AI & Data Analytics Learner

---

# Acknowledgements

Built using:

* Streamlit
* Pandas
* NumPy
* Plotly
* Scikit-learn
* OpenPyXL

Special thanks to the open-source community for making these excellent tools available.
