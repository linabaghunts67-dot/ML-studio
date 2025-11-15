# TabularML Studio  
An interactive Machine Learning platform for **cleaning, exploring, and modeling tabular datasets** across multiple real-world scenarios.

Unlike a single ML script, this project is designed as a **full platform**:
- Choose between different ML scenarios  
- Explore raw data  
- Apply interactive data cleaning  
- Visualize distributions and relationships  
- Train ML models  
- Compare evaluation metrics  
- Download cleaned datasets and predictions  

---

# Supported Scenarios

### **A. Price Prediction (Phones)**
- Dataset with RAM, storage, battery, etc.
- Regression or classification
- Target: price or price range

### **B. Credit Default Risk**
- Loan applications dataset
- Binary classification
- Target: default yes/no

### **C. Customer Churn**
- Telco / subscription dataset  
- Binary classification  
- Target: churn yes/no  

---

# Features

### ✅ **Data Overview**
- Preview raw dataset  
- Missing value summary  
- Basic statistics  
- Target distribution  

### ✅ **Cleaning & Filling**
- Numeric: mean, median, drop rows  
- Categorical: most frequent, “Unknown” category  
- Optional outlier capping  
- Download cleaned dataset  

### ✅ **Data Exploration**
- Distribution plots  
- Feature vs target scatterplots  
- Categorical breakdowns  
- Correlation heatmaps  

### ✅ **Model Training**
- Logistic Regression  
- Linear Regression  
- Random Forest (regression/classification)  
- Adjustable hyperparameters  
- Evaluation metrics: Accuracy, F1, ROC-AUC, RMSE, R²  
- Download predictions  

### ✅ **Interpretability**
- Global feature importance extraction  
- Works for tree-based and linear models  

### ✅ **Software Engineering**
- Modular code in `src/`  
- Preprocessing pipelines  
- Unit tests  
- Clean project structure  

---

# Project Structure

tabularml-studio/
├─ app/
│ ├─ main_app.py
├─ src/
│ ├─ data_loader.py
│ ├─ data_prep.py
│ ├─ features.py
│ ├─ models.py
│ ├─ evaluation.py
│ ├─ explainability.py
├─ tests/
│ ├─ test_data_prep.py
│ ├─ test_models.py
├─ data/
│ ├─ raw/
│ ├─ processed/
├─ docs/
│ ├─ screenshots/
├─ .gitignore
├─ README.md
├─ requirements.txt



---

# ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add datasets

```
Place CSVs into:
data/raw/
```

### 3. Run the app

```
streamlit run app/main_app.py
```
