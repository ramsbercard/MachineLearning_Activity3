# Naive Bayes Health Prediction (From Scratch)

This project implements a **Gaussian Naive Bayes classifier from scratch**
using NumPy and applies it to a real-world daily health dataset.

---

## 📊 Dataset
**File:** `daily_health_dataset.xlsx`

**Features:**
- sleep_hours
- water_liters
- exercise_minutes
- steps
- junk_food
- stress_level

**Target:**
- healthy_day (0 = unhealthy, 1 = healthy)

---

## ⚙️ Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (for evaluation & preprocessing)

---

## 🧠 Model
- Gaussian Naive Bayes (custom implementation)
- Uses:
  - Mean
  - Variance
  - Prior probability
- Log-likelihood for numerical stability

---

## 📈 Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

---

## ▶️ How to Run
1. Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
jupyter notebook FA3_Naive_Bayes_Test_Script_Activity3.ipynb

