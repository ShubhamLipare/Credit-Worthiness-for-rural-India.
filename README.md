# Loan Approval Prediction

## Overview
This project predicts whether a loan application will be approved based on applicant details using machine learning.

## Dataset
- **Source:** [Kaggle/UCI Repository]
- **Features:** Applicant income, credit history, loan amount, etc.
- **Target:** Loan approval (Yes/No)

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn
- FastAPI, Streamlit (for deployment)
- MLflow (for model tracking)

## Approach
1. Data Cleaning & EDA  
2. Feature Engineering  
3. Model Selection (Random Forest, DT etc)  
4. Hyperparameter Tuning  
5. Model Evaluation (Accuracy, F1-score)  
6. Deployment via FastAPI  

## Evaluation Metric
Since goal is to increase loan accessibility for rural India, minimizing the risk of rejecting deserving applicants is critical.
Prioritize Recall to ensure more eligible people get loans.

## 🛠 How to Run
```bash
git clone https://github.com/yourusername/loan-approval.git  
cd loan-approval  
pip install -r requirements.txt  
python train.py  # Train the model  
uvicorn app:app --reload  # Run API  
