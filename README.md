# Customer Churn Predictor
**Live demo:** https://churn-predictor-sarkeengs.streamlit.app
Predicts which telecom customers are likely to leave, with a Streamlit app for live predictions.

## Results
- Dataset: IBM Telco Customer Churn (7,043 customers, 26.5% churned)
- Best model: Logistic Regression (C=5, balanced class weights)
- Churn recall: 0.78 | Precision: 0.51 | ROC-AUC: 0.848
- Caught 293 of 374 churners in the test set and missed 81
- Random Forest scored lower (recall 0.51, ROC-AUC 0.835), so the simpler model won

## Key findings
- Month-to-month contracts churn at 42.7%, versus 11.3% (1-year) and 2.8% (2-year)
- Churners average 18 months of tenure, versus 38 for customers who stay
- Fiber optic users, electronic check payers, and customers without tech support churn more

## Method
Data cleaning, EDA, a one-hot encoding and scaling pipeline, a stratified 80/20 split, and a GridSearchCV-tuned model. Evaluated with precision, recall and ROC-AUC rather than accuracy, because the classes are imbalanced. Leakage columns (Churn Score, CLTV, Churn Value, Churn Reason) were removed.

## Limitations
Precision is 0.51, so about half of the flagged customers would not actually leave. Total Charges is derived from tenure, so individual coefficients should not be read as exact effects.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Screenshots
![High risk](images/screenshot_high_risk.png)
![Low risk](images/screenshot_low_risk.png)

## Data
Download the Telco Customer Churn dataset from Kaggle and place the CSV in `data/`.
