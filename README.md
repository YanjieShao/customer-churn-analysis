# Customer Churn Prediction

## Project Overview
This project predicts customer churn for a telecom company using customer profile, subscription, and billing information.  
The goal is to identify customers at high risk of churn and provide business insights to support customer retention strategies.

## Dataset
The project uses the **Telco Customer Churn** dataset, which contains approximately **7,000 customer records** and a binary target variable:

- `Churn = Yes`: customer left the service
- `Churn = No`: customer stayed with the service

The dataset includes:
- Customer demographics
- Subscription details
- Billing information
- Value-added services

## Project Workflow
1. Data loading and cleaning
2. Exploratory data analysis (EDA)
3. Feature engineering
4. Preprocessing with pipelines
5. Model training:
   - Logistic Regression
   - Random Forest
   - XGBoost
6. Evaluation with ROC-AUC, PR-AUC, confusion matrix, and classification report
7. Model interpretation and feature importance analysis

## Exploratory Data Analysis (EDA)
Initial data exploration showed several clear churn patterns:

- Customer churn rate was approximately **26%–27%**
- Customers with **shorter tenure** were more likely to churn
- **Month-to-month contracts** had the highest churn rate
- Customers with **higher monthly charges** were more likely to churn
- **Longer contracts** were associated with better customer retention

These early findings provided useful hypotheses for the predictive modeling stage.

## Model Performance
Three models were trained and compared for churn prediction:

### Logistic Regression
- **ROC-AUC:** 0.8419
- **PR-AUC:** 0.6351
- Highest recall for churned customers: **0.7941**

### Random Forest
- **ROC-AUC:** 0.8252
- **PR-AUC:** 0.6211

### XGBoost
- **ROC-AUC:** 0.8377
- **PR-AUC:** 0.6467
- Highest overall accuracy: **0.7963**

### Performance Summary
- **Logistic Regression** achieved the highest **ROC-AUC**, indicating the strongest overall ranking ability.
- **XGBoost** achieved the highest **PR-AUC** and the highest **accuracy**, providing a strong balance between precision and ranking performance.
- **Random Forest** performed reasonably well, but was slightly weaker than Logistic Regression and XGBoost on the main ranking metrics.

Overall, all three models achieved **ROC-AUC values above 0.82**, suggesting that churn can be predicted effectively from customer subscription, billing, and service usage features.

## Business Interpretation
The modeling results were broadly consistent with the EDA findings.

Model interpretation suggested that the following factors were associated with **higher predicted churn risk**:

- Month-to-month contracts
- Fiber optic internet service
- Electronic check payment method
- Lack of online security and tech support
- Billing-related variables such as `MonthlyCharges`, `TotalCharges`, and `avg_monthly_spend`

Factors associated with **lower predicted churn risk** included:

- Longer customer tenure
- Two-year contracts
- Having dependents

These findings suggest that **contract structure, tenure, billing patterns, and service adoption** all play an important role in customer retention.

## Important Predictors
Random Forest feature importance showed that churn prediction was strongly influenced by:

- `TotalCharges`
- `tenure`
- `avg_monthly_spend`
- `MonthlyCharges`
- `Contract_Month-to-month`
- `TechSupport_No`
- `OnlineSecurity_No`
- `Contract_Two year`
- `PaymentMethod_Electronic check`
- `InternetService_Fiber optic`

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- XGBoost

## Repository Structure
```text
customer-churn-analysis/
├── data/
├── notebooks/
│   └── churn_eda.ipynb
├── src/
│   └── train.py
├── models/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run
1. Place the dataset under `data/`
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the training script:
```bash
python src/train.py
```

## Outputs
- Trained models saved under `models/`
- Evaluation metrics saved under `results/metrics.json`
- ROC and PR curves saved under `results/`
- Feature importance saved under `results/rf_feature_importance.csv`

## Conclusion
This project shows that telecom customer churn can be modeled effectively using customer subscription, billing, and service-related data.

Both EDA and predictive modeling consistently highlighted the importance of:
- tenure
- contract type
- monthly charges
- support/service adoption

From a business perspective, the results suggest that retention strategies may be especially valuable for:
- short-tenure customers
- customers on month-to-month contracts
- customers with high monthly charges
- customers without value-added support services
