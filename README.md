📡 Telecom Customer Churn Prediction

Introduction
Customer churn refers to the phenomenon where subscribers discontinue their relationship with a company’s services. In the telecom industry, churn is particularly critical because customers can easily switch providers in a highly competitive market.
With annual churn rates ranging from 15–25%, telecom operators face significant revenue losses and increased acquisition costs. Retaining existing customers is far more cost-effective than acquiring new ones.
This project aims to predict customer churn using machine learning models. By identifying high-risk customers early, telecom companies can implement targeted retention strategies, reduce attrition, and strengthen customer loyalty.
Customer churn is a critical challenge in the telecom industry, where customers can easily switch providers in a competitive market. Predicting churn helps companies retain valuable customers, reduce acquisition costs, and improve profitability.
This project demonstrates an end-to-end machine learning pipeline for churn prediction, covering:
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Model Building, Evaluation, and Optimization


 What is Customer Churn?
- Definition: When a customer stops using a company’s services.
- Challenge: Large customer bases make individualized retention impractical.
- Opportunity: Predictive analytics enables companies to focus retention efforts on customers most likely to leave.
The key to success in telecom lies in understanding customer behavior and proactively addressing dissatisfaction.

Project Goals
- Understand the dataset and customer behavior.
- Detect missing values, outliers, and correlations.
- Engineer meaningful features to improve model performance.
- Build and evaluate machine learning models for churn prediction.
- Provide actionable insights for telecom businesses.


 Dataset
Telco Customer Churn Dataset
Includes information on:
- Churn status: Customers who left within the last month.
- Services: Phone, multiple lines, internet, online security, backup, device protection, tech support, streaming services.
- Account details: Tenure, contract type, payment method, billing, monthly and total charges.
- Demographics: Gender, senior citizen status, partner, dependents.

 Implementation
Libraries used: scikit-learn, pandas, numpy, matplotlib, seaborn

 Project Workflow
1. Exploratory Data Analysis (EDA)
- Checked dataset shape (7043, 21) and column types.
- Converted TotalCharges from object → float (handled 11 invalid rows).
- Verified no duplicate rows.
- Missing values visualized using missingno.
- Statistical summaries (describe()) for numerical and categorical features.
- Outlier detection using IQR.
- Visualizations:
- Boxplots & Histograms for numerical features.
- Countplots for categorical features vs churn.
- KDE plots for MonthlyCharges and Tenure vs churn.
- Correlation heatmap (tenure negatively correlated with churn, monthly charges weakly positive).

Key Insights:
- Month-to-month contracts → highest churn.
- Electronic check payment → highest churn.
- Fiber optic users churn more than DSL.
- Customers with dependents, partners, or longer tenure churn less.
- Lack of online security/tech support strongly linked to churn.

Preprocessing & Feature Engineering
Steps applied to clean and enrich the dataset:
- Dropped customerID (not useful for modeling).
- Converted SeniorCitizen (0/1) → categorical (No/Yes).
- Handled missing TotalCharges by imputing tenure * MonthlyCharges.
- Feature Engineering:
- Tenure Group: bucketed into ranges (0–1 yr, 1–2 yrs, etc.).
- FiberOptic: binary flag for fiber optic internet.
- Family: combined partner/dependents.
- ProtectiveServices: count of security/backup/tech support features.
- StreamingAddict: count of streaming services.
- ContractScore: ordinal mapping (Month-to-month=3, One year=2, Two year=1).
- PaymentRisk: ordinal mapping (Electronic check=4 → highest risk).
- HighCharges: flag for monthly charges >70.
- AvgMonthlySpend: TotalCharges / (tenure+1).
- NewCustomer: tenure ≤ 6 months.

 Model Building & Evaluation
- Train-test split: 80/20 stratified.
- Preprocessing pipeline:
- Numerical features: imputation (median) + scaling.
- Categorical features: imputation (most frequent) + one-hot encoding.
- Addressed class imbalance using SMOTE.
Models Evaluated:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- AdaBoost
- Best Model:Random Forest 

Results
- Recall prioritized to catch churners (80%).
- Key churn drivers:
- Short tenure
- Month-to-month contracts
- Electronic check payments
- Fiber optic dissatisfaction
- High monthly charges

Hyperparameter Tuning:
- Used Optuna for Random Forest optimization.
- Best params: n_estimators=844, max_depth=6, min_samples_split=8, min_samples_leaf=6, max_features='log2'.

AUC: 0.845, Best Threshold: 0.51
F1: 0.64, Recall: 0.77, Precision: 0.55
Confusion Matrix:
[[798 237]
 [ 86 288]]

 Visualizations
- Donut charts (churn distribution, gender breakdown).
- Histograms & KDE plots (charges, tenure).
- Countplots (contract, payment method, internet service).
- Boxplots & violin plots (tenure vs churn).
- Correlation heatmap.
- ROC curve & Precision-Recall curve.

Conclusion
-  Optimized Random Forest performed best.
- Recall (80%) is prioritized to catch churners, even at the cost of precision.
- Key churn drivers:
- Month-to-month contracts
- Electronic check payments
- Fiber optic internet dissatisfaction
- Lack of online security/tech support
- High monthly charges & short tenure

Future Work
- Build dashboards for real-time churn monitoring.
- Explore deep learning models for further improvement.

 About Me
Hi, I'm Mahesh — a data science enthusiast focused on building practical machine learning solutions.  
- Skilled in Python, EDA, Machine Learning, and Data Visualization  
- Experienced in creating clean, modular projects with clear workflows  
- Passionate about developing recruiter-ready portfolios and working demos  
- 📌 Email: mahesh0105.m@gmail.com
- 📌 LinkedIn: www.linkedin.com/in/mahesh-m0105 
