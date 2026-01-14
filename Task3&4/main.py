import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import joblib
sns.set(style="whitegrid")

#loading the dataset
df = pd.read_csv("Dataset_task3-4/Task 3 and 4_Loan_Data.csv")
print("\nDataset Head:\n", df.head())
print("\nMissing Values:\n", df.isna().sum())

#adding financial features
df["debt_to_income"] = df["total_debt_outstanding"] / (df["income"] + 1)
df["loan_to_income"] = df["loan_amt_outstanding"] / (df["income"] + 1)
df["utilization_ratio"] = df["loan_amt_outstanding"] / (df["total_debt_outstanding"] + 1)
df["credit_lines_per_year"] = df["credit_lines_outstanding"] / (df["years_employed"] + 1)
FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
    "debt_to_income",
    "loan_to_income",
    "utilization_ratio",
    "credit_lines_per_year"
]
X = df[FEATURES]
y = df["default"]

#coorelation heatmap
plt.figure(figsize=(11,9))
sns.heatmap(df[FEATURES + ["default"]].corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap - Credit Risk Features")
plt.show()

#histogram and dstribution
sns.histplot(data=df, x="fico_score", hue="default", bins=30, kde=True)
plt.title("FICO Score Distribution by Default")
plt.show()
sns.histplot(data=df, x="debt_to_income", hue="default", bins=30, kde=True)
plt.title("Debt-to-Income Ratio by Default")
plt.show()
sns.boxplot(data=df, x="default", y="fico_score")
plt.title("FICO vs Default")
plt.show()
sns.boxplot(data=df, x="default", y="debt_to_income")
plt.title("Debt-to-Income vs Default")
plt.show()

#train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#logistic regression
log_model = LogisticRegression(class_weight="balanced", max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict_proba(X_test_scaled)[:,1]
log_auc = roc_auc_score(y_test, log_preds)
print("\n--- Logistic Regression ---")
print("ROC-AUC:", log_auc)
print(classification_report(y_test, (log_preds > 0.5).astype(int)))

#random forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict_proba(X_test)[:,1]
rf_auc = roc_auc_score(y_test, rf_preds)
print("\n--- Random Forest ---")
print("ROC-AUC:", rf_auc)
print(classification_report(y_test, (rf_preds > 0.5).astype(int)))

#roc curve
fpr_log, tpr_log, _ = roc_curve(y_test, log_preds)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_preds)
plt.figure(figsize=(8,6))
plt.plot(fpr_log, tpr_log, label=f"Logistic (AUC={log_auc:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_auc:.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - PD Models")
plt.legend()
plt.show()

#feature importtance
importances = rf_model.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(9,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [FEATURES[i] for i in indices])
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.show()

#pd distribution
pd_df = pd.DataFrame({"PD": rf_preds, "default": y_test.values})
sns.histplot(data=pd_df, x="PD", hue="default", bins=30, kde=True)
plt.title("Predicted PD Distribution")
plt.show()

#saving the models

joblib.dump(log_model, "pd_logistic_model.pkl")
joblib.dump(rf_model, "pd_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

#loss engine
RECOVERY_RATE = 0.10
LGD = 1 - RECOVERY_RATE
log_model = joblib.load("pd_logistic_model.pkl")
rf_model = joblib.load("pd_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

def expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score,
    model="logistic"
):
    if model not in ["logistic", "rf"]:
        raise ValueError("model must be 'logistic' or 'rf'")
    debt_to_income = total_debt_outstanding / (income + 1)
    loan_to_income = loan_amt_outstanding / (income + 1)
    utilization_ratio = loan_amt_outstanding / (total_debt_outstanding + 1)
    credit_lines_per_year = credit_lines_outstanding / (years_employed + 1)
    X_new = pd.DataFrame([[
        credit_lines_outstanding,
        loan_amt_outstanding,
        total_debt_outstanding,
        income,
        years_employed,
        fico_score,
        debt_to_income,
        loan_to_income,
        utilization_ratio,
        credit_lines_per_year
    ]], columns=FEATURES)
    if model == "logistic":
        X_new = scaler.transform(X_new)
        pd_value = log_model.predict_proba(X_new)[0,1]
    else:
        pd_value = rf_model.predict_proba(X_new)[0,1]
    EL = pd_value * LGD * loan_amt_outstanding
    return round(pd_value,4), round(EL,2)

#example
pd_val, el = expected_loss(
    credit_lines_outstanding=3,
    loan_amt_outstanding=250000,
    total_debt_outstanding=400000,
    income=900000,
    years_employed=6,
    fico_score=680,
    model="rf"
)
print("\nPredicted PD:", pd_val)
print("Expected Loss:", el)
