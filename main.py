# ============================================================
# CUSTOMER CHURN PREDICTION
# Complete Beginner-Friendly Data Science Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("       CUSTOMER CHURN PREDICTION PROJECT")
print("=" * 60)

df = pd.read_csv("customer_churn.csv")

print("\nDataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nNumber of rows and columns:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing numerical values
numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# Fill missing text values
text_columns = df.select_dtypes(
    include="object"
).columns

for column in text_columns:

    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nCustomer churn count:")
print(df["churn"].value_counts())

print("\nChurn percentage:")

churn_percentage = (
    df["churn"].value_counts(normalize=True) * 100
)

print(churn_percentage)


# ============================================================
# GRAPH 1 - CHURN DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

df["churn"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
plt.ylabel("Number of Customers")

plt.xticks(
    [0, 1],
    ["Stayed", "Churned"],
    rotation=0
)

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 2 - CONTRACT TYPE VS CHURN
# ============================================================

contract_churn = df.groupby(
    "contract_type"
)["churn"].mean()


plt.figure(figsize=(7, 5))

contract_churn.plot(
    kind="bar"
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 3 - MONTHLY CHARGES VS CHURN
# ============================================================

plt.figure(figsize=(7, 5))

df.boxplot(
    column="monthly_charges",
    by="churn"
)

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
plt.ylabel("Monthly Charges")

plt.suptitle("")

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 4 - TENURE VS CHURN
# ============================================================

plt.figure(figsize=(7, 5))

df.boxplot(
    column="tenure_months",
    by="churn"
)

plt.title("Customer Tenure vs Churn")
plt.xlabel("Churn (0 = Stayed, 1 = Churned)")
plt.ylabel("Tenure (Months)")

plt.suptitle("")

plt.tight_layout()
plt.show()


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


# Customer Lifetime Value

df["customer_lifetime_value"] = (
    df["monthly_charges"]
    *
    df["tenure_months"]
)


# Support Tickets per Month

df["support_tickets_per_month"] = (
    df["support_tickets"]
    /
    df["tenure_months"].replace(0, 1)
)


# Usage per Login

df["usage_per_login"] = (
    df["monthly_usage_hours"]
    /
    df["login_frequency"].replace(0, 1)
)


print("\nNew features created:")

print(
    df[
        [
            "customer_lifetime_value",
            "support_tickets_per_month",
            "usage_per_login"
        ]
    ].head()
)


# ============================================================
# 6. CONVERT TEXT DATA INTO NUMBERS
# ============================================================

print("\n" + "=" * 60)
print("CONVERTING TEXT DATA INTO NUMBERS")
print("=" * 60)


encoder = LabelEncoder()


df["gender"] = encoder.fit_transform(
    df["gender"]
)

df["city"] = encoder.fit_transform(
    df["city"]
)

df["plan_type"] = encoder.fit_transform(
    df["plan_type"]
)

df["contract_type"] = encoder.fit_transform(
    df["contract_type"]
)

df["payment_method"] = encoder.fit_transform(
    df["payment_method"]
)


print("\nData after encoding:")
print(df.head())


# ============================================================
# 7. SELECT FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 60)
print("SELECTING FEATURES AND TARGET")
print("=" * 60)


# X = input data
X = df.drop(
    columns=[
        "customer_id",
        "churn"
    ]
)


# y = target
y = df["churn"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("churn")


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 9. DECISION TREE MODEL
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE MODEL")
print("=" * 60)


decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)


decision_tree.fit(
    X_train,
    y_train
)


# Prediction

dt_prediction = decision_tree.predict(
    X_test
)


# Accuracy

dt_accuracy = accuracy_score(
    y_test,
    dt_prediction
)


print("\nDecision Tree Accuracy:")
print(round(dt_accuracy * 100, 2), "%")


# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST MODEL")
print("=" * 60)


random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


random_forest.fit(
    X_train,
    y_train
)


# Prediction

rf_prediction = random_forest.predict(
    X_test
)


# Accuracy

rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)


print("\nRandom Forest Accuracy:")
print(round(rf_accuracy * 100, 2), "%")


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)


print(
    "\nDecision Tree:",
    round(dt_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest:",
    round(rf_accuracy * 100, 2),
    "%"
)


# Select better model

if rf_accuracy >= dt_accuracy:

    best_model = random_forest
    best_prediction = rf_prediction
    best_model_name = "Random Forest"

else:

    best_model = decision_tree
    best_prediction = dt_prediction
    best_model_name = "Decision Tree"


print("\nBest Model:")
print(best_model_name)


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


accuracy = accuracy_score(
    y_test,
    best_prediction
)


precision = precision_score(
    y_test,
    best_prediction,
    zero_division=0
)


recall = recall_score(
    y_test,
    best_prediction,
    zero_division=0
)


f1 = f1_score(
    y_test,
    best_prediction,
    zero_division=0
)


print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")


print("\nPrecision:")
print(round(precision * 100, 2), "%")


print("\nRecall:")
print(round(recall * 100, 2), "%")


print("\nF1 Score:")
print(round(f1 * 100, 2), "%")


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)


print(
    classification_report(
        y_test,
        best_prediction,
        target_names=[
            "Stayed",
            "Churned"
        ],
        zero_division=0
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)


cm = confusion_matrix(
    y_test,
    best_prediction
)


print(cm)


# Display confusion matrix

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    "Confusion Matrix - " + best_model_name
)

plt.xlabel("Predicted")

plt.ylabel("Actual")


plt.xticks(
    [0, 1],
    ["Stayed", "Churned"]
)

plt.yticks(
    [0, 1],
    ["Stayed", "Churned"]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()

plt.show()


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


if best_model_name == "Random Forest":

    importance = random_forest.feature_importances_

else:

    importance = decision_tree.feature_importances_


feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": importance

})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print(
    feature_importance
)


# Feature importance graph

plt.figure(figsize=(9, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title(
    "Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ============================================================
# 16. CUSTOMER CHURN PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)


# Select first customer

customer = X.iloc[
    0:1
]


prediction = best_model.predict(
    customer
)


probability = best_model.predict_proba(
    customer
)[0][1]


print("\nCustomer Churn Probability:")

print(
    round(
        probability * 100,
        2
    ),
    "%"
)


if prediction[0] == 1:

    print(
        "\nPrediction: CUSTOMER IS LIKELY TO CHURN"
    )

else:

    print(
        "\nPrediction: CUSTOMER IS LIKELY TO STAY"
    )


# ============================================================
# 17. CUSTOMER RISK LEVEL
# ============================================================

if probability < 0.30:

    risk = "LOW RISK"

elif probability < 0.70:

    risk = "MEDIUM RISK"

else:

    risk = "HIGH RISK"


print("\nCustomer Risk Level:")

print(risk)


# ============================================================
# 18. BUSINESS RECOMMENDATION
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS RECOMMENDATION")
print("=" * 60)


if risk == "HIGH RISK":

    print(
        """
Recommended Actions:

1. Contact the customer.
2. Provide a personalized retention offer.
3. Check recent support problems.
4. Check payment issues.
5. Offer a suitable plan.
"""
    )


elif risk == "MEDIUM RISK":

    print(
        """
Recommended Actions:

1. Monitor customer activity.
2. Send targeted engagement messages.
3. Provide product support.
4. Encourage regular usage.
"""
    )


else:

    print(
        """
Recommended Actions:

1. Continue normal engagement.
2. Encourage customer loyalty.
3. Offer referral benefits.
4. Maintain good customer support.
"""
    )


# ============================================================
# PROJECT COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nBest Model:", best_model_name)

print(
    "Final Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nCustomer Churn Prediction project finished.")