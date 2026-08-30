import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.inspection import permutation_importance


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Telco-Customer-Churn.csv")

    # Remove unnecessary ID
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges sometimes contains blank values
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing numerical values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    return df


df = load_data()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["AverageMonthlySpend"] = (
    df["TotalCharges"] /
    df["tenure"].replace(0, 1)
)

df["TotalServices"] = (
    (df["PhoneService"] == "Yes").astype(int)
    + (df["OnlineSecurity"] == "Yes").astype(int)
    + (df["OnlineBackup"] == "Yes").astype(int)
    + (df["DeviceProtection"] == "Yes").astype(int)
    + (df["TechSupport"] == "Yes").astype(int)
    + (df["StreamingTV"] == "Yes").astype(int)
    + (df["StreamingMovies"] == "Yes").astype(int)
)

# Convert target
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# =========================================================
# MACHINE LEARNING DATA
# =========================================================

X = df.drop(columns=["Churn"])
y = df["Churn"]

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()


# =========================================================
# PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            StandardScaler(),
            numerical_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# =========================================================
# MACHINE LEARNING MODELS
# =========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=120,
            random_state=42
        )
}


# =========================================================
# TRAIN MODELS
# =========================================================

trained_models = {}

results = []


for name, model in models.items():

    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    results.append({

        "Model": name,

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "F1 Score":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "ROC-AUC":
            roc_auc_score(
                y_test,
                probabilities
            )
    })

    trained_models[name] = pipeline


results_df = pd.DataFrame(
    results
).sort_values(
    "ROC-AUC",
    ascending=False
)


best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "📊 Churn Analytics"
)

page = st.sidebar.radio(

    "Select Section",

    [

        "🏠 Dashboard",

        "🧹 Data Preparation",

        "📈 Exploratory Data Analysis",

        "🤖 Model Comparison",

        "🔍 Feature Importance",

        "🔮 Churn Prediction",

        "⚠️ Risk Segmentation",

        "💡 Business Recommendations",

        "📋 Customer Data"

    ]
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "📊 Customer Churn Prediction & Retention Analytics"
)

st.caption(
    "45-Day Data Science Training Capstone Project"
)

st.divider()


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.header(
        "🏠 Project Dashboard"
    )

    total_customers = len(df)

    churned_customers = int(
        df["Churn"].sum()
    )

    retained_customers = (
        total_customers -
        churned_customers
    )

    churn_rate = (
        churned_customers /
        total_customers *
        100
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "👥 Total Customers",
        total_customers
    )

    col2.metric(
        "⚠️ Churned Customers",
        churned_customers
    )

    col3.metric(
        "✅ Retained Customers",
        retained_customers
    )

    col4.metric(
        "📉 Churn Rate",
        f"{churn_rate:.2f}%"
    )


    st.divider()


    col1, col2 = st.columns(2)


    # Churn distribution
    with col1:

        st.subheader(
            "Customer Churn Distribution"
        )

        counts = df["Churn"].value_counts()

        fig, ax = plt.subplots()

        ax.bar(
            ["Retained", "Churned"],
            [
                counts.get(0, 0),
                counts.get(1, 0)
            ]
        )

        ax.set_ylabel(
            "Number of Customers"
        )

        ax.set_title(
            "Churn vs Retained"
        )

        st.pyplot(fig)


    # Contract
    with col2:

        st.subheader(
            "Customers by Contract"
        )

        st.bar_chart(
            df["Contract"].value_counts()
        )


    st.divider()


    st.subheader(
        "🤖 Machine Learning Summary"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Models Trained",
        len(models)
    )

    col2.metric(
        "Best Model",
        best_model_name
    )

    col3.metric(
        "Best ROC-AUC",
        f"{results_df.iloc[0]['ROC-AUC']:.3f}"
    )


    st.info(
        "Complete workflow: "
        "Data Cleaning → EDA → Feature Engineering → "
        "Machine Learning → Model Evaluation → "
        "Feature Importance → Risk Segmentation → "
        "Business Recommendations."
    )


# =========================================================
# DATA PREPARATION
# =========================================================

elif page == "🧹 Data Preparation":

    st.header(
        "🧹 Data Preparation"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(15),
        use_container_width=True
    )


    st.subheader(
        "Dataset Information"
    )

    info_df = pd.DataFrame({

        "Column":
            df.columns,

        "Data Type":
            df.dtypes.astype(str),

        "Missing Values":
            df.isnull().sum().values

    })

    st.dataframe(
        info_df,
        use_container_width=True
    )


    st.subheader(
        "⚙️ Feature Engineering"
    )

    st.write(
        "• Average Monthly Spend = Total Charges ÷ Tenure"
    )

    st.write(
        "• Total Services = Number of subscribed services"
    )


# =========================================================
# EDA
# =========================================================

elif page == "📈 Exploratory Data Analysis":

    st.header(
        "📈 Exploratory Data Analysis"
    )


    st.subheader(
        "Churn by Contract Type"
    )

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    st.bar_chart(
        contract_churn
    )


    st.subheader(
        "Churn by Internet Service"
    )

    internet_churn = (
        df.groupby("InternetService")["Churn"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    st.bar_chart(
        internet_churn
    )


    st.subheader(
        "Churn by Payment Method"
    )

    payment_churn = (
        df.groupby("PaymentMethod")["Churn"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    st.bar_chart(
        payment_churn
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Monthly Charges"
        )

        fig, ax = plt.subplots()

        ax.hist(
            df["MonthlyCharges"],
            bins=25
        )

        ax.set_xlabel(
            "Monthly Charges"
        )

        ax.set_ylabel(
            "Customers"
        )

        st.pyplot(fig)


    with col2:

        st.subheader(
            "Customer Tenure"
        )

        fig, ax = plt.subplots()

        ax.hist(
            df["tenure"],
            bins=20
        )

        ax.set_xlabel(
            "Tenure (Months)"
        )

        ax.set_ylabel(
            "Customers"
        )

        st.pyplot(fig)


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "🤖 Model Comparison":

    st.header(
        "🤖 Machine Learning Model Comparison"
    )


    display_df = results_df.copy()


    for column in [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC-AUC"

    ]:

        display_df[column] = (
            display_df[column] * 100
        ).round(2)


    st.dataframe(
        display_df,
        use_container_width=True
    )


    st.subheader(
        "ROC-AUC Comparison"
    )

    st.bar_chart(
        results_df.set_index("Model")[
            "ROC-AUC"
        ]
    )


    st.success(
        f"Best performing model: {best_model_name}"
    )


    # Confusion Matrix

    predictions = best_model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )


    st.subheader(
        "Confusion Matrix"
    )


    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        f"Confusion Matrix - {best_model_name}"
    )


    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )


    st.pyplot(fig)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

elif page == "🔍 Feature Importance":

    st.header(
        "🔍 Feature Importance"
    )


    st.write(
        "Permutation importance shows which original customer attributes have the greatest impact on model performance."
    )


    importance = permutation_importance(

        best_model,

        X_test,

        y_test,

        n_repeats=5,

        random_state=42,

        scoring="roc_auc"
    )


    importance_df = pd.DataFrame({

        "Feature":
            X_test.columns,

        "Importance":
            importance.importances_mean

    }).sort_values(
        "Importance",
        ascending=False
    )


    st.dataframe(
        importance_df,
        use_container_width=True
    )


    st.subheader(
        "Top Churn Drivers"
    )


    st.bar_chart(
        importance_df.head(10)
        .set_index("Feature")
    )


# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "🔮 Churn Prediction":

    st.header(
        "🔮 Individual Customer Churn Prediction"
    )


    st.write(
        "Enter customer details to estimate the probability of churn."
    )


    input_data = {}


    col1, col2 = st.columns(2)


    with col1:

        for column in categorical_features:

            options = sorted(
                df[column]
                .dropna()
                .astype(str)
                .unique()
            )

            input_data[column] = st.selectbox(

                column.replace(
                    "_", " "
                ).title(),

                options
            )


    with col2:

        for column in numerical_features:

            minimum = float(
                df[column].min()
            )

            maximum = float(
                df[column].max()
            )

            median = float(
                df[column].median()
            )

            input_data[column] = st.number_input(

                column.replace(
                    "_", " "
                ).title(),

                min_value=minimum,

                max_value=maximum,

                value=median
            )


    if st.button(
        "🔮 Predict Churn",
        use_container_width=True
    ):

        customer = pd.DataFrame(
            [input_data]
        )


        probability = best_model.predict_proba(
            customer
        )[0][1]


        prediction = int(
            probability >= 0.5
        )


        if probability < 0.30:

            risk = "Low Risk"

        elif probability < 0.70:

            risk = "Medium Risk"

        else:

            risk = "High Risk"


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


        col2.metric(
            "Risk Level",
            risk
        )


        col3.metric(

            "Prediction",

            "Likely Churn"
            if prediction == 1
            else "Likely Stay"
        )


        if risk == "High Risk":

            st.error(
                "⚠️ High-risk customer. "
                "Proactive retention action is recommended."
            )

        elif risk == "Medium Risk":

            st.warning(
                "⚠️ Medium-risk customer. "
                "Monitor engagement and activity."
            )

        else:

            st.success(
                "✅ Low-risk customer. "
                "Continue normal engagement."
            )


# =========================================================
# RISK SEGMENTATION
# =========================================================

elif page == "⚠️ Risk Segmentation":

    st.header(
        "⚠️ Customer Risk Segmentation"
    )


    probabilities = best_model.predict_proba(
        X
    )[:, 1]


    risk_df = pd.DataFrame({

        "Customer":
            range(1, len(df) + 1),

        "Churn Probability (%)":
            probabilities * 100

    })


    risk_df["Risk Level"] = np.where(

        probabilities < 0.30,

        "Low Risk",

        np.where(

            probabilities < 0.70,

            "Medium Risk",

            "High Risk"
        )
    )


    counts = risk_df[
        "Risk Level"
    ].value_counts()


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "🟢 Low Risk",
        counts.get(
            "Low Risk",
            0
        )
    )


    col2.metric(
        "🟡 Medium Risk",
        counts.get(
            "Medium Risk",
            0
        )
    )


    col3.metric(
        "🔴 High Risk",
        counts.get(
            "High Risk",
            0
        )
    )


    st.subheader(
        "Risk Distribution"
    )


    st.bar_chart(
        counts
    )


    st.subheader(
        "Customer Risk Report"
    )


    risk_df["Churn Probability (%)"] = (
        risk_df["Churn Probability (%)"]
        .round(2)
    )


    st.dataframe(
        risk_df.sort_values(
            "Churn Probability (%)",
            ascending=False
        ),
        use_container_width=True
    )


    st.download_button(

        "⬇️ Download Risk Report",

        risk_df.to_csv(
            index=False
        ).encode("utf-8"),

        "customer_risk_report.csv",

        "text/csv"
    )


# =========================================================
# BUSINESS RECOMMENDATIONS
# =========================================================

elif page == "💡 Business Recommendations":

    st.header(
        "💡 Business Recommendations"
    )


    st.subheader(
        "🔴 High-Risk Customers"
    )

    st.write(
        """
        • Offer personalized retention plans.

        • Contact customers with repeated service issues.

        • Provide targeted discounts where appropriate.

        • Encourage longer-term contracts.

        • Prioritize customers with high churn probability.
        """
    )


    st.subheader(
        "🟡 Medium-Risk Customers"
    )

    st.write(
        """
        • Monitor customer engagement.

        • Send personalized communication.

        • Track support interactions.

        • Encourage greater product/service usage.
        """
    )


    st.subheader(
        "🟢 Low-Risk Customers"
    )

    st.write(
        """
        • Maintain service quality.

        • Encourage loyalty.

        • Promote long-term plans.

        • Continue regular engagement.
        """
    )


    st.divider()


    st.info(
        "The purpose of churn prediction is not only to identify customers likely to leave, but to provide actionable insights for improving customer retention."
    )


# =========================================================
# CUSTOMER DATA
# =========================================================

elif page == "📋 Customer Data":

    st.header(
        "📋 Customer Dataset"
    )


    search = st.text_input(
        "🔎 Search customer data"
    )


    if search:

        filtered_df = df[
            df.astype(str).apply(

                lambda row:

                row.str.contains(
                    search,
                    case=False,
                    na=False
                ).any(),

                axis=1
            )
        ]

    else:

        filtered_df = df


    st.dataframe(
        filtered_df,
        use_container_width=True
    )


    st.download_button(

        "⬇️ Download Dataset",

        df.to_csv(
            index=False
        ).encode("utf-8"),

        "telco_customer_churn_cleaned.csv",

        "text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Customer Churn Prediction & Retention Analytics | "
    "45-Day Data Science Training Project"
)