# ============================================================
# CUSTOMER CHURN PREDICTION & RETENTION ANALYTICS
# 45-DAY DATA SCIENCE TRAINING PROJECT
# ============================================================

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
    confusion_matrix,
    classification_report
)

from sklearn.inspection import permutation_importance


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #666666;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
}

div[data-testid="stMetric"] {
    border: 1px solid #dddddd;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv("customer_churn.csv")

    # Remove duplicate records
    data = data.drop_duplicates()

    return data


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "customer_churn.csv was not found. "
        "Make sure it is in the same folder as app.py."
    )

    st.stop()


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

required_columns = [
    "customer_id",
    "gender",
    "city",
    "tenure_months",
    "plan_type",
    "monthly_charges",
    "contract_type",
    "payment_method",
    "monthly_usage_hours",
    "login_frequency",
    "support_tickets",
    "churn"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "The following required columns are missing from "
        "customer_churn.csv:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

# Fill numerical missing values

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# Fill categorical missing values

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    if not df[column].mode().empty:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# ============================================================
# TARGET CLEANING
# ============================================================

# Convert churn into 0 and 1 if required

if df["churn"].dtype == "object":

    churn_mapping = {
        "Yes": 1,
        "No": 0,
        "yes": 1,
        "no": 0,
        "Churn": 1,
        "Stay": 0,
        "Stayed": 0,
        "Churned": 1
    }

    df["churn"] = df["churn"].map(churn_mapping)


df["churn"] = pd.to_numeric(
    df["churn"],
    errors="coerce"
)

df = df.dropna(
    subset=["churn"]
)

df["churn"] = df["churn"].astype(int)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Customer Lifetime Value

df["customer_lifetime_value"] = (
    df["monthly_charges"] *
    df["tenure_months"]
)


# Support tickets per month

df["support_tickets_per_month"] = (
    df["support_tickets"] /
    df["tenure_months"].replace(0, 1)
)


# Usage per login

df["usage_per_login"] = (
    df["monthly_usage_hours"] /
    df["login_frequency"].replace(0, 1)
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["customer_id", "churn"]
)

y = df["churn"]


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()


# ============================================================
# PREPROCESSING
# ============================================================

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


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=8
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            max_depth=10
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
}


# ============================================================
# TRAIN ALL MODELS
# ============================================================

trained_models = {}

results = []


for model_name, model in models.items():

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )

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


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

    except:

        roc_auc = 0


    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc
    })


    trained_models[model_name] = pipeline


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Churn Analytics")

st.sidebar.write(
    "Customer Churn Prediction & Retention Analytics"
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


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-title">'
    'Customer Churn Prediction & Retention Analytics'
    '</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'End-to-end Data Science and Machine Learning Project'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 1. DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("🏠 Project Dashboard")

    total_customers = len(df)

    churned_customers = int(
        df["churn"].sum()
    )

    retained_customers = (
        total_customers -
        churned_customers
    )

    churn_rate = (
        churned_customers /
        total_customers
    ) * 100


    # Metrics

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "👥 Total Customers",
            total_customers
        )


    with col2:

        st.metric(
            "⚠️ Churned Customers",
            churned_customers
        )


    with col3:

        st.metric(
            "✅ Retained Customers",
            retained_customers
        )


    with col4:

        st.metric(
            "📉 Churn Rate",
            f"{churn_rate:.2f}%"
        )


    st.divider()


    # Churn distribution

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Customer Churn Distribution"
        )

        churn_counts = df["churn"].value_counts()

        labels = ["Retained", "Churned"]

        values = [
            churn_counts.get(0, 0),
            churn_counts.get(1, 0)
        ]

        fig, ax = plt.subplots()

        ax.bar(
            labels,
            values
        )

        ax.set_ylabel(
            "Number of Customers"
        )

        ax.set_title(
            "Churn vs Retained Customers"
        )

        st.pyplot(fig)


    with col2:

        st.subheader(
            "Customers by Plan Type"
        )

        plan_counts = (
            df["plan_type"]
            .value_counts()
        )

        st.bar_chart(
            plan_counts
        )


    st.divider()


    st.subheader(
        "🤖 Machine Learning Summary"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Models Trained",
            len(models)
        )


    with col2:

        st.metric(
            "Best Model",
            best_model_name
        )


    with col3:

        best_auc = results_df.iloc[0]["ROC-AUC"]

        st.metric(
            "Best ROC-AUC",
            f"{best_auc:.3f}"
        )


    st.info(
        "The project follows an end-to-end workflow: "
        "Data Cleaning → EDA → Feature Engineering → "
        "Model Training → Evaluation → Prediction → "
        "Risk Segmentation → Business Recommendations."
    )


# ============================================================
# 2. DATA PREPARATION
# ============================================================

elif page == "🧹 Data Preparation":

    st.header(
        "🧹 Data Collection & Preparation"
    )

    st.write(
        "This section shows the basic data preparation "
        "performed before machine learning."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with col3:

        st.metric(
            "Duplicate Rows",
            df.duplicated().sum()
        )


    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    st.subheader(
        "Missing Values"
    )

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({

        "Column": missing.index,

        "Missing Values": missing.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )


    st.subheader(
        "Data Types"
    )

    dtype_df = pd.DataFrame({

        "Column": df.columns,

        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ]
    })

    st.dataframe(
        dtype_df,
        use_container_width=True
    )


    st.subheader(
        "Engineered Features"
    )

    st.write(
        """
        **Customer Lifetime Value**

        Monthly Charges × Tenure

        **Support Tickets per Month**

        Support Tickets ÷ Tenure

        **Usage per Login**

        Monthly Usage Hours ÷ Login Frequency
        """
    )


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "📈 Exploratory Data Analysis":

    st.header(
        "📈 Exploratory Data Analysis"
    )

    st.write(
        "EDA helps identify patterns and relationships "
        "between customer characteristics and churn."
    )


    # Churn by contract

    st.subheader(
        "Churn Rate by Contract Type"
    )

    contract_churn = (
        df.groupby("contract_type")["churn"]
        .mean() * 100
    )

    st.bar_chart(
        contract_churn
    )


    # Churn by plan

    st.subheader(
        "Churn Rate by Plan Type"
    )

    plan_churn = (
        df.groupby("plan_type")["churn"]
        .mean() * 100
    )

    st.bar_chart(
        plan_churn
    )


    # Churn by payment method

    st.subheader(
        "Churn Rate by Payment Method"
    )

    payment_churn = (
        df.groupby("payment_method")["churn"]
        .mean() * 100
    )

    st.bar_chart(
        payment_churn
    )


    # Monthly charges

    st.subheader(
        "Monthly Charges Distribution"
    )

    fig, ax = plt.subplots()

    ax.hist(
        df["monthly_charges"],
        bins=20
    )

    ax.set_xlabel(
        "Monthly Charges"
    )

    ax.set_ylabel(
        "Number of Customers"
    )

    ax.set_title(
        "Distribution of Monthly Charges"
    )

    st.pyplot(fig)


    # Tenure

    st.subheader(
        "Tenure Distribution"
    )

    fig, ax = plt.subplots()

    ax.hist(
        df["tenure_months"],
        bins=20
    )

    ax.set_xlabel(
        "Tenure in Months"
    )

    ax.set_ylabel(
        "Number of Customers"
    )

    ax.set_title(
        "Customer Tenure Distribution"
    )

    st.pyplot(fig)


# ============================================================
# 4. MODEL COMPARISON
# ============================================================

elif page == "🤖 Model Comparison":

    st.header(
        "🤖 Machine Learning Model Comparison"
    )

    st.write(
        "Four classification algorithms were trained "
        "and compared using multiple evaluation metrics."
    )


    # Results table

    display_results = results_df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]:

        display_results[column] = (
            display_results[column] * 100
        ).round(2)


    st.dataframe(
        display_results,
        use_container_width=True
    )


    st.subheader(
        "ROC-AUC Comparison"
    )

    chart_data = results_df.set_index(
        "Model"
    )["ROC-AUC"]

    st.bar_chart(
        chart_data
    )


    st.success(
        f"Best performing model based on ROC-AUC: "
        f"{best_model_name}"
    )


    # Confusion matrix

    st.subheader(
        "Confusion Matrix – Best Model"
    )

    best_predictions = best_model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        best_predictions
    )

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_title(
        f"Confusion Matrix - {best_model_name}"
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Retained", "Churned"]
    )

    ax.set_yticklabels(
        ["Retained", "Churned"]
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


    # Classification report

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        best_predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )


# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================

elif page == "🔍 Feature Importance":

    st.header(
        "🔍 Important Features"
    )

    st.write(
        "Permutation importance estimates how much each "
        "original feature contributes to model performance."
    )


    with st.spinner(
        "Calculating feature importance..."
    ):

        importance_result = permutation_importance(

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
            importance_result.importances_mean
    })


    importance_df = importance_df.sort_values(

        by="Importance",

        ascending=False
    )


    st.dataframe(
        importance_df,
        use_container_width=True
    )


    st.subheader(
        "Feature Importance Chart"
    )


    chart_df = (
        importance_df
        .head(10)
        .set_index("Feature")
    )

    st.bar_chart(
        chart_df
    )


# ============================================================
# 6. CHURN PREDICTION
# ============================================================

elif page == "🔮 Churn Prediction":

    st.header(
        "🔮 Customer Churn Prediction"
    )

    st.write(
        "Enter customer details to estimate the probability "
        "that the customer may churn."
    )


    input_values = {}


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # CATEGORICAL INPUTS
    # --------------------------------------------------------

    with col1:

        for column in categorical_features:

            options = sorted(
                df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if len(options) > 0:

                input_values[column] = st.selectbox(

                    column.replace(
                        "_",
                        " "
                    ).title(),

                    options
                )


    # --------------------------------------------------------
    # NUMERICAL INPUTS
    # --------------------------------------------------------

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


            if column == "tenure_months":

                input_values[column] = st.number_input(

                    column.replace(
                        "_",
                        " "
                    ).title(),

                    min_value=max(
                        1.0,
                        minimum
                    ),

                    max_value=max(
                        maximum,
                        1.0
                    ),

                    value=max(
                        median,
                        1.0
                    )
                )

            else:

                input_values[column] = st.number_input(

                    column.replace(
                        "_",
                        " "
                    ).title(),

                    min_value=minimum,

                    max_value=maximum,

                    value=median
                )


    st.divider()


    if st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    ):

        customer_input = pd.DataFrame(
            [input_values]
        )


        # Prediction

        prediction = best_model.predict(
            customer_input
        )[0]


        probability = best_model.predict_proba(
            customer_input
        )[0][1]


        probability_percent = (
            probability * 100
        )


        # Risk

        if probability < 0.30:

            risk = "LOW RISK"

        elif probability < 0.70:

            risk = "MEDIUM RISK"

        else:

            risk = "HIGH RISK"


        st.divider()


        result_col1, result_col2, result_col3 = st.columns(3)


        with result_col1:

            st.metric(
                "Churn Probability",
                f"{probability_percent:.2f}%"
            )


        with result_col2:

            st.metric(
                "Risk Level",
                risk
            )


        with result_col3:

            st.metric(
                "Prediction",
                "Likely Churn" if prediction == 1
                else "Likely Stay"
            )


        # Recommendation

        if probability >= 0.70:

            st.error(
                "⚠️ High-risk customer detected."
            )

            st.subheader(
                "Recommended Action"
            )

            st.write(
                """
                • Contact the customer proactively

                • Offer a personalized retention plan

                • Investigate recent support issues

                • Consider a suitable loyalty offer

                • Encourage product engagement
                """
            )


        elif probability >= 0.30:

            st.warning(
                "⚠️ Medium-risk customer detected."
            )

            st.subheader(
                "Recommended Action"
            )

            st.write(
                """
                • Monitor customer activity

                • Send targeted engagement messages

                • Track support requests

                • Offer relevant product information
                """
            )


        else:

            st.success(
                "✅ Low-risk customer detected."
            )

            st.subheader(
                "Recommended Action"
            )

            st.write(
                """
                • Maintain normal engagement

                • Continue quality customer support

                • Encourage regular product usage

                • Consider loyalty benefits
                """
            )


# ============================================================
# 7. RISK SEGMENTATION
# ============================================================

elif page == "⚠️ Risk Segmentation":

    st.header(
        "⚠️ Customer Risk Segmentation"
    )

    st.write(
        "Customers are divided into risk groups using "
        "their predicted churn probabilities."
    )


    all_probabilities = best_model.predict_proba(
        X
    )[:, 1]


    risk_df = df[
        [
            "customer_id"
        ]
    ].copy()


    risk_df["Churn Probability"] = (
        all_probabilities
    )


    risk_df["Churn Probability (%)"] = (
        all_probabilities * 100
    ).round(2)


    def assign_risk(probability):

        if probability < 0.30:

            return "Low Risk"

        elif probability < 0.70:

            return "Medium Risk"

        else:

            return "High Risk"


    risk_df["Risk Level"] = (
        risk_df["Churn Probability"]
        .apply(assign_risk)
    )


    risk_counts = (
        risk_df["Risk Level"]
        .value_counts()
    )


    # Metrics

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🟢 Low Risk",
            risk_counts.get(
                "Low Risk",
                0
            )
        )


    with col2:

        st.metric(
            "🟡 Medium Risk",
            risk_counts.get(
                "Medium Risk",
                0
            )
        )


    with col3:

        st.metric(
            "🔴 High Risk",
            risk_counts.get(
                "High Risk",
                0
            )
        )


    st.divider()


    st.subheader(
        "Risk Distribution"
    )


    st.bar_chart(
        risk_counts
    )


    st.subheader(
        "Risk Segmentation Rules"
    )


    segmentation = pd.DataFrame({

        "Probability":
            [
                "0–30%",
                "30–70%",
                "70–100%"
            ],

        "Risk":
            [
                "Low",
                "Medium",
                "High"
            ],

        "Recommended Action":
            [
                "Normal engagement",
                "Monitor and targeted engagement",
                "Proactive retention action"
            ]
    })


    st.dataframe(
        segmentation,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Customer Risk List"
    )


    st.dataframe(
        risk_df.sort_values(
            "Churn Probability",
            ascending=False
        ),
        use_container_width=True
    )


    # Download risk report

    csv = risk_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(

        "⬇️ Download Risk Report",

        data=csv,

        file_name="customer_risk_segments.csv",

        mime="text/csv"
    )


# ============================================================
# 8. BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "💡 Business Recommendations":

    st.header(
        "💡 Business Recommendations"
    )

    st.write(
        "The purpose of churn prediction is not only "
        "to predict customers who may leave, but also "
        "to help the business take preventive action."
    )


    st.subheader(
        "🔴 High-Risk Customers"
    )

    st.write(
        """
        **Probability: 70–100%**

        Recommended actions:

        • Proactive customer support

        • Personalized retention offers

        • Investigate customer complaints

        • Review payment-related problems

        • Provide loyalty incentives
        """
    )


    st.subheader(
        "🟡 Medium-Risk Customers"
    )

    st.write(
        """
        **Probability: 30–70%**

        Recommended actions:

        • Monitor customer activity

        • Send targeted engagement campaigns

        • Track support tickets

        • Encourage product usage

        • Provide relevant offers
        """
    )


    st.subheader(
        "🟢 Low-Risk Customers"
    )

    st.write(
        """
        **Probability: 0–30%**

        Recommended actions:

        • Maintain normal engagement

        • Continue good customer service

        • Encourage long-term subscriptions

        • Offer loyalty benefits
        """
    )


    st.divider()


    st.subheader(
        "🎯 Business Objective"
    )

    st.info(
        "Use machine learning predictions to identify "
        "customers at risk early enough for the business "
        "to take preventive retention actions."
    )


# ============================================================
# 9. CUSTOMER DATA
# ============================================================

elif page == "📋 Customer Data":

    st.header(
        "📋 Customer Dataset"
    )

    st.write(
        "Explore the cleaned customer dataset used "
        "for analysis and machine learning."
    )


    search = st.text_input(
        "🔎 Search customer data"
    )


    if search:

        filtered_df = df[
            df.astype(str)
            .apply(

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


    csv = df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(

        "⬇️ Download Cleaned Dataset",

        data=csv,

        file_name="cleaned_customer_churn.csv",

        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction & Retention Analytics | "
    "45-Day Data Science Training Project"
)