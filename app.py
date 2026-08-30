import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    font-size: 18px;
    color: #666;
}

.metric-card {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv("customer_churn.csv")

    data = data.drop_duplicates()

    return data


df = load_data()


# =========================================================
# DATA PREPARATION
# =========================================================

# Fill missing numerical values

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# Fill missing categorical values

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    if not df[column].mode().empty:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["customer_lifetime_value"] = (
    df["monthly_charges"] *
    df["tenure_months"]
)

df["support_tickets_per_month"] = (
    df["support_tickets"] /
    df["tenure_months"].replace(0, 1)
)

df["usage_per_login"] = (
    df["monthly_usage_hours"] /
    df["login_frequency"].replace(0, 1)
)


# =========================================================
# ENCODING
# =========================================================

encoders = {}

categorical_features = [
    "gender",
    "city",
    "plan_type",
    "contract_type",
    "payment_method"
]

for column in categorical_features:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

    encoders[column] = encoder


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop(
    columns=[
        "customer_id",
        "churn"
    ]
)

y = df["churn"]


# =========================================================
# TRAIN MODEL
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# =========================================================
# MODEL ACCURACY
# =========================================================

test_prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    test_prediction
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Churn Analytics")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Dashboard",
        "📈 Churn Analysis",
        "🤖 Churn Prediction",
        "📋 Customer Data"
    ]
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<p class="main-title">Customer Churn Analytics</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning based customer retention analysis</p>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# DASHBOARD
# =========================================================

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
            "⚠️ Churned",
            churned_customers
        )


    with col3:

        st.metric(
            "✅ Retained",
            retained_customers
        )


    with col4:

        st.metric(
            "📉 Churn Rate",
            f"{churn_rate:.1f}%"
        )


    st.divider()


    # Charts

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Customer Churn")

        churn_data = pd.DataFrame({
            "Status": [
                "Retained",
                "Churned"
            ],

            "Customers": [
                retained_customers,
                churned_customers
            ]
        })

        st.bar_chart(
            churn_data.set_index("Status")
        )


    with col2:

        st.subheader("Customers by Plan")

        plan_data = (
            df.groupby("plan_type")
            .size()
        )

        st.bar_chart(plan_data)


    st.divider()


    st.subheader("🤖 Machine Learning Model")

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Model",
            "Random Forest"
        )


    with col2:

        st.metric(
            "Test Accuracy",
            f"{accuracy * 100:.2f}%"
        )


# =========================================================
# CHURN ANALYSIS
# =========================================================

elif page == "📈 Churn Analysis":

    st.header("📈 Churn Analysis")

    st.write(
        "Explore patterns that may be associated with customer churn."
    )


    # Contract analysis

    st.subheader("Churn by Contract Type")

    contract_analysis = (
        df.groupby("contract_type")["churn"]
        .mean()
    )

    st.bar_chart(
        contract_analysis
    )


    # Monthly charges

    st.subheader("Monthly Charges Distribution")

    st.line_chart(
        df["monthly_charges"]
    )


    # Tenure

    st.subheader("Customer Tenure")

    st.line_chart(
        df["tenure_months"]
    )


    # Feature importance

    st.subheader("🔍 Important Prediction Features")

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")
    )


# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "🤖 Churn Prediction":

    st.header("🤖 Customer Churn Prediction")

    st.write(
        "Enter customer information to estimate churn risk."
    )


    col1, col2 = st.columns(2)


    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        city = st.selectbox(
            "City",
            ["Amritsar", "Chandigarh", "Delhi",
             "Jalandhar", "Ludhiana", "Mumbai", "Pune"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=1,
            max_value=100,
            value=12
        )

        plan = st.selectbox(
            "Plan Type",
            ["Basic", "Premium", "Standard"]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=100.0,
            max_value=5000.0,
            value=799.0
        )


    with col2:

        contract = st.selectbox(
            "Contract Type",
            ["Monthly", "Quarterly", "Yearly"]
        )

        payment = st.selectbox(
            "Payment Method",
            ["Cash", "Card", "Net Banking",
             "UPI", "Wallet"]
        )

        usage = st.number_input(
            "Monthly Usage Hours",
            min_value=1.0,
            max_value=500.0,
            value=20.0
        )

        login_frequency = st.number_input(
            "Login Frequency",
            min_value=1,
            max_value=100,
            value=15
        )

        support_tickets = st.number_input(
            "Support Tickets",
            min_value=0,
            max_value=50,
            value=2
        )


    st.divider()


    if st.button(
        "🔮 Predict Churn",
        use_container_width=True
    ):

        # Encode values

        input_data = {}

        for column, value in {

            "gender": gender,

            "city": city,

            "plan_type": plan,

            "contract_type": contract,

            "payment_method": payment

        }.items():

            encoder = encoders[column]

            try:

                input_data[column] = (
                    encoder.transform([value])[0]
                )

            except ValueError:

                st.error(
                    f"Unknown value for {column}"
                )

                st.stop()


        # Derived features

        lifetime_value = (
            monthly_charges *
            tenure
        )

        tickets_per_month = (
            support_tickets /
            max(tenure, 1)
        )

        usage_per_login = (
            usage /
            max(login_frequency, 1)
        )


        # Create customer dataframe

        customer = pd.DataFrame({

            "gender": [
                input_data["gender"]
            ],

            "city": [
                input_data["city"]
            ],

            "tenure_months": [
                tenure
            ],

            "plan_type": [
                input_data["plan_type"]
            ],

            "monthly_charges": [
                monthly_charges
            ],

            "contract_type": [
                input_data["contract_type"]
            ],

            "payment_method": [
                input_data["payment_method"]
            ],

            "monthly_usage_hours": [
                usage
            ],

            "login_frequency": [
                login_frequency
            ],

            "support_tickets": [
                support_tickets
            ],

            "customer_lifetime_value": [
                lifetime_value
            ],

            "support_tickets_per_month": [
                tickets_per_month
            ],

            "usage_per_login": [
                usage_per_login
            ]
        })


        # Add missing columns if required

        for column in X.columns:

            if column not in customer.columns:

                customer[column] = 0


        customer = customer[
            X.columns
        ]


        # Prediction

        prediction = model.predict(
            customer
        )[0]

        probability = model.predict_proba(
            customer
        )[0][1]


        # Results

        st.divider()

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )


        with result_col2:

            if probability >= 0.70:

                risk = "HIGH RISK"

            elif probability >= 0.30:

                risk = "MEDIUM RISK"

            else:

                risk = "LOW RISK"


            st.metric(
                "Risk Level",
                risk
            )


        # Result message

        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to churn."
            )

            st.subheader(
                "💡 Recommended Actions"
            )

            st.write(
                """
                • Contact the customer proactively  
                • Offer a personalized retention plan  
                • Check recent support issues  
                • Review payment difficulties  
                • Encourage regular product usage
                """
            )

        else:

            st.success(
                "✅ Customer is likely to stay."
            )

            st.subheader(
                "💡 Recommended Actions"
            )

            st.write(
                """
                • Maintain customer engagement  
                • Continue quality support  
                • Provide loyalty benefits  
                • Encourage regular usage
                """
            )


# =========================================================
# CUSTOMER DATA
# =========================================================

elif page == "📋 Customer Data":

    st.header("📋 Customer Dataset")

    st.write(
        "Explore the customer records used in the project."
    )


    # Search

    search = st.text_input(
        "🔎 Search customer"
    )


    if search:

        filtered_data = df[
            df.astype(str)
            .apply(
                lambda row:
                row.str.contains(
                    search,
                    case=False
                ).any(),
                axis=1
            )
        ]

        st.dataframe(
            filtered_data,
            use_container_width=True
        )

    else:

        st.dataframe(
            df,
            use_container_width=True
        )


    # Download

    csv = df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "⬇️ Download Dataset",
        data=csv,
        file_name="customer_churn_analysis.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Customer Churn Analytics | Machine Learning Project | 45-Day Training"
)