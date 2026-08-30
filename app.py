import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# =========================================
# PAGE SETTINGS
# =========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================
# TITLE
# =========================================

st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a customer is likely to leave the service."
)


# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv("customer_churn.csv")


# =========================================
# DATA PREPARATION
# =========================================

df = df.drop_duplicates()

# Feature engineering

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


# =========================================
# ENCODE CATEGORICAL DATA
# =========================================

encoder = LabelEncoder()

categorical_columns = [
    "gender",
    "city",
    "plan_type",
    "contract_type",
    "payment_method"
]

for column in categorical_columns:

    df[column] = encoder.fit_transform(
        df[column]
    )


# =========================================
# FEATURES AND TARGET
# =========================================

X = df.drop(
    columns=[
        "customer_id",
        "churn"
    ]
)

y = df["churn"]


# =========================================
# TRAIN MODEL
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
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


# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("Customer Information")


gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)


city = st.sidebar.selectbox(
    "City",
    [
        "Amritsar",
        "Ludhiana",
        "Delhi",
        "Chandigarh",
        "Jalandhar",
        "Pune",
        "Mumbai"
    ]
)


tenure = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=1,
    max_value=100,
    value=12
)


plan = st.sidebar.selectbox(
    "Plan Type",
    ["Basic", "Standard", "Premium"]
)


monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=100,
    max_value=5000,
    value=799
)


contract = st.sidebar.selectbox(
    "Contract Type",
    ["Monthly", "Quarterly", "Yearly"]
)


payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "UPI",
        "Card",
        "Cash",
        "Wallet",
        "Net Banking"
    ]
)


usage = st.sidebar.number_input(
    "Monthly Usage Hours",
    min_value=1,
    max_value=200,
    value=20
)


login_frequency = st.sidebar.number_input(
    "Login Frequency",
    min_value=1,
    max_value=100,
    value=15
)


support_tickets = st.sidebar.number_input(
    "Support Tickets",
    min_value=0,
    max_value=50,
    value=2
)


payment_delays = st.sidebar.number_input(
    "Payment Delays",
    min_value=0,
    max_value=20,
    value=0
)


session_minutes = st.sidebar.number_input(
    "Average Session (Minutes)",
    min_value=1,
    max_value=300,
    value=30
)


# =========================================
# PREDICTION BUTTON
# =========================================

if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    # Encode user inputs using dataset categories

    gender_value = (
        1 if gender == "Male" else 0
    )

    city_value = encoder.fit(
        df["city"]
    )

    # Create mapping from original dataset

    city_mapping = {
        "Amritsar": 0,
        "Chandigarh": 1,
        "Delhi": 2,
        "Jalandhar": 3,
        "Ludhiana": 4,
        "Mumbai": 5,
        "Pune": 6
    }

    plan_mapping = {
        "Basic": 0,
        "Premium": 1,
        "Standard": 2
    }

    contract_mapping = {
        "Monthly": 0,
        "Quarterly": 1,
        "Yearly": 2
    }

    payment_mapping = {
        "Cash": 0,
        "Card": 1,
        "Net Banking": 2,
        "UPI": 3,
        "Wallet": 4
    }


    city_value = city_mapping[city]

    plan_value = plan_mapping[plan]

    contract_value = contract_mapping[contract]

    payment_value = payment_mapping[payment]


    # Feature engineering for new customer

    lifetime_value = (
        monthly_charges * tenure
    )

    tickets_per_month = (
        support_tickets /
        max(tenure, 1)
    )

    usage_per_login = (
        usage /
        max(login_frequency, 1)
    )


    # Create input dataframe

    customer_data = pd.DataFrame({

        "gender": [gender_value],

        "city": [city_value],

        "tenure_months": [tenure],

        "plan_type": [plan_value],

        "monthly_charges": [monthly_charges],

        "contract_type": [contract_value],

        "payment_method": [payment_value],

        "monthly_usage_hours": [usage],

        "login_frequency": [login_frequency],

        "support_tickets": [support_tickets],

        "payment_delays": [payment_delays],

        "avg_session_minutes": [session_minutes],

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


    # =========================================
    # MAKE PREDICTION
    # =========================================

    prediction = model.predict(
        customer_data
    )

    probability = model.predict_proba(
        customer_data
    )[0][1]


    # =========================================
    # DISPLAY RESULT
    # =========================================

    st.subheader("Prediction Result")


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


    with col2:

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


    if prediction[0] == 1:

        st.error(
            "⚠️ Customer is likely to CHURN."
        )

        st.write(
            "Recommended Action:"
        )

        st.write(
            """
            • Contact the customer  
            • Offer a personalized retention plan  
            • Check support issues  
            • Review payment problems  
            • Encourage service usage
            """
        )

    else:

        st.success(
            "✅ Customer is likely to STAY."
        )

        st.write(
            "Recommended Action:"
        )

        st.write(
            """
            • Maintain customer engagement  
            • Continue good customer support  
            • Offer loyalty benefits  
            • Encourage regular usage
            """
        )


# =========================================
# DATASET SECTION
# =========================================

st.divider()

st.subheader("📋 Customer Dataset")

st.dataframe(
    df,
    use_container_width=True
)


# =========================================
# CHURN SUMMARY
# =========================================

st.subheader("📈 Churn Summary")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Customers",
        len(df)
    )

with col2:

    st.metric(
        "Churned Customers",
        int(df["churn"].sum())
    )

with col3:

    churn_rate = (
        df["churn"].mean() * 100
    )

    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )