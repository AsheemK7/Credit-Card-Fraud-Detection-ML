
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

st.set_page_config(page_title="Credit Card Fraud Detection",
                   page_icon="💳", layout="wide")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE, "credit_fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE, "credit_fraud_scaler.pkl"))
dataset_path = os.path.join(BASE, "creditcard.csv")
df = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else None

if "history" not in st.session_state:
    st.session_state.history = []

st.title("💳 Credit Card Fraud Detection")
st.caption("Random Forest | Machine Learning | Interactive Fraud Analysis")

pages = ["Dashboard", "Custom Prediction"] if df is None else ["Dashboard", "Test Transactions", "Custom Prediction"]
page = st.sidebar.radio("Navigation", pages)

if page == "Dashboard":
    st.subheader("Fraud Monitoring Dashboard")

    total = len(st.session_state.history)
    fraud = sum(x["Prediction"] == "Fraudulent" for x in st.session_state.history)

    a, b, c = st.columns(3)
    a.metric("Analyzed Transactions", total)
    b.metric("Fraud Detected", fraud)
    c.metric("Legitimate", total - fraud)

    if total:
        hist = pd.DataFrame(st.session_state.history)
        st.plotly_chart(px.pie(hist, names="Prediction",
                               title="Prediction Distribution", hole=0.4),
                        use_container_width=True)
        st.dataframe(hist, use_container_width=True)
    else:
        st.info("Analyze transactions to populate the dashboard.")

elif page == "Test Transactions":
    st.subheader("Test Using Real Dataset Transactions")

    kind = st.selectbox("Select actual transaction type",
                        ["Legitimate", "Fraudulent"])
    label = 0 if kind == "Legitimate" else 1

    samples = df[df["Class"] == label]

    if st.button("Load and Analyze Sample Transaction"):
        sample = samples.sample(1).drop(columns=["Class"])
        scaled = scaler.transform(sample)
        pred = int(model.predict(scaled)[0])
        prob = float(model.predict_proba(scaled)[0][1])

        result = "Fraudulent" if pred == 1 else "Legitimate"

        st.session_state.history.append({
            "Prediction": result,
            "Actual": kind,
            "Fraud Probability": round(prob * 100, 2),
            "Amount": float(sample["Amount"].iloc[0])
        })

        st.write("Actual Class:", kind)
        st.write("Predicted Class:", result)
        st.metric("Fraud Probability", f"{prob*100:.2f}%")

        if pred == label:
            st.success("Correct Prediction!")
        else:
            st.warning("Incorrect Prediction!")

        st.dataframe(sample)

elif page == "Custom Prediction":
    st.subheader("Enter Custom Transaction Features")
    st.warning("V1–V28 must be valid PCA-transformed values from compatible data.")

    with st.form("custom"):
        time = st.number_input("Time", min_value=0.0)
        amount = st.number_input("Amount", min_value=0.0)
        features = {}
        cols = st.columns(4)

        for i in range(1, 29):
            with cols[(i-1) % 4]:
                features[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

        submit = st.form_submit_button("Predict")

    if submit:
        data = pd.DataFrame([{
            "Time": time,
            **features,
            "Amount": amount
        }])
        data = data[["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]]

        scaled = scaler.transform(data)
        pred = int(model.predict(scaled)[0])
        prob = float(model.predict_proba(scaled)[0][1])

        st.metric("Fraud Probability", f"{prob*100:.2f}%")
        if pred == 1:
            st.error("Fraudulent Transaction Detected")
        else:
            st.success("Legitimate Transaction")

st.sidebar.caption("Credit Card Fraud Detection | Random Forest")
