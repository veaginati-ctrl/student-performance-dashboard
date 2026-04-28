# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(page_title="Student Dashboard", layout="wide")

# ---------------- LOAD FILES ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

df = pd.read_csv("StudentPerformanceFactors.csv")
df = df.dropna()

target_column = "Exam_Score"
feature_columns = [col for col in df.columns if col != target_column]

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio("Go to", ["Dashboard", "Prediction"])

# Filters (only for Dashboard)
if page == "Dashboard":
    st.sidebar.subheader("🔎 Filters")

    # Example filters (adjust based on your dataset)
    for col in df.select_dtypes(include='object').columns:
        options = ["All"] + list(df[col].unique())
        selected = st.sidebar.selectbox(f"{col}", options)

        if selected != "All":
            df = df[df[col] == selected]

# ---------------- DASHBOARD PAGE ----------------
if page == "Dashboard":

    st.title("🎓 Student Performance Dashboard")

    # KPIs
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Average Score", round(df[target_column].mean(), 2))
    col3.metric("Max Score", df[target_column].max())

    st.markdown("---")

    # ---------------- PLOTLY CHARTS ----------------

    st.subheader("📊 Score Distribution")
    fig1 = px.histogram(df, x=target_column)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📈 Score vs Study Hours")
    if "Hours_Studied" in df.columns:
        fig2 = px.scatter(df, x="Hours_Studied", y=target_column)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 Average Score by Category")

    for col in df.select_dtypes(include='object').columns:
        fig = px.bar(
            df.groupby(col)[target_column].mean().reset_index(),
            x=col,
            y=target_column,
            title=f"{col} vs Score"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":

    st.title("🔮 Predict Student Score")

    input_data = []

    for col in feature_columns:
        if col in encoders:
            options = list(encoders[col].classes_)
            val = st.selectbox(f"{col}", options)
            val = encoders[col].transform([val])[0]
        else:
            val = st.number_input(f"{col}", value=float(df[col].mean()))
        
        input_data.append(val)

    if st.button("Predict Score"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]

        st.success(f"📊 Predicted Score: {round(prediction, 2)}")