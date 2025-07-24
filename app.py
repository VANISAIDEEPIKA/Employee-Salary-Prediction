import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import clean_data, apply_encoders

# Load dataset and models
df = pd.read_csv("Employee_Salary_Dataset.csv")
reg_model = joblib.load("best_regression_model.pkl")
clf_model = joblib.load("best_classification_model.pkl")
encoders = joblib.load("encoders.pkl")
model_features = joblib.load("model_features.pkl")

# 🔁 Education to EducationNum Mapping
education_map = {
    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4,
    '9th': 5, '10th': 6, '11th': 7, '12th': 8,
    'HS-grad': 9, 'Some-college': 10, 'Assoc-acdm': 11, 'Assoc-voc': 12,
    'Bachelors': 13, 'Masters': 14, 'Doctorate': 16, 'PhD': 16
}

# Streamlit page config
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="wide")

# App title and description
st.title("💼 Employee Salary Predictor")
st.markdown("""
This project aims to develop a machine learning model that predicts employee salaries based on various attributes such as age, experience, education, job role, industry, and location.

By analyzing historical salary data, this app enables data-driven decisions in hiring and compensation by:
- 💰 **Regression**: Predicting the **exact salary amount** based on the given inputs.
- 📊 **Classification**: Predicting whether an employee's salary falls into **>50K or ≤50K** category.
""")

# Visualizations
with st.expander("📊 Data Insights & Visualizations"):
    st.subheader("Education Level vs Salary")
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="Education", y="Salary", data=df)
    st.pyplot(plt)

    st.subheader("Workclass Distribution")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Workclass", order=df["Workclass"].value_counts().index)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu")
    st.pyplot(plt)

# Sidebar form
st.sidebar.header("📝 Enter Employee Details")

# Input fields
age = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education", list(education_map.keys()))  # 👈 simplified
workclass = st.sidebar.selectbox("Workclass", ['Private', 'Self-employed', 'Government', 'Unemployed'])
occupation = st.sidebar.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Sales', 'Managerial', 'Clerical', 'Engineer'])
industry = st.sidebar.selectbox("Industry", ['IT', 'Finance', 'Healthcare', 'Education', 'Manufacturing'])
location = st.sidebar.selectbox("Location", ['Delhi', 'Mumbai', 'Hyderabad', 'Bangalore', 'Chennai'])
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
hours = st.sidebar.slider("Hours Per Week", 20, 60, 40)
marital = st.sidebar.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
race = st.sidebar.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 10000, 0)
country = st.sidebar.selectbox("Native Country", ['India', 'United States', 'Canada', 'Germany', 'Australia'])
weight = st.sidebar.number_input("Final Weight", 50000, 150000, 90000)

# Auto compute EducationNum 👇
education_num = education_map.get(education, 10)

# Create input dataframe
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Education': [education],
    'EducationNum': [education_num],
    'Workclass': [workclass],
    'Occupation': [occupation],
    'Industry': [industry],
    'Location': [location],
    'YearsExperience': [experience],
    'HoursPerWeek': [hours],
    'MaritalStatus': [marital],
    'Race': [race],
    'CapitalGain': [capital_gain],
    'CapitalLoss': [capital_loss],
    'NativeCountry': [country],
    'FinalWeight': [weight]
})

# Apply label encoders (like Gender)
input_df = apply_encoders(input_df, encoders)

# Apply one-hot encoding (must match training)
input_df = pd.get_dummies(input_df)

# Align with training model features
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_features]

# Prediction task selection
task = st.radio("Select Prediction Type", ["Regression (Exact Salary)", "Classification (>50K or ≤50K)"])

# Prediction button
if st.button("🔍 Predict Salary"):
    if task == "Regression (Exact Salary)":
        salary = reg_model.predict(input_df)[0]
        group = "Lower Income Group" if salary <= 50000 else "Upper Income Group"
        st.success(f"💰 Estimated Salary: ₹{int(salary):,}/month ({group})")
    else:
        result = clf_model.predict(input_df)[0]
        label = ">50K" if result == 1 else "≤50K"
        st.success(f"📊 Predicted Salary Category: **{label}**")
