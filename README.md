# Employee-Salary-Prediction
# 💼 Employee Salary Prediction using Machine Learning

An interactive Streamlit web application for predicting employee salary using machine learning algorithms. This project supports both **regression** (to estimate exact salary value) and **classification** (to predict if salary is >50K or ≤50K). It is built on a clean data pipeline with intuitive UI design, suitable for HR analytics or decision support systems.

🔗 **Live App:**  
[👉 Open the Streamlit App](https://employee-salary-prediction-4yhrhrenebabwcmwtstjbb.streamlit.app/)

---

## 🎓 Internship Program

- **Internship Title:** IBM SkillsBuild - Artificial Intelligence  
- **Duration:** 6 Weeks (June - July 2025)  
- **Organization:** Edunet Foundation  
- **Platform:** IBM SkillsBuild + AICTE  
- **Student Name:** Vani Sai Deepika  
- **College:** Rishi MS Institute of Engineering & Technology for Women  
- **Department:** Computer Science and Engineering  
- **Email:** saideepikavani@gmail.com  
- **AICTE Student ID:** STU6641f91d5732e1715599645  

---

## 🔍 Project Objective

To build a predictive model using machine learning algorithms that:
- Estimates employee salary in currency terms using **regression**
- Predicts whether the salary is greater than or less than 50K using **classification**

This project demonstrates real-world application of AI in HR analytics and compensation modeling.

---

## 🌟 Key Learnings

- End-to-end machine learning pipeline from data cleaning to deployment
- Handling categorical variables using Label Encoding and One-Hot Encoding
- Training and evaluating both **regression** and **classification** models
- Deploying machine learning models using **Streamlit**
- Building modular and reusable code for production-ready apps

---

## 🚀 Features of the Web App

- 🎯 **Dual Prediction Modes**:  
  - Regression: Predict actual salary value (e.g., ₹30,000/month)  
  - Classification: Predict salary category (>50K or ≤50K)

 - 💬 **User-Friendly UI**:  
  - Intuitive form layout  
  - Instant feedback and formatted salary prediction

- 🧮 **Input Features**:  
  The following features are collected from the user to predict salary value or salary category:

- **Age** – Age of the employee.
- **Education** – Highest education qualification (e.g., Bachelors, Masters, PhD).
- **Occupation** – Type of job or profession (e.g., Tech-support, Sales, Executive).
- **Gender** – Gender identity of the individual.
- **Experience** – Overall years of work experience.
- **Industry** – Domain or sector of employment (e.g., IT, Healthcare, Finance).
- **Workclass** – Type of employment (e.g., Private, Self-employed, Government).
- **Final Weight** – A weighting factor used in census data (used as-is for modeling).
- **Native Country** – Country of origin/residence of the individual.
- **Capital Loss** – Capital loss recorded in the previous financial year.
- **Capital Gain** – Capital gain recorded in the previous financial year.
- **Hours Per Week** – Average number of hours worked per week.
- **Race** – Racial category of the employee (used for demographic grouping).
- **Marital Status** – Marital status (e.g., Married, Single, Divorced).
- **Years of Experience** – Total number of years in the current industry or job type.
- **Location** – City/region of employment or residence.

  ## 📈 Model Results

After training and evaluating the models, the following results are displayed in the Streamlit web app:

- ✅ **Predicted Salary Output**:  
  Displays either the estimated salary value (₹ amount/month) or a category label (e.g., `>50K`, `≤50K`, `Lower Income Group`) based on user input.

- 📊 **Performance Metrics**:
  - **Regression Metrics** (for salary value prediction):
    - RMSE (Root Mean Squared Error)
    - R² Score (Coefficient of Determination)
  - **Classification Metrics** (for predicting >50K or ≤50K):
    - Accuracy Score
    - Confusion Matrix (shows TP, TN, FP, FN for binary classification)

---

## 📊 Data Insights & Visualizations

Before building the models, exploratory data analysis (EDA) was performed to understand feature relationships and distributions:

- **Education Level vs Salary**:  
  Visualizes how average salary values vary across different education levels.

- **Workclass Distribution**:  
  A bar chart showing the frequency of different work classes (e.g., Private, Government, Self-employed).

- **Correlation Heatmap**:  
  A heatmap that shows correlations between numeric features (e.g., Age, Experience, Capital Gain/Loss, Salary) to identify patterns and dependencies.

> These insights helped in feature selection and understanding data trends before modeling.


---

## 🧠 Tech Stack Used

### 📌 Languages & Libraries
- **Python** – Core programming language for data analysis, modeling, and app development.
- **pandas** – Used for data loading, preprocessing, and manipulation of structured datasets.
- **numpy** – Handles efficient numerical computations and array-based operations.
- **joblib** – Saves and loads trained models and encoders for consistent predictions.
- **scikit-learn (sklearn)** – A comprehensive machine learning library used for:
  - **Preprocessing** – Handling label encoding, train-test splits, and feature scaling.
  - **Model Training** – Implements models like Linear Regression, Random Forest Regressor, and Random Forest Classifier.
  - **Model Evaluation** – Calculates performance metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score, Accuracy, and Confusion Matrix.
- **Matplotlib** & **Seaborn** – Used for visualizing distributions, relationships, and model evaluation plots.

### 📌 ML Models
- **Linear Regression** – Predicts continuous salary values based on input features.
- **Random Forest Regressor** – An ensemble-based model used to improve regression accuracy and reduce overfitting.
- **Random Forest Classifier** – A robust classifier used to predict if an employee's salary is >50K or ≤50K.

### 📌 Tools & Platforms
- **Jupyter Notebook** – Used for EDA, preprocessing, feature engineering, and model experimentation.
- **Streamlit (VS Code)** – For building the interactive web UI that takes user input and shows predictions.
- **Streamlit Cloud** – Enables public deployment of the app directly via GitHub.
- **GitHub** – Hosts the full project code and files for collaboration and version control.


---
## 🗂️ Project Structure

```bash
employee-salary-prediction/
├── app.py                         # Main Streamlit app (UI logic)
├── salary_model_dev.ipynb         # Jupyter Notebook for training, EDA, and model building
├── requirements.txt               # Python dependencies for deployment
├── Employee_Salary_Dataset.csv    # Original raw dataset
├── Encoded_Salary_Dataset.csv     # Cleaned & encoded dataset used for modeling
├── best_regression_model.pkl      # Trained regression model (predicts salary value)
├── best_classification_model.pkl  # Trained classification model (predicts >50K or ≤50K)
├── encoders.pkl                   # LabelEncoders/dummy encoders used during training
├── utils.py                       # Helper functions (cleaning, encoding, etc.)
└── README.md                      # Project documentation (GitHub README)
        
                                          
