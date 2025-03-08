# Employee Attrition Analysis

## 📌 Project Overview

This project aims to analyze employee attrition using Python. 
We explore key factors influencing attrition, perform visualizations using Plotly Express, and insights for HR decision making.

## 📊 Data Preprocessing & Cleaning

### 1️⃣ **Data Source**

- The dataset contains various employee attributes such as **age, job role, salary, work-life balance, and attrition status**.

### 2️⃣ **Cleaning Steps**

- Dropped unnecessary columns: `EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`.
- Ensured data consistency and handled missing values (if any).

## 📈 Exploratory Data Analysis (EDA)

### 🔹 **Attrition Distribution**

```
fig_attrition = px.bar(
    df_cleaned["Attrition"].value_counts().reset_index(),
    x="index", y="Attrition", color="index",
    title="Attrition Distribution",
    labels={"index": "Attrition Status", "Attrition": "Count"}
)
fig_attrition.show()
```

### 🔹 **Numeric Feature Analysis**

#### Age Distribution 📊

```
fig_age = px.histogram(df_cleaned, x="Age", nbins=30, title="Age Distribution")
fig_age.show()
```

#### Monthly Income Distribution 💰

```
fig_income = px.histogram(df_cleaned, x="MonthlyIncome", nbins=30, title="Monthly Income Distribution")
fig_income.show()
```

#### Years at Company ⏳

```
fig_years = px.histogram(df_cleaned, x="YearsAtCompany", nbins=30, title="Years at Company Distribution")
fig_years.show()
```

### 🔹 **Categorical Feature Analysis**

#### Attrition by Department 🏢

```
fig_dept = px.bar(df_cleaned.groupby("Department")["Attrition"].value_counts().unstack())
fig_dept.show()
```

#### Attrition by Job Role 👔

```
fig_role = px.bar(df_cleaned.groupby("JobRole")["Attrition"].value_counts().unstack())
fig_role.show()
```

#### Attrition by Overtime Work ⏳

```
fig_overtime = px.bar(df_cleaned.groupby("OverTime")["Attrition"].value_counts().unstack())
fig_overtime.show()
```

## 🔍 Key Insights

- **Attrition Rate:** Employees with **low salary, high overtime, and poor work-life balance** tend to leave more.
- **Departmental Differences:** Certain departments experience **higher attrition rates**.
- **Job Satisfaction Impact:** Employees with **lower job satisfaction tend to leave** more often.
- **Work-Life Balance Matters:** Employees with **poor work-life balance** are more likely to quit.
---

📊 Simple Predictive Model for Employee Attrition (Python)

Model Overview

A basic classification model is built using Logistic Regression to predict employee attrition based on key features.

Python Code for Predictive Model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Select relevant features
features = ["Age", "MonthlyIncome", "JobSatisfaction", "OverTime", "YearsAtCompany"]
df = df[features + ["Attrition"]]

# Encode categorical variables
le = LabelEncoder()
df["Attrition"] = le.fit_transform(df["Attrition"])
df["OverTime"] = le.fit_transform(df["OverTime"])

# Split data
X = df[features]
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

Insights from the Model

The model predicts whether an employee will leave the company based on key attributes.

OverTime, JobSatisfaction, and YearsAtCompany have a strong impact on attrition.

Future improvements can include more features and advanced machine learning techniques.

📝 Key Takeaways

✅ Work-life balance, job satisfaction, and salary are the most significant attrition drivers.
✅ Departments with high attrition risk need targeted HR strategies.
✅ The predictive model helps identify employees at risk before they leave.

📌 This report helps HR teams make data driven decisions to reduce attrition and improve employee retention.

💡 **Project by:** [Hossam Hassan] | 🔗 **GitHub:** [[Your Repository](https://github.com/HossamHassan20990/Employee-Attrition-Analysis/tree/main)]


