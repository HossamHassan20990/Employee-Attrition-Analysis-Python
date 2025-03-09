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
fig = px.pie(df, names="Attrition", title="Employee Attrition Distribution", hole=0.3)

# Show Plot
fig.show()
```

### 🔹 **Numeric Feature Analysis**

#### Age Distribution 📊

```
fig_age = px.histogram(
    df_cleaned, x="Age", nbins=30, 
    title="Age Distribution of Employees", 
    labels={"Age": "Employee Age"}, 
    color_discrete_sequence=["#636EFA"],
    text_auto= True
)
fig_age.show()
```

#### Monthly Income Distribution 💰

```
fig_income = px.histogram(
    df_cleaned, x="MonthlyIncome", nbins=30, 
    title="Monthly Income Distribution", 
    labels={"MonthlyIncome": "Monthly Income ($)"}, 
    color_discrete_sequence=["#EF553B"],
    text_auto= True
)
fig_income.show()
```

#### Years at Company ⏳

```
fig_years = px.histogram(
    df_cleaned, x="YearsAtCompany", nbins=30, 
    title="Years at Company Distribution", 
    labels={"YearsAtCompany": "Years at Company"}, 
    color_discrete_sequence=["#00CC96"],
    text_auto= True
)
fig_years.show()
```

### 🔹 **Categorical Feature Analysis**

#### Attrition by Department 🏢

```
fig_dept = px.bar(
    df_cleaned.groupby("Department")["Attrition"].value_counts().unstack(),
    title="Attrition by Department",
    labels={"value": "Frequency", "Department": "Department"},
    barmode="group",
    color_discrete_sequence=["#636EFA", "#EF553B"],
    text_auto= True
)
fig_dept.show()
```

#### Attrition by Job Role 👔

```
fig_role = px.bar(
    df_cleaned.groupby("JobRole")["Attrition"].value_counts().unstack(),
    title="Attrition by Job Role",
    labels={"value": "Frequency", "JobRole": "Job Role"},
    barmode="group",
    color_discrete_sequence=["#636EFA", "#EF553B"],
    text_auto= True
)
fig_role.show()
```
#### Attrition by Marital Status 💍
```
fig_marital = px.bar(
    df_cleaned.groupby("MaritalStatus")["Attrition"].value_counts().unstack(),
    title="Attrition by Marital Status",
    labels={"value": "frequency", "MaritalStatus": "Marital Status"},
    barmode="group",
    color_discrete_sequence=["#636EFA", "#EF553B"],
    text_auto= True
)
fig_marital.show()
```

#### Attrition by Overtime Work ⏳

```
fig_overtime = px.bar(
    df_cleaned.groupby("OverTime")["Attrition"].value_counts().unstack(),
    title="Attrition by Overtime Work",
    labels={"value": "frequency", "OverTime": "Overtime"},
    barmode="group",
    color_discrete_sequence=["#636EFA", "#EF553B"],
    text_auto= True
)
fig_overtime.show()
```
### 🔹 **Correlation Analysis**
- Correlation Heatmap Shows relationships between numeric features
- Attrition vs. Salary, Job Satisfaction, and Work-Life Balance

#### Correlation Heatmap
```
import plotly.figure_factory as ff
import numpy as np

# Select only important numeric columns for attrition
important_features = [
    "Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", 
    "JobSatisfaction", "WorkLifeBalance"
]

df_selected = df_cleaned[important_features]

# Compute correlation matrix
correlation_matrix = df_selected.corr()

# Format numbers to 2 decimal places
z_text = np.around(correlation_matrix.values, decimals=2).astype(str)

# Create heatmap
fig_corr = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=list(correlation_matrix.columns),
    y=list(correlation_matrix.index),
    annotation_text=z_text,  # Add formatted text
    colorscale="Blues",
    showscale=True
)

# Set title
fig_corr.update_layout(title_text="Correlation Heatmap (Key Features)")

# Show figure
fig_corr.show()
```
#### Attrition vs. Monthly Income 
```
fig_income_attrition = px.box(
    df_cleaned, x="Attrition", y="MonthlyIncome", 
    color="Attrition",
    title="Attrition vs. Monthly Income",
    labels={"MonthlyIncome": "Monthly Income ($)", "Attrition": "Attrition Status"},
    color_discrete_sequence=["#636EFA", "#EF553B"],
    
)
fig_income_attrition.show()
```
### Attrition vs. Job Satisfaction
```
fig_job_satisfaction = px.histogram(
    df_cleaned, x="JobSatisfaction", color="Attrition",
    title="Attrition vs. Job Satisfaction",
    labels={"JobSatisfaction": "Job Satisfaction Level", "Attrition": "Attrition Status"},
    barmode="group",
    color_discrete_sequence=["#636EFA", "#EF553B"],
    text_auto= True
)
fig_job_satisfaction.show()
```

### Attrition vs. Work-Life Balance
```
fig_work_life = px.histogram(
    df_cleaned, x="WorkLifeBalance", color="Attrition",
    title="Attrition vs. Work-Life Balance",
    labels={"WorkLifeBalance": "Work-Life Balance Level", "Attrition": "Attrition Status"},
    barmode="group",
    color_discrete_sequence=["#EF553B", "#636EFA"],
        text_auto= True
)
fig_work_life.show()

```
## 🔍 Key Insights

- **Attrition Rate:** Employees with **low salary, high overtime, and poor work-life balance** tend to leave more.
- **Departmental Differences:** Certain departments experience **higher attrition rates**.
- **Job Satisfaction Impact:** Employees with **lower job satisfaction tend to leave** more often.
- **Work-Life Balance Matters:** Employees with **poor work-life balance** are more likely to quit.
---

📊 Simple Predictive Model for Employee Attrition 

Model Overview

A basic classification model is built using Logistic Regression to predict employee attrition based on key features.

### Encode categorical variables

```
df_encoded = df_cleaned.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
```

### Define features and target variable
```
X = df_encoded.drop(columns=["Attrition"])  # Features
y = df_encoded["Attrition"]  # Target variable
```

### Split data into training and test sets (80% train, 20% test)

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Scale the features
```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Train Logistic Regression Model
```
model = LogisticRegression()
model.fit(X_train, y_train)
```

# Predictions
```
y_pred = model.predict(X_test_scaled)
```

# Model Evaluation
```
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
```

### ✅ Model Results
    Accuracy: 89.5%
    Precision & Recall:
    Employees Staying (0): 91% precision, 98% recall
    Employees Leaving (1): 70% precision, 36% recall
    🔹 Interpretation: The model predicts employees staying well but struggles with employees leaving (attrition cases are fewer)
```

📝 Key Takeaways

✅ Work-life balance, job satisfaction, and salary are the most significant attrition drivers.
✅ Departments with high attrition risk need targeted HR strategies.
✅ The predictive model helps identify employees at risk before they leave.

📌 This report helps HR teams make data driven decisions to reduce attrition and improve employee retention.

💡 **Project by:** [Hossam Hassan] | 🔗 **GitHub:** [(https://github.com/HossamHassan20990/Employee-Attrition-Analysis/tree/main)]


