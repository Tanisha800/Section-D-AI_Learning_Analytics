# Project: AI Learning Analytics – Student Performance Prediction System

## From Predictive Analytics to Data-Driven Academic Intervention



## 📘 Project Overview

This project focuses on the design and implementation of an **AI-based Learning Analytics System** that predicts student academic performance and supports early identification of at-risk learners. The system leverages classical machine learning techniques to analyse demographic, socio-economic, and behavioural data and generate actionable insights for educators.

The primary goal of the project is to move beyond reactive academic evaluation systems and enable **proactive, data-driven academic intervention** through interpretable machine learning models.

### Key Objectives
- Predict student academic performance (pass/fail and score estimation)
- Identify students at academic risk at an early stage
- Categorise learners into meaningful performance-based groups
- Demonstrate real-world usability through deployment

---

##  Project Milestones

### 🔹 Milestone 1: Machine Learning–Based Performance Prediction
**Objective:**  
Develop a complete classical machine learning pipeline to predict student performance using structured educational data.

**Key Outcomes:**
- Data preprocessing and feature engineering
- Supervised learning models for classification and regression
- Performance evaluation using standard metrics
- Exploratory data analysis for insight generation

### 🔹 Milestone 2: Deployment & Practical Usability
**Objective:**  
Deploy the trained models through a user-friendly web application to demonstrate real-world applicability.

**Key Outcomes:**
- Interactive web interface for predictions
- Support for individual and bulk student analysis
- Cloud deployment for remote accessibility

---

##  Constraints & Requirements

- **Team Size:** 4 Students  
- **API Budget:** Free Tier Only  
- **Machine Learning Focus:** Classical ML (No paid APIs)  
- **Hosting:** Mandatory (Render)  
- **UI Framework:** Streamlit  

---

##  Technology Stack

| Component | Technology |
|--------|------------|
| **Programming Language** | Python |
| **Data Analysis** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Models Used** | Logistic Regression, Linear Regression, K-Means |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Deployment Platform** | Render |
| **Version Control** | GitHub |

---

##  Dataset Description

- **Dataset:** Students Performance in Exams – Expanded Dataset  
- **Source:** Kaggle  
- **Size:** 30,000+ student records  

### Key Features
- Academic scores (Math, Reading, Writing)
- Demographic attributes
- Socio-economic indicators
- Behavioural factors (study hours, test preparation)

### Derived Features
- `AverageScore`: Mean of subject-wise scores  
- `Result`: Binary pass/fail label  

---

##  Exploratory Data Analysis (EDA)

EDA was conducted to understand:
- Score distributions and variability
- Correlation between study habits and performance
- Socio-economic performance differences
- Natural clustering patterns among students

Key observations include:
- Academic scores follow approximately normal distributions
- Weekly study hours show a positive correlation with performance
- Clear performance segmentation exists among learners

---

##  Methodology

The system follows a structured ML pipeline:

1. **Data Cleaning & Preprocessing**
   - Handling missing values
   - Encoding categorical features
   - Feature scaling

2. **Model Training**
   - **Logistic Regression:** Pass/Fail classification
   - **Linear Regression:** Score prediction
   - **K-Means Clustering:** Learner categorisation

3. **Evaluation**
   - Classification: Accuracy, Precision, Recall, F1-score
   - Regression: RMSE, MAE

4. **Deployment**
   - Streamlit-based web application
   - Cloud hosting on Render

---

##  Evaluation Results

### Classification (Logistic Regression)
- **Accuracy:** 73.33%
- **F1-Score:** 0.8462
- **Recall:** 1.00 (High priority on identifying at-risk students)

### Regression (Linear Regression)
- **RMSE:** 16.18
- **MAE:** 14.37

These results indicate reliable predictive performance given the 0–100 scoring scale.

---

##  Deployment

The application is deployed as a **Streamlit web app** and supports:
- Individual student performance prediction
- Bulk CSV uploads for cohort-level analysis
- Real-time prediction and categorisation

The system is hosted on **Render**, ensuring accessibility without local setup.

---

## 👥 Team Contributions

- **Adamya Tiwari**  
  Data preprocessing, feature engineering, model training, clustering, and optimisation.

- **Nachiket Amlekar**  
  Developed the Streamlit-based web application in Python and deployed the system on the Render cloud platform.

- **Tanisha Dhiman**  
  GitHub repository management and preparation of technical documentation.

- **Aditi Singh**  
  Dataset sourcing and project video editing.

---

##  Future Scope

Potential enhancements include:
- Ensemble learning methods for improved accuracy
- Temporal analysis using longitudinal student data
- Integration with real institutional datasets
- Adaptive learning recommendation engines
- Interactive dashboards for educators

---
