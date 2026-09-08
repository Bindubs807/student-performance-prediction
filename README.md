# Predictive Analysis of Student Performance

A data science project that analyzes student academic data and predicts student GPA using a Linear Regression model.

## Project Overview

Educational institutions can use student-performance data to understand the factors associated with academic outcomes and identify students who may benefit from additional support.

This project performs exploratory data analysis (EDA), data preprocessing, regression modeling, model evaluation, and GPA prediction for a new student. The target variable is **GPA**, which ranges from 0.00 to 4.00.

## Objectives

- Analyze a student-performance dataset using Python and Pandas
- Perform exploratory data analysis using statistical summaries and visualizations
- Check data quality and identify missing values
- Preprocess data for machine learning
- Train a Linear Regression model to predict GPA
- Evaluate model performance using MAE, RMSE, and R² score
- Predict GPA and performance level for a new student

## Dataset

The dataset contains **2,392 student records** and **15 columns**.

### Target Variable

- `GPA`: Student Grade Point Average, ranging from 0.00 to 4.00

### Input Features Used

- `Age`
- `Gender`
- `Ethnicity`
- `ParentalEducation`
- `StudyTimeWeekly`
- `Absences`
- `Tutoring`
- `ParentalSupport`
- `Extracurricular`
- `Sports`
- `Music`
- `Volunteering`

### Excluded Columns

- `StudentID` was excluded because it is only a unique identifier.
- `GradeClass` was excluded because it is related to GPA and may cause data leakage during model training.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Visual Studio Code

## Project Structure

```text
student-performance-prediction/
│
├── data/
│   └── student_performance.csv
│
├── report/
│   └── Project Report.pdf
│
├── screenshots/
│   ├── 01_dataset_preview.png
│   ├── 02_data_quality_check.png
│   ├── 05_model_performance.png
│   ├── 06_prediction_output.png
│   ├── absences_vs_gpa.png
│   ├── actual_vs_predicted_gpa.png
│   ├── feature_importance.png
│   ├── gpa_distribution.png
│   └── study_time_vs_gpa.png
│
├── src/
│   ├── 01_load_and_explore.ipynb
│   ├── 02_preprocess_and_model.ipynb
│   └── 03_predict_new_students.ipynb
│
├── student_performance_prediction.py
└── README.md
```

## Methodology

1. Loaded the dataset using Pandas.
2. Examined dataset shape, column names, data types, and missing values.
3. Performed EDA using GPA distribution, study-time versus GPA, and absences versus GPA visualizations.
4. Removed `StudentID` and `GradeClass` before model training.
5. Split the data into 80% training data and 20% testing data using `random_state=42`.
6. Trained a Linear Regression model.
7. Evaluated the model on unseen test data.
8. Predicted GPA for a hypothetical new student.

## Model Performance

The Linear Regression model produced the following results on the test dataset:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.155 |
| Root Mean Squared Error (RMSE) | 0.197 |
| R² Score | 0.953 |

The R² score of **0.953** indicates that the model explains approximately **95.3% of the variation in GPA** for the test data.

> Note: This is a regression project. R² is not classification accuracy.

## Key Insights

- Tutoring had the strongest positive association with predicted GPA in the Linear Regression model.
- Extracurricular participation, sports participation, music participation, and parental support were positively associated with predicted GPA.
- Absences had a negative association with GPA. Students with more absences generally had lower predicted GPA.
- Weekly study time had a positive, though comparatively smaller, association with GPA.
- The actual-versus-predicted GPA plot showed that most predictions were close to the ideal prediction line.

## New Student Prediction

The model was tested using a hypothetical new student with the following example inputs:

- Age: 17
- Weekly study time: 12 hours
- Absences: 5
- Tutoring: Yes
- Parental support: High
- Extracurricular participation: Yes
- Sports participation: Yes

**Predicted GPA: 3.45 out of 4.00**  
**Predicted Performance Level: Excellent**

## How to Run

### 1. Clone the repository

```bash
git clone [https://github.com/Bindubs807/student-performance-prediction.git](https://github.com/Bindubs807/student-performance-prediction.git)
cd student-performance-prediction
```

### 2. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Python file

```bash
python student_performance_prediction.py
```

You can also open and run the notebooks inside the `src` folder using Jupyter Notebook or Visual Studio Code.

## Future Scope

- Compare Linear Regression with advanced models such as Random Forest Regressor and Gradient Boosting Regressor.
- Apply cross-validation to evaluate model stability across different data splits.
- Use real institutional data with appropriate privacy and ethical safeguards.
- Include additional factors such as previous grades, assignment scores, examination marks, and attendance trends.
- Develop a simple web application or dashboard to predict GPA and identify students who may need academic support.

## Author

**Bindu BS**

