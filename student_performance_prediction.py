import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
project_root = Path(__file__).resolve().parent
data_path = project_root / "data" / "student_performance.csv"
screenshots_path = project_root / "screenshots"

# Load dataset
df = pd.read_csv(data_path)

print("DATASET OVERVIEW")
print("-" * 50)
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMISSING VALUES")
print("-" * 50)
print(df.isnull().sum())
print("Total missing values:", df.isnull().sum().sum())

# Select features and target
# StudentID is only an identifier.
# GradeClass is excluded because it is related to the GPA target.
X = df.drop(columns=["StudentID", "GPA", "GradeClass"])
y = df["GPA"]

# Split dataset into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 50)
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R²) Score: {r2:.3f}")

# Actual versus predicted GPA graph
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.65)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.title("Actual GPA vs Predicted GPA")
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.legend()
plt.tight_layout()
plt.savefig(screenshots_path / "actual_vs_predicted_gpa_final.png", dpi=300)
plt.show()

# Feature importance table
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

feature_importance["AbsoluteCoefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(
    by="AbsoluteCoefficient",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("-" * 50)
print(feature_importance[["Feature", "Coefficient"]].to_string(index=False))

# New student input
new_student = pd.DataFrame({
    "Age": [17],
    "Gender": [1],
    "Ethnicity": [0],
    "ParentalEducation": [2],
    "StudyTimeWeekly": [12.0],
    "Absences": [5],
    "Tutoring": [1],
    "ParentalSupport": [3],
    "Extracurricular": [1],
    "Sports": [1],
    "Music": [0],
    "Volunteering": [0]
})

predicted_gpa = model.predict(new_student)[0]

if predicted_gpa >= 3.0:
    performance_level = "Excellent"
elif predicted_gpa >= 2.0:
    performance_level = "Good"
elif predicted_gpa >= 1.0:
    performance_level = "Average"
else:
    performance_level = "Needs Improvement"

print("\nNEW STUDENT PREDICTION")
print("-" * 50)
print(f"Predicted GPA: {predicted_gpa:.2f} out of 4.00")
print(f"Predicted Performance Level: {performance_level}")