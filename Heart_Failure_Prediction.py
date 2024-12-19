
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# Set Kaggle credentials
os.environ['KAGGLE_USERNAME'] = "alaaeddinebessghaier"
os.environ['KAGGLE_KEY'] = "8a16700523f6ec4d10e2d400d2c21a41"

# Download dataset

import zipfile
with zipfile.ZipFile("heart-failure-prediction.zip", 'r') as zip_ref:
    zip_ref.extractall("heart_failure_dataset")

data = pd.read_csv("heart_failure_dataset/heart.csv")
print(data.head())

print(data.info())
print(data.describe())

print(data.isnull().sum())

numeric_summary = data.describe()
print(numeric_summary)

categorical_summary = data.select_dtypes(include=['object']).nunique()
print(categorical_summary)

# Distribution of numeric variables
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

n_cols = 2  # Number of columns for subplots
n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Round up to get enough rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_columns):
    sns.histplot(data[col], bins=30, ax=axes[i], kde=True)
    axes[i].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    sns.countplot(data=data, x=col)
    plt.title(f"Distribution of {col}")
    plt.show()

numeric_data = data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
data['HeartDisease'] = data['HeartDisease'].astype('category')
sns.countplot(x='HeartDisease', data=data)
plt.title("Heart Disease Distribution")
plt.show()

X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression())])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {roc_auc}")

# ROC Curve
RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
plt.title("ROC Curve")
plt.show()
new_data = {
    'Age': [30],
    'Sex': ['M'],
    'ChestPainType': ['ATA'],
    'RestingBP': [90],
    'Cholesterol': [60],
    'FastingBS': [0],
    'RestingECG': ['Normal'],
    'MaxHR': [60],
    'ExerciseAngina': ['N'],
    'Oldpeak': [2],
    'ST_Slope': ['Up']
}

new_data_df = pd.DataFrame(new_data)
prediction = pipeline.predict(new_data_df)

prediction_prob = pipeline.predict_proba(new_data_df)[:, 1]

if prediction[0] == 1:
    print(f"Prediction: Heart disease detected (Probability: {prediction_prob[0]:.2f})")
else:
    print(f"Prediction: No heart disease detected (Probability: {prediction_prob[0]:.2f})")
