import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn

# Load the dataset
file_path = "train_u6lujuX_CVtuZ9i (1).csv"
loan_data = pd.read_csv(file_path)

# Step 1: Data Preprocessing
# Handle missing values
num_imputer = SimpleImputer(strategy='median')
loan_data[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']] = num_imputer.fit_transform(
    loan_data[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

cat_imputer = SimpleImputer(strategy='most_frequent')
loan_data[['Gender', 'Married', 'Dependents', 'Self_Employed']] = cat_imputer.fit_transform(
    loan_data[['Gender', 'Married', 'Dependents', 'Self_Employed']])

loan_data['Dependents'] = loan_data['Dependents'].replace('3+', 3).astype(int)

label_encoder = LabelEncoder()
for column in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    loan_data[column] = label_encoder.fit_transform(loan_data[column])

scaler = StandardScaler()
loan_data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']] = scaler.fit_transform(
    loan_data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']])

# Split the data into features (X) and target (y)
X = loan_data.drop(columns=['Loan_ID', 'Loan_Status'])
y = loan_data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Loan Approval Prediction")

with mlflow.start_run():
    # Step 4: Model Training
    logistic_model = LogisticRegression(random_state=42, max_iter=1000)
    logistic_model.fit(X_train, y_train)

    # Step 5: Model Prediction
    y_pred = logistic_model.predict(X_test)

    # Step 6: Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Log parameters
    mlflow.log_param("random_state", 42)
    mlflow.log_param("max_iter", 1000)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(logistic_model, "model")

    # Display results
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Log Confusion Matrix as an artifact
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=logistic_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Feature Importance
    feature_importance = pd.Series(logistic_model.coef_[0], index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar', color='skyblue')
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Coefficient Values")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
