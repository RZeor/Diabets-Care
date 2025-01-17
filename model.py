import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from joblib import dump

# Load dataset
file_path = 'diabetes data upload.csv'  # Ubah dengan lokasi file Anda
data = pd.read_csv(file_path)

# Preprocessing the data: Encoding categorical variables
data_encoded = data.copy()
for column in data_encoded.columns:
    if data_encoded[column].dtype == 'object':
        data_encoded[column] = data_encoded[column].astype('category').cat.codes

# Splitting data into features (X) and target (y)
X = data_encoded.drop(columns=['class'])
y = data_encoded['class']

# Creating a variable for train-test split ratio
split_ratio = 0.3  # This can be adjusted as needed

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions
y_pred = rf_model.predict(X_test)

# Manually calculating confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

# Calculating precision and accuracy
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Displaying results
results = {
    "Confusion Matrix": conf_matrix.tolist(),
    "Precision": precision,
    "Accuracy": accuracy,
    "True Negative": tn,
    "False Positive": fp,
    "False Negative": fn,
    "True Positive": tp,
}

# Print results
print("Confusion Matrix:")
print(conf_matrix)
print("\nPrecision:", precision)
print("Accuracy:", accuracy)

# Saving the trained Random Forest model to a .pkl file
model_path = 'random_forest_model.pkl'  # Ubah dengan lokasi penyimpanan Anda
dump(rf_model, model_path)
print(f"\nModel saved to {model_path}")
