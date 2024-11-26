import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import os

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("C:\\Users\\tanvi\\Desktop\\Deploy2\\agricrop.csv")

# Preprocess the data
print("Preprocessing the data...")
data['commodity_name'] = data['commodity_name'].str.lower()
data['state'] = data['state'].str.lower()
data['district'] = data['district'].str.lower()
data['market'] = data['market'].str.lower()
data = data.dropna(subset=['modal_price'])

# Encode categorical variables
print("Encoding categorical variables...")
label_encoders = {
    'commodity_name': LabelEncoder().fit(data['commodity_name']),
    'state': LabelEncoder().fit(data['state']),
    'district': LabelEncoder().fit(data['district']),
    'market': LabelEncoder().fit(data['market'])
}

data['commodity_name'] = label_encoders['commodity_name'].transform(data['commodity_name'])
data['state'] = label_encoders['state'].transform(data['state'])
data['district'] = label_encoders['district'].transform(data['district'])
data['market'] = label_encoders['market'].transform(data['market'])

# Features and target
X = data[['commodity_name', 'state', 'district', 'market', 'min_price', 'max_price']]
y = data['modal_price']

# Train-test split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
print("Training Random Forest model...")
random_forest_model = Pipeline([
    ('model', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))
])

# Train the model
random_forest_model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = random_forest_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Random Forest R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Save the best model and label encoders
print("Saving the model and encoders...")
try:
    # Save paths
    output_dir = os.path.dirname(os.path.abspath(__file__))  # Save in current script directory
    model_path = os.path.join(output_dir, 'best_model_pipeline.pkl')
    encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
    
    # Save the model and encoders
    joblib.dump(random_forest_model, model_path)
    joblib.dump(label_encoders, encoders_path)
    
    print(f"Model saved successfully at: {model_path}")
    print(f"Label encoders saved successfully at: {encoders_path}")
except Exception as e:
    print("Error saving files:", e)

print("Training script completed!")
