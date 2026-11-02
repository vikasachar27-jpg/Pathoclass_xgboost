import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Loading data...")
# Added low_memory=False to fix the Dtype warning
df = pd.read_csv("clinvar_conflicting.csv", low_memory=False)

# 1. THE FIX: The dataset uses 'CLASS' instead of 'CLNSIG'
# We will use this pre-engineered binary column as our target
df['Label'] = df['CLASS']

# 2. Get numeric columns and drop the target from the features
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Label' in numeric_columns:
    numeric_columns.remove('Label')
if 'CLASS' in numeric_columns:
    numeric_columns.remove('CLASS')

X = df[numeric_columns]
y = df['Label']

print(f"Training on {len(numeric_columns)} features...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train XGBoost
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate and save
preds = model.predict(X_test)
print(f"Model trained! Accuracy: {accuracy_score(y_test, preds):.2f}")

joblib.dump(model, "xgb_model.pkl")
joblib.dump(numeric_columns, "feature_names.pkl")
print("Saved xgb_model.pkl and feature_names.pkl")