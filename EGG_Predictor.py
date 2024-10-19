import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Step 1: Prepare the data (example data)
df = pd.read_csv('data.csv')
df.head()
labels = df["specific.disorder"].unique().tolist()
x = []
y = []
for i, row in df.iterrows():
    y.append(labels.index(row["specific.disorder"]))
    x.append(row.tolist()[8:])

X = np.array(x)
y = np.array(y)
# Hello this is an update
# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create the LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Step 4: Define the model parameters
params = {
    'objective': 'multiclass',  # For multiclass classification
    'num_class': 12,             # Number of classes
    'metric': 'multi_logloss',   # Loss function for multiclass
    'learning_rate': 0.01,
    'num_leaves': 64,
    'boosting_type': 'gbdt',
    'max_depth': 7,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8
}

# Step 5: Train the model
bst = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

# Step 6: Predict class probabilities
y_pred_proba = bst.predict(X_test)

# Step 7: Convert probabilities to class predictions (one-hot encoding)
y_pred = np.argmax(y_pred_proba, axis=1)  # Choose the class with highest probability

# Step 8: Convert to one-hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
y_pred_onehot = onehot_encoder.fit_transform(y_pred.reshape(-1, 1))

# Output results
print("Predicted one-hot encoded classes:\n", y_pred_onehot)

# Optional: Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
