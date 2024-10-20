import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data.csv')
labels = df["specific.disorder"].unique().tolist()
x = []
y = []

for i, row in df.iterrows():
    idx = labels.index(row.tolist()[7])  # Assuming "specific.disorder" is at index 7
    y.append(idx)  # Using integer labels directly
    x.append(row.tolist()[8:])  # Assuming features start at index 8
X = np.array(x)
Y = np.array(y)

# Step 1.5: Scale Features and get rid of missing values
imputer = SimpleImputer(strategy='mean')  # Use 'mean', 'median', or another strategy if needed
X_imputed = imputer.fit_transform(X)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, LogisticRegression
from catboost import CatBoostClassifier

# Create directory for saving training data
os.makedirs('training_data', exist_ok=True)

# Assuming X_imputed and Y are provided as NumPy arrays
# X_imputed shape: (941, 1144)
# Y shape: (941,)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

class EN(nn.Module):
    def __init__(self):
        super(EN, self).__init__()
        # Define your layers here
        self.fc = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)

    def forward(self, x):
        # Define forward pass
        return self.fc(x)
    
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1,2,3,4,5,6,7,8,9,10,11]), y=Y)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
class CatBoost(nn.Module):
    def __init__(self):
        super(CatBoost, self).__init__()
        # Define your layers here
        self.fc = CatBoostClassifier(class_weights=[class_weights[0].item(), class_weights[1].item()], iterations=100, verbose=False)

    def forward(self, x):
        # Define forward pass
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get last output of the sequence
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def evaluate(csv_file):
    scaled_case = csv_file.reshape(1,-1)
    #df = pd.read_csv(csv_file, header=None)

    # Extract the first (and only) row and convert it into a vector (as a NumPy array)
    vector = df.iloc[0].values

    #x = np.array(vector)
    #case = imputer.fit_transform(vector[8:].reshape(1,-1))
    #scaled_case = scaler.fit_transform(case)


    base = [0,0,0,0,0,0,0,0,0,0,0,0]
    # Define the architecture
    model = EN()
    # Load the state dictionary
    model = torch.load("EN_disorder_0.pt")  
    output = model.predict(scaled_case)
    print(0, output)
    base[0] = output.tolist()[0]

    model = torch.load("EN_disorder_2.pt")  
    output = model.predict(scaled_case)
    print(2, output)
    base[2]=output.tolist()[0]

    model = torch.load("EN_disorder_4.pt")  
    output = model.predict(scaled_case)
    print(4, output)
    base[4]=output.tolist()[0]

    model = torch.load("EN_disorder_5.pt")  
    output = model.predict(scaled_case)
    print(5, output)
    base[5]=output.tolist()[0]

    model = torch.load("EN_disorder_6.pt")  
    output = model.predict(scaled_case)
    print(6, output)
    base[6]=output.tolist()[0]

    model = torch.load("EN_disorder_8.pt")  
    output = model.predict(scaled_case)
    print(8, output)
    base[8]=output.tolist()[0]

    model = torch.load("EN_disorder_10.pt")  
    output = model.predict(scaled_case)
    print(10, output)
    base[10]=output.tolist()[0]

    model = torch.load("EN_disorder_11.pt")  
    output = model.predict(scaled_case)
    print(11, output)
    base[11] = output.tolist()[0]

    lst = LSTMModel(input_size=1140)
    lst = torch.load("LSTM_disorder_1.pt") 
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(scaled_case, dtype=torch.float32), torch.tensor(scaled_case, dtype=torch.float32))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    for inputs, _ in test_loader:
        inputs = inputs.unsqueeze(1)
        outputs = lst(inputs)
        print(1, output)
        base[1] = outputs.tolist()[0][0]

    lst = torch.load("LSTM_disorder_9.pt") 
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(scaled_case, dtype=torch.float32), torch.tensor(scaled_case, dtype=torch.float32))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    for inputs, _ in test_loader:
        inputs = inputs.unsqueeze(1)
        outputs = lst(inputs)
        base[9] = outputs.tolist()[0][0]

    cat = CatBoost()
    cat = torch.load("CatBoost_disorder_7.pt")  
    output = cat.predict(scaled_case)
    print(7, output)
    base[7] = output.tolist()[0]

    out = np.array(base)

    def softmax(x):
        # Subtracting the maximum value of x from each element for numerical stability
        e_x = np.exp(x)
        return e_x / e_x.sum()

    final = softmax(out)

    #print(out)

    return final.tolist()

for i in range(10):
    print(evaluate(X_scaled[i]))
evaluate('sample2.csv')
evaluate('sample1.csv')