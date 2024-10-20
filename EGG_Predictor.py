import lightgbm as lgb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Step 1: Prepare the data (example data)
df = pd.read_csv('data.csv')
labels = df["specific.disorder"].unique().tolist()
x = []
y = []

for i, row in df.iterrows():
    idx = labels.index(row.tolist()[7])  # Assuming "specific.disorder" is at index 7
    y.append(idx)  # Using integer labels directly
    x.append(row.tolist()[8:])  # Assuming features start at index 8

print(labels)
X = np.array(x)
Y = np.array(y)

# Step 1.5: Scale Features and get rid of missing values
imputer = SimpleImputer(strategy='mean')  # Use 'mean', 'median', or another strategy if needed
X_imputed = imputer.fit_transform(X)
print("NaN values in X:", np.isnan(X_imputed).sum())
print("X Shape:", X_imputed.shape)
print("Y Shape:", Y.shape)

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

# Convert labels to integers if necessary
Y = Y.astype(int)
num_classes = len(np.unique(Y))

# Split the data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
)
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)
# Now, X_train_full is 60%, X_val_full is 20%, X_test is 20%

def prepare_binary_labels(y, class_index):
    return np.where(y == class_index, 1, 0)

# Define PyTorch models

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
        return x.squeeze()

class TransformerModel(nn.Module):
    def __init__(self, input_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average over sequence length
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze()

# Training and evaluation functions

def train_pytorch_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, model_name):
    num_epochs = 1
    batch_size = 32
    patience = 10
    trigger_times = 0
    best_val_loss = float('inf')
    
    # Prepare DataLoaders
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.unsqueeze(1)  # Add sequence length dimension for LSTM/Transformer
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            predictions = (outputs >= 0.5).float()
            correct_preds += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_preds / total_samples
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct_preds = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                predictions = (outputs >= 0.5).float()
                val_correct_preds += (predictions == labels).sum().item()
                val_total_samples += labels.size(0)
                print(predictions)
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct_preds / val_total_samples
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            best_model_state = model.state_dict()
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    return train_loss, val_loss, train_acc, val_acc

def evaluate_pytorch_model(model, criterion, X_test, y_test):
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    model.eval()
    test_running_loss = 0.0
    test_correct_preds = 0
    test_total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * inputs.size(0)
            predictions = (outputs >= 0.5).float()
            test_correct_preds += (predictions == labels).sum().item()
            test_total_samples += labels.size(0)
    
    test_loss = test_running_loss / len(test_loader.dataset)
    test_acc = test_correct_preds / test_total_samples
    return test_loss, test_acc

def train_sklearn_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)[:]
    val_preds = model.predict(X_val)[:]
    train_loss = log_loss(y_train, train_preds)
    val_loss = log_loss(y_val, val_preds)
    train_acc = accuracy_score(y_train, (train_preds >= 0.5).astype(int))
    val_acc = accuracy_score(y_val, (val_preds >= 0.5).astype(int))
    return train_loss, val_loss, train_acc, val_acc

def evaluate_sklearn_model(model, X_test, y_test):
    test_preds = model.predict(X_test)[:]
    test_loss = log_loss(y_test, test_preds)
    test_acc = accuracy_score(y_test, (test_preds >= 0.5).astype(int))
    return test_loss, test_acc

def train_evaluate_model(model_name, disorder_index):
    # Prepare binary labels for the disorder
    y_train = prepare_binary_labels(y_train_full, disorder_index)
    y_val = prepare_binary_labels(y_val_full, disorder_index)
    y_test_disorder = prepare_binary_labels(y_test, disorder_index)
    
    # Handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Initialize model
    if model_name == 'LSTM':
        model = LSTMModel(input_size=X_train_full.shape[1]).to(device)
        criterion = nn.BCELoss(weight=class_weights[1].to(device))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_loss, val_loss, train_acc, val_acc = train_pytorch_model(model, criterion, optimizer, X_train_full, y_train, X_val_full, y_val, model_name)
    elif model_name == 'Transformer':
        model = TransformerModel(input_size=X_train_full.shape[1]).to(device)
        criterion = nn.BCELoss(weight=class_weights[1].to(device))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_loss, val_loss, train_acc, val_acc = train_pytorch_model(model, criterion, optimizer, X_train_full, y_train, X_val_full, y_val, model_name)
    elif model_name == 'EN':
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
        train_loss, val_loss, train_acc, val_acc = train_sklearn_model(model, X_train_full, y_train, X_val_full, y_val)
    elif model_name == 'LR':
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        train_loss, val_loss, train_acc, val_acc = train_sklearn_model(model, X_train_full, y_train, X_val_full, y_val)
    elif model_name == 'LightGBM':
        model = lgb.LGBMClassifier(class_weight='balanced', n_estimators=100)
        train_loss, val_loss, train_acc, val_acc = train_sklearn_model(model, X_train_full, y_train, X_val_full, y_val)
    elif model_name == 'CatBoost':
        model = CatBoostClassifier(class_weights=[class_weights[0].item(), class_weights[1].item()], iterations=100, verbose=False)
        train_loss, val_loss, train_acc, val_acc = train_sklearn_model(model, X_train_full, y_train, X_val_full, y_val)
    else:
        raise ValueError('Invalid model name')
    
    # Evaluate on test set
    if model_name in ['LSTM', 'Transformer']:
        test_loss, test_acc = evaluate_pytorch_model(model, criterion, X_test, y_test_disorder)
    else:
        test_loss, test_acc = evaluate_sklearn_model(model, X_test, y_test_disorder)
    
    
    return {
        'model_name': model_name,
        'disorder_index': disorder_index,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'model': model  # Save the trained model
    }

# Main code to train and evaluate models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = []
model_names = ['LSTM', 'Transformer', 'EN', 'LR', 'LightGBM', 'CatBoost']

for disorder_index in range(num_classes):
    print(f"Training models for disorder {disorder_index}")
    for model_name in model_names:
        print(f"  Model: {model_name}")
        result = train_evaluate_model(model_name, disorder_index)
        torch.save(result['model'], f'{model_name}_disorder_{disorder_index}.pt')
        results.append(result)

# Create a DataFrame of results

results_df = pd.DataFrame(results)

# Plotting bar charts for validation accuracy

for disorder_index in range(num_classes):
    disorder_results = results_df[results_df['disorder_index'] == disorder_index]
    plt.figure(figsize=(10, 5))
    plt.bar(disorder_results['model_name'], disorder_results['val_acc'])
    plt.title(f'Validation Accuracy for Disorder {disorder_index}')
    plt.xlabel('Model')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'training_data/val_accuracy_disorder_{disorder_index}.png')
    plt.close()

# Select the best models for each disorder

best_models = {}
for disorder_index in range(num_classes):
    disorder_results = results_df[results_df['disorder_index'] == disorder_index]
    best_row = disorder_results.loc[disorder_results['val_acc'].idxmax()]
    best_models[disorder_index] = {
        'model_name': best_row['model_name'],
        'model': best_row['model']
    }
    print(f"Best model for disorder {disorder_index}: {best_row['model_name']} with validation accuracy {best_row['val_acc']:.4f}")

# Function to get disorder probabilities

def get_disorder_probabilities(best_models, X):
    probabilities = []
    for disorder_index in range(num_classes):
        model_info = best_models[disorder_index]
        model = model_info['model']
        model_name = model_info['model_name']
        if model_name in ['LSTM', 'Transformer']:
            model.eval()
            dataset = torch.tensor(X, dtype=torch.float32)
            loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
            disorder_probs = []
            with torch.no_grad():
                for inputs in loader:
                    inputs = inputs.to(device)
                    inputs = inputs.unsqueeze(1)
                    outputs = model(inputs)
                    disorder_probs.extend(outputs.cpu().numpy())
            probabilities.append(np.array(disorder_probs))
        else:
            disorder_probs = model.predict(X)[:]
            probabilities.append(disorder_probs)
    return np.column_stack(probabilities)  # Shape: (num_samples, num_classes)

# Generate probabilities for training, validation, and test sets

train_probs = get_disorder_probabilities(best_models, X_train_full)
val_probs = get_disorder_probabilities(best_models, X_val_full)
test_probs = get_disorder_probabilities(best_models, X_test)

# Define the final prediction model

class FinalPredictionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FinalPredictionModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        return self.fc(x)

# Train the final model

def train_final_model(X_train, y_train, X_val, y_val):
    model = FinalPredictionModel(input_size=num_classes, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 20
    patience = 10
    trigger_times = 0
    best_val_loss = float('inf')
    
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_preds / total_samples
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct_preds = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct_preds += (predicted == labels).sum().item()
                val_total_samples += labels.size(0)
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct_preds / val_total_samples
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            best_model_state = model.state_dict()
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    # Plotting
    epochs = range(1, len(train_losses) + 1)
    
    # Plot Loss
    plt.figure()
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Final Model Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_data/final_model_loss.png')
    plt.close()
    
    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Final Model Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_data/final_model_accuracy.png')
    plt.close()
    
    return model

# Train the final model

final_model = train_final_model(train_probs, y_train_full, val_probs, y_val_full)

# Evaluate the final model on the test set

def evaluate_final_model(model, X_test, y_test):
    model.eval()
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    test_running_loss = 0.0
    test_correct_preds = 0
    test_total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_correct_preds += (predicted == labels).sum().item()
            test_total_samples += labels.size(0)
    
    test_loss = test_running_loss / len(test_loader.dataset)
    test_acc = test_correct_preds / test_total_samples
    return test_loss, test_acc

test_loss, test_acc = evaluate_final_model(final_model, test_probs, y_test)
print(f'Final Model Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')