import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10,2)

    def forward(self, x):
        return self.fc(x)
    
model = Model()

def evaluate(csv_file):
    output = []
    x = pd.read_csv(csv_file)
    model = torch.load('EN_disorder_0.pt')
    with torch.no_grad():
        out = model(x)
    print(out)

evaluate('data.csv')

