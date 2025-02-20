import pandas as pd
import torch
from models.interatomic_nn import InteratomicPotentialNN

# Load dataset.csv
df = pd.read_csv("data/dataset.csv")
print("Dataset loaded successfully.")

# Select only the first 6 features (positional data)
X_test = torch.tensor(df.iloc[:, 3:9].values, dtype=torch.float32)  # x1, y1, z1, x2, y2, z2

# Initialize model with correct input dimensions
input_dim = 6  # Trained model expects 6 features
model = InteratomicPotentialNN(input_dim=input_dim)

# Load model weights
model.load_state_dict(torch.load("results/model_weights.pth"))
print("Model loaded successfully.")

# Run predictions
predictions = model(X_test).detach().numpy()
print("Predictions generated:", predictions)

