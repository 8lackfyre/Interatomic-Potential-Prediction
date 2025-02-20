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

import matplotlib.pyplot as plt

# Scatter plot of predictions
plt.figure(figsize=(8, 6))
plt.scatter(range(len(predictions)), predictions, color='blue', label='Predictions')
plt.xlabel("Sample Index")
plt.ylabel("Predicted Value")
plt.title("Interatomic Potential Predictions")
plt.legend()

# Save the figure as an image
plt.savefig("results/predictions_plot.png")  
print("Plot saved as 'results/predictions_plot.png'")

