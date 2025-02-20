import torch
import torch.nn as nn

class InteratomicPotentialNN(nn.Module):
    def __init__(self, input_dim):
        super(InteratomicPotentialNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output: predicted energy
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Example usage
if __name__ == "__main__":
    model = InteratomicPotentialNN(input_dim=6)
    print(model)

