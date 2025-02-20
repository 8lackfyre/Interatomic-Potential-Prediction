from ase import Atoms
from ase.calculators.emt import EMT
import pandas as pd
import numpy as np

# Create a dataset
data = []

for i in range(1000):  # Generate 1000 samples
    # Generate a random atomic system (Oxygen, Hydrogen, Carbon, etc.)
    atoms = Atoms('H2O', positions=np.random.rand(3, 3) * 5)  
    atoms.set_calculator(EMT())  # Use EMT potential to approximate energy

    # Compute descriptors
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions().flatten()
    energy = atoms.get_potential_energy()  # Energy label

    data.append(list(atomic_numbers) + list(positions) + [energy])

# Convert to Pandas DataFrame
columns = ['Atom1', 'Atom2', 'Atom3', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'Energy']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("data/dataset.csv", index=False)
print("Synthetic dataset created.")

