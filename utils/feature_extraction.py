import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.emt import EMT
from ase.neighborlist import NeighborList

def compute_descriptors(atoms):
    """Compute atomic descriptors: Atomic number + distances from neighbors"""
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    nl = NeighborList([1.5] * len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)

    descriptors = []
    for i, atom in enumerate(atoms):
        indices, offsets = nl.get_neighbors(i)
        distances = np.linalg.norm(positions[indices] - positions[i], axis=1)
        feature_vector = [atomic_numbers[i]] + list(distances[:5])  # Top 5 neighbors
        descriptors.append(feature_vector)

    return np.array(descriptors)

# Example usage
atoms = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.2]])
atoms.set_calculator(EMT())
descriptors = compute_descriptors(atoms)
print("Descriptors:", descriptors)

