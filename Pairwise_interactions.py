import numpy as np
from Bio.PDB import PDBParser

# Set the cut-off distance and force constant
Rc = 10.0  # Å
k = 10.0   # kcal/mol/Å^2

# Parse the PDB structure
parser = PDBParser()
structure = parser.get_structure('protein', r'C:\Users\Imaging\Documents\Oleg_python\PROJECTS\MarkovStateModel\2zwh_fixed_pH_7.pdb')

# Get the list of central carbon atoms (CA) in the structure
central_carbons = []
for model in structure:
    for chain in model:
        for residue in chain:
            if residue.has_id('CA'):
                central_carbons.append(residue['CA'])

# Calculate the pairwise distances between all central carbon atoms
n_cc = len(central_carbons)
d_cc = np.zeros((n_cc, n_cc))
for i, cci in enumerate(central_carbons):
    for j, ccj in enumerate(central_carbons[i+1:], i+1):
        d_cc[i, j] = d_cc[j, i] = cci - ccj

# Calculate the harmonic potential energy due to pairwise interactions
energy = 0.0
for i in range(n_cc):
    for j in range(i+1, n_cc):
        if d_cc[i, j] < Rc:
            energy += 0.5 * k * (d_cc[i, j] - d_cc[i, j]**2 / Rc**2)**2

print(f"The pairwise interaction energy due to central carbon atoms within {Rc} Å is {energy:.2f} kcal/mol.")
