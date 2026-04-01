import os
import numpy as np
import torch
import random

# --- PARAMETERS ---
# Define fN epsN vectors
fN_vec = torch.linspace(0.01, 0.08, 10)
epsN_vec = torch.linspace(0.005, 0.2, 10)

num_pairs = 5  # Number of couples (j,k) to select
# if you select 100, you obtain the full database, i.e. brute force

# Folder path with files .txt
# path = os.path.join(os.getcwd(), "Database_MeshFact_1") # OLD works

# folder with file .py
base_dir = os.path.dirname(os.path.abspath(__file__))

# folder with database
data_path = os.path.join(base_dir, "..", "Database_MeshFact_1", "Database_MeshFact_1")
path = os.path.abspath(data_path)  # normalize path

files = os.listdir(path)

# --- EXTRACT INDEXES J,K from file names ---
indices = []
for file in files:
    if file.endswith(".txt"):
        name = file.replace("simtrazout_", "").replace(".txt", "")
        try:
            j, k = map(int, name.split("_"))
            indices.append((j, k))
        except:
            print(f"File name not recognised: {file}")

# --- Random choice of 5 files ---
random.seed()
selected = random.sample(indices, num_pairs)

print("File random chosen:")
print(" j | k |     fN     |    epsN")
print("--------------------------------")
for j, k in selected:
    print(f"{j:2d} | {k:2d} | {fN_vec[j - 1]:.4f} | {epsN_vec[k - 1]:.4f}")
print("--------------------------------\n")

# --- CONSTRUCTION REDUCED DATABSE --- #
database = {}

for j, k in selected:
    filename = f"simtrazout_{j}_{k}.txt"
    filepath = os.path.join(path, filename)

    # Read the file
    data = np.loadtxt(filepath, skiprows=1)

    # Save in dictionary
    database[(fN_vec[j - 1], epsN_vec[k - 1])] = torch.tensor(data[:, 1])

# --- SAVE --- #
displacements = torch.tensor(data[:, 0], dtype=torch.float32)

torch.save(database, "Database_MeshFact_1_subset.pt")
torch.save(displacements, "Displacements.pt")
torch.save(fN_vec, "fN_vec.pt")
torch.save(epsN_vec, "epsN_vec.pt")

print(" Random Database saved in 'Database_MeshFact_1_subset.pt'")
