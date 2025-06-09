import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
import torch
import numpy as np
import random
from typing import List
import os
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

def load_hiv(path: str, load_on_device: bool = False, convert_fingerprints_to_float: bool = False, fpSize: int = 2048, radius: int = 3, device = None):

    hiv_dataset = pd.read_csv(path)
    smiles = hiv_dataset["smiles"]
    labels = hiv_dataset["HIV_active"]

    smiles_succeded = []
    labels_succeded = []
    fingerprints = []

    fpgen = AllChem.GetMorganGenerator(fpSize = fpSize, radius=radius)

    for id, smile in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol)
            fingerprints.append(fpgen.GetFingerprintAsNumPy(mol))
            smiles_succeded.append(smile)
            labels_succeded.append(labels[id])
        except:
            print(smile)

    fingerprints = torch.from_numpy(np.array(fingerprints))
    labels_succeded = torch.from_numpy(np.array(labels_succeded))

    if convert_fingerprints_to_float:
        fingerprints = fingerprints.float()

    if load_on_device:
        fingerprints = fingerprints.to(device)
        labels_succeded = labels_succeded.to(device)

    return fingerprints, labels_succeded, smiles_succeded

def find_repeating_fingerprints(fingerprints, labels, smiles):

    # Step 1: Find unique rows and their inverse indices
    _, inverse_indices = torch.unique(fingerprints, dim=0, return_inverse=True)

    # Step 2: Count how often each row appears using the inverse indices
    row_counts = torch.bincount(inverse_indices)

    # Step 3: Find the indices of repeating rows (i.e., rows that appear more than once)
    repeating_rows_indices = (row_counts > 1).nonzero(as_tuple=True)[0]

    # Step 4: Get the indices of the repeating rows in the original tensor
    repeating_row_indices_in_tensor = [torch.where(inverse_indices == idx)[0] for idx in repeating_rows_indices]

    repeating_labels = [[labels[id].item() for id in idx] for idx in repeating_row_indices_in_tensor]

    return repeating_row_indices_in_tensor, repeating_labels

def drop_duplicates(fingerprints, labels, smiles):

    repeating_row_indices_in_tensor, repeating_labels = find_repeating_fingerprints(fingerprints, labels, smiles)
    indices_to_drop = []

    for indices, same_fingerprint_labels in zip(repeating_row_indices_in_tensor, repeating_labels):

        indices = indices.tolist()

        # drop all records
        if len(set(same_fingerprint_labels)) > 1:
            new_indices_to_drop = indices

        # leave only one record
        else:
            new_indices_to_drop = random.sample(indices, len(indices)-1)

        indices_to_drop = indices_to_drop + new_indices_to_drop

    # new dataset
    cleared_smiles = []
    indices_to_drop_set = set(indices_to_drop)
    for id in range(len(smiles)):
        if id not in indices_to_drop_set:
            cleared_smiles.append(smiles[id])

    mask = torch.ones(fingerprints.size(0), dtype=torch.bool)
    mask[indices_to_drop] = False
    cleared_fingerprints = fingerprints[mask]
    cleared_labels = labels[mask]

    print(f"Dataset reduced from {fingerprints.shape[0]} rows to {cleared_fingerprints.shape[0]}")

    return cleared_fingerprints, cleared_labels, cleared_smiles


def extract_validation_part(X: torch.Tensor, y: torch.Tensor, smiles: List[str], similarity_matrix: torch.Tensor, validation_size: float, device):
    """Splits the molecules into validation part and the rest of the dataset"""

    n_molecules_validation = int(validation_size * X.shape[0])

    # find the minimal distances for each molecule
    min_distances = torch.min(similarity_matrix, dim=1).values

    # find indices of the molecules with the biggest distances to their nearest neighbours
    _, val_indices = torch.topk(min_distances, largest = True, k=n_molecules_validation)
    val_indices_list = val_indices.tolist()
    print(f"val_indices_list: {val_indices_list}")

    # split the data on validation and rest
    X_val, y_val, smiles_val, similarity_matrix_val = X[val_indices], y[val_indices], [smiles[id] for id in val_indices_list], similarity_matrix[val_indices]
    print(f"len(smiles_val): {len(smiles_val)}")

    mask = torch.ones(X.shape[0])
    mask[val_indices] = 0
    mask = mask.bool()
    X_rest, y_rest, similarity_matrix_rest = X[mask], y[mask], similarity_matrix[mask]
    smiles_rest = [smiles[id] for id in range(len(smiles)) if id not in set(val_indices_list)]
    print(f"len(smiles_rest): {len(smiles_rest)}")

    return X_val, X_rest, y_val, y_rest, smiles_val, smiles_rest, similarity_matrix_val, similarity_matrix_rest


def jaccard_similarity(X: torch.Tensor, similarity_matrix_path: str, device):
    """
    Compute the Jaccard similarity matrix for all pairs of binary vectors in the X.
    This function processes the X in smaller chunks to avoid memory issues.
    """

    if os.path.isfile(similarity_matrix_path):
        print(f"Loading similarity matrix from {similarity_matrix_path}")
        return torch.load(similarity_matrix_path, map_location=device, weights_only=True)

    print(f"Started calculating the Jaccard similarity matrix")

    # Initialize a tensor to hold the similarity matrix (empty for now)
    similarity_matrix = torch.zeros(X.size(0), X.size(0), device=device)

    # Step 2: Process the X in smaller chunks to avoid GPU memory overflow
    chunk_size = 256  # You can reduce this if needed, depending on your GPU memory
    for i in range(0, X.size(0), chunk_size):
        start = i
        end = min(i + chunk_size, X.size(0))

        # Get a chunk of the X
        chunk = X[start:end]

        # Calculate the pairwise Jaccard similarity for this chunk
        chunk_similarity = torch.mm(chunk, X.T)  # Intersection matrix (chunk vs. all)
        chunk_union = chunk.sum(dim=1).view(-1, 1) + X.sum(dim=1).view(1, -1) - chunk_similarity

        # Compute Jaccard similarity for this chunk
        similarity_matrix[start:end, :] = chunk_similarity.float() / chunk_union.float()

    # Free unused memory
    torch.cuda.empty_cache()

    torch.save(similarity_matrix, similarity_matrix_path)
    print(f"Similarity matrix saved on {similarity_matrix_path}")

    return similarity_matrix

def load_clusterization_data(path: str, reset_cluster_labels: bool):

    def find_matching_path(kern_name: str, files: List[str], path: str):
        for file in files:
            if kern_name in file:
                return f"{path}/{file}"
        return None

    files = os.listdir(path)

    centroids = torch.load(find_matching_path("centroids", files, path), torch.device("cpu"))
    cluster_labels = torch.load(find_matching_path("labels", files, path), torch.device("cpu"))
    silhouette_scores = torch.load(find_matching_path("silhouette_scores", files, path), torch.device("cpu"))
    centroid_shifts = torch.load(find_matching_path("centroid_shifts", files, path), torch.device("cpu"))

    if reset_cluster_labels:
        _, cluster_labels = cluster_labels.unique(sorted=True, return_inverse=True)

    return centroids, cluster_labels, silhouette_scores, centroid_shifts

def map_scaffolds(smiles, scaffolds_as_fingerprints: bool, fpsize: int, radius: int):
    scaffolds = []
    scaffold_smiles = []
    scaffold_fingerprints = []
    fpgen = AllChem.GetMorganGenerator(fpSize = fpsize, radius=radius) if scaffolds_as_fingerprints else None

    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)

        scaffold = GetScaffoldForMol(mol)
        scaffold = Chem.AddHs(scaffold)

        scaffolds.append(scaffold)
        scaffold_smiles.append(Chem.MolToSmiles(scaffold))

        if scaffolds_as_fingerprints:
            scaffold_fingerprint = fpgen.GetFingerprintAsNumPy(scaffold)
            scaffold_fingerprints.append(scaffold_fingerprint)

    return scaffolds, scaffold_smiles, scaffold_fingerprints