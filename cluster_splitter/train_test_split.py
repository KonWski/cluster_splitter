import torch
from typing import List
import numpy as np
import math
import itertools
import sys
import os

def train_test_split(method: str, **kwargs):

    def scaffold_split(X: torch.tensor, y: torch.tensor, smiles: List[str], cluster_labels_tensor: torch.tensor, test_size: float, random_state: int = None):

        if random_state:
            np.random.seed(random_state)

        unique_cluster_labels = torch.unique(cluster_labels_tensor).tolist()
        test_cluster_labels = []

        approximate_test_size = int(len(unique_cluster_labels) * test_size)
        current_test_size = 0

        while current_test_size < approximate_test_size:
            cluster_label = np.random.choice(unique_cluster_labels)
            if cluster_label not in test_cluster_labels:
                test_cluster_labels.append(cluster_label)
                cluster_size = (cluster_labels_tensor == cluster_label).sum()
                current_test_size += cluster_size


        test_indices = torch.isin(cluster_labels_tensor, torch.tensor(test_cluster_labels))
        train_indices = ~test_indices

        return X[train_indices], X[test_indices], y[train_indices], y[test_indices], [smiles[id] for id in train_indices], \
         [smiles[id] for id in test_indices], cluster_labels_tensor[train_indices], cluster_labels_tensor[test_indices], train_indices, test_indices

    def cluster_split(X: torch.tensor, y: torch.tensor, smiles: List[str], distance_matrix: torch.tensor, cluster_labels_tensor: torch.tensor, 
                      test_size: float, path_g_matrix: str, path_last_pair: str, device):
        """The average distance between clusters within the cluster set K

        Parameters
        ----------
        X: torch.Tensor
            fingerprints of molecules
        cluster_labels_tensor: torch.Tensor
            ids of clusters associated to each fingerprint
        test_size: float
        g_matrix: torch.Tensor
            matrix of g distances between clusters
        """

        unique_cluster_labels = torch.unique(cluster_labels_tensor).tolist()
        test_cluster_labels = []
        approximate_test_size = int(len(unique_cluster_labels) * test_size)
        g_matrix = get_g_matrix(distance_matrix, cluster_labels_tensor, len(unique_cluster_labels), path_g_matrix, path_last_pair, device)

        # initiate test set
        initial_cluster_label = initiate_test_set(g_matrix)
        test_cluster_labels.append(initial_cluster_label)
        unique_cluster_labels.remove(initial_cluster_label)
        current_test_size = cluster_labels_tensor[cluster_labels_tensor == test_cluster_labels[0]].shape[0]
        print("Initiated test set")

        while current_test_size < approximate_test_size:

            print(f"current_test_size: {current_test_size}, approximate_test_size: {approximate_test_size}")
            print(f"len(test_cluster_labels): {len(test_cluster_labels)}")
            min_split_value, min_split_value_cluster_label = split_value(test_cluster_labels, unique_cluster_labels, g_matrix)

            test_cluster_labels.append(min_split_value_cluster_label)
            unique_cluster_labels.remove(min_split_value_cluster_label)

            min_split_value_cluster_size = cluster_labels_tensor[cluster_labels_tensor == min_split_value_cluster_label].shape[0]
            current_test_size += min_split_value_cluster_size

        test_indices = torch.isin(cluster_labels_tensor, torch.tensor(test_cluster_labels))
        train_indices = ~test_indices

        return X[train_indices], X[test_indices], y[train_indices], y[test_indices], [smiles[id] for id in train_indices], \
         [smiles[id] for id in test_indices], cluster_labels_tensor[train_indices], cluster_labels_tensor[test_indices], train_indices, test_indices

    if method == 'scaffold':
        return scaffold_split(**kwargs)

    elif method == "cluster":
        return cluster_split(**kwargs)

    else:
        raise ValueError(f"Unknown method: {method}")

def initiate_test_set(g_matrix: torch.tensor):
    initial_cluster_id = g_matrix.min(dim=1).values.max(0).indices.item()
    print(f"Test set initiated with cluster id: {initial_cluster_id}")
    return initial_cluster_id

def get_distance_matrix(X: torch.Tensor, path_distance_matrix: str, device = None):
    """
    Parameters
    ----------
    """
    if not device:
        device = torch.device("cpu")

    distance_matrix_loaded = False
    if os.path.exists(path_distance_matrix):
        print(f"Loading distance_matrix from {path_distance_matrix}")
        distance_matrix_loaded = True
        distance_matrix = torch.load(path_distance_matrix, map_location = device)

    else:
        distance_matrix = torch.cdist(X, X).to(device)

    if not distance_matrix_loaded:
        torch.save(distance_matrix, path_distance_matrix)
        print(f"distance_matrix saved on {path_distance_matrix}")

    return distance_matrix


def get_g_matrix(distance_matrix: torch.Tensor, cluster_labels_tensor: torch.Tensor, n_unique_cluster_labels: int,  path_g_matrix: str, path_last_pair: str, device):
    """Generate matrix indicating g distances between clusters

    Parameters
    ----------
    distance_matrix: torch.Tensor
        matrix of distance between molecules' fingerprints
    cluster_labels_tensor: torch.Tensor
        ids of clusters associated to each fingerprint
    n_unique_cluster_labels: int
        number of unique clusters' labels
    path_g_matrix: str
        path to the g_matrix
    """

    # g_cluster
    if os.path.exists(path_g_matrix):
        print(f"Loading g_matrix from {path_g_matrix}")
        g_clusters = torch.load(path_g_matrix, map_location=device)

    else:
        g_clusters = torch.zeros(n_unique_cluster_labels, n_unique_cluster_labels)
        g_clusters.to(device)

    if (g_clusters == 0).sum().item() == n_unique_cluster_labels:
        return g_clusters

    # last_pair
    if os.path.exists(path_last_pair):
        print(f"Loading last pair of clusters from {path_last_pair}")
        with open(path_last_pair, 'r') as f:
            line = f.readline().strip()
            a, b = map(int, line.split(','))
            last_pair = (a, b)
    else:
        last_pair = None

    clusters_pairs = list(itertools.combinations(range(n_unique_cluster_labels), 2))
    n_clusters_pairs = len(clusters_pairs)
    it = 0

    if last_pair:
        it = clusters_pairs.index(last_pair)
        clusters_pairs = clusters_pairs[it:]
        print(f"Starting from cluster pair: {it}")

    fully_loaded = True if it + 1 == n_clusters_pairs else False

    # optimization maneuver
    previous_cluster_0 = None

    for cluster_0, cluster_1 in clusters_pairs:

        # optimization maneuver
        if cluster_0 != previous_cluster_0:
            cluster_elements_0_indices = cluster_labels_tensor == cluster_0
            filtered_distance_matrix = distance_matrix[cluster_elements_0_indices, :]
            n_cluster_elements_0 = filtered_distance_matrix.shape[0]
            previous_cluster_0 = cluster_0

        cluster_elements_1_indices = cluster_labels_tensor == cluster_1
        n_cluster_elements_1 = cluster_elements_1_indices.sum()

        if n_cluster_elements_0 > 1 or n_cluster_elements_1 > 1:
            g_value = (filtered_distance_matrix[:, cluster_elements_1_indices]).sum().item() / (n_cluster_elements_0 * n_cluster_elements_1)
        else:
            g_value = (filtered_distance_matrix[:, cluster_elements_1_indices]).item()

        g_clusters[cluster_0, cluster_1] = g_value
        g_clusters[cluster_1, cluster_0] = g_value

        # progress bar
        it += 1
        if it % 100000 == 0 or it + 1 == n_clusters_pairs:
            print(f"Progress: {round(it/n_clusters_pairs, 3)}")

            # save g_matrix
            torch.save(g_clusters, path_g_matrix)
            print(f"g_matrix saved on {path_g_matrix}")

            with open(path_last_pair, "w") as f:
                f.write(f"{cluster_0},{cluster_1}")

    if not fully_loaded:
        torch.save(g_clusters, path_g_matrix)
        print(f"g_matrix saved on {path_g_matrix}")

        with open(path_last_pair, "w") as f:
            f.write(f"{clusters_pairs[-1][0]},{clusters_pairs[-1][1]}")

    return g_clusters


def g(cluster_0: int, cluster_1: int, g_matrix: torch.tensor):
    """Average distance between elements belonging to distinct clusters

    Parameters
    ----------
    cluster_0: int
        id of cluster 0
    cluster_1: int
        id of cluster 1
    g_matrix: torch.Tensor
        matrix of g distances between clusters
    """
    return g_matrix[cluster_0, cluster_1]


def h(unique_cluster_labels: List[int], g_matrix: torch.tensor):
    """The average distance between clusters within the cluster set K

    Parameters
    ----------
    unique_cluster_labels: List[int]
        unique clusters' labels of clusters inside of the set K
    g_matrix: torch.Tensor
        matrix of g distances between clusters
    """

    summed_distances = 0.0
    clusters_pairs = list(itertools.combinations(unique_cluster_labels, 2))
    n_clusters_pairs = len(clusters_pairs)

    # initial, one-element set
    if n_clusters_pairs == 0:
        return summed_distances

    rows, cols = zip(*clusters_pairs)
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)

    summed_distances = g_matrix[rows, cols].sum().item()

    return summed_distances / n_clusters_pairs


def l(cluster_0: int, cluster_labels: List[int], g_matrix: torch.tensor):
    """The average distance between cluster cluster_elements_0 and the other clusters in the set K

    Parameters
    ----------
    cluster_0: int
        id of cluster 0
    cluster_labels: List[int]
        clusters' labels from the set K
    g_matrix: torch.Tensor
        matrix of g distances between clusters
    """

    return g_matrix[cluster_0, cluster_labels].sum().item() / len(cluster_labels)

    # summed_distances = 0.0

    # for cluster_1 in cluster_labels:
    #     summed_distances += g(cluster_0, cluster_1, g_matrix)

    # return summed_distances / len(cluster_labels)


def m(cluster_labels_0: List[int], cluster_labels_1: List[int], g_matrix: torch.Tensor):
    """The average distance between two sets of clusters K’ and K’’,
    calculated as the average distance between pairs of clusters from opposite sets.

    Parameters
    ----------
    cluster_labels_0: List[int]
        clusters' labels belonging to the set K'
    cluster_labels_1: List[int]
        clusters' labels belonging to the set K''
    g_matrix: torch.Tensor
        matrix of g distances between clusters
    """
    ls = g_matrix[torch.tensor(cluster_labels_0), :][:,torch.tensor(cluster_labels_1)].sum(1) / len(cluster_labels_1)
    return ls.sum() / ls.shape[0]

    # n_cluster_labels_0 = len(set(cluster_labels_0))
    # summed_distances = 0.0

    # for cluster_0 in cluster_labels_0:
    #     summed_distances += l(cluster_0, cluster_labels_1, g_matrix)

    # return summed_distances / n_cluster_labels_0


def split_value(test_cluster_labels: List[int], unique_cluster_labels: List[int], g_matrix: torch.tensor):
    """chooses the optimal cluster - the one with the smallest split_value

    Parameters
    ----------
    test_cluster_labels: List[int]
        list of clusters labels belonging to the test part of the dataset
    unique_cluster_labels: List[int]
        list of clusters labels belonging to the rest of the dataset
    g_matrix: torch.Tensor
        matrix of g distances between clusters
    """

    min_split_value = sys.float_info.max
    min_split_value_cluster_label = None

    # softmax part
    h_k_test = h(test_cluster_labels, g_matrix)
    # print(f"Calculated h_k_test")

    m_k_rest_k_test = m(test_cluster_labels, unique_cluster_labels, g_matrix)
    # print(f"Calculated h_k_test, m_k_rest_k_test")

    for id, cluster_label in enumerate(unique_cluster_labels):

        # if id % 1000 == 0:
        #     print(f"Split_value progress: {round(id/n_unique_cluster_labels, 3)}")

        # cluster part
        l_cluster_k_test = l(cluster_label, test_cluster_labels, g_matrix)
        # print(f"Calculated l_cluster_k_test")

        test_cluster_labels_expanded = test_cluster_labels + [cluster_label]
        unique_cluster_labels_reduced = list(set(unique_cluster_labels) - set([cluster_label]))
        m_k_rest_without_cluster_k_test_with_cluster = m(test_cluster_labels_expanded, unique_cluster_labels_reduced, g_matrix)
        # print(f"Calculated m_k_rest_without_cluster_k_test_with_cluster")

        exp_h_k_test = math.exp(h_k_test)
        exp_m_k_rest_k_test = math.exp(m_k_rest_k_test)

        split_value = (1 - exp_h_k_test / (exp_h_k_test + exp_m_k_rest_k_test)) * l_cluster_k_test + \
          (1 - exp_m_k_rest_k_test / (exp_h_k_test + exp_m_k_rest_k_test)) * (m_k_rest_without_cluster_k_test_with_cluster - m_k_rest_k_test)
        # print(f"split_value: {split_value}")

        if split_value < min_split_value:
            min_split_value = split_value
            min_split_value_cluster_label = cluster_label

    return min_split_value, min_split_value_cluster_label