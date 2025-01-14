import numpy as np

def generate_erdos_graphs(k: int, n: int, pi: float) -> np.array:
    """
    Generate k Erdos-Renyi graphs with n nodes and edge probability p

    Params:
        k: int. Number of graphs to generate
        n: int. Number of nodes
        pi: float. Edge probability
    
    Returns:
        np.array. Array of adjacency matrices of shape (k, n, n)
    """
    # Generate a sample of binomial random variables and create the upper triangular part of the adjacency matrix
    sample = np.triu(np.random.binomial(1, pi, [k, n, n]), 1)
    # Return the adjacency matrix by adding its transpose to the upper triangular part
    return sample + np.transpose(sample, (0, 2, 1))


def compute_ns_mats(adj_mats, s):
    """
    Compute the neighbour set matrices up to stage s from the adjacency matrix A

    Params:
        A: np.array. Adjacency matrix. Shape (n, n)
        s: int. Maximum stage of neighbour dependence
    
    Returns:
        A_tensor: np.array. Tensor of powers of the adjacency matrix. Shape (s, n, n)
    """
    k, n, _ = np.shape(adj_mats)
    # Create the tensor containing the adjacency matrices for each stage of neighbour dependence up to stage s
    ns_mats = np.zeros([k, s, n, n])
    # Compute the stage 1 adjacency matrices
    n_nodes = np.sum(adj_mats, axis=2).reshape(k, 1, n)
    ns_mats[:, 0] = np.divide(adj_mats, n_nodes, out=ns_mats[:, 0], where=(n_nodes!=0))
    stage_s_mats = adj_mats.copy()
    # Initialise see matrix to keep track of nodes that have been visited to avoid cycles and adding nodes multiple times
    seen = np.array([np.eye(n)] * k)
    # Compute the adjacency matrices for each stage of neighbour dependence up to stage s
    for i in range(1, s):
        seen = seen + stage_s_mats
        stage_s_mats = np.clip(stage_s_mats @ adj_mats, 0, 1)
        stage_s_mats[seen > 0] = 0
        n_nodes = np.sum(stage_s_mats, axis=2).reshape(k, 1, n)
        ns_mats[:, i] = np.divide(stage_s_mats, n_nodes, out=ns_mats[:, i], where=(n_nodes!=0))
    return ns_mats