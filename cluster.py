"""
cluster.py: four community‐detection methods on a vote‐graph.

Exports:
    - girvan_newman(G, k)  
    - infomap_communities(G, flags)  
    - label_propagation(G)  
    - spectral(G, k, seed)
"""

import networkx as nx
from networkx.algorithms.community import girvan_newman as gn
from networkx.algorithms.community import label_propagation_communities
from sklearn.cluster import SpectralClustering
from infomap import Infomap

from typing import Tuple, Dict

ClusteringResult = Tuple[nx.Graph, Dict[str, int]]

def girvan_newman(G:nx.Graph, k:int=3) -> ClusteringResult:
    """
    Perform Girvan–Newman community detection on graph G.

    This algorithm iteratively removes edges of highest betweenness centrality
    until at least k communities remain.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (can be directed or undirected).
    k : int, optional (default=3)
        The desired number of communities to extract.

    Returns
    -------
    G : networkx.Graph
        The (unchanged) input graph, returned for API consistency.
    colors : dict
        Mapping node → matplotlib color code (e.g. 'C0', 'C1', …) indicating
        the community assignment.
    """
    G = G.copy()

    comp_gen = gn(G)  
    for communities in comp_gen:
        if len(communities) >= k:
            break

    colors = {node: i for i, comm in enumerate(communities) for node in comm}
    return G, colors

def infomap_communities(G:nx.Graph) -> ClusteringResult:
    """
    Detect communities in a directed, weighted graph using Infomap.

    Maps node names to integer IDs, feeds edges (with weights) into Infomap,
    runs the clustering, and then maps results back to original node names.

    Parameters
    ----------
    G : networkx.DiGraph
        A directed graph with optional 'weight' attributes on edges.

    Returns
    -------
    G : networkx.DiGraph
        The (unchanged) input graph, returned for consistency.
    color_map : dict
        Mapping node → matplotlib color code indicating the Infomap module
        to which each node belongs.
    """
    G = G.copy()

    # 1. Create a mapping: player_name → unique int ID
    node_to_id = {name: idx for idx, name in enumerate(G.nodes(), start=1)}
    id_to_node = {idx: name for name, idx in node_to_id.items()}

    # Initialize Infomap (directed, two-level module)
    im = Infomap("--two-level --directed")

    # 2a. (Optional) Explicitly add nodes
    for name, idx in node_to_id.items():
        im.addNode(idx)

    # 2b. Add links using integer IDs and weights
    for u, v, data in G.edges(data=True):
        uid = node_to_id[u]
        vid = node_to_id[v]
        weight = float(data.get('weight', 1))
        im.addLink(uid, vid, weight)

    # Run the clustering
    im.run()

    # Build clusters: moduleIndex() → list of node_ids
    clusters = {}
    for node in im.nodes:
        module = node.moduleIndex()
        clusters.setdefault(module, []).append(id_to_node[node.node_id])

    # Assign a distinct color to each module
    color_map = {}
    for i, members in enumerate(clusters.values()):
        for name in members:
            color_map[name] = i

    return G, color_map

def label_propagation(G:nx.Graph) -> ClusteringResult:
    """
    Find communities via label propagation on the undirected version of G.

    Converts G to an undirected graph, then repeatedly relabels each node
    to the most frequent label among its neighbors until convergence.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        The input graph. Directionality will be ignored.

    Returns
    -------
    H : networkx.Graph
        The undirected copy of G used for propagation.
    colors : dict
        Mapping node → matplotlib color code indicating its final label
        propagation community.
    """
    G = G.to_undirected()

    communities = list(label_propagation_communities(G))
    colors = {}
    for i, comm in enumerate(communities):
        for node in comm:
            colors[node] = i

    return G, colors

def spectral(G:nx.Graph, k:int=3, seed=None) -> ClusteringResult:
    """
    Cluster nodes using Spectral Clustering on the adjacency matrix of G.

    Builds a precomputed affinity matrix from G’s weighted edges, performs
    spectral embedding followed by k‑means, and assigns each node a cluster.

    Parameters
    ----------
    G : networkx.Graph
        The input graph (directed or undirected) with optional 'weight' edges.
    k : int, optional (default=3)
        Number of clusters for the spectral algorithm.
    seed : int or None, optional
        Random seed for k‑means initialization (for reproducibility).

    Returns
    -------
    G : networkx.Graph
        The (unchanged) input graph, returned for API consistency.
    colors : dict
        Mapping node → matplotlib color code indicating its spectral cluster.
    """
    G = G.copy()

    # Build adjacency matrix A
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight='weight')

    sc = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=seed)
    labels = sc.fit_predict(A)

    # Map labels back to nodes
    colors = {nodes[i]: lab for i, lab in enumerate(labels)}

    return G, colors
