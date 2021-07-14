from decimal import Decimal

from network_dismantling.common.config import output_path

threshold = {
    "train": Decimal("0.18"),
    "test": Decimal("0.1")  # 0.05
}

all_features = ["num_vertices", "num_edges", "degree", "clustering_coefficient", "eigenvectors", "chi_degree",
            "chi_lcc", "pagerank_out", "betweenness_centrality", "kcore"]

# "mean_chi_degree", "mean_chi_lcc",

features_indices = dict(zip(all_features, range(len(all_features))))

base_models_path = output_path / "models/"
