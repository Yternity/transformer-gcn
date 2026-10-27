"""Data filtering, splitting, and spatial graph-construction utilities."""

import torch
from torch_geometric.data import Data
import math



def stratified_ratio_masks(
    data: Data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    print_stats: bool = False,
    per_class: bool = True,
    min_per_split: int = 0,   #
) -> Data:

    assert hasattr(data, "y"), "Data.y 缺失"
    y = data.y.view(-1).to(torch.long)
    N = y.numel()

    uniques, counts = y.unique(sorted=True, return_counts=True)
    if print_stats:
        print("Per-class counts BEFORE split:")
        for lbl, cnt in zip(uniques.tolist(), counts.tolist()):
            print(f"  class {lbl}: {cnt}")

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)

    g = torch.Generator().manual_seed(seed)

    assert 0.0 <= train_ratio <= 1.0 and 0.0 <= val_ratio <= 1.0
    assert train_ratio + val_ratio <= 1.0 + 1e-9
    # test_ratio = 1 - train_ratio - val_ratio

    # 分层按类切分
    for lbl in uniques.tolist():
        idx = torch.nonzero(y == lbl, as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        n = perm.numel()

        n_train = int(math.floor(train_ratio * n))
        n_val   = int(math.floor(val_ratio   * n))

        if min_per_split > 0:
            n_train = max(min_per_split, n_train)
            n_val   = max(min_per_split, n_val)

        # 修正防溢出
        n_train = min(n_train, n)
        n_val   = min(n_val, max(n - n_train, 0))
        n_test  = max(n - n_train - n_val, 0)


        tr = perm[:n_train]
        va = perm[n_train:n_train + n_val]
        te = perm[n_train + n_val:]

        train_mask[tr] = True
        val_mask[va]   = True
        test_mask[te]  = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

from typing import Dict, Tuple, Optional
def filter_small_classes(
    data: Data,
    min_samples: int = 200,
    remap_labels: bool = True,
) -> Tuple[Data, Dict]:
    assert hasattr(data, "x") and hasattr(data, "y"), "data 必须包含 x 与 y"
    y = data.y
    x = data.x

    unique, counts = torch.unique(y, return_counts=True)
    orig_num_classes = unique.numel()
    orig_num_samples = y.numel()


    valid_mask = counts >= min_samples
    valid_classes = unique[valid_mask]  # Tensor of classes to keep

    keep_mask = torch.isin(y, valid_classes)
    x_f = x[keep_mask]
    y_f = y[keep_mask]


    label_mapping = {}
    if remap_labels:
        new_classes = torch.unique(y_f)
        label_mapping = {int(old.item()): int(i) for i, old in enumerate(new_classes)}
        y_f = torch.tensor([label_mapping[int(t.item())] for t in y_f], dtype=torch.long)
    else:
        new_classes = torch.unique(y_f)

    filtered_data = Data(x=x_f, y=y_f)

    stats = {
        "orig_num_samples": int(orig_num_samples),
        "orig_num_classes": int(orig_num_classes),
        "kept_num_samples": int(y_f.numel()),
        "kept_num_classes": int(new_classes.numel()),
        "removed_num_classes": int(orig_num_classes - new_classes.numel()),
        "removed_ratio_samples": float(1 - (y_f.numel() / orig_num_samples)),
        "min_samples_threshold": int(min_samples),
        "label_mapping": label_mapping,  #
        "kept_classes_raw": [int(c.item()) for c in new_classes],  #
    }
    return filtered_data, stats

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
import torch
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops
from torch_geometric.data import Data


def build_graph_from_coords(
    data: Data,
    k: int = 1,
    sigma_km: float = 2.0,
    symmetric: bool = True,
    add_loops: bool = True,
    drop_zero_weight: bool = True,
) -> Data:

    device = data.x.device
    coords_deg = data.x[:, -2:].detach().cpu().numpy()  # [N, 2]
    # Keep coordinate embedding and strip coords from node features for all paths,
    # including tiny grids (N<=1), so downstream feature dims stay consistent.
    data.pe = data.x[:, -2:]
    data.x = data.x[:, :-2]

    if coords_deg.ndim != 2 or coords_deg.shape[1] != 2:
        raise ValueError("data.x 的最后两列必须是 [lon, lat] 两个维度。")

    N = coords_deg.shape[0]
    if N <= 1:
        data.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        data.edge_weight = torch.empty((0,), dtype=torch.float32, device=device)
        return data

    coords_rad = np.radians(coords_deg[:, [1, 0]])


    k_eff = min(k, N - 1)

    nbrs = NearestNeighbors(
        n_neighbors=k_eff + 1,
        algorithm="ball_tree",
        metric="haversine",
    )
    nbrs.fit(coords_rad)
    dist_rad, idx = nbrs.kneighbors(coords_rad, return_distance=True)

    dist_rad = dist_rad[:, 1:]
    idx = idx[:, 1:]

    dist_km = dist_rad * 6371.0

    sim = np.exp(-(dist_km ** 2) / (2.0 * sigma_km ** 2))

    if drop_zero_weight:
        eps = 1e-8
        mask = sim > eps
    else:
        mask = np.ones_like(sim, dtype=bool)

    rows = np.repeat(np.arange(N), k_eff)[mask.reshape(-1)]
    cols = idx.reshape(-1)[mask.reshape(-1)]
    vals = sim.reshape(-1)[mask.reshape(-1)]

    A = coo_matrix((vals, (rows, cols)), shape=(N, N))

    if symmetric:
        A = A.maximum(A.transpose())

    edge_index, edge_weight = from_scipy_sparse_matrix(A)

    if add_loops:
        edge_index, edge_weight = add_self_loops(
            edge_index,
            edge_weight,
            fill_value=1.0,
            num_nodes=N,
        )
    data.y = data.y
    data.edge_index = edge_index


    return data
