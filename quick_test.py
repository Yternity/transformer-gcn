"""Synthetic-data smoke test for the Transformer-GCN code package.

This test verifies that the main model can be imported, constructed, and run
for one forward pass without requiring the manuscript data files.
"""

import torch

from core.mode import ourmodel


def main() -> None:
    torch.manual_seed(42)

    num_nodes = 6
    input_dim = 8
    num_classes = 4

    features = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 3, 5],
            [1, 2, 3, 4, 5, 0, 2, 4, 0, 3, 5, 1],
        ],
        dtype=torch.long,
    )

    model = ourmodel(
        input_dim=input_dim,
        hide=num_classes,
        method="ours",
        layers_count=1,
        A=edge_index,
    )
    model.eval()

    with torch.no_grad():
        output = model(features)

    expected_shape = (num_nodes, num_classes)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"Unexpected output shape {tuple(output.shape)}; expected {expected_shape}.")
    if not torch.isfinite(output).all():
        raise RuntimeError("Model output contains non-finite values.")

    row_sums = output.exp().sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        raise RuntimeError("Log-softmax output rows do not sum to one after exponentiation.")

    print(f"quick_test passed: Transformer-GCN forward pass output shape = {expected_shape}")


if __name__ == "__main__":
    main()
