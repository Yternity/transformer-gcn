"""Training workflow for Transformer-GCN and neural baseline models."""

import argparse
import copy
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

try:
    from .config import (
        DATA_DIR,
        DEFAULT_EPOCHS,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_K,
        DEFAULT_LAYERS_COUNT,
        DEFAULT_LEARNING_RATE,
        DEFAULT_MIN_SAMPLES,
        DEFAULT_MODEL_NAME,
        DEFAULT_PATIENCE,
        DEFAULT_SEED,
        DEFAULT_SIGMA_KM,
        DEFAULT_SPLIT_SEED,
        DEFAULT_TRAIN_RATIO,
        DEFAULT_VAL_RATIO,
        DEFAULT_WEIGHT_DECAY,
        RESULTS_DIR,
        ensure_runtime_dirs,
    )
    from .eval import test
    from .load_data import build_graph_from_coords, filter_small_classes, stratified_ratio_masks
    from .mode import GCNModel, GraphSAGEModel, MLPModel, ourmodel
    from .per import per_class_accuracy, print_per_class
except ImportError:
    from config import (
        DATA_DIR,
        DEFAULT_EPOCHS,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_K,
        DEFAULT_LAYERS_COUNT,
        DEFAULT_LEARNING_RATE,
        DEFAULT_MIN_SAMPLES,
        DEFAULT_MODEL_NAME,
        DEFAULT_PATIENCE,
        DEFAULT_SEED,
        DEFAULT_SIGMA_KM,
        DEFAULT_SPLIT_SEED,
        DEFAULT_TRAIN_RATIO,
        DEFAULT_VAL_RATIO,
        DEFAULT_WEIGHT_DECAY,
        RESULTS_DIR,
        ensure_runtime_dirs,
    )
    from eval import test
    from load_data import build_graph_from_coords, filter_small_classes, stratified_ratio_masks
    from mode import GCNModel, GraphSAGEModel, MLPModel, ourmodel
    from per import per_class_accuracy, print_per_class


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = DATA_DIR / "ours_data.pt"


def prepare_data(
    min_samples: int = DEFAULT_MIN_SAMPLES,
    k: int = DEFAULT_K,
    sigma_km: float = DEFAULT_SIGMA_KM,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    split_seed: int = DEFAULT_SPLIT_SEED,
):
    data = torch.load(DATA_PATH, weights_only=False)
    data, info = filter_small_classes(data, min_samples=min_samples, remap_labels=True)
    data = build_graph_from_coords(data, k=k, sigma_km=sigma_km)
    data = stratified_ratio_masks(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=split_seed,
        print_stats=True,
    )

    tensor_x = torch.nn.functional.normalize(data.x.to(DEVICE), dim=0, p=2)
    tensor_y = data.y.long().to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)
    tensor_train_mask = data.train_mask.to(DEVICE)
    tensor_val_mask = data.val_mask.to(DEVICE)
    tensor_test_mask = data.test_mask.to(DEVICE)
    num_classes = int(data.y.max().item()) + 1

    print(data)
    print(int(tensor_train_mask.sum().item()))
    print(int(tensor_val_mask.sum().item()))
    print(int(tensor_test_mask.sum().item()))
    print("============================================================================")

    return {
        "data": data,
        "info": info,
        "x": tensor_x,
        "y": tensor_y,
        "edge_index": edge_index,
        "train_mask": tensor_train_mask,
        "val_mask": tensor_val_mask,
        "test_mask": tensor_test_mask,
        "num_classes": num_classes,
    }


def build_model(
    model_name: str,
    in_dim: int,
    hidden_dim: int,
    num_classes: int,
    layers_count: int,
    edge_index: torch.Tensor,
):
    name = model_name.lower()
    if name == "ours":
        model = ourmodel(in_dim, hidden_dim, "Trans", layers_count, edge_index)
    elif name == "mlp":
        model = MLPModel(in_dim, hidden_dim, num_classes)
    elif name == "gcn":
        model = GCNModel(in_dim, hidden_dim, num_classes, edge_index)
    elif name in ("sage", "graphsage"):
        model = GraphSAGEModel(in_dim, hidden_dim, num_classes, edge_index)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model.to(DEVICE)


def train(
    iteration_seed: int = DEFAULT_SEED,
    model_name: str = DEFAULT_MODEL_NAME,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    layers_count: int = DEFAULT_LAYERS_COUNT,
):
    ensure_runtime_dirs()

    torch.manual_seed(iteration_seed)
    np.random.seed(iteration_seed)
    random.seed(iteration_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    bundle = prepare_data()
    tensor_x = bundle["x"]
    tensor_y = bundle["y"]
    edge_index = bundle["edge_index"]
    tensor_train_mask = bundle["train_mask"]
    tensor_val_mask = bundle["val_mask"]
    tensor_test_mask = bundle["test_mask"]
    num_classes = bundle["num_classes"]
    info = bundle["info"]

    model = build_model(
        model_name=model_name,
        in_dim=tensor_x.shape[1],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        layers_count=layers_count,
        edge_index=edge_index,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    cross_loss = nn.CrossEntropyLoss()
    epoch_test_acc_history = [[] for _ in range(epochs)]
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "test_acc": []}

    best_val_acc = -1.0
    best_test_acc = 0.0
    best_epoch = -1
    best_state_dict = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        logits = model(tensor_x)
        loss = cross_loss(logits[tensor_train_mask], tensor_y[tensor_train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, _ = test(model, tensor_train_mask, tensor_x, tensor_y)
        val_acc, val_loss = test(model, tensor_val_mask, tensor_x, tensor_y)
        test_acc, _ = test(model, tensor_test_mask, tensor_x, tensor_y)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_loss.item()))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["test_acc"].append(float(test_acc))
        epoch_test_acc_history[epoch].append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[{model_name}] Epoch {epoch}: "
            f"Loss {loss.item():.4f}, Train Acc {train_acc:.4f}, "
            f"Val Loss {val_loss.item():.4f}, Val Acc {val_acc:.4f}, Test Acc {test_acc:.4f}"
        )

        if patience_counter >= patience:
            print(
                f"[{model_name}] Early stopping at epoch {epoch}, best epoch={best_epoch}, "
                f"best val acc={best_val_acc:.4f}, best test acc={best_test_acc:.4f}"
            )
            break

    per_class_train = None
    per_class_val = None
    per_class_test = None
    y_pred_all_cpu = None

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            f"[{model_name}] Loaded best model from epoch {best_epoch}, "
            f"best val acc={best_val_acc:.4f}, corresponding test acc={best_test_acc:.4f}"
        )
        with torch.no_grad():
            logits_best = model(tensor_x)
            y_pred_all_cpu = logits_best.argmax(dim=1).cpu()
            per_class_train = per_class_accuracy(logits_best, tensor_y, mask=tensor_train_mask, class_labels=num_classes)
            per_class_val = per_class_accuracy(logits_best, tensor_y, mask=tensor_val_mask, class_labels=num_classes)
            per_class_test = per_class_accuracy(logits_best, tensor_y, mask=tensor_test_mask, class_labels=num_classes)
        print_per_class(per_class_test, title=f"Test (best epoch, {model_name})")

    model_tag = model_name.lower()
    if model_tag == "ours":
        results_fname = f"train_results_seed_{iteration_seed}.pt"
        best_model_fname = f"best_model_seed_{iteration_seed}.pt"
    else:
        results_fname = f"{model_tag}_train_results_seed_{iteration_seed}.pt"
        best_model_fname = f"{model_tag}_best_model_seed_{iteration_seed}.pt"

    results_to_save = {
        "seed": iteration_seed,
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "best_test_acc": float(best_test_acc),
        "history": history,
        "num_classes": num_classes,
        "filter_small_classes_info": info,
    }
    if per_class_train is not None:
        results_to_save["per_class_train"] = per_class_train
        results_to_save["per_class_val"] = per_class_val
        results_to_save["per_class_test"] = per_class_test
        results_to_save["y_true"] = tensor_y.cpu()
        results_to_save["y_pred"] = y_pred_all_cpu
        results_to_save["train_mask"] = tensor_train_mask.cpu()
        results_to_save["val_mask"] = tensor_val_mask.cpu()
        results_to_save["test_mask"] = tensor_test_mask.cpu()

    results_path = RESULTS_DIR / results_fname
    torch.save(results_to_save, results_path)
    print(f"[{model_name}] 训练过程结果已保存到: {results_path}")

    if best_state_dict is not None:
        best_model_path = RESULTS_DIR / best_model_fname
        torch.save(best_state_dict, best_model_path)
        print(f"[{model_name}] 最佳模型参数已保存到: {best_model_path}")

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return epoch_test_acc_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train model in core pipeline.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, choices=["ours", "mlp", "gcn", "sage"])
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--layers-count", type=int, default=DEFAULT_LAYERS_COUNT)
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        iteration_seed=args.seed,
        model_name=args.model_name,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        layers_count=args.layers_count,
    )


if __name__ == "__main__":
    main()
