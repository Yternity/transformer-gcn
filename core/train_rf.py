"""Training workflow for the random-forest baseline model."""

import torch
import numpy as np

try:
    from .config import (
        DATA_DIR,
        DEFAULT_K,
        DEFAULT_MIN_SAMPLES,
        DEFAULT_SEED,
        DEFAULT_SIGMA_KM,
        DEFAULT_SPLIT_SEED,
        DEFAULT_TRAIN_RATIO,
        DEFAULT_VAL_RATIO,
        RESULTS_DIR,
        ensure_runtime_dirs,
    )
    from .load_data import build_graph_from_coords, stratified_ratio_masks, filter_small_classes
    from .per import per_class_accuracy, print_per_class
except ImportError:
    from config import (
        DATA_DIR,
        DEFAULT_K,
        DEFAULT_MIN_SAMPLES,
        DEFAULT_SEED,
        DEFAULT_SIGMA_KM,
        DEFAULT_SPLIT_SEED,
        DEFAULT_TRAIN_RATIO,
        DEFAULT_VAL_RATIO,
        RESULTS_DIR,
        ensure_runtime_dirs,
    )
    from load_data import build_graph_from_coords, stratified_ratio_masks, filter_small_classes
    from per import per_class_accuracy, print_per_class

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


DEVICE = torch.device("cpu")  # RF 用 CPU 即可


DATA_PATH = DATA_DIR / "ours_data.pt"


def train_rf(seed: int = DEFAULT_SEED, n_estimators: int = 300, n_jobs: int = 1):
    """
    训练 Random Forest 作为传统机器学习 baseline。
    与深度模型保持相同的数据过滤和划分方式。
    """
    ensure_runtime_dirs()
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = torch.load(DATA_PATH, weights_only=False)

    # 与 train.py 保持一致的类别过滤与标签重映射
    data, info = filter_small_classes(data, min_samples=DEFAULT_MIN_SAMPLES, remap_labels=True)

    # 与 train.py 保持一致：先按坐标构图，再做分层划分
    data = build_graph_from_coords(data, k=DEFAULT_K, sigma_km=DEFAULT_SIGMA_KM)
    data = stratified_ratio_masks(
        data,
        train_ratio=DEFAULT_TRAIN_RATIO,
        val_ratio=DEFAULT_VAL_RATIO,
        seed=DEFAULT_SPLIT_SEED,
        print_stats=True
    )

    # 与 train.py 保持一致：按列 L2 归一化
    x = torch.nn.functional.normalize(data.x, dim=0, p=2).cpu().numpy()
    y = data.y.cpu().numpy().astype(int)

    train_mask = data.train_mask.cpu().numpy().astype(bool)
    val_mask = data.val_mask.cpu().numpy().astype(bool)
    test_mask = data.test_mask.cpu().numpy().astype(bool)

    print("X shape:", x.shape, " y shape:", y.shape)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=n_jobs,
        random_state=seed,
        class_weight="balanced_subsample",
    )

    print("\n[rf] 训练 RandomForestClassifier ...")
    rf.fit(x[train_mask], y[train_mask])

    # 在三个子集上评估精度
    y_pred_all = rf.predict(x)

    train_acc = accuracy_score(y[train_mask], y_pred_all[train_mask])
    val_acc = accuracy_score(y[val_mask], y_pred_all[val_mask])
    test_acc = accuracy_score(y[test_mask], y_pred_all[test_mask])

    print(f"[rf] Train Acc = {train_acc:.4f}")
    print(f"[rf] Val   Acc = {val_acc:.4f}")
    print(f"[rf] Test  Acc = {test_acc:.4f}")

    # 构造“伪 logits”，以复用 per_class_accuracy 计算按类精度
    num_classes = int(y.max()) + 1
    logits_dummy = torch.zeros((len(y), num_classes), dtype=torch.float32)
    idx_all = torch.arange(len(y))
    logits_dummy[idx_all, torch.from_numpy(y_pred_all)] = 1.0

    y_t = torch.from_numpy(y).long()
    train_mask_t = torch.from_numpy(train_mask)
    val_mask_t = torch.from_numpy(val_mask)
    test_mask_t = torch.from_numpy(test_mask)

    rpt_tr = per_class_accuracy(logits_dummy, y_t, mask=train_mask_t, class_labels=num_classes)
    rpt_val = per_class_accuracy(logits_dummy, y_t, mask=val_mask_t, class_labels=num_classes)
    rpt_te = per_class_accuracy(logits_dummy, y_t, mask=test_mask_t, class_labels=num_classes)

    print_per_class(rpt_te, title="RF Test per-class accuracy")

    results = {
        "seed": seed,
        "model_name": "rf",
        "best_epoch": 0,
        "best_val_acc": float(val_acc),
        "best_test_acc": float(test_acc),
        "history": {
            "epoch": [0],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [float(train_acc)],
            "val_acc": [float(val_acc)],
            "test_acc": [float(test_acc)],
        },
        "num_classes": num_classes,
        "filter_small_classes_info": info,
        "per_class_train": rpt_tr,
        "per_class_val": rpt_val,
        "per_class_test": rpt_te,
        "y_true": y_t,
        "y_pred": torch.from_numpy(y_pred_all),
        "train_mask": train_mask_t,
        "val_mask": val_mask_t,
        "test_mask": test_mask_t,
    }

    results_path = RESULTS_DIR / f"rf_train_results_seed_{seed}.pt"
    torch.save(results, results_path)
    print(f"[rf] 结果已保存到: {results_path}")

    model_path = RESULTS_DIR / f"rf_model_seed_{seed}.pkl"
    joblib.dump(rf, model_path)
    print(f"[rf] 模型已保存到: {model_path}")

    return results


if __name__ == "__main__":
    train_rf(DEFAULT_SEED)
