"""Per-class accuracy utilities for imbalanced classification outputs."""

import torch
import math

import math
import torch

def per_class_accuracy(logits: torch.Tensor,
                       y: torch.Tensor,
                       mask: torch.Tensor | None = None,
                       class_labels=None):
    """
    统计每个类别的准确率。
    - logits: [N, C]
    - y:      [N]
    - mask:   [N] bool，可选
    - class_labels:
        * None  -> 仅统计 y 中实际出现的类
        * int   -> 视为类别数 C，统计 range(C)
        * list/tuple/torch.Tensor -> 迭代这些类
    返回: dict {cls: {"acc": float或nan, "correct":int, "total":int}}
    """
    if mask is not None:
        idx = mask.nonzero(as_tuple=False).view(-1)
        logits = logits.index_select(0, idx)
        y = y.index_select(0, idx)

    preds = logits.argmax(dim=1)

    # 统一 class_labels
    if class_labels is None:
        classes = torch.unique(y).tolist()
    elif isinstance(class_labels, int):
        classes = list(range(class_labels))
    elif isinstance(class_labels, torch.Tensor):
        classes = class_labels.view(-1).tolist()
    else:
        classes = list(class_labels)

    report = {}
    for c in classes:
        c = int(c)
        cls_mask = (y == c)
        total = int(cls_mask.sum().item())
        if total == 0:
            report[c] = {"acc": float("nan"), "correct": 0, "total": 0}
        else:
            correct = int((preds[cls_mask] == y[cls_mask]).sum().item())
            report[c] = {"acc": correct / total, "correct": correct, "total": total}
    return report

def print_per_class(report: dict, title: str = ""):
    if title:
        print(f"\n{title} per-class accuracy:")
    else:
        print("\nPer-class accuracy:")
    for c in sorted(report.keys()):
        r = report[c]
        acc = r["acc"]
        if isinstance(acc, float) and math.isnan(acc):
            print(f"  class {c}: n=0  acc=—")
        else:
            print(f"  class {c}: n={r['total']:>4}  acc={acc:.4f} ({r['correct']}/{r['total']})")
