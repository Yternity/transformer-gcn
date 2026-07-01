"""Evaluation helpers for classification experiments."""

import torch
import torch.nn as nn

@torch.no_grad()
def test(model,mask,tensor_x,tensor_y):
    model.eval()
    with torch.no_grad():
        out = model(tensor_x)
        loss = nn.CrossEntropyLoss()(out[mask], tensor_y[mask])
        pred = out.argmax(dim=1)
        correct = pred[mask].eq(tensor_y[mask]).sum().item()
        acc = correct / mask.sum().item()
    return acc,loss
