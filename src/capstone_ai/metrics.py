from __future__ import annotations

import numpy as np
import torch


def classification_calibration(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> dict:
    """Return accuracy, NLL, Brier score, and expected calibration error."""
    probabilities = probabilities.detach().float().cpu()
    labels = labels.detach().long().cpu()
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels).float()
    ece = torch.zeros(())
    for lower in torch.linspace(0, 1, bins + 1)[:-1]:
        upper = lower + 1 / bins
        mask = (confidence >= lower) & ((confidence < upper) if upper < 1 else (confidence <= upper))
        if mask.any():
            ece += mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs()
    nll = -torch.log(probabilities[torch.arange(labels.numel()), labels].clamp_min(1e-12)).mean()
    one_hot = torch.nn.functional.one_hot(labels, num_classes=probabilities.size(1)).float()
    brier = ((probabilities - one_hot) ** 2).sum(dim=1).mean()
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1).mean()
    return {
        "accuracy": float(correct.mean()),
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
        "mean_predictive_entropy": float(entropy),
        "samples": int(labels.numel()),
    }


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    per_target = {}
    names = ("next_day_ETa", "next_day_Ks")
    for index, name in enumerate(names[: target.shape[1]]):
        y, e = target[:, index], error[:, index]
        ss_total = ((y - y.mean()) ** 2).sum()
        per_target[name] = {
            "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e**2).mean())),
            "r2": float(1 - (e**2).sum() / ss_total) if ss_total else None,
        }
    return {"per_target": per_target, "mean_mae": float(np.abs(error).mean()), "mean_rmse": float(np.sqrt((error**2).mean()))}
