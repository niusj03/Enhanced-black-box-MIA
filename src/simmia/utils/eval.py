import numpy as np
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import os

__all__ = [
    "evaluate",
]


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def evaluate(prediction, answers, output_dir, result_name=None):
    fpr, tpr, auc, acc = sweep(np.array(prediction), np.array(answers, dtype=bool))
    low1 = tpr[np.where(fpr < 0.01)[0][-1]]
    low5 = tpr[np.where(fpr < 0.05)[0][-1]]
    low10 = tpr[np.where(fpr < 0.10)[0][-1]]

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    if result_name is None:
        img_name = "roc_tpr_at_5_fpr.png"
    else:
        img_name = f"{result_name}_roc_tpr_at_5_fpr.png"
    img_path = os.path.join(output_dir, img_name)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    title_part = output_dir.split("/")[1]

    # find exact point used for low5
    idx_5 = np.where(fpr < 0.05)[0][-1]
    fpr_5 = fpr[idx_5]
    tpr_5 = tpr[idx_5]

    ax.scatter([fpr_5], [tpr_5], marker="o", label=f"TPR@5%FPR = {tpr_5:.3f}")
    ax.axvline(0.05, linestyle="--", alpha=0.5)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title_part} — ROC (TPR@5%FPR)")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(img_path, dpi=200)
    plt.close(fig)
    return auc, acc, low1, low5, low10
