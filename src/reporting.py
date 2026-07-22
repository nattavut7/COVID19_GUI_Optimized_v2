from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)


def save_evaluation(result, class_names, output_dir, prefix):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        result["y_true"],
        result["y_pred"],
        target_names=class_names,
        digits=5,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(
        out / f"{prefix}_classification_report.csv"
    )

    cm = confusion_matrix(result["y_true"], result["y_pred"])
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    ).plot(xticks_rotation=30, values_format="d")

    plt.tight_layout()
    plt.savefig(out / f"{prefix}_confusion_matrix.png", dpi=300)
    plt.close()


def plot_history(history, output_file):
    df = pd.DataFrame(history)
    df.to_csv(Path(output_file).with_suffix(".csv"), index=False)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="Train loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_file).with_name("loss_curve.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_accuracy"], label="Train accuracy")
    plt.plot(df["epoch"], df["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_file).with_name("accuracy_curve.png"), dpi=300)
    plt.close()
