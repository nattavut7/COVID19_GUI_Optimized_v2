from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from src.common import (
    device_info, ensure_dir, load_config, save_json, set_seed
)
from src.data import build_loaders
from src.engine import benchmark_latency, evaluate, train_one_epoch
from src.models import build_model
from src.reporting import plot_history, save_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xray_3class.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. Run scripts\\check_gpu.py first."
        )

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = bool(tr.get("cudnn_benchmark", True)) if "tr" in locals() else True

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ensure_dir(Path(cfg.get("output_root", "outputs")) / f"{cfg.get("experiment_name", "experiment")}_{stamp}")
    ckpt_dir = ensure_dir(out / "checkpoints")
    report_dir = ensure_dir(out / "reports")
    writer = SummaryWriter(out / "tensorboard")

    train_loader, val_loader, test_loader, class_names = build_loaders(cfg)
    save_json(class_names, out / "class_names.json")
    save_json(device_info(), out / "system_info.json")

    tr = cfg["training"]
    torch.backends.cudnn.benchmark = bool(tr.get("cudnn_benchmark", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(tr.get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(tr.get("allow_tf32", True))
    model = build_model(
        tr["model"], len(class_names), bool(tr.get("pretrained", True))
    ).to(device)

    channels_last = bool(tr.get("channels_last", True))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(tr.get("label_smoothing", 0.0))
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(tr["learning_rate"]),
        weight_decay=float(tr["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, int(tr["epochs"]))
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(tr.get("mixed_precision", True))
    )

    start_epoch = 1
    best_acc = -1.0
    best_state = None
    patience_count = 0
    history = []

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_acc = float(state.get("best_acc", -1.0))
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, int(tr["epochs"]) + 1):
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            bool(tr.get("mixed_precision", True)), channels_last
        )
        val_result = evaluate(
            model, val_loader, criterion, device,
            bool(tr.get("mixed_precision", True)), channels_last
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "val_precision": val_result["precision"],
            "val_recall": val_result["recall"],
            "val_f1": val_result["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        writer.add_scalar("Loss/train", row["train_loss"], epoch)
        writer.add_scalar("Loss/val", row["val_loss"], epoch)
        writer.add_scalar("Accuracy/train", row["train_accuracy"], epoch)
        writer.add_scalar("Accuracy/val", row["val_accuracy"], epoch)

        print(
            f"Epoch {epoch:03d}/{tr['epochs']} | "
            f"train_acc={row['train_accuracy']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_f1={row['val_f1']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "class_names": class_names,
            "config": cfg,
        }
        torch.save(checkpoint, ckpt_dir / "last.pt")

        if row["val_accuracy"] > best_acc:
            best_acc = row["val_accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model": best_state,
                    "class_names": class_names,
                    "config": cfg,
                    "best_acc": best_acc,
                },
                ckpt_dir / "best.pt",
            )
            patience_count = 0
        else:
            patience_count += 1

        if bool(tr.get("early_stopping", False)) and patience_count >= int(tr.get("patience", 15)):
            print("Early stopping.")
            break

    writer.close()
    plot_history(history, out / "training_history.csv")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_result = evaluate(
        model, test_loader, criterion, device,
        bool(tr.get("mixed_precision", True)), channels_last
    )
    save_evaluation(test_result, class_names, report_dir, "original")

    latency = benchmark_latency(model, test_loader, device, channels_last)
    result_row = {
        "model": "original",
        "accuracy": test_result["accuracy"],
        "precision": test_result["precision"],
        "recall": test_result["recall"],
        "f1": test_result["f1"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "latency_ms": latency,
        "experiment_dir": str(out),
    }
    pd.DataFrame([result_row]).to_csv(
        out / "original_results.csv", index=False
    )

    torch.save(
        {
            "model": model.state_dict(),
            "class_names": class_names,
            "config": cfg,
        },
        out / "original_model.pt",
    )

    print("\nTraining completed.")
    print(pd.DataFrame([result_row]).to_string(index=False))
    print("\nExperiment directory:", out)


if __name__ == "__main__":
    main()
