from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.optim import AdamW

from src.common import ensure_dir, load_config
from src.data import build_loaders
from src.engine import benchmark_latency, evaluate, train_one_epoch
from src.models import build_model
from src.reporting import save_evaluation


def global_prune(model, amount: float):
    parameters = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data = module.weight.data.contiguous()
            parameters.append((module, "weight"))

    prune.global_unstructured(
        parameters,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return parameters


def remove_pruning(parameters):
    for module, name in parameters:
        prune.remove(module, name)


def sparsity(model):
    zeros = 0
    total = 0
    for p in model.parameters():
        zeros += int(torch.sum(p == 0).item())
        total += p.numel()
    return zeros / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xray_3class.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected.")

    device = torch.device("cuda")
    out = ensure_dir(args.output or cfg.get("output_dir", "outputs/pruning_result"))
    report_dir = ensure_dir(out / "reports")

    train_loader, val_loader, test_loader, class_names = build_loaders(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")

    model = build_model(
        cfg["training"]["model"], len(class_names), pretrained=False
    )
    model.load_state_dict(state["model"])
    model.to(device)
    model = model.to(memory_format=torch.contiguous_format)

    channels_last = bool(cfg["training"].get("channels_last", True))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    p_cfg = cfg["pruning"]
    parameters = global_prune(model, float(p_cfg["amount"]))
    print(f"Applied pruning amount: {p_cfg['amount']}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=float(p_cfg["learning_rate"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, int(p_cfg["fine_tune_epochs"]) + 1):
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            True, channels_last
        )
        val_result = evaluate(
            model, val_loader, criterion, device, True, channels_last
        )
        print(
            f"Pruning epoch {epoch:02d} | "
            f"train_acc={train_result['accuracy']:.4f} | "
            f"val_acc={val_result['accuracy']:.4f}"
        )
        if val_result["accuracy"] > best_acc:
            best_acc = val_result["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    remove_pruning(parameters)
    current_sparsity = sparsity(model)
    pruned_path = out / "pruned_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "class_names": class_names,
            "config": cfg,
            "sparsity": current_sparsity,
        },
        pruned_path,
    )

    result = evaluate(
        model, test_loader, criterion, device, True, channels_last
    )
    save_evaluation(result, class_names, report_dir, "pruned")

    latency = benchmark_latency(model, test_loader, device, channels_last)
    rows = [{
        "model": "pruned",
        "accuracy": result["accuracy"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"],
        "sparsity": current_sparsity,
        "parameters": sum(p.numel() for p in model.parameters()),
        "latency_ms": latency,
        "model_size_mb": pruned_path.stat().st_size / (1024 ** 2),
    }]

    size = int(cfg["dataset"]["image_size"])
    dummy = torch.randn(1, 3, size, size, device=device)

    if bool(cfg["export"].get("onnx", True)):
        onnx_path = out / "pruned_model.onnx"
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["image"],
            output_names=["probabilities"],
            dynamic_axes={
                "image": {0: "batch"},
                "probabilities": {0: "batch"},
            },
            opset_version=18,
        )
        print("ONNX exported:", onnx_path)

    if bool(cfg["export"].get("dynamic_quantization", True)):
        cpu_model = copy.deepcopy(model).cpu().eval()
        quantized = torch.ao.quantization.quantize_dynamic(
            cpu_model, {nn.Linear}, dtype=torch.qint8
        )
        quant_path = out / "dynamic_int8_model.pt"
        torch.save(
            {
                "model": quantized.state_dict(),
                "class_names": class_names,
                "config": cfg,
            },
            quant_path,
        )
        rows.append({
            "model": "dynamic_int8_cpu",
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "sparsity": None,
            "parameters": sum(p.numel() for p in quantized.parameters()),
            "latency_ms": None,
            "model_size_mb": quant_path.stat().st_size / (1024 ** 2),
        })

    pd.DataFrame(rows).to_csv(
        out / "pruning_export_results.csv", index=False
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
