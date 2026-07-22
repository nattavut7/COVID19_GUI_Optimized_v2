from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


def _move(x, y, device, channels_last=False):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    return x, y


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device,
    use_amp: bool, channels_last: bool
):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = _move(x, y, device, channels_last)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.detach().cpu().numpy())
        y_pred.extend(logits.argmax(1).detach().cpu().numpy())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
    }


@torch.inference_mode()
def evaluate(model, loader, criterion, device, use_amp=True, channels_last=False):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for x, y in tqdm(loader, desc="Evaluate", leave=False):
        x, y = _move(x, y, device, channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = model(x)
            loss = criterion(logits, y)

        prob = torch.softmax(logits.float(), dim=1)
        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())
        y_prob.extend(prob.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_prob": np.asarray(y_prob),
    }


@torch.inference_mode()
def benchmark_latency(model, loader, device, channels_last=False, runs=100):
    model.eval()
    samples = []
    for x, _ in loader:
        for item in x:
            samples.append(item.unsqueeze(0))
            if len(samples) >= runs + 10:
                break
        if len(samples) >= runs + 10:
            break

    times = []
    for i, x in enumerate(samples):
        x = x.to(device)
        if channels_last and device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if i >= 10:
            times.append(elapsed)

    return float(np.mean(times)) if times else None
