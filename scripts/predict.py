from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import torch
from PIL import Image
from torchvision import transforms
from src.models import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu")
    cfg, names = state["config"], state["class_names"]
    model = build_model(cfg["training"]["model"], len(names), pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    size = int(cfg["dataset"]["image_size"])
    tfm = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    image = Image.open(args.image).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)
    if device.type == "cuda": torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(): prob = torch.softmax(model(x), dim=1)[0]
    if device.type == "cuda": torch.cuda.synchronize()
    latency = (time.perf_counter()-start)*1000
    values, indices = prob.sort(descending=True)
    result = {
        "prediction": names[int(indices[0])],
        "confidence": float(values[0]),
        "latency_ms": latency,
        "device": str(device),
        "scores": [{"class": names[int(i)], "probability": float(v)} for v,i in zip(values,indices)]
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
if __name__ == "__main__": main()
