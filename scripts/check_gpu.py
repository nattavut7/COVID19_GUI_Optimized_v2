import torch
print("PyTorch version :", torch.__version__)
print("CUDA build      :", torch.version.cuda)
print("CUDA available  :", torch.cuda.is_available())
print("GPU count       :", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU was not detected.")
print("GPU name        :", torch.cuda.get_device_name(0))
print("Capability      :", torch.cuda.get_device_capability(0))
props = torch.cuda.get_device_properties(0)
print("VRAM            :", round(props.total_memory / 1024**3, 2), "GB")
x = torch.randn(4096, 4096, device="cuda")
y = x @ x
print("CUDA tensor test:", y.shape, y.device)
