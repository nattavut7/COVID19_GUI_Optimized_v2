# Q1 COVID-19 PyTorch GUI v2 — RTX 5070 Ti Optimized

ปรับสำหรับ:
- RTX 5070 Ti 16 GB
- Intel Core Ultra 9 285K
- RAM DDR5 32 GB
- WD Black SN850X NVMe

จุดปรับหลัก:
- Batch size 64
- DataLoader workers 8, prefetch 4, pin memory, persistent workers
- Mixed Precision, channels-last, cuDNN benchmark และ TF32
- ปิด Early Stopping เป็นค่าเริ่มต้น
- ค่า Pruning เริ่มต้น 50% แทน 98%
- GUI Prediction ใหม่: รูป Preview ใหญ่, ชื่อผลทำนายตัวใหญ่, สีตามคลาส, Confidence progress bar, Top-3 scores และ Latency

ติดตั้ง:
```bat
install_windows_conda.bat
```
เปิด GUI:
```bat
conda activate q1covid-torch-v2
python research_gui.py
```

ใช้ Dataset เดิมโดยคัดลอก `xray_3class` มาไว้ที่ `data\prepared\xray_3class` หรือเลือกผ่าน GUI
