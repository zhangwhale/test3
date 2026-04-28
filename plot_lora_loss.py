import re
from pathlib import Path
import matplotlib.pyplot as plt

log_path = Path("logs/lora_hust_handbook.log")
save_path = Path("figures/lora_hust_handbook_loss.png")

pattern = re.compile(r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([0-9.]+)")

steps = []
losses = []
epoch_offset = 0
last_epoch = None
last_total = None

with log_path.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pattern.search(line)
        if not m:
            continue

        epoch = int(m.group(1))
        step = int(m.group(3))
        total = int(m.group(4))
        loss = float(m.group(5))

        if last_epoch is None:
            last_epoch = epoch
            last_total = total
        elif epoch != last_epoch:
            epoch_offset += last_total
            last_epoch = epoch
            last_total = total

        steps.append(epoch_offset + step)
        losses.append(loss)

if not steps:
    print("未解析到 loss，请检查 logs/lora_hust_handbook.log")
else:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, linewidth=1.2)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("MiniMind LoRA Fine-tuning Loss on HUST Graduate Handbook")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"已保存：{save_path}")
