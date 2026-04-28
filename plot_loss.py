import re
from pathlib import Path
import matplotlib.pyplot as plt

def parse_loss(log_path):
    steps = []
    losses = []
    pattern = re.compile(r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([0-9.]+)")

    epoch_offset = 0
    last_epoch = None
    last_total_steps = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            epoch = int(m.group(1))
            step = int(m.group(3))
            total_steps = int(m.group(4))
            loss = float(m.group(5))

            if last_epoch is None:
                last_epoch = epoch
                last_total_steps = total_steps
            elif epoch != last_epoch:
                epoch_offset += last_total_steps
                last_epoch = epoch
                last_total_steps = total_steps

            steps.append(epoch_offset + step)
            losses.append(loss)

    return steps, losses

def plot_loss(log_path, save_path, title):
    steps, losses = parse_loss(log_path)

    if not steps:
        print(f"未能从 {log_path} 中解析到 loss，请检查日志内容。")
        return

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, linewidth=1.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"已保存：{save_path}")

if __name__ == "__main__":
    plot_loss(
        "logs/full_sft.log",
        "figures/full_sft_loss.png",
        "MiniMind Full SFT Loss Curve"
    )
