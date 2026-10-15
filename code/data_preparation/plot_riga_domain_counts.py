"""Generate the audited RIGA+ labelled-domain composition figure."""
from pathlib import Path

import matplotlib.pyplot as plt

domains = ["BinRushed", "Magrabia", "MESSIDOR\nBase 1", "MESSIDOR\nBase 2", "MESSIDOR\nBase 3"]
counts = [195, 95, 173, 148, 133]

fig, ax = plt.subplots(figsize=(7.2, 3.8))
bars = ax.bar(domains, counts, color=["#3769a3", "#52967d", "#7961a8", "#c47b3a", "#5685a8"], width=0.7)
ax.set_ylabel("Labelled images")
ax.set_ylim(0, 220)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#d9dee5", linewidth=0.7, alpha=0.8)
ax.set_axisbelow(True)
for bar, value in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, value + 4, str(value), ha="center", va="bottom", fontsize=9)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "supplementary" / "figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "figS1_riga_domain_counts.pdf", bbox_inches="tight")
fig.savefig(out / "figS1_riga_domain_counts.png", dpi=300, bbox_inches="tight")

