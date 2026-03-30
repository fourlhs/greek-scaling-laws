import numpy as np
import psutil
import os
import matplotlib.pyplot as plt

cache_path = "dataset_cache.npy"
process = psutil.Process(os.getpid())

def rss_mb():
    return process.memory_info().rss / 1024 ** 2

# Without mmap
baseline = rss_mb()
data = np.load(cache_path)
peak_full_mb = rss_mb() - baseline
del data

# With mmap
baseline = rss_mb()
data = np.load(cache_path, mmap_mode='r')
peak_mmap_mb = rss_mb() - baseline
del data

print(f"Full load peak: {peak_full_mb:.1f} MB")
print(f"mmap peak:      {peak_mmap_mb:.1f} MB")

plt.bar(["Full load", "mmap_mode='r'"], [peak_full_mb, peak_mmap_mb], color=["tomato", "steelblue"])
plt.ylabel("Peak memory allocated (MB)")
plt.title("np.load memory usage: full vs mmap")
for i, v in enumerate([peak_full_mb, peak_mmap_mb]):
    plt.text(i, v + 0.5, f"{v:.1f} MB", ha="center")
plt.tight_layout()
plt.savefig("memory_comparison.png", dpi=150)
plt.show()
print("Saved to memory_comparison.png")
