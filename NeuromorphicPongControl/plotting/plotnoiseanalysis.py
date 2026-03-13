import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})
plt.rcParams['font.weight'] = 'bold'

# Plot the noise analysis results
noiselevels = np.loadtxt("../data/noisetest/noiselevels_ratio.txt")
averagehitrates = np.loadtxt("../data/noisetest/averagehitrates.txt")

plt.figure(figsize=(8, 6))
plt.plot(noiselevels, averagehitrates, marker='o')
plt.xscale("log")
plt.xlabel("Noise Level", fontweight='bold')
plt.ylabel("Average Hit Rate", fontweight='bold')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("../data/imgs/9_noise_analysis_plot.jpeg", format="jpeg", dpi=300)
plt.savefig("../data/imgs/9_noise_analysis_plot.tiff", format="tiff", dpi=300)
plt.savefig("../data/imgs/9_noise_analysis_plot.eps", format="eps", dpi=300)
plt.show()