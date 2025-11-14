import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as ticker

plt.rcParams.update({'font.size': 25})
plt.rcParams['font.weight'] = 'bold'

# === Load data ===
data = np.loadtxt("parameter_error_table.txt", delimiter=" ")
# assuming your file has a header; remove skiprows=1 if no header
onlydecay = data[data[:,-1] != 1]
print(onlydecay)
accuracy = np.round(onlydecay[:, 0], 3)
accuracy += 0.0001 # Ensures the right value sin the right bins
print(len(accuracy))
print(np.sum(accuracy > 0.85))


# Get unique elig values


fig, ax = plt.subplots(1,1, figsize=(8,6))

# === For each elig, plot histogram and smooth line ===


# Histogram (shared bins across groups for fairness)
counts, bin_edges = np.histogram(accuracy, bins=20, range=(0, 1))
plt.hist(accuracy, bins=bin_edges, edgecolor='white')
# Labels and legend
plt.xlabel("Hit rate", weight='bold')
plt.ylabel("Count", weight='bold')
plt.axvline(0.85, color='r', linestyle='dashed', linewidth=2)
plt.text(0,28, "A")
plt.tight_layout()

plt.savefig("PerformanceHistogram.jpeg")
