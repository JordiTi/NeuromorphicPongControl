'''Visualize the hit rates per parameter combination using a heatmap and circles that indicate the fraction of good trials.
This is only possible using single runs. Adjust the "varnames" variable as needed.'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 25})
plt.rcParams['font.weight'] = 'bold'

# Declare variable names for the axis labels
varnames = ["Acc", r"$\alpha$", r"$V_{thr}$", "A", "#Sns", "R", "S", "L", "e"]

# Load data
parametermat = np.loadtxt("../data/traindata/parameter_error_table.txt")
print("Loaded parameter matrix")

# Combine variable names and data
df = pd.DataFrame(parametermat, columns=varnames)
params = [col for col in df.columns if col != "Acc"]

# Build multi-index for parameter combinations
totalcombis = 1
uniqueparametervalues = []
row_labels = []
for p in params:
    totalcombis *= len(df[p].unique())
    uniqueparametervalues.append(len(df[p].unique()))
    for v in sorted(df[p].unique()):
        row_labels.append((p, v))


# Create matrices for mean accuracy and good fraction
heatmap_data = pd.DataFrame(index=pd.MultiIndex.from_tuples(row_labels),
                            columns=pd.MultiIndex.from_tuples(row_labels))
good_fraction = pd.DataFrame(index=pd.MultiIndex.from_tuples(row_labels),
                             columns=pd.MultiIndex.from_tuples(row_labels))

print("Generated heatmap data")

# Fill matrices. Threshold accuracy values with 0.85 to distinguish  between good and bad hit rates.
for idx_1, (i_p, i_v) in enumerate(row_labels):
    for idx_2, (j_p, j_v) in enumerate(row_labels):
        mask = (df[i_p] == i_v) & (df[j_p] == j_v)
        if mask.any():
            acc_values = df.loc[mask, "Acc"]
            heatmap_data.loc[(i_p, i_v), (j_p, j_v)] = acc_values.mean()

            if i_p == j_p and i_v == j_v:
                good_fraction.loc[(i_p, i_v), (j_p, j_v)] = (acc_values >= 0.85).sum()/(totalcombis/(uniqueparametervalues[varnames.index(i_p)-1]))
            else:
                good_fraction.loc[(i_p, i_v), (j_p, j_v)] = (acc_values >= 0.85).sum()/(totalcombis/(uniqueparametervalues[varnames.index(i_p)-1]*uniqueparametervalues[varnames.index(j_p)-1]))

print("Calculated mean accuracies and good fractions")

# Clean up code
heatmap_data = heatmap_data.astype(float)
good_fraction = good_fraction.astype(float)

heatmap_data = heatmap_data.dropna(how="all", axis=0).dropna(how="all", axis=1)
good_fraction = good_fraction.loc[heatmap_data.index, heatmap_data.columns]

heatmap_data.index = [f"{p}={v}" for p, v in heatmap_data.index]
heatmap_data.columns = [f"{p}={v}" for p, v in heatmap_data.columns]
good_fraction.index = heatmap_data.index
good_fraction.columns = heatmap_data.columns

avg_values = heatmap_data.values
good_values = good_fraction.values

# Plot heatmap
fig, ax = plt.subplots(figsize=(14, 10))

# Create colormap 
cmap = plt.get_cmap("viridis")
rgba_img = cmap(avg_values)

# Make all same-parameter blocks white, as there is no interaction between the two
row_params = [s.split('=')[0] for s in heatmap_data.index]
col_params = [s.split('=')[0] for s in heatmap_data.columns]
unique_params = sorted(set(row_params))

for p in unique_params:
    row_idx = [i for i, rp in enumerate(row_params) if rp == p]
    col_idx = [j for j, cp in enumerate(col_params) if cp == p]
    for i in row_idx:
        for j in col_idx:
            if i != j:
                rgba_img[i, j] = [1.0, 1.0, 1.0, 1.0]  # white, opaque

# Display base heatmap
im = ax.imshow(rgba_img, interpolation="none", aspect="auto")

# Marker overlay (shows fraction of good runs)
# Marker size scaled by good fraction
marker_sizes = 1000 * np.nan_to_num(good_values, nan=0.0)

for i in range(avg_values.shape[0]):
    for j in range(avg_values.shape[1]):
        frac = good_values[i, j]
        if np.isnan(frac):  # skip empty or white cells
            continue
        if frac > 0:  # draw only where there is at least one good run
            ax.scatter(j, i, s=marker_sizes[i, j],
                       c='black', alpha=0.7, edgecolors='white', linewidths=1.5)
            

# add colorbar for reference
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
cbar = plt.colorbar(sm, ax=ax)
cbar.ax.set_ylabel("Accuracy", fontweight="bold")

# Axis labels and ticks
ax.set_xticks(np.arange(len(heatmap_data.columns)))
ax.set_yticks(np.arange(len(heatmap_data.index)))
ax.set_xticklabels(heatmap_data.columns, rotation=90)
ax.set_yticklabels(heatmap_data.index)

# plt.text(0,-1, "B")
plt.tight_layout()

plt.savefig("../data/imgs/ParameterCombiMatrix_vanilla_circles.jpeg")
