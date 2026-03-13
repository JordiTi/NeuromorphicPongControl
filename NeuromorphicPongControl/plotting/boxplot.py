import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})
plt.rcParams['font.weight'] = 'bold'

# --------------------------------------------------
# Load data
# --------------------------------------------------
datapath = "../data/plottingdata/SIF-EFA.txt"
outputpath = "6_AccuracyBoxplot_SIF-EFA"
# For LFA and DFA
# varnames = ["Acc", r"$\alpha$", r"$V_{thr}$", "A", "#Sns", "R", "S"]
# For EFA
varnames = ["Acc", r"$\alpha$", r"$V_{thr}$", "A", "#Sns", "R", "S", "e"]
parametermat = np.loadtxt(datapath)

df = pd.DataFrame(parametermat, columns=varnames)

# --------------------------------------------------
# Sort learning rates nicely
# --------------------------------------------------
alpha_col = r"$\alpha$"
df = df.sort_values(alpha_col)

# --------------------------------------------------
# Plot box plot
# --------------------------------------------------
plt.figure(figsize=(10,6))

sns.boxplot(
    data=df,
    x=alpha_col,
    y="Acc",
    showfliers=False,
    width=0.6
)

# Show individual points
sns.stripplot(
    data=df,
    x=alpha_col,
    y="Acc",
    color="black",
    alpha=0.8,
    jitter=0.15
)

# Draw good-accuracy threshold
plt.axhline(0.85, linestyle="--", linewidth=2, color="red")
# plt.text(
#     len(df[alpha_col].unique()) - 0.5,
#     0.855,
#     "Good hit rate = 0.85",
#     ha="right",
#     fontsize=20
# )
# plt.text(0,1.05, "B")
# Labels
plt.xlabel(r"Learning rate $\alpha$", fontsize=25, fontweight='bold')
plt.ylabel("Hit rate", fontsize=25, fontweight='bold')
plt.tight_layout()
plt.savefig(f"../data/imgs/{outputpath}.jpeg")
plt.savefig(f"../data/imgs/{outputpath}.tiff", format="tiff", dpi=300)
plt.savefig(f"../data/imgs/{outputpath}.eps", format="eps", dpi=300)
