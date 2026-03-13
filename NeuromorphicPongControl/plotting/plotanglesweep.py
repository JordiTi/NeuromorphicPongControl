import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../utils/")
import tools
import re
import numpy as np
import random

plt.rcParams.update({'font.size': 25})
plt.rcParams['font.weight'] = 'bold'
parametermat = []
errormat = []

datapath="anglesweep_995"
imgname = "7_anglesweep995"
resultfilenames = os.listdir(f"../data/errorfiles/{datapath}/data")
plt.figure(figsize=(10, 8))
for f in resultfilenames:

    if f.startswith("trial"):
        parts = f.strip(".txt").split("_")

        for i, p in enumerate(parts):
            p = re.sub(r"[^\d.e\-]", "", p)
            p = re.sub(r"\Ae+", "", p)
            if i == 0:
                trial = int(p)
            elif i == 1:
                angle = float(p)

        parametervalues = [angle]
        parametermat.append(parametervalues)
        with open(f"../data/errorfiles/{datapath}/data/{f}" , "r") as errorfile:
            data = errorfile.readlines()
            coll = []
            for d in data:
                dist = d.split(",")[0]
                hit = d.split(",")[1]
                d = int(hit)
                coll.append(d)

        ma = tools.moving_average(coll, 10000)
        errormat.append(ma)

parametermat = np.asarray(parametermat)


errormat = np.asarray(errormat)

parametermat_errormat = np.insert(parametermat, 0, errormat[:, -1], axis=1)
sorted_by_col = parametermat_errormat[parametermat_errormat[:, 0].argsort()]
# np.savetxt("bestninefive.txt",np.array(sorted_by_col))

uniqueangle = np.unique(parametermat[:, 0])



bestfinish = 0
bestparams = []
# Plot lr depdencency
plt.figure(1)
lastvals = []
sterrs = []
avgs = []
xcoord = np.linspace(0, len(errormat[0]), num=len(errormat[0]))
# Create 11 distinct colors for the lines
colors = [(0.1215686, 0.4666667, 0.7058824),
(1.0, 0.4980392, 0.0549020),
(0.1725490, 0.6274510, 0.1725490),
(0.8392157, 0.1529412, 0.1568627),
(0.5803922, 0.4039216, 0.7411765),
(0.5490196, 0.3372549, 0.2941176),
(0.8901961, 0.4666667, 0.7607843),
(0.4980392, 0.4980392, 0.4980392),
(0.7372549, 0.7411765, 0.1333333),
(0.0901961, 0.7450980, 0.8117647),
(0.0, 0.0, 0.0)]
idx = 0
for ua in uniqueangle:
    

    indices = np.where(np.all(parametermat == [ ua], axis=1))
    print(indices)
    data = []
    for i in indices:
        data.append(errormat[i, :])
    data = np.asarray(data)[0]
    print(data)

    avg = np.average(data, axis=0)
    avgs.append(np.round(avg[-1], 3))
    lastvals.append(avg[-1])
    sterr = np.std(data, axis=0)/np.sqrt(4)
    sterrs.append(np.round(sterr[-1], 3))


    plt.plot(avg, label=r"$\phi_{max}=$"+f"{ua}", color=colors[idx])
    plt.fill_between(xcoord, avg+sterr, avg-sterr, alpha=0.5, color=colors[idx])
    idx += 1
plt.legend(prop={'size':20})
plt.ylim([0.4, 1])
plt.xlabel("Iteration", weight="bold")
plt.ylabel("Hit rate", weight="bold")

plt.tight_layout()


plt.savefig(f"../data/imgs/{imgname}.jpg", format="jpg", dpi=300)
plt.savefig(f"../data/imgs/{imgname}.tiff", format="tiff", dpi=300)
plt.savefig(f"../data/imgs/{imgname}.eps", format="eps", dpi=300)
plt.clf()
# np.savetxt("elig_ninefive_onlyhitrate.txt",np.array(lastvals))
print(bestfinish, bestparams)

# eligtable = np.array([uniqueangle, avgs, sterrs]).T
# np.savetxt(f"../errorfiles/plots/resulttable_1", eligtable, delimiter=' & ', fmt='%.3f', newline=' \\\\\n')

