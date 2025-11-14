import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../../")
import utils
import re
import numpy as np
import random

plt.rcParams.update({'font.size': 25})
plt.rcParams['font.weight'] = 'bold'
parametermat = []
errormat = []

resultfilenames = os.listdir("../errorfiles/data/")
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
        with open(f"../errorfiles/data/{f}", "r") as errorfile:
            data = errorfile.readlines()
            coll = []
            for d in data:
                dist = d.split(",")[0]
                hit = d.split(",")[1]
                d = int(hit)
                coll.append(d)

        ma = utils.moving_average(coll, 10000)
        errormat.append(ma)

parametermat = np.asarray(parametermat)


errormat = np.asarray(errormat)
uniqueangle = np.unique(parametermat[:, 0])



bestfinish = 0
bestparams = []
# Plot lr depdencency
plt.figure(1)
lastvals = []
sterrs = []
avgs = []
xcoord = np.linspace(0, len(errormat[0]), num=len(errormat[0]))
for ua in uniqueangle:
    

    indices = np.where(np.all(parametermat == [ ua], axis=1))
    print(indices)
    data = []
    for i in indices:
        data.append(errormat[i, :])
    data = np.asarray(data)[0]

    avg = np.average(data, axis=0)
    avgs.append(np.round(avg[-1], 3))
    lastvals.append(avg[-1])
    sterr = np.std(data, axis=0)/np.sqrt(4)
    sterrs.append(np.round(sterr[-1], 3))

    color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

    plt.plot(avg, label=f"{ua}", color=color)
    plt.fill_between(xcoord, avg+sterr, avg-sterr, alpha=0.5, color=color)
plt.legend(prop={'size':20})
plt.ylim([0.4, 1])
plt.xlabel("Iteration", weight="bold")
plt.ylabel("Hit rate", weight="bold")

plt.tight_layout()


plt.savefig(f"../errorfiles/plots/Anglesweep_0.995_4.jpg")
plt.clf()
# np.savetxt("elig_ninefive_onlyhitrate.txt",np.array(lastvals))
print(bestfinish, bestparams)

eligtable = np.array([uniqueangle, avgs, sterrs]).T
np.savetxt(f"../errorfiles/plots/resulttable_1", eligtable, delimiter=' & ', fmt='%.3f', newline=' \\\\\n')

