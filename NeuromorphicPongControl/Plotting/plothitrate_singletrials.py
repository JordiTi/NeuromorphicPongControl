import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../../")
import utils
import re
import numpy as np
import random

plt.rcParams.update({'font.size': 20})
parametermat = []
errormat = []

resultfilenames = os.listdir("../errorfiles/data/")
plt.figure(figsize=(10, 8))
for f in resultfilenames:

    if f.startswith("lr"):
        parts = f.strip(".txt").split("_")
        for i, p in enumerate(parts):

            p = p.replace("2=", "=")
            p = p.replace("3=", "=")
            p = re.sub(r"[^\d.e\-]", "", p)
            p = re.sub(r"\Ae+", "", p)
            if i == 0:
                lr = float(p)
            elif i == 1:
                thr = int(p)
            elif i == 2:
                amp = int(p)
            elif i == 3:
                nsensorneurons = int(p)
            elif i == 4:
                exponent = int(float(p))
            elif i == 5:
                div = int(p)

        parametervalues = [lr, thr, amp, nsensorneurons, exponent, div]
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


parametermat_errormat = np.insert(parametermat, 0, errormat[:, -1], axis=1)
sorted_by_col = parametermat_errormat[parametermat_errormat[:, 0].argsort()]
np.savetxt("parameter_error_table.txt",np.array(sorted_by_col))

uniquelr = np.unique(parametermat[:, 0])
uniquethr = np.unique(parametermat[:, 1])
uniqueamp = np.unique(parametermat[:, 2])
uniquensn = np.unique(parametermat[:, 3])
uniqueexp = np.unique(parametermat[:, 4])
uniquediv = np.unique(parametermat[:, 5])

bestfinish = 0
bestparams = []
# Plot lr depdencency
plt.figure(1)
xcoord = np.linspace(0, len(errormat[0]), num=len(errormat[0]))
lastvals = []
for uth in uniquethr:
    for uamp in uniqueamp:
        for unsn in uniquensn:
            for uexp in uniqueexp:
                for udiv in uniquediv:
                    for ulr in uniquelr:
                        indices = np.where(np.all(parametermat == [ulr, uth, uamp, unsn, uexp, udiv], axis=1))
                        data = []
                        for i in indices:
                            data.append(errormat[i, :])
                        data = np.asarray(data)[0]
                        print(f"{[ulr, uth, uamp, unsn, uexp, udiv]}")
                        if not np.sum(data):
                            print("NOT")
                        avg = np.average(data, axis=0)
                        lastvals.append(avg[-1])
                        sterr = np.std(data, axis=0)/np.sqrt(2)

                        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

                        plt.plot(avg, label=f"{ulr}", color=color)
                    plt.legend()
                    plt.ylim([0, 1])
                    plt.xlabel("Iteration")
                    plt.ylabel("Hitrate")
                    plt.tight_layout()
                    plt.savefig(f"../errorfiles/plots/thr={uth}_amp={uamp}_sn={unsn}_exp={uexp}_udiv={udiv}.png")
                    plt.clf()
np.savetxt("vanilla_onlyhitrate.txt",np.array(lastvals))
print(bestfinish, bestparams)

