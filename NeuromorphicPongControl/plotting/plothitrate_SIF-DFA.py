import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../utils/")
import tools
import re
import numpy as np
import random

plt.rcParams.update({'font.size': 20})
parametermat = []
errormat = []

resultfilenames = os.listdir("../data/errorfiles/errorfiles_SIF-DFA/data/")
plt.figure(figsize=(10, 8))
for f in resultfilenames:

    if f.startswith("lr"):
        parts = f.strip(".txt").split("_")

        for i, p in enumerate(parts):
            p = re.sub(r"[^\d.e\-]", "", p)
            p = re.sub(r"\Ae+", "", p)
            if i == 0:
                lr = float(p)
            elif i == 1:
                threshold = int(p)
            elif i == 2:
                maxamp = int(p)
            elif i == 3:
                exponents = int(float(p))
            elif i == 4:
                nsensorneurons = int(p)
            elif i == 5:
                div = int(p)

        parametervalues = [lr, threshold, maxamp, exponents, nsensorneurons, div]
        parametermat.append(parametervalues)
        with open(f"../data/errorfiles/errorfiles_SIF-DFA/data/{f}", "r") as errorfile:
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
np.savetxt("../data/plottingdata/SIF-DFA.txt",np.array(sorted_by_col))

# uniquelr = np.unique(parametermat[:, 0])
# uniquethr = np.unique(parametermat[:, 1])
# uniqyemaxamp = np.unique(parametermat[:, 2])
# uniqueexp = np.unique(parametermat[:, 3])
# uniquensn = np.unique(parametermat[:, 4])
# uniquediv = np.unique(parametermat[:,5])
# uniqueelig = np.unique(parametermat[:,6])



# bestfinish = 0
# bestparams = []
# # Plot lr depdencency
# plt.figure(1)
# lastvals = []
# xcoord = np.linspace(0, len(errormat[0]), num=len(errormat[0]))
# for uth in uniquethr:
#     for uamp in uniqyemaxamp:
#         for uexp in uniqueexp:
#             for unsn in uniquensn:
#                 for udiv in uniquediv:
#                     for ulr in uniquelr:
#                         for uel in uniqueelig:

#                             indices = np.where(np.all(parametermat == [ ulr, uth, uamp, uexp, unsn, udiv, uel], axis=1))
#                             data = []
#                             for i in indices:
#                                 data.append(errormat[i, :])
#                             data = np.asarray(data)[0]
#                             print(f"{[ulr, uth, uamp, unsn, uexp, udiv, uel]}")
#                             if not np.sum(data):
#                                 print("NOT")

#                             avg = np.average(data, axis=0)
#                             lastvals.append(avg[-1])
#                             sterr = np.std(data, axis=0)/np.sqrt(2)

#                             color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

#                             plt.plot(avg, label=f"{uel}", color=color)
#                             # plt.fill_between(xcoord, avg+sterr, avg-sterr, alpha=0.5, color=color)
#                         plt.legend()
#                         plt.ylim([0, 1])
#                         plt.xlabel("Iteration")
#                         plt.ylabel("Hitrate")
#                         plt.tight_layout()
#                         plt.savefig(f"../errorfiles/plots/amp={uamp}_thr={uth}_uexp={uexp}_usn={unsn}_udiv={udiv}_ulr={ulr}.png")
#                         plt.clf()
# np.savetxt("finalerrors.txt",np.array(lastvals))
# print(bestfinish, bestparams)

