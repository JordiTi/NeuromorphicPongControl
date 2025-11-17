'''Plot the training data. Adjust the file as necessary (see comments in file)'''
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../")
import utils.tools as ut
import re
import numpy as np
import random

plt.rcParams.update({'font.size': 20})
parametermat = []
errormat = []
averagekernel = 100 # pt moving average

print("Collecting data...")
resultfilenames = os.listdir("../data/traindata/errorfiles/")
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
            elif i == 6:
                log = int(p)
            elif i == 7:
                elig = float(p)

        parametervalues = [lr, thr, amp, nsensorneurons, exponent, div, log, elig]
        parametermat.append(parametervalues)
        with open(f"../data/traindata/errorfiles/{f}", "r") as errorfile:
            data = errorfile.readlines()
            coll = []
            for d in data:
                dist = d.split(",")[0]
                hit = d.split(",")[1]
                d = int(hit)
                coll.append(d)
        if averagekernel > len(coll):
            raise Exception(f"ma ({averagekernel}) is bigger than length of the data ({len(coll)})")
        ma = ut.moving_average(coll, averagekernel)
        errormat.append(ma)

parametermat = np.asarray(parametermat)
errormat = np.asarray(errormat)

print("Saving averaged data...")
parametermat_errormat = np.insert(parametermat, 0, errormat[:, -1], axis=1)
sorted_by_col = parametermat_errormat[parametermat_errormat[:, 0].argsort()]
np.savetxt("../data/traindata/parameter_error_table.txt",np.array(sorted_by_col))

print("Extracting unique parameters...")
uniquelr = np.unique(parametermat[:, 0])
uniquethr = np.unique(parametermat[:, 1])
uniqueamp = np.unique(parametermat[:, 2])
uniquensn = np.unique(parametermat[:, 3])
uniqueexp = np.unique(parametermat[:, 4])
uniquediv = np.unique(parametermat[:, 5])
uniquelog = np.unique(parametermat[:, 6])
uniqueelig = np.unique(parametermat[:, 7])

print("Plotting...")
plt.figure(1)
xcoord = np.linspace(0, len(errormat[0]), num=len(errormat[0]))
lastvals = []
for uth in uniquethr:
    for uamp in uniqueamp:
        for unsn in uniquensn:
            for uexp in uniqueexp:
                for udiv in uniquediv:
                    for ulr in uniquelr:
                        for ulo in uniquelog:
                            for uel in uniqueelig:
                                indices = np.where(np.all(parametermat == [ulr, uth, uamp, unsn, uexp, udiv, log, uel], axis=1))
                                data = []
                                for i in indices:
                                    data.append(errormat[i, :])
                                data = np.asarray(data)[0]
                                if not np.sum(data):
                                    print(f"No data in thr={uth}_amp={uamp}_sn={unsn}_exp={uexp}_udiv={udiv}_ulog={ulo}_uelig={uel}")
                                avg = np.average(data, axis=0)
                                lastvals.append(avg[-1])


                                color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
                                plt.plot(avg, label=f"{uel}", color=color)
                                if len(data) > 1:
                                    sterr = np.std(data, axis=0)/np.sqrt(len(data)-1)
                                    plt.fill_between(xcoord, avg+sterr, avg-sterr, alpha=0.5, color=color)
                            plt.legend()
                            plt.ylim([0, 1])
                            plt.xlabel("Iteration")
                            plt.ylabel("Hitrate")
                            plt.tight_layout()
                            plt.savefig(f"../data/imgs/thr={uth}_amp={uamp}_sn={unsn}_exp={uexp}_udiv={udiv}_ulog={ulo}.png")
                            plt.clf()

