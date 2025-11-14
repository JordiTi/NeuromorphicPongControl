import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patch

plt.rcParams.update({'font.size': 15})
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
width = 2000
height = 1601

nsensorneurons = 10
exponent = -25
sensorcenters = np.linspace(-500, height+500, nsensorneurons)/height

hitpoint = 0.2

fig, ax = plt.subplots(1)
points = np.arange(-1000, height + 500 + 500)/height
for i, s in enumerate(sensorcenters):
    c1 = plt.Circle((0.2, math.exp(exponent * (s - 0.2) ** 2)), radius=0.02, color="orange")
    ax.add_patch(c1)
    sensoroutputs = []
    for p in points:
        sensoroutputs.append(math.exp(exponent * (s - p) ** 2))
    if i % 2 == 0:

        plt.plot(points, sensoroutputs, color="royalblue")
    else:
        plt.plot(points, sensoroutputs, color="green")


plt.vlines([0, 1], 0, 1, color="black", linewidth=3, ls=":")
plt.vlines([0.2], 0, 1 , color="red", linewidth=2)
plt.xlabel("Relative environment height")
plt.ylabel("Sensor value")

plt.tight_layout()
plt.show()
