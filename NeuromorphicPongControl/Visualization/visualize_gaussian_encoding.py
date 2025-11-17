'''Visualizes encoding from sensor values to spiking signal probability through Gaussian encoding'''
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patch

plt.rcParams.update({'font.size': 15})
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
width = 2000
height = 1601

# Set your initial values here, nsensorneurons is the number of sensor neurons,
# exponent is the width of the gaussian,
# Sensorcenters is the location of the means.
nsensorneurons = 10
exponent = -25
sensorcenters = np.linspace(-500, height+500, nsensorneurons)/height

# Relative hit height of the ball (between 0 and 1)
hitheight = 0.2


fig, ax = plt.subplots(1)
# Resolution
points = np.arange(-1000, height + 500 + 500)/height
for i, s in enumerate(sensorcenters):
    
    # Encoding avlues 
    c1 = plt.Circle((hitheight, math.exp(exponent * (s - hitheight) ** 2)), radius=0.02, color="orange")
    ax.add_patch(c1)

    # Plot Gaussians
    sensoroutputs = []
    for p in points:
        sensoroutputs.append(math.exp(exponent * (s - p) ** 2))
    if i % 2 == 0:
        plt.plot(points, sensoroutputs, color="royalblue")
    else:
        plt.plot(points, sensoroutputs, color="green")

# Plot environment width and ball height indicator 
plt.vlines([0, 1], 0, 1, color="black", linewidth=3, ls=":")
plt.vlines([hitheight], 0, 1 , color="red", linewidth=2)
plt.xlabel("Relative environment height")
plt.ylabel("Sensor value")

plt.tight_layout()
plt.show()
