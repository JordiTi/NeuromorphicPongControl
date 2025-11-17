'''Tool to determine what the error is given non-overlapping positional coding.
This is not direclty applicable to the paper but might give you intuition on how big the paddle
needs to be.'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Slider

# Plot the trajectory of the ball
def getballtrajectory(ballangle, ballstart):
    upperbounce = 0
    lowerbounce = 0
    bouncex = None
    if ballangle * 2000 + ballstart > 1600:
        bouncex = (1600 - ballstart) / ballangle
        upperbounce = 1
    elif ballangle * 2000 + ballstart < 0:
        bouncex = - ballstart / ballangle
        lowerbounce = 1
    if not bouncex is None:
        if upperbounce:
            return [[0, bouncex], [ballstart, 1600 ]], [[bouncex, 2000], [1600, -ballangle * (2000-bouncex) + 1600 ]]
        elif lowerbounce:
            return [[0, bouncex], [ballstart, 0 ]], [[bouncex, 2000], [0, -ballangle * 2000 - ballstart ]]
    else:
        return [ballstart, ballangle * 2000 + ballstart]

# Given initial position of sensors, create sensor lines for plotting
def getsensorlines(sensors):
    sensorlines = []
    for s in sensors:
        sensorlines.append([s, s])
        sensorlines.append([0, 1600])
    return sensorlines

# Compute the uncertainty given the resolution of the sensor and the positions where the ball passes
def computeuncertaintylines(sensor1, sensor2, passingpositions, sensoredges):

    linex1 = sensors[sensor1]
    linex2 = sensors[sensor2]

    if passingpositions[sensor1] == 0:
        liney11 = 0
        liney12 = sensorwidth
    elif passingpositions[sensor1] == 1600:
        liney11 = 1600 - sensorwidth
        liney12 = 1600
    else:
        for i, sedge in enumerate(sensoredges):
            if sedge >= passingpositions[sensor1]:
                liney11 = sensoredges[i-1]
                liney12 = sensoredges[i]
                break

    if passingpositions[sensor2] == 0:
        liney21 = 0
        liney22 = sensorwidth
    elif passingpositions[sensor2] == 1600:
        liney21 = 1600 - sensorwidth
        liney22 = 1600
    else:
        for j, sedge in enumerate(sensoredges):
            if sedge >= passingpositions[sensor2]:
                liney21 = sensoredges[j-1]
                liney22 = sensoredges[j]
                break
    return linex1, linex2, liney11, liney12, liney21, liney22


# Determine on what y positions the ball passes the sensor
def computepassingpositions(ballangle, sensors, ballstart):
    passingpositions = []
    for sensor in sensors:
        passingposition = ballangle * sensor + ballstart
        if passingposition > 1600:
            passingposition = 1600 * 2 - passingposition
        elif passingposition < 0:
            passingposition = abs(passingposition)
        passingpositions.append(passingposition)
    return passingpositions

# Creates lines for drawing 
def getuncertaintylines(bouncex, sensors, passingpositions, sensoredges):
    if bouncex is None or bouncex <= sensors[0] or bouncex > sensors[2]:
        sensor1 = 0
        sensor2 = 2
        linex1, linex2, liney11, liney12, liney21, liney22 = (
            computeuncertaintylines(sensor1, sensor2, passingpositions, sensoredges))

    # Ball bounces between sensors 1 and 2, now only sensors 2 and 3 determine uncertainty
    elif sensors[0] < bouncex <= sensors[1]:
        sensor1 = 1
        sensor2 = 2
        linex1, linex2, liney11, liney12, liney21, liney22 = (
            computeuncertaintylines(sensor1, sensor2, passingpositions, sensoredges))
    # Ball bounces between sensors 2 and 3. Now only sensors 1 and 2 determine uncertainty
    else:
        sensor1 = 0
        sensor2 = 1
        linex1, linex2, liney11, liney12, liney21, liney22 = (
            computeuncertaintylines(sensor1, sensor2, passingpositions, sensoredges))
    return linex1, linex2, liney11, liney12, liney21, liney22


# Some initialization
xvals = [0, 2000]
ballstart = 801
ballangle = 0
sensors = [100, 800, 1500]
sensorwidth = 50

bouncex = None
if ballangle * 2000 + ballstart > 1600:
    bouncex = (1600 - ballstart) / ballangle
elif ballangle * 2000 + ballstart < 0:
    bouncex = - ballstart / ballangle


fig, ax = plt.subplots()
ax.set_ylim(0, 1600)
ax.set_xlim(0, 2000)
ax.tick_params(axis='y', which='both', labelleft=True, labelright=True)

# Plot initial horizontal sensor lines
sensorlines = getsensorlines(sensors)
slineplot = ax.plot(*sensorlines)

# Plot initial ball positions
balltrajectory = [ax.plot(xvals, getballtrajectory(ballangle, ballstart), "--", color="brown")]

# Plot initial sensor positions
sensoredges = np.arange(0, 1600 + sensorwidth, sensorwidth)
sensoredgeplots = []
for sensorx in sensors:
    plotspersensors = []
    for sensoredge in sensoredges:
        edgeplot = ax.plot([sensorx-10, sensorx+10], [sensoredge, sensoredge], color='red')
        plotspersensors.append(edgeplot)
    sensoredgeplots.append(plotspersensors)

# Plot initial uncertainty lines
# In nobounce cases or cases where paddle bounces behind 3rd lidar or paddle bounces before 1st lidar:
passingpositions = computepassingpositions(ballangle, sensors, ballstart)


linex1, linex2, liney11, liney12, liney21, liney22 = getuncertaintylines(bouncex, sensors, passingpositions, sensoredges)
firstliney = (liney22 - liney11)/(linex2 - linex1) * (2000-linex1) + liney11
secondliney = (liney21 - liney12)/(linex2 - linex1) * (2000-linex1) + liney12

ax.set_title(f"Uncertainty: {int(abs(firstliney - secondliney))}")
lowerline = ax.plot([linex1, 2000], [liney11, firstliney], color="green")
topline = ax.plot([linex1, 2000], [liney12, secondliney], color="blue")

fig.subplots_adjust(left=0.25, bottom=0.5)


# Ball trajectory sliders
ballstartslider_ax = fig.add_axes([0.25, 0.2, 0.65, 0.03])
ballstartslider = Slider(ballstartslider_ax, "start", 0, 1600, valinit=800)
ballangleslider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
ballangleslider = Slider(ballangleslider_ax, "Angle", -0.5, 0.5, valinit=0)

# Sensor width sliders
sensorwidthslider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
sensorwidthslider = Slider(sensorwidthslider_ax, "Sensorwidth", 10, 100, valinit=40)

# Sensor position sliders
sensor1slider_ax = fig.add_axes([0.25, 0.25, 0.65, 0.03])
sensor2slider_ax = fig.add_axes([0.25, 0.30, 0.65, 0.03])
sensor3slider_ax = fig.add_axes([0.25, 0.35, 0.65, 0.03])

sensor1slider = Slider(sensor1slider_ax, "Lidar1", 0, 2000, valinit=100)
sensor2slider = Slider(sensor2slider_ax, "Lidar2", 0, 2000, valinit=800)
sensor3slider = Slider(sensor3slider_ax, "Lidar3", 0, 2000, valinit=1500)
sensorsliders = [sensor1slider, sensor2slider, sensor3slider]


# Update plot when sliders change
def sliders_on_changed(hi):

    # Update ball trajectory
    bouncex = None
    if ballangleslider.val * 2000 + ballstartslider.val > 1600:
        bouncex = (1600 - ballstartslider.val) / ballangleslider.val
    elif ballangleslider.val * 2000 + ballstartslider.val < 0:
        bouncex = - ballstartslider.val / ballangleslider.val

    # Since its either 1 or 2 lines, first remove trajectory lines then plot new ones
    for m, trajectoryline in enumerate(balltrajectory):
        balltrajectory[m][0].remove()
    balltrajectory.clear()
    if bouncex is None:
        balltrajectory.append(ax.plot(xvals, getballtrajectory(ballangleslider.val, ballstartslider.val), "--", color="brown"))
    else:
        l1, l2 = getballtrajectory(ballangleslider.val, ballstartslider.val)
        balltrajectory.append(ax.plot(l1[0], l1[1], "--", color="brown"))
        balltrajectory.append(ax.plot(l2[0], l2[1], "--", color="brown"))

    # Update sensor line data
    for i, s in enumerate(slineplot):
        s.set_data([sensorsliders[i].val, sensorsliders[i].val], [0, 1600])

    # Update sensor lines, since the number of lines is dynamic, remove all lines and replot new ones
    for j, persensor in enumerate(sensoredgeplots):
        for k, individualsensoredge in enumerate(persensor):

            if len(ax.lines) > 4:
                individualsensoredge[0].remove()
    sensoredgeplots.clear()

    # Plot slider positions
    sensoredges = np.arange(0, 1600 + sensorwidthslider.val, sensorwidthslider.val)
    for slidernum in sensorsliders:
        plotspersensors = []
        sensorposition = slidernum.val
        for sensoredge in sensoredges:
            edgeplot = ax.plot([sensorposition-10, sensorposition+10], [sensoredge, sensoredge], color='red')
            plotspersensors.append(edgeplot)
        sensoredgeplots.append(plotspersensors)

    # Plot uncertainty lines

    for k, sens in enumerate(sensors):
        sensors[k] = sensorsliders[k].val
    passingpositions = computepassingpositions(ballangleslider.val, sensors, ballstartslider.val)
    linex1, linex2, liney11, liney12, liney21, liney22 = getuncertaintylines(bouncex, sensors, passingpositions, sensoredges)
    firstliney = (liney22 - liney11)/(linex2 - linex1) * (2000-linex1) + liney11
    secondliney = (liney21 - liney12)/(linex2 - linex1) * (2000-linex1) + liney12

    ax.set_title(f"Uncertainty: {int(abs(firstliney - secondliney))}")
    lowerline[0].set_data([linex1, 2000], [liney11, firstliney])
    topline[0].set_data([linex1, 2000], [liney12, secondliney])

    fig.canvas.draw_idle()


ballstartslider.on_changed(sliders_on_changed)
ballangleslider.on_changed(sliders_on_changed)
sensor1slider.on_changed(sliders_on_changed)
sensor2slider.on_changed(sliders_on_changed)
sensor3slider.on_changed(sliders_on_changed)
sensorwidthslider.on_changed(sliders_on_changed)


plt.show()
