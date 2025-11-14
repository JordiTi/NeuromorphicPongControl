import ContinuousObjects as co
import utils
from ContinuousEnvironment import Environment
from ContinuousObjects import Ball, Paddle
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
import getopt
import pandas as pd
import matplotlib.animation as animation
import matplotlib.patches as patches

'''
Visualize the paddle's behaviour in a live plot by loading all the network parameters and hyperparameters
'''
# Command line parsing
opts, args = getopt.getopt(sys.argv[1:], "l:a:d:t:m:e:n:i:", ["lr=", "angle=", "dx=", "th1=", "maxamp=", "exponent=", "nsensorneurons=", "inhib="])
for o, a in opts:
    if o == "-l" or o == "--lr":
        lr = float(a)
    elif o == "-a" or o == "--angle":
        ballangle_max = float(a)
    elif o == "-d" or o == "--dx":
        dx = int(a)
    elif o == "-t" or o == "--th1":
        th1 = int(a)
    elif o == "-m" or o == "--maxamp":
        maxamp = int(a)
    elif o == "-e" or o == "--exponent":
        exponent = float(a)
    elif o == "-n" or o == "--nsensorneurons":
        nsensorneurons = int(a)
    elif o == "-i" or o == "--inhib":
        inhib = int(a)

mainpath = "../fba_pos_inhib_prob"
# Initialize environment
width = 2000
height = 1601
env = Environment(height, width)
dt = 1
laserlist = utils.laserpositions(100, dx, 3)

ball = Ball()
ball.size = 1
ballangle_max = ballangle_max

paddle = Paddle()
paddle.height = 256
paddle.width = 1

# Initialize network
outputsize = 100
networkneurons = [nsensorneurons*(len(laserlist)+1), 50, outputsize]

# Specify neuron behaviour per layer
thresholds = [th1, 20, 20]
leaktimeconstants = [50, 50, 50]
thresholdlist = [np.ones(value)*thresholds[idx] for idx, value in enumerate(networkneurons)]
leakfactorlist = [np.exp(-1/(np.ones(value)*leaktimeconstants[idx])) for idx, value in enumerate(networkneurons)]

# Generate neuron weights
l1l2weights = np.loadtxt(mainpath + "/l1l2weights/" + f"lr={lr}_angle={ballangle_max}_dx={dx}_threshold={th1}_maxamp={maxamp}_exponents={exponent}_nsensorneurons={nsensorneurons}_inhib_prob={inhib}.txt")
l2l3weights = np.loadtxt(mainpath + "/l2l3weights/" + f"lr={lr}_angle={ballangle_max}_dx={dx}_threshold={th1}_maxamp={maxamp}_exponents={exponent}_nsensorneurons={nsensorneurons}_inhib_prob={inhib}.txt")

# Generate DFA weights
l1l2feedbackweights = np.loadtxt(mainpath + "/feedbackweights/" + f"lr={lr}_angle={ballangle_max}_dx={dx}_threshold={th1}_maxamp={maxamp}_exponents={exponent}_nsensorneurons={nsensorneurons}_inhib_prob={inhib}.txt")

# Generate muscle fiber amplitudes
amplitudes = np.loadtxt(mainpath + "/amplitudes/" + f"lr={lr}_angle={ballangle_max}_dx={dx}_threshold={th1}_maxamp={maxamp}_exponents={exponent}_nsensorneurons={nsensorneurons}_inhib_prob={inhib}.txt")

l1 = co.Layer(networkneurons[0], thresholdlist[0],  layertype="input", weights=l1l2weights, neurontypes="IF", thresholdtype="static", elig=1)
l2 = co.Layer(networkneurons[1], thresholdlist[1],  layertype="hidden", weights=l2l3weights, neurontypes="IF", thresholdtype="static", elig=1)
l3 = co.Layer(networkneurons[2], thresholdlist[2],layertype="output", neurontypes="IF", elig=1)


alphas = np.loadtxt(mainpath + "/alphas/" + f"lr={lr}_angle={ballangle_max}_dx={dx}_threshold={th1}_maxamp={maxamp}_exponents={exponent}_nsensorneurons={nsensorneurons}_inhib_prob={inhib}.txt")
betas = np.loadtxt(mainpath + "/betas/" + f"lr={lr}_angle={ballangle_max}_dx={dx}_threshold={th1}_maxamp={maxamp}_exponents={exponent}_nsensorneurons={nsensorneurons}_inhib_prob={inhib}.txt")
mf_agonist = co.Musclefibers(int(outputsize/2), alphas[0:int(outputsize/2)], betas[0:int(outputsize/2)])
mf_antagonist = co.Musclefibers(int(outputsize/2), alphas[int(outputsize/2):int(outputsize)], betas[int(outputsize/2):int(outputsize)])

layers = [l1, l2, l3]

inputhist1 = np.zeros([networkneurons[0], int(width*1.25)])
inputhist2 = np.zeros([networkneurons[1], int(width*1.25)])
inputhist3 = np.zeros([networkneurons[2], int(width*1.25)])

outputhist1 = np.zeros([networkneurons[0], int(width*1.25)])
outputhist2 = np.zeros([networkneurons[1], int(width*1.25)])
outputhist3 = np.zeros([networkneurons[2], int(width*1.25)])

chargehist1 = np.zeros([networkneurons[0], int(width*1.25)])
chargehist2 = np.zeros([networkneurons[1], int(width*1.25)])
chargehist3 = np.zeros([networkneurons[2], int(width*1.25)])


iterations = 100000

sensorcenters = np.linspace(-500, height+500, nsensorneurons)/height
indices = []

hits = []
errors = []

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(0, width), ylim=(0, height))
plt.axis('scaled')
paddleimg = ax.add_patch(patches.Rectangle((1975, 1), 25, 250))
ballimg = ax.add_patch(patches.Rectangle((0, 0), 40, 40))
p1 = ax.add_patch(patches.Rectangle((100, 0), 10, 1600, color="red"))
p2 = ax.add_patch(patches.Rectangle((500, 0), 10, 1600, color="red"))
p3 = ax.add_patch(patches.Rectangle((900, 0), 10, 1600, color="red"))
score = ax.set_title("0  0        0%")


for j in range(iterations):

    t = 0
    # Flags
    bounced = 0
    reset = 0
    calculated = 0
    ball.position = [0, np.random.uniform(0, height - 1)]
    ballstart = ball.position[1]
    # ball.position = [0, 100]
    previousballposition = ball.position.copy()
    ballhit = -1
    ballangle = random.uniform(-ballangle_max, ballangle_max)
    ballangle = math.atan(ballangle)
    vvert = math.sin(ballangle)
    vhor = math.cos(ballangle)
    ball.speed = [vhor, vvert]
    paddle.position = startingdist = np.random.uniform(0, height - paddle.height)
    # paddle.position = startingdist = 130
    passedlist = np.zeros(len(laserlist) + 1)
    passedlist[-1] = paddle.position/env.height

    # Reset everything
    for l in layers:
        l.reset()
    mf_agonist.reset()
    mf_antagonist.reset()

    # Keep track of things
    paddleposhist = []
    ballposhist = []

    # Keep track of neural activation
    l2activityhist = np.zeros(l2.nneurons)
    l3activityhist = np.zeros(l3.nneurons)

    l1outspikehist = np.zeros(networkneurons[0])
    l2outspikehist = np.zeros(networkneurons[1])
    l3outspikehist = np.zeros(networkneurons[2])

    sensorsignals = np.zeros(len(sensorcenters))
    sensoryneuronsignals = np.zeros(nsensorneurons*4)

    aghist = np.zeros([int(outputsize/2), int(width*1.25)])
    anhist = np.zeros([int(outputsize/2), int(width*1.25)])

    while not reset:
        t += dt
        # initialize ball and paddle

        ball.move()
        idx, position = utils.passedlaser(previousballposition, ball.position, laserlist)
        previousballposition = ball.position.copy()

        if position:
            passedlist[idx] = position/env.height

        if idx == len(laserlist)-1 and not calculated:
            sensoryneuronsignals = []
            for p in passedlist:
                sensoroutputs = []
                for s in sensorcenters:
                    sensoroutputs.append(math.exp(exponent*(s-p)**2))
                sensoryneuronsignals.append(sensoroutputs.copy())
            sensoryneuronsignals = np.array(sensoryneuronsignals).flatten()
            calculated = 1

        a1, outputspikesl1 = l1.update_neurons(sensoryneuronsignals, prob=1)
        a2, outputspikesl2 = l2.update_neurons(a1)

        # Charge output layer and update outputs
        outputspikesl3 = l3.update_neurons(a2)
        mf_agonist.update(outputspikesl3[0:int(outputsize/2)], amplitudes[0:int(outputsize/2)])
        mf_antagonist.update(outputspikesl3[int(outputsize/2):int(outputsize)], amplitudes[int(outputsize/2):int(outputsize)])

        # Move paddle
        dxagonist = np.sum(mf_agonist.fibers_c2)
        dxantagonist = np.sum(mf_antagonist.fibers_c2)
        paddle.position += (dxagonist - dxantagonist)

        if t%50 == 0:
            ballimg.set_xy([ball.position[0], ball.position[1]])
            paddleimg.set_xy([1975, paddle.position ])
            plt.draw()
            if sum(hits):
                score.set_text(f"{sum(hits)}  {len(hits)-sum(hits)}        {sum(hits)/len(hits)*100:.0f}%")
            if plt.waitforbuttonpress(0.01): quit()
        #Limit paddle
        # paddle.position = min(max(paddle.position, 0), height-paddle.height)

        # Save history

        # If ball has touched any side.
        while ball.position[0] + int(ball.size/2) > (env.width - 1) or \
            ball.position[1] + int(ball.size/2) > (env.height - 1) or \
            ball.position[0] - int(ball.size/2) < 0 or \
            ball.position[1] - int(ball.size/2) < 0:

            # Ball hits front or paddle
            if ball.position[0] + int(ball.size / 2) > (width - 1 ):

                ballhit = ball.position[1]
                middleofpaddle = paddle.position + paddle.height/2

                if paddle.position <= ballhit <= paddle.position + paddle.height:
                    distance_error = (ball.position[1] - middleofpaddle)/height
                    hits.append(1)
                    hit = 1
                else:
                    hits.append(0)
                    hit = 0
                    distance_error = (ball.position[1] - middleofpaddle)/height
                reset = 1
                errors.append(abs(distance_error))
                print(sum(hits), len(hits)-sum(hits), sum(hits)/len(hits))
                break
            else:
                bounced = 1
            ball.bounce(env)

        if reset:

            break

