from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import utils.ContinuousObjects as co
import utils.tools as tools
from utils.ContinuousEnvironment import Environment
from utils.ContinuousObjects import Ball, Paddle
import numpy as np
import math
import random
import getopt
import sys


# Command line parsing
opts, args = getopt.getopt(sys.argv[1:], "h:o:t:")
for o, a in opts:
    if o == "-h":
        thr_hidden = int(a)
    elif o == "-o":
        thr_out = int(a)
    elif o == "-t":
        trial = int(a)
        
lr = 0.01
threshold = 2
maxamp = 25
nsensorneurons = 50
exponent = -50
div = 100
elig = 0.995
ballangle_max = 0.5

# Initialize environment
dx = 400
width = 2000
height = 1601

env = Environment(height, width)
dt = 1
laserlist = utils.laserpositions(100, dx, 3)

ball = Ball()
ball.size = 1

paddle = Paddle()
paddle.height = 256
paddle.width = 1

# Initialize network
outputsize = 100
networkneurons = [nsensorneurons*(len(laserlist)+1), 50, outputsize]

# Specify neuron behaviour per layer
thresholds = [threshold, thr_hidden, thr_out]
thresholdlist = [np.ones(value)*thresholds[idx] for idx, value in enumerate(networkneurons)]

# Generate muscle fiber amplitudes
amplitudes = np.random.uniform(1, maxamp, outputsize)

l1 = co.Layer(networkneurons[0], networkneurons[0], networkneurons[2],  "input", thresholdlist[0], div, elig)
l2 = co.Layer(networkneurons[0], networkneurons[1],  networkneurons[2], "hidden", thresholdlist[1], div, elig)
l3 = co.Layer(networkneurons[1], networkneurons[2], networkneurons[2], "output", thresholdlist[2], div, elig)

tau_rise = np.random.uniform(50, 200, outputsize)
tau_decay = np.random.uniform(50, 200, outputsize)
alphas = np.exp(-1/tau_rise)
betas = np.exp(-1/tau_decay)
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

agonisthist = np.zeros([int(networkneurons[2]/2), int(width*1.25)])
antagonisthist = np.zeros([int(networkneurons[2]/2), int(width*1.25)])

iterations = 200000

sensorcenters = np.linspace(-500, height+500, nsensorneurons)/height

hits = []
errors = []
initializationhist = []
finalpositionhist = []

for j in range(iterations):

    t = 0
    # Flags
    bounced = 0
    reset = 0
    calculated = 0
    ball.position = [0, np.random.uniform(0, height - 1)]
    previousballposition = ball.position.copy()
    ballhit = -1
    ballangle = random.uniform(-ballangle_max, ballangle_max)
    ballangle = math.atan(ballangle)
    vvert = math.sin(ballangle)
    vhor = math.cos(ballangle)
    ball.speed = [vhor, vvert]
    paddle.position = startingdist = np.random.uniform(0, height - paddle.height)
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
            inputspikerate = sensoryneuronsignals/threshold

        outputspikesl1 = l1.update_neurons(sensoryneuronsignals, prob=1)
        outputspikesl2 = l2.update_neurons(outputspikesl1, l1.thresholds)
        outputspikesl3 = l3.update_neurons(outputspikesl2, l2.thresholds)

        mf_agonist.update(outputspikesl3[0:int(outputsize/2)], amplitudes[0:int(outputsize/2)])
        mf_antagonist.update(outputspikesl3[int(outputsize/2):int(outputsize)], amplitudes[int(outputsize/2):int(outputsize)])

        # Move paddle
        dxagonist = np.sum(mf_agonist.fibers_c2)
        dxantagonist = np.sum(mf_antagonist.fibers_c2)
        paddle.position += (dxagonist - dxantagonist)
        paddleposhist.append(paddle.position)
        # If ball has touched any side.
        while ball.position[0] + int(ball.size/2) > (env.width - 1) or \
            ball.position[1] + int(ball.size/2) > (env.height - 1) or \
            ball.position[0] - int(ball.size/2) < 0 or \
            ball.position[1] - int(ball.size/2) < 0:

            # Ball hits front or paddle
            if ball.position[0] + int(ball.size / 2) > (width - 1):

                ballhit = ball.position[1]
                middleofpaddle = paddle.position + paddle.height/2

                if paddle.position <= ballhit <= paddle.position + paddle.height:
                    distance_error = (ball.position[1] - middleofpaddle)/height
                    hits.append(1)
                else:
                    hits.append(0)
                    distance_error = (ball.position[1] - middleofpaddle)/height
                reset = 1
                errors.append(abs(distance_error))
                break
            else:
                bounced = 1
            ball.bounce(env)

        if reset:

            # Average activity per neuron
            individualbadness = distance_error * np.log(l3.sumofspikes + 1 ) * amplitudes
            individualbadness[int(outputsize / 2):] *= -1
            l2.update_weights(individualbadness, lr, inputspikerate, limit=1)
            l3.update_weights(individualbadness, lr, previousactivation=1/(1 + np.exp(-l2.activityhistory_scaled)), limit=1)
            break

np.savetxt(
    f"/scratch/p309238/fba_thresholdsweep/l2weights/thr_hid={thr_hidden}_thr_out={thr_out}_trial={trial}.txt"
    , l2.weightmatrix)
np.savetxt(
    f"/scratch/p309238/fba_thresholdsweep/l3weights/thr_hid={thr_hidden}_thr_out={thr_out}_trial={trial}.txt"
    , l3.weightmatrix)
np.savetxt(
    f"/scratch/p309238/fba_thresholdsweep/feedbackweights/thr_hid={thr_hidden}_thr_out={thr_out}_trial={trial}.txt"
    , l2.feedbackweights)
np.savetxt(
    f"/scratch/p309238/fba_thresholdsweep/amplitudes/thr_hid={thr_hidden}_thr_out={thr_out}_trial={trial}.txt"
    , amplitudes)
np.savetxt(
    f"/scratch/p309238/fba_thresholdsweep/alphas/thr_hid={thr_hidden}_thr_out={thr_out}_trial={trial}.txt"
    , alphas)
np.savetxt(
    f"/scratch/p309238/fba_thresholdsweep/betas/thr_hid={thr_hidden}_thr_out={thr_out}_trial={trial}.txt"
    , betas)

with open(f"/scratch/p309238/fba_thresholdsweep/errorfiles/thr_hid={thr_hidden}_thr_out={thr_out}_trial={trial}.txt",
      'w') as errorfile:
    for id, err in enumerate(errors):
        errorfile.write(f"{err},{hits[id]}\n")
