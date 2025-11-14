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
opts, args = getopt.getopt(sys.argv[1:], "a:l:t:m:n:e:d:o:v:")
for o, a in opts:
    if o == "-a":
        ballangle_max = float(a)
    elif o == "-l":
        lr = float(a)
    elif o == "-t":
        threshold = int(a)
    elif o == "-m":
        maxamp = int(a)
    elif o == "-n":
        nsensorneurons = int(a)
    elif o == "-e":
        exponent = int(float(a))
    elif o == "-d":
        div = int(a)
    elif o == "-o":
        log = int(a)
    elif o == "-v":
        elig = float(a)

# Initialize environment
dx = 400
width = 2000
height = 1601
env = Environment(height, width)
dt = 1

# Initialize LiDAR's and encoders
laserlist = tools.laserpositions(100, dx, 3)
sensorcenters = np.linspace(-500, height+500, nsensorneurons)/height

# Initialize ball
ball = Ball()
ball.size = 1
ballangle_max = ballangle_max

# Initialize paddle
paddle = Paddle()
paddle.height = 256
paddle.width = 1

# Initialize network
outputsize = 100
networkneurons = [nsensorneurons*(len(laserlist)+1), 50, outputsize]

# Specify neuron thresholds per layer
thresholds = [threshold, 20, 20]
thresholdlist = [np.ones(value)*thresholds[idx] for idx, value in enumerate(networkneurons)]

# Initialize network
l1 = co.Layer(networkneurons[0], networkneurons[0], networkneurons[2],  "input", thresholdlist[0], div, elig)
l2 = co.Layer(networkneurons[0], networkneurons[1],  networkneurons[2], "hidden", thresholdlist[1], div, elig)
l3 = co.Layer(networkneurons[1], networkneurons[2], networkneurons[2], "output", thresholdlist[2], div, elig)
layers = [l1, l2, l3]

# Initialize musclefibers
amplitudes = np.random.uniform(1, maxamp, outputsize)
tau_rise = np.random.uniform(50, 200, outputsize)
tau_decay = np.random.uniform(50, 200, outputsize)
alphas = np.exp(-dt/tau_rise)
betas = np.exp(-dt/tau_decay)
mf_agonist = co.Musclefibers(int(outputsize/2), alphas[0:int(outputsize/2)], betas[0:int(outputsize/2)])
mf_antagonist = co.Musclefibers(int(outputsize/2), alphas[int(outputsize/2):int(outputsize)], betas[int(outputsize/2):int(outputsize)])

iterations = 200000


# Track scores
hits = []
errors = []

for j in range(iterations):
    if j%200 == 0:
        print(f"Iteration {j}")
        
    t = 0

    # Flags
    reset = 0
    calculated = 0
    ballhit = -1

    # Random ball initialization
    ball.position = [0, np.random.uniform(0, height - 1)]
    previousballposition = ball.position.copy()
    ballangle = random.uniform(-ballangle_max, ballangle_max)
    ballangle = math.atan(ballangle)
    vvert = math.sin(ballangle)
    vhor = math.cos(ballangle)
    ball.speed = [vhor, vvert]

    # Random paddle initialization
    paddle.position = startingdist = np.random.uniform(0, height - paddle.height)

    # Tracks what LiDAR's are passed
    passedlist = np.zeros(len(laserlist) + 1)
    passedlist[-1] = paddle.position/env.height

    # Reset state variables
    for l in layers:
        l.reset()
    mf_agonist.reset()
    mf_antagonist.reset()

    # Gaussian encoder values
    sensoryneuronsignals = np.zeros(nsensorneurons*4)


    while not reset:

        t += dt

        # Move  ball, check whether ball passed LiDAR's
        ball.move()
        idx, position = tools.passedlaser(previousballposition, ball.position, laserlist)
        previousballposition = ball.position.copy()
        if position:
            passedlist[idx] = position/env.height

        # Ball passes last LiDAR, sensor value to spike encoding is performed
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

        # Update system state variables
        outputspikesl1 = l1.update_neurons(sensoryneuronsignals, prob=1)
        outputspikesl2 = l2.update_neurons(outputspikesl1, l1.thresholds)
        outputspikesl3 = l3.update_neurons(outputspikesl2, l2.thresholds)
        mf_agonist.update(outputspikesl3[0:int(outputsize/2)], amplitudes[0:int(outputsize/2)])
        mf_antagonist.update(outputspikesl3[int(outputsize/2):int(outputsize)], amplitudes[int(outputsize/2):int(outputsize)])

        # Move paddle
        dxagonist = np.sum(mf_agonist.fibers_c2)
        dxantagonist = np.sum(mf_antagonist.fibers_c2)
        paddle.position += (dxagonist - dxantagonist)

        # If ball has touched any wall.
        while ball.position[0] + int(ball.size/2) > (env.width - 1) or \
            ball.position[1] + int(ball.size/2) > (env.height - 1) or \
            ball.position[0] - int(ball.size/2) < 0 or \
            ball.position[1] - int(ball.size/2) < 0:

            # Ball hits front or paddle, hit/miss is determined, and game reset is initialized.
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
            
            ball.bounce(env)

        # Weights are updated, game is reset
        if reset:

            # Log error calculation
            if log:
                individualbadness = distance_error * np.log(l3.sumofspikes + 1 ) * amplitudes
            else:
                individualbadness = distance_error * l3.sumofspikes * amplitudes

            # Weight update
            individualbadness[int(outputsize / 2):] *= -1
            l2.update_weights(individualbadness, lr, inputspikerate, limit=1)
            l3.update_weights(individualbadness, lr, previousactivation=1/(1 + np.exp(-l2.activityhistory_scaled)), limit=1)
            break

extension = "lr={lr}_threshold={threshold}_maxamp={maxamp}_nsensorneurons={nsensorneurons}_exponent={exponent}_div={div}_log={log}_elig={elig}.txt"
np.savetxt(
    f"./l2weights/{extension}"
    , l2.weightmatrix)
np.savetxt(
    f"./l3weights/{extension}"
    , l3.weightmatrix)
np.savetxt(
    f"./feedbackweights/{extension}"
    , l2.feedbackweights)
np.savetxt(
    f"./amplitudes/{extension}"
    , amplitudes)
np.savetxt(
    f"./alphas/{extension}"
    , alphas)
np.savetxt(
    f"./betas/{extension}"
    , betas)

with open(f"./errorfiles/{extension}",
      'w') as errorfile:
    for id, err in enumerate(errors):
        errorfile.write(f"{err},{hits[id]}\n")
