'''Visualization of the Pong game, given parameters. Press any button to quit the simulation.
Run the following to see the best model: python3 visualize_game.py -a 0.5 -l 0.01 -t 2 -m 25 -n 50 -e -50 -d 100 -o 1 -e 0.995 -r 0'''

import imageio
record_gif = False
gif_frames = []
gif_name = "pong.gif"

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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''
Visualize the paddle's behaviour in a live plot by loading all the network parameters and hyperparameters
'''
# Command line parsing
opts, args = getopt.getopt(sys.argv[1:], "a:l:t:m:n:e:d:o:v:r:g:")
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
        #Steepness
        div = int(a)
    elif o == "-o":
        log = int(a)
    elif o == "-v":
        elig = float(a)
    elif o == "-r":
        run = int(a)
    elif o == "-g":
        record_gif = True

mainpath = "./Visualizationdata"
# extension = f"lr={lr}_threshold={threshold}_maxamp={maxamp}_nsensorneurons={nsensorneurons}_exponent={exponent}_div={div}_log={log}_elig={elig}_run={run}.txt"
extension = f"lr={lr}_threshold={threshold}_maxamp={maxamp}_nsensorneurons={nsensorneurons}_exponent={exponent}_div={div}_elig={elig}.txt"

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

# Generate neuron weights
l2weights = np.loadtxt(mainpath + "/l2weights/" + extension)
l3weights = np.loadtxt(mainpath + "/l3weights/" + extension)

# Generate DFA weights
feedbackweights = np.loadtxt(mainpath + "/feedbackweights/" + extension)

# Generate muscle fiber amplitudes
amplitudes = np.loadtxt(mainpath + "/amplitudes/" + extension)

l1 = co.Layer(networkneurons[0], networkneurons[0], networkneurons[2],  "input", thresholdlist[0], div, elig)
l2 = co.Layer(networkneurons[0], networkneurons[1],  networkneurons[2], "hidden", thresholdlist[1], div, elig)
l3 = co.Layer(networkneurons[1], networkneurons[2], networkneurons[2], "output", thresholdlist[2], div, elig)
l2.weightmatrix = l2weights
l3.weightmatrix = l3weights
layers = [l1, l2, l3]

alphas = np.loadtxt(mainpath + "/alphas/" + extension)
betas = np.loadtxt(mainpath + "/betas/" + extension)
mf_agonist = co.Musclefibers(int(outputsize/2), alphas[0:int(outputsize/2)], betas[0:int(outputsize/2)])
mf_antagonist = co.Musclefibers(int(outputsize/2), alphas[int(outputsize/2):int(outputsize)], betas[int(outputsize/2):int(outputsize)])

iterations = 20

sensorcenters = np.linspace(-500, height+500, nsensorneurons)/height
indices = []

hits = []
errors = []

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(0, width), ylim=(0, height))
plt.axis('scaled')
paddleimg = ax.add_patch(patches.Rectangle((1975, 1), 25, 250, color="red"))
ballimg = ax.add_patch(patches.Rectangle((0, 0), 40, 40, color="green"))
p1 = ax.add_patch(patches.Rectangle((100, 0), 10, 1600, color="purple"))
p2 = ax.add_patch(patches.Rectangle((500, 0), 10, 1600, color="purple"))
p3 = ax.add_patch(patches.Rectangle((900, 0), 10, 1600, color="purple"))
vline = ax.vlines(1000, 0, 1600, colors="white", linestyles="dashed")
vline.set_linewidth(4)
ax.set_xticks([])
ax.set_yticks([])
score = ax.set_title(f"HIT:0 | MISS:0 | 100%")
ax.set_facecolor("Black")


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

    sensorsignals = np.zeros(len(sensorcenters))
    sensoryneuronsignals = np.zeros(nsensorneurons*4)

    aghist = np.zeros([int(outputsize/2), int(width*1.25)])
    anhist = np.zeros([int(outputsize/2), int(width*1.25)])

    while not reset:
        t += dt
        # initialize ball and paddle

        ball.move()
        idx, position = tools.passedlaser(previousballposition, ball.position, laserlist)
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

        if t%50 == 0:
            ballimg.set_xy([ball.position[0], ball.position[1]])
            paddleimg.set_xy([1975, paddle.position ])
            plt.draw()

            # --- Record GIF frame ---
            if record_gif:
                fig.canvas.draw()

                # Get ARGB buffer
                buf = fig.canvas.get_renderer().tostring_argb()
                w, h = fig.canvas.get_width_height()

                # Convert to numpy array
                frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))

                # Convert ARGB → RGB (drop alpha)
                frame = frame[:, :, 1:4]

                gif_frames.append(frame)
            # -------------------------

            if sum(hits):
                score.set_text(f"HIT:{sum(hits)} | MISS:{len(hits)-sum(hits)} | {sum(hits)/len(hits)*100:.0f}%")

            if not record_gif:
                if plt.waitforbuttonpress(0.01):
                    quit()
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

if record_gif:
    print(f"Saving GIF: {gif_name}")
    imageio.mimsave(gif_name, gif_frames, fps=30)
    print("GIF saved.")