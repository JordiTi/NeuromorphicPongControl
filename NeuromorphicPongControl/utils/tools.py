import numpy as np
import os
from random import randint
import scipy
import random

def passedlaser(ballposition_previous, ballposition_current, laserpositions):

    passedidx = 0
    passingposition = None
    for lasernumber, laserposition in enumerate(laserpositions):
        if ballposition_previous[0] < laserposition <= ballposition_current[0]:
            passingposition = ballposition_previous[1] + \
                       (laserposition - ballposition_previous[0])/(ballposition_current[0] - ballposition_previous[0]) * \
                       (ballposition_current[1] - ballposition_previous[1])

            if passingposition == 0:
                passingposition = 0.001
            passedidx = lasernumber
            break
    return passedidx, passingposition

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


