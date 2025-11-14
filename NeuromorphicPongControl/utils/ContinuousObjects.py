import random
import math
import sys
import numpy as np



class Ball:
    def __init__(self):
        self.speed = [0, 0] # x speed, y speed
        self.position = [0, 0]
        self.positionHistory = []
        self.prediction = 0
        self.size = 1

    def move(self):

        self.position[0] += self.speed[0]
        self.position[1] += self.speed[1]

    def bounce(self, env, paddle=0):
        # Ball hit the back
        if self.position[0] + int(self.size/2) > (env.width - 1):
            self.speed[0] = -self.speed[0]
            self.position[0] = 2 * (env.width - 1) - self.position[0] - 2*int(self.size/2)

        # Ball hit the top
        if self.position[1] + int(self.size/2) > (env.height - 1):
            self.speed[1] = -self.speed[1]
            self.position[1] = 2*(env.height - 1) - self.position[1] - 2*int(self.size/2)

        # Ball hit the bottom
        if self.position[1] - int(self.size/2) < 0:
            self.speed[1] = - self.speed[1]
            self.position[1] = - self.position[1] + 2*int(self.size/2)

        # Ball hit the front, changes direction always
        if self.position[0] - int(self.size/2) < 0:
            self.speed[0] = -self.speed[0]
            self.position[0] = -self.position[0] + 2*int(self.size/2)
            if self.speed[1] == 0:
                self.speed[1] = random.choices([-1, 1])[0]

            self.speed[1] = math.copysign(1, self.speed[1]) * random.uniform(0, 1)
            if paddle:
                if paddle.position + paddle.height > self.position[1]-self.size/2 and\
                    self.position[1] + self.size/2 >= paddle.position:
                    env.hits += 1
                else:
                    env.misses += 1
        self.position = self.position


class Paddle:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.position = 0
        self.speed = 0
        self.mass = 0
        self.acceleration = 0

    def move(self, quantity, dt, limit, controlled = "force"):
        blocked = 0 # Hits top or bottom edge
        if controlled == "position":
            self.acceleration = 0
            self.speed = 0
            self.position = min(limit-self.height, max(quantity, 0))

        elif controlled == "force":
            self.acceleration = quantity/self.mass
            self.speed = self.speed + self.acceleration*dt
            self.position = min(limit - self.height, max(0, self.position + self.speed*dt))
        elif controlled == "velocity":
            self.speed = quantity
            dx = self.speed * dt
            newposition = self.position + dx
            if newposition < 0 or newposition > self.position + self.height:
                pass
            else:
                self.position = newposition
        else:
            sys.exit(f"{controlled} control is not implemented")

    def reset(self):
        self.position = 0
        self.speed = 0


class Layer:
    def __init__(self, ninputneurons, noutputneurons, noutputs, layertype, thresholds, div, elig):
        self.ninputneurons = ninputneurons
        self.thresholds = thresholds
        self.weightmatrix = np.random.uniform(-1, 1, [ninputneurons, noutputneurons])
        self.feedbackweights = np.random.uniform(-1, 1, [noutputneurons, noutputs])

        self.layertype = layertype
        self.noutputneurons = noutputneurons
        self.V = np.zeros(self.noutputneurons)
        self.sumofspikes = np.zeros(self.noutputneurons)
        self.activityhistory = np.zeros(self.noutputneurons)
        self.inputhistory = np.zeros(self.ninputneurons)
        self.div = div
        self.elig = elig
        self.eligmat = np.ones(self.noutputneurons)*elig

        if self.layertype != "input":

            self.zeros = np.ones(self.weightmatrix.shape)
            self.zeros_arr = np.ones(self.noutputneurons)

        self.resetones = np.ones(noutputneurons)

    def update_neurons(self, inputs, threshold_previouslayer=None, prob=0):

        if self.layertype == "input":
            # Implement like in fba vanilla
            if prob:
                inputs_weighted = self.thresholds * (np.random.uniform(0, 1, inputs.shape) < (inputs / self.thresholds))
                self.V += inputs_weighted


            else:
                inputs_weighted = inputs
                self.V += inputs_weighted
            self.activityhistory += inputs_weighted
            self.V = np.maximum(0, self.V)
            # thresholds crossings spike and reset neuroncharges
            outputspikes = self.V >= self.thresholds

        else:
            # Neuron membrane potential
            self.inputhistory += np.divide(inputs, threshold_previouslayer)
            inputs_weighted = np.dot(inputs, self.weightmatrix)
            self.V += inputs_weighted
            self.V = np.maximum(0, self.V)
            self.activityhistory += inputs_weighted
            outputspikes = self.V >= self.thresholds
            # thresholds crossings spike and reset neuroncharges


        # Reset by reversing outputspikes
        self.sumofspikes += outputspikes
        self.sumofspikes *= self.eligmat
        self.activityhistory *= self.eligmat
        resets = np.abs(outputspikes - self.resetones)
        self.V *= resets

        return outputspikes

    def update_weights(self, individualerror, lr, inputspikerate=None, previousactivation=None, limit=0):
        self.activityhistory_scaled = self.activityhistory/self.div
        activation_negativeexponent = np.exp(-self.activityhistory_scaled)
        activation_sigmoid = 1/(1+activation_negativeexponent)
        activation_sigmoid_deriv = (activation_sigmoid) * ( 1 - activation_sigmoid )

        if self.layertype == "output":
            dw = lr*np.outer(individualerror.T, previousactivation.T)
            self.weightmatrix += dw.T

        elif self.layertype == "hidden":
            da = np.matmul(self.feedbackweights, individualerror)*activation_sigmoid_deriv

            if inputspikerate is not None:
                dw = lr*-np.outer(da, inputspikerate)

            else:
                dw = lr*-np.outer(da, np.transpose(self.inputhistory))
            self.weightmatrix += dw.T

        else:

            print(f"You are trying to update a {self.layertype} layer")
            pass
        if limit:
            self.weightmatrix = np.maximum(np.minimum(self.weightmatrix, 1), -1)

    def reset(self):

        self.V = np.zeros(self.noutputneurons)
        self.sumofspikes = np.zeros(self.noutputneurons)
        self.activityhistory = np.zeros(self.noutputneurons)
        self.inputhistory = np.zeros(self.ninputneurons)


class Musclefibers:
    def __init__(self, numberoffibers, alphas, betas):
        self.fiberreset = np.zeros(numberoffibers)
        self.fibers_c1 = np.zeros(numberoffibers)
        self.fibers_c2 = np.zeros(numberoffibers)
        self.alphas = alphas
        self.betas = betas

    def update(self, inputs, amplitudes):

        self.fibers_c1 = self.fibers_c1 * self.alphas + (1-self.alphas)*inputs*amplitudes
        self.fibers_c2 = self.fibers_c2 * self.betas + (1-self.betas) * self.fibers_c1

    def reset(self):
        self.fibers_c1 = self.fiberreset
        self.fibers_c2 = self.fiberreset




