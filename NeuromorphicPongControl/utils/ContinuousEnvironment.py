import numpy as np

class Environment:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.grid = None
        self.hits = 0
        self.misses = 0

    def fillgrid(self, ball, paddle, l1p,l2p, l3p):
        ballposition = ball.position
        ballsize = ball.size
        paddleposition = paddle.position_raw
        paddleheight = paddle.height
        paddlewidth = paddle.width
        self.grid = np.zeros((self.height, self.width))

        # Draw ball
        for i in range(ballsize):
            for j in range(ballsize):
                xpos = ballposition[0] - ballsize/2 + i
                ypos = ballposition[1] - ballsize/2 + j
                self.grid[min(max(0, int(ypos)), self.height - 1),
                          min(max(0,int(xpos)), self.width - 1)] = 1

        # Draw paddle
        for n in range(paddleheight):
            for m in range(paddlewidth):

                verticalposition = int(paddleposition) + n
                self.grid[verticalposition, m] = 1

        # Draw lasers
        for l in range(self.height):
            self.grid[l, l1p] = 1
            self.grid[l, l2p] = 1
            self.grid[l, l3p] = 1


    def spike(self, previousposition, newposition,  laser1, laser2, previoustime, currenttime):

        # Interpolate laser 2 crossing position and time
        if previousposition[0] <= laser2.position < newposition[0] or \
                newposition[0] < laser2.position <= previousposition[0]:
            laser2.toggle = 1
            rc = (previousposition[1] - newposition[1])/(previousposition[0] - newposition[0])
            laser2.yvalue = rc*(laser2.position - newposition[0]) + newposition[1]
            timefraction = (previousposition[0] - laser2.position)/(previousposition[0] - newposition[0])
            laser2.crossingtime = previoustime + (timefraction*(abs(currenttime-previoustime)))

        # Interpolate laser 1 crossing position
        if (previousposition[0] <= laser1.position < newposition[0] or
            newposition[0] < laser1.position <= previousposition[0]) and\
                laser2.toggle:

            rc = (previousposition[1] - newposition[1])/(previousposition[0] - newposition[0])
            laser1.yvalue = rc*(laser1.position - newposition[0]) + newposition[1]
            laser1.toggle = 1
            timefraction = (previousposition[0] - laser1.position)/(previousposition[0] - newposition[0])
            laser1.crossingtime = previoustime + (timefraction*(abs(currenttime-previoustime)))

        if laser1.crossingtime and laser2.crossingtime and laser1.toggle and laser2.toggle:
            dt = abs(laser1.crossingtime-laser2.crossingtime)
        else:
            dt = 0
        return dt

    def reset(self):
        self.hits = 0
        self.misses = 0
