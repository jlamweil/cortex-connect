from numpy.random import binomial
from random import choice
from numpy import array, dot, random
import numpy as np


#probability 0< f <= 0.5
f = 0.5

#number of neurons
N = 30

#Decision boundary (arbitrary and held constant)
T = 5

#Robustness
#Either defined with K, or with rho

K = T/10
wAverage = T/(f*N)
# for different rescaled robustness parameters:
rho =K / (wAverage * np.sqrt(f*(1-f)*N))
"""
wAverage = T/(f*N)
rho = 0.
K = rho * wAverage * np.sqrt(f*(1-f)*N)
"""


#number of random patterns
#http://www.cell.com/cms/attachment/572560/4239133/mmc1.pdf
#the choice of alpha_c = H(B)/[f'H(tauMinus) + (1-f')H(tauPlus)]
#Like in Gutfreund and Stein (1990), we suppose f'=f


def generate_data():
    eta = binomial(1,f,(1,N))
    return(eta)

def testLearn(training_data, w):
    for mu in range(training_data.shape[0]):
        for i in range(training_data.shape[1]):
            x = training_data[mu,np.arange(training_data.shape[1])!=i]
            wi = w[i,np.arange(training_data.shape[1])!=i]
            expected = training_data[mu,i]
            result = dot(x, wi)
            if unit_step(result) != expected:
                return False
    return True

#Network storing fixed-point attractors

# https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
# http://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/


# Unit_step does not include K, the robustness.
# A preprocessing step will ignore data too close to the boundary.
unit_step = lambda x: 0 if x < T else 1

training_data = generate_data()

w = random.uniform(0,2*T/(f*N),(N,N))
l_rate = 0.01*T

while l_rate > T * 1e-6:
    iteration = 0
    converged = testLearn(training_data, w)
    while (not converged) and (iteration < 1e6):
        mu = random.randint(training_data.shape[0])
        for i in range(training_data.shape[1]):
            x = training_data[mu,np.arange(training_data.shape[1])!=i]
            wi = w[i,np.arange(training_data.shape[1])!=i]
            expected = training_data[mu,i]
            result = dot(wi, x)
            if (2*expected - 1)*(result - T) <= K:
                wi += l_rate * (2*expected - 1) * x
                #Non-negativity constraint
                w[i,np.arange(training_data.shape[1])!=i] = (wi>0) * wi

        converged = testLearn(training_data, w)
        iteration += 1
    if converged:
        print("converged")
        training_data = np.vstack((training_data,generate_data()))
    else:
        print("not")
        l_rate /= 2
    print(training_data.shape[0])