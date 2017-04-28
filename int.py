import csv
import numpy as np
from numpy import genfromtxt
import json 
import time
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.misc import derivative
from scipy.optimize import brentq

#GLD GDX
theta = 0.5388
mu = 16.6677
sigma = 0.1599

r = 0.05
c = 0.05

def SymDiff(f,x,h):
    return (f(x+h) - f(x-h)) / (2*h)

def F(x):
    def f(u,x):
        return u**(r/mu - 1.0) * np.exp( np.sqrt( (2.0*mu / (sigma**2.0) ) ) * (x - theta) * u - (u**2.0)/2.0 )
    return integrate.quad(f, 0.0, np.inf, args=(x,))[0]

def Fprime(x):
    return SymDiff(F,x,10e-6)

def G(x):
    def g(u,x):
        return u**(r/mu - 1.0) * np.exp( np.sqrt( (2.0*mu / (sigma**2.0) ) ) * (theta - x) * u - (u**2.0)/2.0 )
    return integrate.quad(g, 0.0, np.inf, args=(x,))[0]

def Gprime(x):
    return SymDiff(G,x,10e-6)

def Psi(x):
    return F(x) / G(x)

bstar = None
def CalcExitLevel():
    global bstar
    if not (bstar is None):
        return bstar
    f = lambda b: F(b) - (b-c)*Fprime(b)
    bstar = brentq(f, 0.001, 0.999)
    return bstar

def V(x):
    b = CalcExitLevel()
    if x < b: return (b-c)*F(x)/F(b)
    return x-c

def Vprime(x):
    return SymDiff(V,x,10e-6)

def CalcEntryLevel():
    b = CalcExitLevel()
    f = lambda d: G(d)*(Vprime(d)-1) - Gprime(d)*(V(d)-d-c)
    return brentq(f, 0.0001, b)


print("Test: " + str(F(0.5)) )
print("FTest: " + str(Fprime(0.2)) )
print("GTest: " + str(Gprime(0.2)) )

print("b test: " + str(CalcExitLevel() ))
print("d test: " + str(CalcEntryLevel() ))


xs = np.linspace(0,0.6,100)
ys = [F(x) for x in xs]

plt.plot(xs,ys)
plt.show()
