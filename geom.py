import csv
import numpy as np
from numpy import genfromtxt
import json 
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Brownian Motion

def GeometricBrownianMotion(start_time, end_time, time_increment, start_price, mu, sigma):
    ts = []
    Bs = [0.0]
    # 0 to 100s with 100 points
    ts = np.arange(start_time, end_time, time_increment)
    delta = ts[1] - ts[0] # The period
    
    for t in ts[1::]:
        variance = delta
        n = np.random.normal(0, np.sqrt(variance))
        Bs.append(Bs[-1] + n)
    
    # Geometric Brownian Motion
    St = [start_price]
    for i in range(1,len(ts)):
        Sti = St[0] * np.exp( (mu - (sigma**2.0)/2.0 ) * ts[i] + sigma*Bs[i] )
        St.append(Sti)
    
    return (ts, St)
   

(ts, St) = GeometricBrownianMotion(0.0, 1.0, 0.001, 1.0, 2.0, 1.0)
plt.plot(ts, St)
plt.show()

