import csv
import numpy as np
from numpy import genfromtxt
import json 
import time
from datetime import datetime
import matplotlib.pyplot as plt

# s = np.random.poisson(5,1000)
# print(s)
# 
# plt.hist(s, 14, normed=True)
# plt.show()


# def lam(s):
#     A = 1.0
#     k = 0.05
#     return A * np.exp(-k * s)
# 
# 
# xs = []
# ys = []
# for s in np.linspace(0,100,1000):
#     xs.append(s)
#     ys.append(lam(s))
# 
# print(xs)
# print(ys)
# 
# plt.plot(xs, ys)
# plt.show()

def get_poloniex_file(filename):
    xs = []
    ys = []
    with open(filename) as data_file:    
      data = json.load(data_file)
      for point in data:
        time = datetime.fromtimestamp( point["date"] )
        price = point["close"]
        xs.append(time)
        ys.append(price)
    return (xs,ys)

(prices_x, prices_y) = get_poloniex_file('btc.json')
# x = x[3000:-1]
# y = y[3000:-1]


data = genfromtxt('BCHAIN-HRATE.csv', delimiter=',')

hashes_x = []
hashes_y = []
with open('BCHAIN-HRATE.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        hashes_x.append(row[0])
        hashes_y.append(row[1])

hashes_x = [datetime.strptime(s, "%Y-%m-%d") for s in hashes_x[1::] ]
hashes_y = [float(s) for s in hashes_y[1::] ]

max_hy = np.max(hashes_y)
max_py = np.max(prices_y)

hashes_y = [y / max_hy for y in hashes_y]
prices_y = [y / max_py for y in prices_y]

hys = []
hxs = []

for (hx,hy) in zip(hashes_x,hashes_y):
    if hx >= prices_x[0]:
        hxs.append(hx)
        hys.append(hy)
hashes_x = hxs
hashes_y = hys

plt.plot(hashes_x,hashes_y)
plt.plot(prices_x,prices_y)
plt.show()
