import numpy as np
import math

#reading the income file and calclute The Means
data = np.loadtxt('income.txt', skiprows=1)

fSizes = data[:, 0]
fIncomes =  data[:, 1]
n = len(data)

mean1 = np.mean(fIncomes/fSizes)
mean2 = np.mean(fIncomes) / np.mean(fSizes)

print(f"The 1st Mean(I/S) : {mean1:.4f}")
print(f"The 2nd Mean(I)/Mean(S) : {mean2:.4f}")

#now Bootstrap
rng = np.random.default_rng(seed=42)
bootN = 100000
bootMeans1 = np.empty(bootN)
bootMeans2 = np.empty(bootN)
 
for i in range(bootN):
    bootFSize = rng.choice(fSizes, size=n, replace=True)
    bootFIncome = rng.choice(fIncomes, size=n, replace=True)

    bootMeans1[i] = np.mean(bootFIncome / bootFSize)
    bootMeans2[i] = np.mean(bootFIncome) / np.mean(bootFSize)

varOfMeans1 = np.var(bootMeans1, ddof=1)
varOfMeans2 = np.var(bootMeans2, ddof=1)

print(f"The Bootstrap var of 1st Mean(I/S) : {varOfMeans1:.4f}")
print(f"The Bootstrap var of 2nd Mean(I)/Mean(S) : {varOfMeans2:.4f}")
print(f"so The 2nd one is better Becuse have smaller var")
