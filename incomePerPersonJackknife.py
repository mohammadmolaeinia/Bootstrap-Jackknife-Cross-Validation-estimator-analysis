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

#now Jackknife
rng = np.random.default_rng(seed=42)
jKMeans1 = np.empty(n)
jKMeans2 = np.empty(n)
 
for i in range(n):
    jKFSize = np.delete(fSizes, i)
    jKFIncome = np.delete(fIncomes, i)

    jKMeans1[i] = np.mean(jKFIncome / jKFSize)
    jKMeans2[i] = np.mean(jKFIncome) / np.mean(jKFSize)

jKMeanBar1 = np.mean(jKMeans1)
jKMeanBar2 = np.mean(jKMeans2)

jKVar1 = ((n - 1) / n) * np.sum((jKMeans1 - jKMeanBar1)**2)
jKVar2 = ((n - 1) / n) * np.sum((jKMeans2 - jKMeanBar2)**2)

jKbias1 = (n - 1) * (jKMeanBar1 - mean1)
jKbias2 = (n - 1) * (jKMeanBar2 - mean2)

print(f"The Jackknife var of 1st Mean(I/S) : {jKVar1:.4f}")
print(f"The Jackknife var of 2nd Mean(I)/Mean(S) : {jKVar2:.4f}")
print(f"The Jackknife Bias of 1st Mean(I/S) : {jKbias1:.4f}")
print(f"The Jackknife Bias of 2nd Mean(I)/Mean(S) : {jKbias2:.4f}")
