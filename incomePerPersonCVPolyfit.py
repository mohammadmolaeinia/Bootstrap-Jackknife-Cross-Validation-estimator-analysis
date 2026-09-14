import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

#reading the income file and calclute The Means
data = np.loadtxt('income.txt', skiprows=1)

fSizes = data[:, 0]
fIncomes =  data[:, 1]
n = len(data)

#fit and predict
def fitAndPredict(x, y, xTest, degree):
    coefficents = np.polyfit(x, y, degree)

    yPred = np.polyval(coefficents, xTest)
    return yPred

#now Cross Validation
degrees = [1, 2, 3, 4, 5]
cvErrors = []

for d in degrees:
    squarErrors = []

    for i in range(n):
        xTest = fSizes[i]
        yTest = fIncomes[i]

        x = np.delete(fSizes, i)
        y = np.delete(fIncomes, i)

        yPred = fitAndPredict(x, y, xTest, d)

        error = (yTest - yPred) ** 2
        squarErrors.append(error)

    cvErrorsMean = np.mean(squarErrors)
    cvErrors.append(cvErrorsMean)
    print(f"deg {d}: MSE: {cvErrorsMean:.4f}")
    

bestIndex = np.argmin(cvErrors)
bestDegree = degrees[bestIndex]
bestDegreeCVErrors = cvErrors[bestIndex]

print(f"The best Degree is : {bestDegree:.4f} and the CV Error is : {bestDegreeCVErrors:.4f} ")

finalCoefficents = np.polyfit(fSizes, fIncomes, bestDegree)

xCurve = np.linspace(np.min(fSizes), np.max(fSizes), 100)
yCurve = np.polyval(finalCoefficents, xCurve)

#making figure of PDF
chartErrors, ax3 = plt.subplots(figsize=(10, 6))
ax3.scatter(fSizes, fIncomes, color='blue', alpha=1, s=1, label='data point')
ax3.plot(xCurve, yCurve, color='red', linewidth=1, linestyle='-', label='fitted line')
ax3.set_xlabel('Family Size')
ax3.set_ylabel('Family Income')
ax3.xaxis.set_tick_params(labelsize=6.5)
ax3.yaxis.set_tick_params(labelsize=6.5)
plt.title('Income vs. Size and Fitted line')
ax3.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
chartErrors.tight_layout()

plt.show()
