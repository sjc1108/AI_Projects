import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = sys.argv[1] 

data = pd.read_csv(filename)
years = data["year"].values
days = data["days"].values

plt.plot(years, days)
plt.xlabel("Year")
plt.ylabel("Number of Frozen Days")
plt.title("Lake Mendota: Year vs Number of Frozen Days")
plt.savefig("plot.jpg")

X = np.vstack([np.ones(len(years)), years]).T.astype('int64')
Y = days.reshape(-1, 1).astype('int64')

print("Q3a:")
print(X)

print("Q3b:")
print(Y)

Z = np.dot(X.T, X).astype('int64')
print("Q3c:")
print(Z)

I = np.linalg.inv(Z)
print("Q3d:")
print(I)

PI = np.dot(I, X.T)
print("Q3e:")
print(PI)

beta_hat = np.dot(PI, Y)
print("Q3f:")
print(beta_hat)

x_test = 2022
y_hat_t = beta_hat[0] + beta_hat[1] * x_test
print(f"Q4: {y_hat_t[0]}")

sign = ""
sign = '>' if beta_hat[1] > 0 else '<' if beta_hat[1] < 0 else '='
print("Q5a: " + sign)
print("Q5b: If beta1 sign is positive, it suggests the number of ice days is increasing over years. If negative, it's decreasing. \
      If it is \"=\" and Beta(1) is 0, then the year has no impact on frozen days.")

x_st = -beta_hat[0] / beta_hat[1]
print(f"Q6a: {x_st[0]}")
print("Q6b: This prediction might seem far-fetched, as it's unlikely that Wisconsin's weather will shift \
      dramatically enough to never reach freezing temperatures within a few centuries. However, if we ignore \
      this and solely focus on the data showing a gradual decrease in frozen days, the trend suggests that,\
       at this rate, the number of frozen days could near zero by 2455.")