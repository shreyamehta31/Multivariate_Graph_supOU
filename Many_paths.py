from typing import Optional

import numpy as np
from scipy.stats import gamma

nu = 0.1
a = 3  # shape for U
b = 0.05  # scale for U
B = 0.1  # in A
an = 1.95  # shape for A
mu = nu * (a / (1 / b))  # mean of Levy
var = nu * (a * (1 + a) / (1 / b) ** 2)  # variance of Levy

mean_theoretical = -(mu / B * (an - 1))

variance_theoretical = -(var / 2 * B * (an - 1))


# Function for generating Ui rv
def get_gamma(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.gamma(a, b, s)


Us = {}

for i in range(5):
    Ui = get_gamma(s=2000, random_state=42 + i)  # Ui for positive index
    Us[f"U_{i}"] = Ui

Uns = {}

for i in range(5):
    Uni = get_gamma(s=2000, random_state=50 + i)  # Ui for negative index
    Uns[f"Un_{i}"] = Uni


# print(Us[f"U_{2}"]);

def get_gammaP(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return 0.1 * np.random.gamma(an, 1, s)


As = {}  # Create an empty dictionary to store the values

for i in range(5):
    Ai = -get_gammaP(s=2000, random_state=34 + i)  # Ui for positive index
    As[f"A_{i}"] = Ai

Ans = {}  # Create an empty dictionary to store the values

for i in range(5):
    Ani = -get_gammaP(s=2000, random_state=62 + i)
    Ans[f"An_{i}"] = Ani


# Function for generating Ti rv
def get_exp(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.exponential(nu, s)


Ts = {}  # Create an empty dictionary to store the values

for i in range(5):
    Ti = get_exp(s=2000, random_state=22 + i)  # Ui for positive index
    Ts[f"T_{i}"] = Ti

Tns = {}  # Create an empty dictionary to store the values

for i in range(5):
    Tni = get_exp(s=2000, random_state=72 + i)
    Tns[f"Tn_{i}"] = Tni


# Function for generating tau_i's rv
def get_tau(arr):
    cumulative_array = []
    running_sum = 0
    for num in arr:
        running_sum += num
        cumulative_array.append(running_sum)
    return cumulative_array


taus = {}  # Create an empty dictionary to store the values

for i in range(5):
    taui = get_tau(Ts[f"T_{i}"])
    taus[f"tau_{i}"] = taui

tauns = {}  # Create an empty dictionary to store the values

for i in range(5):
    tauni = get_tau(Tns[f"Tn_{i}"])
    tauns[f"taun_{i}"] = tauni


def get_I1(Z, k: int, Us, As, Ts, taus, i: int) -> np.ndarray:
    U = Us[f"U_{i}"]
    A = As[f"A_{i}"]
    T = Ts[f"T_{i}"]
    tau = taus[f"tau_{i}"]
    Sum = 0
    for j in range(0, k):
        Sum = Sum + np.exp(A[j] * (Z - tau[j])) * U[j]
    return (
        Sum
    )


def get_I2(Z, Uns, Ans, Tns, tauns, i: int) -> np.ndarray:
    Un = Uns[f"Un_{i}"]
    An = Ans[f"An_{i}"]
    Tn = Tns[f"Tn_{i}"]
    taun = tauns[f"taun_{i}"]
    Sum2 = 0
    for j in range(0, 2000):
        Sum2 = Sum2 + np.exp(An[j] * (Z + taun[j])) * Un[j]
    return (
        Sum2
    )


def get_supOU(Z: int, i, Us, As, Ts, taus, Uns, Ans, Tns, tauns):
    tau = taus[f"tau_{i}"]
    index = 0;
    for l in tau:
        if l <= Z:
            index = index + 1
    k = index;

    I1 = get_I1(Z, k, Us, As, Ts, taus, i)
    I2 = get_I2(Z, Uns, Ans, Tns, tauns, i)
    return (
            I1 + I2
    )


t_values = np.arange(110)


def process_supOU(t_values, Us, As, Ts, taus, Uns, Ans, Tns, tauns, i) -> np.ndarray:
    sup_OU = np.zeros_like(t_values,dtype=float)   # Initialize array to store function outputs

    for m, t in enumerate(t_values):
        sup_OU[m] = get_supOU(t, i, Us, As, Ts, taus, Uns, Ans, Tns, tauns)

    return (sup_OU)


supOUs = {}
mean = []
variance = []
num_runs = 5
for run in range(num_runs):
    supOU_run = process_supOU(t_values, Us, As, Ts, taus, Uns, Ans, Tns, tauns, run)
    supOUs[f"supOUs_{run}"] = supOU_run
    mean.append(np.mean(supOU_run))
    variance.append(np.var(supOU_run))

# mean=[]
# variance=[]
# for run in range(num_runs):


import matplotlib.pyplot as plt

num_runs = 5
for run in range(num_runs):
    plt.plot(t_values, supOUs[f"supOUs_{run}"])
plt.xlabel('t')
plt.ylabel('sup_OU')
plt.title('Multiple Runs of sup_OU')
plt.legend(['Run {}'.format(run + 1) for run in range(num_runs)])
plt.show()
