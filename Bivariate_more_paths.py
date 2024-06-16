import numpy as np
from scipy.stats import gamma
from numpy.random import poisson, gamma, uniform
from typing import Optional

import time

time_taken = time.time()

nu = 10  # rate
a = 3  # shape for U
b = 0.05  # scale for U
c = 0.5
Adj = np.array([[0, 1],
                [1, 0]])  # in K

cAdj = c * Adj
I2 = np.identity(2)
cAdj += I2
k = -cAdj

eigenvalues, eigenvectors = np.linalg.eig(k)

lambda1 = eigenvalues[0]
lambda2 = eigenvalues[1]

O = np.linalg.inv(eigenvectors)

an = 1.95  # shape for A
mu = (nu) * (a / (1 / b))  # mean of Levy
var = (nu) * ((a * (a + 1) / (1 / b) ** 2))  # variance of Levy

tv = 1000
t_values = np.arange(tv)
num_paths = 200
sup_OUb_paths = np.zeros((num_paths, tv, 2, 1), dtype=float)


# Function for generating gamma distributed random variables
def get_gamma(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.gamma(a, b, s)


# Function for generating gammaP distributed random variables
def get_gammaP(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.gamma(an, 1, s)


# Function for generating exponential random variables
def get_exp(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.exponential(1 / nu, s)  # scale=1/rate


# Function for generating cumulative sum (tau_i's)
def get_tau(arr):
    cumulative_array = []
    running_sum = 0
    for num in arr:
        running_sum += num
        cumulative_array.append(running_sum)
    return cumulative_array


for i in range(num_paths):
    U1 = get_gamma(s=20000, random_state=42 + i)  # Ui for positive index for 1 Poisson
    Un1 = get_gamma(s=20000, random_state=43 + i)  # Ui for negative index for 1 Poisson

    U2 = get_gamma(s=20000, random_state=60 + i)  # Ui for positive index for 2 Poisson
    Un2 = get_gamma(s=20000, random_state=61 + i)  # Ui for negative index for 2 Poisson

    Ufj = get_gamma(s=20000, random_state=62 + i)  # Ui for positive index for || Poisson first component
    Unfj = get_gamma(s=20000, random_state=63 + i)  # Ui for negative index for || Poisson first component

    Usj = get_gamma(s=20000, random_state=64 + i)  # Ui for positive index for || Poisson second component
    Unsj = get_gamma(s=20000, random_state=65 + i)  # Ui for negative index for || Poisson second component

    theta1 = get_gammaP(s=20000, random_state=45 + i)  # Ai for positive index for 1 Poisson
    thetan1 = get_gammaP(s=20000, random_state=44 + i)  # Ai for negative index for 1 Poisson

    theta2 = get_gammaP(s=20000, random_state=66 + i)  # Ai for positive index for 2 Poisson
    thetan2 = get_gammaP(s=20000, random_state=67 + i)  # Ai for negative index for 2 Poisson

    thetaj = get_gammaP(s=20000, random_state=68 + i)  # Ai for positive index for || Poisson
    thetanj = get_gammaP(s=20000, random_state=69 + i)  # Ai for negative index for || Poisson

    T1 = get_exp(s=20000, random_state=46 + i)  # Ti for positive index for 1 Poisson
    Tn1 = get_exp(s=20000, random_state=47 + i)  # Ti for negative index for 1 Poisson

    T2 = get_exp(s=20000, random_state=70 + i)  # Ti for positive index for 2 Poisson
    Tn2 = get_exp(s=20000, random_state=71 + i)  # Ti for negative index for 2 Poisson

    Tj = get_exp(s=20000, random_state=72 + i)  # Ti for positive index for || Poisson
    Tnj = get_exp(s=20000, random_state=73 + i)  # Ti for negative index for || Poisson

    tau1 = get_tau(T1)  # tau_i for positive index for 1 Poisson
    taun1 = get_tau(Tn1)  # tau_i for negative index for 1 Poisson

    tau2 = get_tau(T2)  # tau_i for positive index for 2 Poisson
    taun2 = get_tau(Tn2)  # tau_i for negative index for 2 Poisson

    tauj = get_tau(Tj)  # tau_i for positive index for || Poisson
    taunj = get_tau(Tnj)  # tau_i for negative index for || Poisson

    # Generate compound Levy process
    compound_levy = []
    for j in range(len(tau1)):
        if j == 0:
            compound_levy.append(np.sum(U1[:int(tau1[j])]))
        else:
            compound_levy.append(np.sum(U1[int(tau1[j - 1]):int(tau1[j])]))

    compound_levy = np.array(compound_levy)

    # Calculate empirical mean and variance of the compound Levy process
    empirical_mean_levy = np.mean(compound_levy)
    empirical_variance_levy = np.var(compound_levy)


    # Functions for calculating I1, I2, I3, I4, I5, I6
    def get_I1(Z: np.ndarray, k: int) -> np.ndarray:
        Sum = np.array([0, 0])
        for j in range(k):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * thetaj[j] * (Z - tauj[j])), 0],
                                                        [0, np.exp(lambda2 * thetaj[j] * (Z - tauj[j]))]]) @ O,
                               np.array([Ufj[j], Usj[j]]))
        return Sum


    def get_I2(Z: np.ndarray, k: int) -> np.ndarray:
        Sum = np.array([0, 0])
        for j in range(k):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * theta1[j] * (Z - tau1[j])), 0],
                                                        [0, np.exp(lambda2 * theta1[j] * (Z - tau1[j]))]]) @ O,
                               np.array([U1[j], 0]))
        return Sum


    def get_I3(Z: np.ndarray, k: int) -> np.ndarray:
        Sum = np.array([0, 0])
        for j in range(k):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * theta2[j] * (Z - tau2[j])), 0],
                                                        [0, np.exp(lambda2 * theta2[j] * (Z - tau2[j]))]]) @ O,
                               np.array([0, U2[j]]))
        return Sum


    def get_I4(Z: np.ndarray) -> np.ndarray:
        Sum = np.array([0, 0])
        for j in range(200):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * thetanj[j] * (Z - taunj[j])), 0],
                                                        [0, np.exp(lambda2 * thetanj[j] * (Z - taunj[j]))]]) @ O,
                               np.array([Unfj[j], Unsj[j]]))
        return Sum


    def get_I5(Z: np.ndarray) -> np.ndarray:
        Sum = np.array([0, 0])
        for j in range(200):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * thetan1[j] * (Z - taun1[j])), 0],
                                                        [0, np.exp(lambda2 * thetan1[j] * (Z - taun1[j]))]]) @ O,
                               np.array([Un1[j], 0]))
        return Sum


    def get_I6(Z: np.ndarray) -> np.ndarray:
        Sum = np.array([0, 0])
        for j in range(200):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * thetan2[j] * (Z - taun2[j])), 0],
                                                        [0, np.exp(lambda2 * thetan2[j] * (Z - taun2[j]))]]) @ O,
                               np.array([0, Un2[j]]))
        return Sum


    def get_supOU(Z: int, tau1: np.ndarray, tau2: np.ndarray, tauj: np.ndarray) -> np.ndarray:
        index1 = 0
        for l in tau1:
            if l <= Z:
                index1 += 1
        k1 = index1

        index2 = 0
        for l in tau2:
            if l <= Z:
                index2 += 1
        k2 = index2

        indexj = 0
        for l in tauj:
            if l <= Z:
                indexj += 1
        kj = indexj

        I1 = get_I1(Z, kj)
        I2 = get_I2(Z, k1)
        I3 = get_I3(Z, k2)
        I4 = get_I4(Z)
        I5 = get_I5(Z)
        I6 = get_I6(Z)

        return np.expand_dims(I1 + I2 + I3 + I4 + I5 + I6, axis=1)


    supOUb = np.zeros((tv, 2, 1), dtype=float)
    for t in range(tv):
        supOUbv = get_supOU(t, tau1, tau2, tauj)
        supOUb[t] = supOUbv
    sup_OUb_paths[i, :] = supOUb

# Extract x and y components for all vectors at all time steps
x_component_values = [[] for _ in range(num_paths)]
y_component_values = [[] for _ in range(num_paths)]
meanx = np.zeros(num_paths)
meany = np.zeros(num_paths)
variancex = [np.zeros((2, 2)) for _ in range(num_paths)]
variancey = [np.zeros((2, 2)) for _ in range(num_paths)]

for i in range(num_paths):
    for t in t_values:
        vector = sup_OUb_paths[i, t].flatten()
        x_component = vector[0]
        y_component = vector[1]

        x_component_values[i].append(x_component)
        y_component_values[i].append(y_component)

    # Select data for time range t=25 to t=1000
    start_time = 25
    end_time = 1000
    selected_t_values = t_values[start_time:end_time]
    selected_x_components = np.array(x_component_values[i][start_time:end_time])
    selected_y_components = np.array(y_component_values[i][start_time:end_time])

    meanx[i] = np.mean(selected_x_components)
    meany[i] = np.mean(selected_y_components)
    variancex[i] = np.var(selected_x_components)
    variancey[i] = np.var(selected_y_components)

    print("Mean of x component of ", i, "th path:", meanx[i])
    print("Mean of y component of ", i, "th path:", meany[i])
    print("Variance of x component of ", i, "th path:", variancex[i])
    print("Variance  of y component of ", i, "th path:", variancey[i])

#  import matplotlib.pyplot as plt

# Uncomment the following sections to add more functionality like plotting and saving results
'''  def autocorr2(x, lags):
        mean = np.mean(x)
        var = np.var(x)
        xp = x - mean
        corr = [1. if l == 0 else np.sum(xp[l:] * xp[:-l]) / len(x) / var for l in lags]
        return np.array(corr)

    lags = np.arange(0, 100)

    autocorrx = autocorr2(selected_x_components[i], lags)
    autocorry = autocorr2(selected_y_components[i], lags)

#theoretical
D1_values = [[] for i in range(num_paths)]
D2_values = [[] for i in range(num_paths)]
b=np.array( [[ 0.3419,-0.0526 ],
      [ -0.0526,0.3419]] )'''

'''   for h in lags:
        a = np.array([[1 + h, 0.5 * h],
                  [0.5 * h, 1 + h]])
        C=np.dot(scipy.linalg.fractional_matrix_power(a, -0.95),b)
        D=np.diag(C)

        D1_values[i].append(D[0])
        D2_values[i].append(D[1])'''
'''
for i in range(num_paths):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    title = "Bivariate supOU "
    axes[0].plot(selected_t_values, selected_x_components[i], label='X Component')
    axes[0].set_title("X Component", fontsize=15)
    axes[1].plot(selected_t_values, selected_y_components[i], label='Y Component')
    axes[1].set_title("Y Component", fontsize=15)

    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
plt.show()

for i in range(num_paths):
    plt.stem(lags, autocorrx, use_line_collection=True)
    plt.plot(lags, D1_values[i], color='black', label='Diagonal Element 1')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation of x component')
    plt.title('Autocorrelation Function of X component')
plt.show(block=True)

for i in range(num_paths):
    plt.stem(lags, autocorry, use_line_collection=True)
    plt.plot(lags, D2_values[i], color='black', label='Diagonal Element 2')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation of y component')
    plt.title('Autocorrelation Function of Y component')
    plt.show(block=True)
'''
np.savez('Bivariatesup_OU_more(nu=10,a=3,b=0.05,B=-0.1,an=1.95).npz', selected_x_components=selected_x_components,
         selected_y_components=selected_y_components)

print("My program took", time.time() - time_taken, "to run")