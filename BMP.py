from typing import Optional
import numpy as np
from numpy.random import gamma, exponential
import scipy.linalg

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

tv = 1000
t_values = np.arange(tv)
num_paths = 5
sup_OUb_paths = np.zeros((num_paths, tv, 2, 1), dtype=float)

for i in range(num_paths):

    def get_gamma(s: int, random_state: Optional[int] = None) -> np.ndarray:
        np.random.seed(random_state)
        return np.random.gamma(a, b, s)


    U1 = get_gamma(s=20000, random_state=42 + i)  # Ui for positive index for 1 Poisson
    Un1 = get_gamma(s=20000, random_state=43 + i)  # Ui for negative index for 1 Poisson

    U2 = get_gamma(s=20000, random_state=60 + i)  # Ui for positive index for 2 Poisson
    Un2 = get_gamma(s=20000, random_state=61 + i)  # Ui for negative index for 2 Poisson

    Ufj = get_gamma(s=20000, random_state=62 + i)  # Ui for positive index for || Poisson first component
    Unfj = get_gamma(s=20000, random_state=63 + i)  # Ui for negative index for || Poisson first component

    Usj = get_gamma(s=20000, random_state=64 + i)  # Ui for positive index for || Poisson second component
    Unsj = get_gamma(s=20000, random_state=65 + i)  # Ui for negative index for || Poisson second component


    def get_gammaP(s: int, random_state: Optional[int] = None) -> np.ndarray:
        np.random.seed(random_state)
        return np.random.gamma(an, 1, s)


    theta1 = get_gammaP(s=20000, random_state=45 + i)  # Ai for positive index for 1 Poisson
    thetan1 = get_gammaP(s=20000, random_state=44 + i)  # Ai for negative index for 1 Poisson

    theta2 = get_gammaP(s=20000, random_state=66 + i)  # Ai for positive index for 2 Poisson
    thetan2 = get_gammaP(s=20000, random_state=67 + i)  # Ai for negative index for 2 Poisson

    thetaj = get_gammaP(s=20000, random_state=68 + i)  # Ai for positive index for || Poisson
    thetanj = get_gammaP(s=20000, random_state=69 + i)  # Ai for negative index for || Poisson


    def get_exp(s: int, random_state: Optional[int] = None) -> np.ndarray:
        np.random.seed(random_state)
        return np.random.exponential(1 / nu, s)  # scale=1/rate


    T1 = get_exp(s=20000, random_state=46 + i)  # Ti for positive index for 1 Poisson
    Tn1 = get_exp(s=20000, random_state=47 + i)  # Ti for negative index for 1 Poisson

    T2 = get_exp(s=20000, random_state=70 + i)  # Ti for positive index for 2 Poisson
    Tn2 = get_exp(s=20000, random_state=71 + i)  # Ti for negative index for 2 Poisson

    Tj = get_exp(s=20000, random_state=72 + i)  # Ti for positive index for || Poisson
    Tnj = get_exp(s=20000, random_state=73 + i)  # Ti for negative index for || Poisson


    def get_tau(arr):
        cumulative_array = []
        running_sum = 0
        for num in arr:
            running_sum += num
            cumulative_array.append(running_sum)
        return cumulative_array


    tau1 = get_tau(T1)  # tau_i for positive index for 1 Poisson
    taun1 = get_tau(Tn1)  # tau_i for negative index for 1 Poisson

    tau2 = get_tau(T2)  # tau_i for positive index for 2 Poisson
    taun2 = get_tau(Tn2)  # tau_i for negative index for 2 Poisson

    tauj = get_tau(Tj)  # tau_i for positive index for || Poisson
    taunj = get_tau(Tnj)  # tau_i for negative index for || Poisson

    compound_levy = []
    for i in range(len(tau1)):
        if i == 0:
            compound_levy.append(np.sum(U1[:int(tau1[i])]))
        else:
            compound_levy.append(np.sum(U1[int(tau1[i - 1]):int(tau1[i])]))

    compound_levy = np.array(compound_levy)

    empirical_mean_levy = np.mean(compound_levy)
    empirical_variance_levy = np.var(compound_levy)

    def get_I1(Z: np.ndarray, k: int) -> np.ndarray:
        Sum = np.array([0, 0])
        for i in range(0, k):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * thetaj[i] * (Z - tauj[i])), 0],
                                                        [0, np.exp(lambda2 * thetaj[i] * (Z - tauj[i]))]]) @ O,
                               np.array([Ufj[i], Usj[i]]))
        return Sum

    def get_I2(Z: np.ndarray, k: int) -> np.ndarray:
        Sum = np.array([0, 0])
        for i in range(0, k):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * theta1[i] * (Z - tau1[i])), 0],
                                                        [0, np.exp(lambda2 * theta1[i] * (Z - tau1[i]))]]) @ O,
                               np.array([U1[i], 0]))
        return Sum

    def get_I3(Z: np.ndarray, k: int) -> np.ndarray:
        Sum = np.array([0, 0])
        for i in range(0, k):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * theta2[i] * (Z - tau2[i])), 0],
                                                        [0, np.exp(lambda2 * theta2[i] * (Z - tau2[i]))]]) @ O,
                               np.array([0, U2[i]]))
        return Sum

    def get_I4(Z: np.ndarray) -> np.ndarray:
        Sum = np.array([0, 0])
        for i in range(0, 200):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * thetanj[i] * (Z - taunj[i])), 0],
                                                        [0, np.exp(lambda2 * thetanj[i] * (Z - taunj[i]))]]) @ O,
                               np.array([Unfj[i], Unsj[i]]))
        return Sum

    def get_I5(Z: np.ndarray) -> np.ndarray:
        Sum = np.array([0, 0])
        for i in range(0, 200):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * thetan1[i] * (Z - taun1[i])), 0],
                                                        [0, np.exp(lambda2 * thetan1[i] * (Z - taun1[i]))]]) @ O,
                               np.array([Un1[i], 0]))
        return Sum

    def get_I6(Z: np.ndarray) -> np.ndarray:
        Sum = np.array([0, 0])
        for i in range(0, 200):
            Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * thetan2[i] * (Z - taun2[i])), 0],
                                                        [0, np.exp(lambda2 * thetan2[i] * (Z - taun2[i]))]]) @ O,
                               np.array([0, Un2[i]]))
        return Sum

    def get_supOU(Z: int, tau1: np.ndarray, tau2: np.ndarray, tauj: np.ndarray) -> np.ndarray:
        index1 = 0
        for l in tau1:
            if l <= Z:
                index1 = index1 + 1
        k1 = index1
        index2 = 0
        for l in tau2:
            if l <= Z:
                index2 = index2 + 1
        k2 = index2
        indexj = 0
        for l in tauj:
            if l <= Z:
                indexj = indexj + 1
        kj = indexj
        I1 = get_I1(Z, kj)
        I2 = get_I2(Z, k1)
        I3 = get_I3(Z, k2)
        I4 = get_I4(Z)
        I5 = get_I5(Z)
        I6 = get_I6(Z)
        return np.expand_dims(I1 + I2 + I3 + I4 + I5 + I6, axis=1)

    supOUb = np.zeros((tv, 2, 1), dtype=float)

    for index, t in enumerate(t_values):
        supOUbv = get_supOU(t, tau1, tau2, tauj)
        supOUb[index] = supOUbv
        sup_OUb_paths[i] = supOUb

# Iterate through each time value in t_values
x_component_values = [[] for i in range(num_paths)]
y_component_values = [[] for i in range(num_paths)]
meanx = np.zeros(num_paths)
meany = np.zeros(num_paths)
variancex = [np.zeros((2, 2)) for i in range(num_paths)]
variancey = [np.zeros((2, 2)) for i in range(num_paths)]

for i in range(num_paths):
    # Extract x and y components for each vector at this time step
    vector = supOUbv.flatten()
    x_component = vector[0]
    y_component = vector[1]
    x_component_values[i].append(x_component)
    y_component_values[i].append(y_component)

    # Select data for time range t=25 to t=1000
    start_time = 25
    end_time = 1000
    selected_t_values = t_values[start_time:end_time]
    selected_x_components[i] = x_component_values[i][start_time:end_time]
    selected_y_components[i] = y_component_values[i][start_time:end_time]

    meanx[i] = np.mean(selected_x_components[i])
    meany[i] = np.mean(selected_y_components[i])

    print("Mean of x component of ", i, "th path:", meanx[i])
    print("Mean of y component of ", i, "th path:", meany[i])

    # Autocorrelation calculation (if needed)
    # Define the autocorr2 function
    # lags = np.arange(0, 100)
    # autocorrx = autocorr2(selected_x_components, lags)
    # autocorry = autocorr2(selected_y_components, lags)

    # Plotting (if needed)
    # fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    # axes[0].plot(selected_t_values, selected_x_components, label='X Component')
    # axes[1].plot(selected_t_values, selected_y_components, label='Y Component')

    # Save data (if needed)
    # np.savez('Bivariatesup_OU(nu=10,a=3,b=0.05,B=-0.1,an=1.95).npz', selected_x_components=selected_x_components, selected_y_components=selected_y_components)

