from typing import Optional

import numpy as np
from scipy.stats import gamma
import math


nu=10 #rate
a=3     #shape for U
b=0.05     #scale for U
c=0.5
Adj = np.array( [[ 0,1 ],
      [ 1,0 ]] ) #in K

cAdj = c * Adj
I2= np.identity(2)
cAdj += I2
k=-cAdj

eigenvalues, eigenvectors = np.linalg.eig(k)

lambda1 = eigenvalues[0]
lambda2 = eigenvalues[1]

O = np.linalg.inv(eigenvectors)

an=1.95  #shape for A
mu=(nu)*(a/(1/b)) #mean of Levy
var= (nu)*((a*(a+1)/(1/b)**2))   #variance of Levy



def get_gamma(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.gamma(a, b, s)


U1 = get_gamma(s=20000,random_state=42) #Ui for positive index for 1 Poisson
Un1=get_gamma(s=20000,random_state=43) #Ui for negative index for 1 Poisson

U2 = get_gamma(s=20000,random_state=60) #Ui for positive index for 2 Poisson
Un2=get_gamma(s=20000,random_state=61) #Ui for negative index for 2 Poisson

Ufj = get_gamma(s=20000,random_state=62) #Ui for positive index for || Poisson first component
Unfj=get_gamma(s=20000,random_state=63) #Ui for negative index for || Poisson first component

Usj= get_gamma(s=20000,random_state=64) #Ui for positive index for || Poisson second component
Unsj=get_gamma(s=20000,random_state=65) #Ui for negative index for || Poisson second component

#Function for generating Ai rv
def get_gammaP(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.gamma(an, 1, s)


theta1=get_gammaP(s=20000,random_state=45) #Ai for positive index for 1 Poisson
thetan1=get_gammaP(s=20000,random_state=44) #Ai for negative index for 1 Poisson

theta2=get_gammaP(s=20000,random_state=66) #Ai for positive index for 2 Poisson
thetan2=get_gammaP(s=20000,random_state=67) #Ai for negative index for 2 Poisson

thetaj=get_gammaP(s=20000,random_state=68) #Ai for positive index for || Poisson
thetanj=get_gammaP(s=20000,random_state=69) #Ai for negative index for || Poisson


#Function for generating Ti rv
def get_exp(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.exponential(1/nu, s) #scale=1/rate

T1=get_exp(s=20000,random_state=46)#Ti for positive index for 1 Poisson
Tn1=get_exp(s=20000,random_state=47)#Ti for negative index for 1 Poisson

T2=get_exp(s=20000,random_state=70)#Ti for positive index for 2 Poisson
Tn2=get_exp(s=20000,random_state=71)#Ti for negative index for 2 Poisson

Tj=get_exp(s=20000,random_state=72)#Ti for positive index for || Poisson
Tnj=get_exp(s=20000,random_state=73)#Ti for negative index for || Poisson


#Function for generating tau_i's rv
def get_tau(arr):
    cumulative_array = []
    running_sum = 0
    for num in arr:
        running_sum += num
        cumulative_array.append(running_sum)
    return cumulative_array

tau1=get_tau(T1)#tau_i for positive index for 1 Poisson
taun1=get_tau(Tn1)#tau_i for negative index for 1 Poisson

tau2=get_tau(T2)#tau_i for positive index for 2 Poisson
taun2=get_tau(Tn2)#tau_i for negative index for 2 Poisson

tauj=get_tau(Tj)#tau_i for positive index for || Poisson
taunj=get_tau(Tnj)#tau_i for negative index for || Poisson

# Generate array of function outputs for t values 0, 1, ..., 99
tv=300
t_values = np.arange(tv)
supOUb = np.zeros((tv, 2, 1))

def get_I1(Z:np.ndarray,k:int) -> np.ndarray:
   Sum=np.array([0,0])
   for i in range(0,k):
       Sum=Sum+np.dot(eigenvectors@np.array([[np.exp(lambda1*thetaj[i]*(Z-tauj[i])), 0],
              [0, np.exp(lambda2*thetaj[i]*(Z-tauj[i]))]])@O,np.array([Ufj[i],Usj[i]]))
   return(
       Sum
        )

def get_I2(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * theta1[i] * (Z - tau1[i])), 0],
                                                    [0, np.exp(lambda2 * theta1[i] * (Z - tau1[i]))]]) @ O,np.array([U1[i],0]))

    return (
    Sum
        )


def get_I3(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * theta2[i] * (Z - tau2[i])), 0],
                                                    [0, np.exp(lambda2 * theta2[i] * (Z - tau2[i]))]]) @ O,
                           np.array([0, U2[i]]))


    return (
    Sum
)


def get_I4(Z:np.ndarray) -> np.ndarray:
    Sum=np.array([0,0])
    for i in range(0,200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * thetanj[i] * (Z - taunj[i])), 0],
                                                    [0, np.exp(lambda2 * thetanj[i] * (Z - taunj[i]))]]) @ O,
                           np.array([Unfj[i], Unsj[i]]))

    return (
        Sum
    )

def get_I5(Z:np.ndarray) -> np.ndarray:
    Sum=np.array([0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * thetan1[i] * (Z - taun1[i])), 0],
                                                    [0, np.exp(lambda2 * thetan1[i] * (Z - taun1[i]))]]) @ O,
                           np.array([Un1[i], 0]))


    return (
    Sum
)


def get_I6(Z:np.ndarray) -> np.ndarray:
    Sum=np.array([0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1 * thetan2[i] * (Z - taun2[i])), 0],
                                                    [0, np.exp(lambda2 * thetan2[i] * (Z - taun2[i]))]]) @ O,
                           np.array([0, Un2[i]]))
    return (
            Sum
        )



def get_supOU(Z:int,tau1:np.ndarray,tau2:np.ndarray,tauj:np.ndarray) -> np.ndarray:
   # t= np.arange(Z, dtype=np.float128)
   index1=0;
   for l in tau1:
       if l<=Z:
           index1=index1+1
   k1=index1;
   index2 = 0;
   for l in tau2:
       if l <= Z:
           index2 = index2 + 1
   k2 = index2;
   indexj = 0;
   for l in tauj:
       if l <= Z:
           indexj = indexj + 1
   kj = indexj;
   I1=get_I1(Z, kj)
   I2=get_I2(Z,k1)
   I3 = get_I3(Z,k2)
   I4=get_I4(Z)
   I5=get_I5(Z)
   I6 = get_I6(Z)
   return(
     np.expand_dims(I1 + I2 + I3 + I4 + I5 + I6, axis=1)
        )

for t in t_values:
    supOUbv =  get_supOU(t,tau1,tau2,tauj)
    supOUb[t] = supOUbv


# Iterate through each time value in t_values
for t in t_values:
    print(f"Vector at time {t}:\n{supOUb[t]}")

    import matplotlib.pyplot as plt

    # Define lists outside the loop to store x and y components for all vectors at all time steps
    x_component_values = []
    y_component_values = []

    # Assuming t_values is defined somewhere above
    for t in t_values:
        print(f"Vector at time {t}:\n{supOUb[t]}")

        # Extract x and y components for each vector at this time step
        vector = supOUb[t].flatten()
        x_component = vector[0]
        y_component = vector[1]

        x_component_values.append(x_component)
        y_component_values.append(y_component)

    # Select data for time range t=10 to t=200
    start_time = 25
    end_time = 1000
    selected_t_values = t_values[start_time:end_time]
    selected_x_components = x_component_values[start_time:end_time]
    selected_y_components = y_component_values[start_time:end_time]

    meanx = np.mean(selected_x_components)
    variancex = np.var(selected_x_components)

    print("Mean of x component:", meanx)
    print("Variance of x component:", variancex)

    meany = np.mean(selected_y_components)
    variancey = np.var(selected_y_components)

    print("Mean of y component:", meany)
    print("Variance of y component:", variancey)


    def autocorr2(x, lags):
        '''manualy compute, non partial'''

        mean = np.mean(x)
        var = np.var(x)
        xp = x - mean
        corr = [1. if l == 0 else np.sum(xp[l:] * xp[:-l]) / len(x) / var for l in lags]

        return np.array(corr)


    lags = np.arange(0, 100)

    autocorrx = autocorr2(selected_x_components, lags)
    autocorry = autocorr2(selected_y_components, lags)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))  # Create subplots for X and Y components
    title = "Bivariate supOU "
    # title += r"$\alpha=0.07$, $\gamma = 0$, $\beta = 0.001$"
    axes[0].plot(selected_t_values, selected_x_components, label='X Component')
    axes[0].set_title("X Component", fontsize=15)
    axes[1].plot(selected_t_values, selected_y_components, label='Y Component')
    axes[1].set_title("Y Component", fontsize=15)

    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()

    plt.stem(lags, autocorrx, use_line_collection=True)
   # plt.plot(lags, (1 - B * lags) ** (1 - an), color='black', label='Theoretical Autocorrelation Function')

    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation of x component')
    plt.title('Autocorrelation Function of X component')
    plt.show(block=True)

    plt.stem(lags, autocorry, use_line_collection=True)
    # plt.plot(lags, (1 - B * lags) ** (1 - an), color='black', label='Theoretical Autocorrelation Function')

    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation of y component')
    plt.title('Autocorrelation Function of Y component')
    plt.show(block=True)
