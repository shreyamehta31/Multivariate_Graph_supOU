from typing import Optional

import numpy as np
from scipy.stats import gamma
import math
from numpy.random import poisson, gamma, uniform
import scipy
from scipy import linalg
#from scipy.linalg import solve_sylvester
import time
time_taken = time.time()

nu=10 #rate
a=3     #shape for U
b=0.05     #scale for U
c=0.5
Adj = np.array([
    [0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
]) #in K

cAdj = c * Adj
I2= np.identity(10)
cAdj += I2
k=-cAdj

eigenvalues, eigenvectors = np.linalg.eig(k)

lambda1 = eigenvalues[0]
lambda2 = eigenvalues[1]
lambda3 = eigenvalues[2]
lambda4 = eigenvalues[3]
lambda5 = eigenvalues[4]
lambda6 = eigenvalues[5]
lambda7 = eigenvalues[6]
lambda8 = eigenvalues[7]
lambda9 = eigenvalues[8]
lambda10 = eigenvalues[9]

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

U3 = get_gamma(s=20000,random_state=30) #Ui for positive index for 1 Poisson
Un3=get_gamma(s=20000,random_state=31) #Ui for negative index for 1 Poisson

U4 = get_gamma(s=20000,random_state=32) #Ui for positive index for 2 Poisson
Un4=get_gamma(s=20000,random_state=33) #Ui for negative index for 2 Poisson

U5 = get_gamma(s=20000,random_state=34) #Ui for positive index for 1 Poisson
Un5=get_gamma(s=20000,random_state=35) #Ui for negative index for 1 Poisson

U6 = get_gamma(s=20000,random_state=36) #Ui for positive index for 2 Poisson
Un6=get_gamma(s=20000,random_state=37) #Ui for negative index for 2 Poisson

U7 = get_gamma(s=20000,random_state=38) #Ui for positive index for 1 Poisson
Un7=get_gamma(s=20000,random_state=39) #Ui for negative index for 1 Poisson

U8 = get_gamma(s=20000,random_state=40) #Ui for positive index for 2 Poisson
Un8=get_gamma(s=20000,random_state=41) #Ui for negative index for 2 Poisson

U9 = get_gamma(s=20000,random_state=52) #Ui for positive index for 1 Poisson
Un9=get_gamma(s=20000,random_state=53) #Ui for negative index for 1 Poisson

U10 = get_gamma(s=20000,random_state=54) #Ui for positive index for 2 Poisson
Un10=get_gamma(s=20000,random_state=55) #Ui for negative index for 2 Poisson

U1j = get_gamma(s=20000,random_state=62) #Ui for positive index for || Poisson first component
Un1j=get_gamma(s=20000,random_state=63) #Ui for negative index for || Poisson first component

U2j= get_gamma(s=20000,random_state=64) #Ui for positive index for || Poisson second component
Un2j=get_gamma(s=20000,random_state=65) #Ui for negative index for || Poisson second component

U3j = get_gamma(s=20000,random_state=62) #Ui for positive index for || Poisson first component
Un3j=get_gamma(s=20000,random_state=63) #Ui for negative index for || Poisson first component

U4j= get_gamma(s=20000,random_state=64) #Ui for positive index for || Poisson second component
Un4j=get_gamma(s=20000,random_state=65) #Ui for negative index for || Poisson second component

U5j = get_gamma(s=20000,random_state=62) #Ui for positive index for || Poisson first component
Un5j=get_gamma(s=20000,random_state=63) #Ui for negative index for || Poisson first component

U6j= get_gamma(s=20000,random_state=64) #Ui for positive index for || Poisson second component
Un6j=get_gamma(s=20000,random_state=65) #Ui for negative index for || Poisson second component

U7j = get_gamma(s=20000,random_state=62) #Ui for positive index for || Poisson first component
Un7j=get_gamma(s=20000,random_state=63) #Ui for negative index for || Poisson first component

U8j= get_gamma(s=20000,random_state=64) #Ui for positive index for || Poisson second component
Un8j=get_gamma(s=20000,random_state=65) #Ui for negative index for || Poisson second component

U9j = get_gamma(s=20000,random_state=62) #Ui for positive index for || Poisson first component
Un9j=get_gamma(s=20000,random_state=63) #Ui for negative index for || Poisson first component

U10j= get_gamma(s=20000,random_state=64) #Ui for positive index for || Poisson second component
Un10j=get_gamma(s=20000,random_state=65) #Ui for negative index for || Poisson second component

#Function for generating Ai rv
def get_gammaP(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.gamma(an, 1, s)


theta1=get_gammaP(s=20000,random_state=21) #Ai for positive index for 1 Poisson
thetan1=get_gammaP(s=20000,random_state=22) #Ai for negative index for 1 Poisson

theta2=get_gammaP(s=20000,random_state=23) #Ai for positive index for 2 Poisson
thetan2=get_gammaP(s=20000,random_state=24) #Ai for negative index for 2 Poisson

theta3=get_gammaP(s=20000,random_state=25) #Ai for positive index for 1 Poisson
thetan3=get_gammaP(s=20000,random_state=26) #Ai for negative index for 1 Poisson

theta4=get_gammaP(s=20000,random_state=27) #Ai for positive index for 2 Poisson
thetan4=get_gammaP(s=20000,random_state=28) #Ai for negative index for 2 Poisson

theta5=get_gammaP(s=20000,random_state=29) #Ai for positive index for 1 Poisson
thetan5=get_gammaP(s=20000,random_state=10) #Ai for negative index for 1 Poisson

theta6=get_gammaP(s=20000,random_state=11) #Ai for positive index for 2 Poisson
thetan6=get_gammaP(s=20000,random_state=12) #Ai for negative index for 2 Poisson

theta7=get_gammaP(s=20000,random_state=13) #Ai for positive index for 1 Poisson
thetan7=get_gammaP(s=20000,random_state=14) #Ai for negative index for 1 Poisson

theta8=get_gammaP(s=20000,random_state=15) #Ai for positive index for 2 Poisson
thetan8=get_gammaP(s=20000,random_state=16) #Ai for negative index for 2 Poisson

theta9=get_gammaP(s=20000,random_state=17) #Ai for positive index for 1 Poisson
thetan9=get_gammaP(s=20000,random_state=18) #Ai for negative index for 1 Poisson

theta10=get_gammaP(s=20000,random_state=19) #Ai for positive index for 2 Poisson
thetan10=get_gammaP(s=20000,random_state=20) #Ai for negative index for 2 Poisson

thetaj=get_gammaP(s=20000,random_state=68) #Ai for positive index for || Poisson
thetanj=get_gammaP(s=20000,random_state=69) #Ai for negative index for || Poisson


#Function for generating Ti rv
def get_exp(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.exponential(1/nu, s) #scale=1/rate

T1=get_exp(s=20000,random_state=146)#Ti for positive index for 1 Poisson
Tn1=get_exp(s=20000,random_state=147)#Ti for negative index for 1 Poisson

T2=get_exp(s=20000,random_state=170)#Ti for positive index for 2 Poisson
Tn2=get_exp(s=20000,random_state=171)#Ti for negative index for 2 Poisson

T3=get_exp(s=20000,random_state=246)#Ti for positive index for 1 Poisson
Tn3=get_exp(s=20000,random_state=247)#Ti for negative index for 1 Poisson

T4=get_exp(s=20000,random_state=370)#Ti for positive index for 2 Poisson
Tn4=get_exp(s=20000,random_state=371)#Ti for negative index for 2 Poisson

T5=get_exp(s=20000,random_state=446)#Ti for positive index for 1 Poisson
Tn5=get_exp(s=20000,random_state=447)#Ti for negative index for 1 Poisson

T6=get_exp(s=20000,random_state=570)#Ti for positive index for 2 Poisson
Tn6=get_exp(s=20000,random_state=571)#Ti for negative index for 2 Poisson

T7=get_exp(s=20000,random_state=646)#Ti for positive index for 1 Poisson
Tn7=get_exp(s=20000,random_state=647)#Ti for negative index for 1 Poisson

T8=get_exp(s=20000,random_state=770)#Ti for positive index for 2 Poisson
Tn8=get_exp(s=20000,random_state=771)#Ti for negative index for 2 Poisson

T9=get_exp(s=20000,random_state=846)#Ti for positive index for 1 Poisson
Tn9=get_exp(s=20000,random_state=847)#Ti for negative index for 1 Poisson

T10=get_exp(s=20000,random_state=970)#Ti for positive index for 2 Poisson
Tn10=get_exp(s=20000,random_state=971)#Ti for negative index for 2 Poisson

Tj=get_exp(s=20000,random_state=170)#Ti for positive index for || Poisson
Tnj=get_exp(s=20000,random_state=179)#Ti for negative index for || Poisson


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

tau3=get_tau(T3)#tau_i for positive index for 1 Poisson
taun3=get_tau(Tn3)#tau_i for negative index for 1 Poisson

tau4=get_tau(T4)#tau_i for positive index for 2 Poisson
taun4=get_tau(Tn4)#tau_i for negative index for 2 Poisson

tau5=get_tau(T5)#tau_i for positive index for 1 Poisson
taun5=get_tau(Tn5)#tau_i for negative index for 1 Poisson

tau6=get_tau(T6)#tau_i for positive index for 2 Poisson
taun6=get_tau(Tn6)#tau_i for negative index for 2 Poisson

tau7=get_tau(T7)#tau_i for positive index for 1 Poisson
taun7=get_tau(Tn7)#tau_i for negative index for 1 Poisson

tau8=get_tau(Tn8)#tau_i for positive index for 2 Poisson
taun8=get_tau(Tn8)#tau_i for negative index for 2 Poisson

tau9=get_tau(T9)#tau_i for positive index for 1 Poisson
taun9=get_tau(Tn9)#tau_i for negative index for 1 Poisson

tau10=get_tau(T10)#tau_i for positive index for 2 Poisson
taun10=get_tau(Tn10)#tau_i for negative index for 2 Poisson

tauj=get_tau(Tj)#tau_i for positive index for || Poisson
taunj=get_tau(Tnj)#tau_i for negative index for || Poisson


compound_levy = []
for i in range(len(tau1)):
    if i == 0:
        compound_levy.append(np.sum(U1[:int(tau1[i])]))
    else:
        compound_levy.append(np.sum(U1[int(tau1[i-1]):int(tau1[i])]))
       # print(U1[int(tau1[i-1]):int(tau1[i])])
compound_levy = np.array(compound_levy)


# Calculate empirical mean and variance of the compound Levy process
empirical_mean_levy = np.mean(compound_levy)
empirical_variance_levy = np.var(compound_levy)

print("Empirical Mean of Compound Levy Process:", empirical_mean_levy)
print("Empirical Variance of Compound Levy Process:", empirical_variance_levy)
'''
def get_L(k:int):
   Sum=0
   for i in range(0,k):
       Sum=Sum+U1[i]
   return(
       Sum
        )
def get_CP(Z:int,tau1:np.ndarray) -> np.ndarray:
   # t= np.arange(Z, dtype=np.float128)
   index1=0;
   for l in tau1:
       if l<=Z:
           index1=index1+1
   k1=index1;

   CP=get_L(k1)

   return(
     CP
        )
l=[]

for Z in range(1000):
    l.append(get_CP(Z,tau1))
print(l)
print(np.mean(l))
print(np.var(l))'''

# Generate array of function outputs for t values 0, 1, ..., 99
tv=1000
t_values = np.arange(tv)
supOUb = np.zeros((tv, 10, 1))

def get_I1(Z:np.ndarray,k:int) -> np.ndarray:
   Sum=np.array([0,0,0,0,0,0,0,0,0,0])
   for i in range(0,k):
       Sum=Sum+np.dot(eigenvectors@np.array([[np.exp(lambda1*thetaj[i]*(Z-tauj[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*thetaj[i]*(Z-tauj[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetaj[i]*(Z-tauj[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*thetaj[i]*(Z-tauj[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetaj[i]*(Z-tauj[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*thetaj[i]*(Z-tauj[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetaj[i]*(Z-tauj[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetaj[i]*(Z-tauj[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetaj[i]*(Z-tauj[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetaj[i]*(Z-tauj[i]))]
                                             ])@O,np.array([U1j[i],U2j[i],U3j[i],U4j[i],U5j[i],U6j[i],U7j[i],U8j[i],U9j[i],U10j[i]]))
   return(
       Sum
        )

def get_I2(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*theta1[i]*(Z-tau1[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*theta1[i]*(Z-tau1[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*theta1[i]*(Z-tau1[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*theta1[i]*(Z-tau1[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*theta1[i]*(Z-tau1[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*theta1[i]*(Z-tau1[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*theta1[i]*(Z-tau1[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*theta1[i]*(Z-tau1[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*theta1[i]*(Z-tau1[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*theta1[i]*(Z-tau1[i]))]
                                             ]) @ O,np.array([U1[i],0,0,0,0,0,0,0,0,0]))

    return (
    Sum
        )


def get_I3(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*theta2[i]*(Z-tau2[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*theta2[i]*(Z-tau2[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*theta2[i]*(Z-tau2[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*theta2[i]*(Z-tau2[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*theta2[i]*(Z-tau2[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*theta2[i]*(Z-tau2[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*theta2[i]*(Z-tau2[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*theta2[i]*(Z-tau2[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*theta2[i]*(Z-tau2[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*theta2[i]*(Z-tau2[i]))]
                                             ]) @ O,
                           np.array([0, U2[i],0,0,0,0,0,0,0,0]))


    return (
    Sum
)

def get_I4(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*theta3[i]*(Z-tau3[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*theta3[i]*(Z-tau3[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*theta3[i]*(Z-tau3[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*theta3[i]*(Z-tau3[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*theta3[i]*(Z-tau3[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*theta3[i]*(Z-tau3[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*theta3[i]*(Z-tau3[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*theta3[i]*(Z-tau3[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*theta3[i]*(Z-tau3[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*theta3[i]*(Z-tau3[i]))]
                                             ]) @ O,
                           np.array([0,0, U3[i],0,0,0,0,0,0,0]))


    return (
    Sum
)

def get_I5(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*theta4[i]*(Z-tau4[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*theta4[i]*(Z-tau4[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*theta4[i]*(Z-tau4[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*theta4[i]*(Z-tau4[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*theta4[i]*(Z-tau4[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*theta4[i]*(Z-tau4[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*theta4[i]*(Z-tau4[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*theta4[i]*(Z-tau4[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*theta4[i]*(Z-tau4[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*theta4[i]*(Z-tau4[i]))]
                                             ]) @ O,
                           np.array([0,0,0, U4[i],0,0,0,0,0,0]))


    return (
    Sum
)

def get_I6(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*theta5[i]*(Z-tau5[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*theta5[i]*(Z-tau5[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*theta5[i]*(Z-tau5[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*theta5[i]*(Z-tau5[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*theta5[i]*(Z-tau5[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*theta5[i]*(Z-tau5[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*theta5[i]*(Z-tau5[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*theta5[i]*(Z-tau5[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*theta5[i]*(Z-tau5[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*theta5[i]*(Z-tau5[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0, U5[i],0,0,0,0,0]))


    return (
    Sum
)

def get_I7(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*theta6[i]*(Z-tau6[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*theta6[i]*(Z-tau6[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*theta6[i]*(Z-tau6[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*theta6[i]*(Z-tau6[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*theta6[i]*(Z-tau6[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*theta6[i]*(Z-tau6[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*theta6[i]*(Z-tau6[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*theta6[i]*(Z-tau6[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*theta6[i]*(Z-tau6[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*theta6[i]*(Z-tau6[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0,0, U6[i],0,0,0,0]))


    return (
    Sum
)

def get_I8(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*theta7[i]*(Z-tau7[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*theta7[i]*(Z-tau7[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*theta7[i]*(Z-tau7[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*theta7[i]*(Z-tau7[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*theta7[i]*(Z-tau7[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*theta7[i]*(Z-tau7[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*theta7[i]*(Z-tau7[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*theta7[i]*(Z-tau7[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*theta7[i]*(Z-tau7[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*theta7[i]*(Z-tau7[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0,0,0,U7[i],0,0,0]))


    return (
    Sum
)

def get_I9(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*theta8[i]*(Z-tau8[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*theta8[i]*(Z-tau8[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*theta8[i]*(Z-tau8[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*theta8[i]*(Z-tau8[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*theta8[i]*(Z-tau8[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*theta8[i]*(Z-tau8[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*theta8[i]*(Z-tau8[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*theta8[i]*(Z-tau8[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*theta8[i]*(Z-tau8[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*theta8[i]*(Z-tau8[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0,0,0,0,U8[i],0,0]))


    return (
    Sum
)

def get_I10(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*theta9[i]*(Z-tau9[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*theta9[i]*(Z-tau9[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*theta9[i]*(Z-tau9[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*theta9[i]*(Z-tau9[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*theta9[i]*(Z-tau9[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*theta9[i]*(Z-tau9[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*theta9[i]*(Z-tau9[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*theta9[i]*(Z-tau9[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*theta9[i]*(Z-tau9[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*theta9[i]*(Z-tau9[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0,0,0,0,0,U9[i],0]))


    return (
    Sum
)

def get_I11(Z:np.ndarray,k:int) -> np.ndarray:
    Sum=np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, k):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*theta10[i]*(Z-tau10[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*theta10[i]*(Z-tau10[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*theta10[i]*(Z-tau10[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*theta10[i]*(Z-tau10[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*theta10[i]*(Z-tau10[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*theta10[i]*(Z-tau10[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*theta10[i]*(Z-tau10[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*theta10[i]*(Z-tau10[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*theta10[i]*(Z-tau10[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*theta10[i]*(Z-tau10[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0,0,0,0,0,0,U10[i]]))


    return (
    Sum
)

def get_I12(Z:np.ndarray) -> np.ndarray:
    Sum=np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0,200):
        Sum = Sum + np.dot(eigenvectors@np.array([[np.exp(lambda1*thetanj[i]*(Z-taunj[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*thetanj[i]*(Z-taunj[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetanj[i]*(Z-taunj[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*thetanj[i]*(Z-taunj[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetanj[i]*(Z-taunj[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*thetanj[i]*(Z-taunj[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetanj[i]*(Z-taunj[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetanj[i]*(Z-taunj[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetanj[i]*(Z-taunj[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetanj[i]*(Z-taunj[i]))]
                                             ])@O,np.array([Un1j[i],Un2j[i],Un3j[i],Un4j[i],Un5j[i],Un6j[i],Un7j[i],Un8j[i],Un9j[i],Un10j[i]]))

    return (
        Sum
    )


def get_I13(Z:np.ndarray) -> np.ndarray:
    Sum = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*thetan1[i]*(Z-taun1[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*thetan1[i]*(Z-taun1[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetan1[i]*(Z-taun1[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*thetan1[i]*(Z-taun1[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetan1[i]*(Z-taun1[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*theta1[i]*(Z-taun1[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetan1[i]*(Z-tau1[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetan1[i]*(Z-taun1[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetan1[i]*(Z-taun1[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetan1[i]*(Z-taun1[i]))]
                                             ]) @ O,np.array([Un1[i],0,0,0,0,0,0,0,0,0]))

    return (
    Sum
        )


def get_I14(Z:np.ndarray) -> np.ndarray:
    Sum = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*thetan2[i]*(Z-taun2[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*thetan2[i]*(Z-taun2[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetan2[i]*(Z-taun2[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*thetan2[i]*(Z-taun2[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetan2[i]*(Z-taun2[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*thetan2[i]*(Z-taun2[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetan2[i]*(Z-taun2[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetan2[i]*(Z-taun2[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetan2[i]*(Z-taun2[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetan2[i]*(Z-taun2[i]))]
                                             ]) @ O,
                           np.array([0, Un2[i],0,0,0,0,0,0,0,0]))


    return (
    Sum
)

def get_I15(Z:np.ndarray) -> np.ndarray:
    Sum = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*thetan3[i]*(Z-taun3[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*thetan3[i]*(Z-taun3[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetan3[i]*(Z-taun3[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*thetan3[i]*(Z-taun3[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetan3[i]*(Z-taun3[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*thetan3[i]*(Z-taun3[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetan3[i]*(Z-taun3[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetan3[i]*(Z-taun3[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetan3[i]*(Z-taun3[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetan3[i]*(Z-taun3[i]))]
                                             ]) @ O,
                           np.array([0,0, Un3[i],0,0,0,0,0,0,0]))


    return (
    Sum
)

def get_I16(Z:np.ndarray) -> np.ndarray:
    Sum = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*thetan4[i]*(Z-taun4[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*thetan4[i]*(Z-taun4[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetan4[i]*(Z-taun4[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*thetan4[i]*(Z-taun4[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetan4[i]*(Z-taun4[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*thetan4[i]*(Z-taun4[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetan4[i]*(Z-taun4[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetan4[i]*(Z-taun4[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetan4[i]*(Z-taun4[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetan4[i]*(Z-taun4[i]))]
                                             ]) @ O,
                           np.array([0,0,0 ,Un4[i],0,0,0,0,0,0]))


    return (
    Sum
)

def get_I17(Z:np.ndarray) -> np.ndarray:
    Sum = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*theta5[i]*(Z-tau5[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*theta5[i]*(Z-tau5[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetan5[i]*(Z-taun5[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*theta5[i]*(Z-tau5[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetan5[i]*(Z-taun5[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*thetan5[i]*(Z-taun5[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetan5[i]*(Z-taun5[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetan5[i]*(Z-taun5[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetan5[i]*(Z-taun5[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetan5[i]*(Z-taun5[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0, Un5[i],0,0,0,0,0]))


    return (
    Sum
)

def get_I18(Z:np.ndarray) -> np.ndarray:
    Sum = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*thetan6[i]*(Z-taun6[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*thetan6[i]*(Z-taun6[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetan6[i]*(Z-taun6[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*thetan6[i]*(Z-taun6[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetan6[i]*(Z-taun6[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*thetan6[i]*(Z-taun6[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetan6[i]*(Z-taun6[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetan6[i]*(Z-taun6[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetan6[i]*(Z-taun6[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetan6[i]*(Z-taun6[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0,0, Un6[i],0,0,0,0]))


    return (
    Sum
)

def get_I19(Z:np.ndarray) -> np.ndarray:
    Sum = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*thetan7[i]*(Z-taun7[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*thetan7[i]*(Z-taun7[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetan7[i]*(Z-taun7[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*thetan7[i]*(Z-taun7[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetan7[i]*(Z-taun7[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*thetan7[i]*(Z-taun7[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetan7[i]*(Z-taun7[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetan7[i]*(Z-taun7[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetan7[i]*(Z-taun7[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetan7[i]*(Z-taun7[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0,0,0,Un7[i],0,0,0]))


    return (
    Sum
)

def get_I20(Z:np.ndarray) -> np.ndarray:
    Sum = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*thetan8[i]*(Z-taun8[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*thetan8[i]*(Z-taun8[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetan8[i]*(Z-taun8[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*thetan8[i]*(Z-taun8[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetan8[i]*(Z-taun8[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*thetan8[i]*(Z-taun8[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetan8[i]*(Z-taun8[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetan8[i]*(Z-taun8[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetan8[i]*(Z-taun8[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetan8[i]*(Z-taun8[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0,0,0,0,Un8[i],0,0]))


    return (
    Sum
)

def get_I21(Z:np.ndarray) -> np.ndarray:
    Sum = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*thetan9[i]*(Z-taun9[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*thetan9[i]*(Z-taun9[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetan9[i]*(Z-taun9[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*thetan9[i]*(Z-taun9[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetan9[i]*(Z-taun9[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*thetan9[i]*(Z-taun9[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetan9[i]*(Z-taun9[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetan9[i]*(Z-taun9[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetan9[i]*(Z-taun9[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetan9[i]*(Z-taun9[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0,0,0,0,0,Un9[i],0]))


    return (
    Sum
)

def get_I22(Z:np.ndarray) -> np.ndarray:
    Sum = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0, 200):
        Sum = Sum + np.dot(eigenvectors @ np.array([[np.exp(lambda1*thetan10[i]*(Z-taun10[i])), 0,0,0,0,0,0,0,0,0],
              [0, np.exp(lambda2*thetan10[i]*(Z-taun10[i])),0,0,0,0,0,0,0,0],[0,0, np.exp(lambda3*thetan10[i]*(Z-taun10[i])),0,0,0,0,0,0,0],
                                             [0,0,0, np.exp(lambda4*thetan10[i]*(Z-taun10[i])),0,0,0,0,0,0],[0,0,0,0, np.exp(lambda5*thetan10[i]*(Z-taun10[i])),0,0,0,0,0],
                                             [0,0,0,0,0,np.exp(lambda6*thetan10[i]*(Z-taun10[i])),0,0,0,0],[0,0,0,0,0,0, np.exp(lambda7*thetan10[i]*(Z-taun10[i])),0,0,0],
                                             [0,0,0,0,0,0,0, np.exp(lambda8*thetan10[i]*(Z-taun10[i])),0,0],[0,0,0,0,0,0,0,0, np.exp(lambda9*thetan10[i]*(Z-taun10[i])),0],
                                             [0,0,0,0,0,0,0,0,0, np.exp(lambda10*thetan10[i]*(Z-taun10[i]))]
                                             ]) @ O,
                           np.array([0,0,0,0,0,0,0,0,0,Un10[i]]))


    return (
    Sum
)

def get_supOU(Z:int,tau1:np.ndarray,tau2:np.ndarray,tau3:np.ndarray,tau4:np.ndarray,tau5:np.ndarray
              ,tau6:np.ndarray,tau7:np.ndarray,tau8:np.ndarray,tau9:np.ndarray,tau10:np.ndarray,tauj:np.ndarray) -> np.ndarray:
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
   index3 = 0;
   for l in tau3:
       if l <= Z:
           index3 = index3 + 1
   k3 = index3;
   index4 = 0;
   for l in tau4:
       if l <= Z:
           index4 = index4 + 1
   k4 = index4;
   index5 = 0;
   for l in tau5:
       if l <= Z:
           index5 = index5 + 1
   k5 = index5;
   index6 = 0;
   for l in tau6:
       if l <= Z:
           index6 = index6 + 1
   k6 = index6;
   index7 = 0;
   for l in tau7:
       if l <= Z:
           index7 = index7 + 1
   k7 = index7;
   index8 = 0;
   for l in tau8:
       if l <= Z:
           index8 = index8 + 1
   k8 = index8;
   index9 = 0;
   for l in tau9:
       if l <= Z:
           index9 = index9 + 1
   k9 = index9;
   index10 = 0;
   for l in tau10:
       if l <= Z:
           index10 = index10 + 1
   k10 = index10;
   indexj = 0;
   for l in tauj:
       if l <= Z:
           indexj = indexj + 1
   kj = indexj;
   I1=get_I1(Z, kj)
   I2=get_I2(Z,k1)
   I3 = get_I3(Z,k2)
   I4 = get_I4(Z, k3)
   I5 = get_I5(Z, k4)
   I6 = get_I6(Z, k5)
   I7 = get_I7(Z, k6)
   I8 = get_I8(Z, k7)
   I9 = get_I9(Z, k8)
   I10 = get_I10(Z, k9)
   I11 = get_I11(Z, k10)
   I12=get_I12(Z)
   I13=get_I13(Z)
   I14 = get_I14(Z)
   I15= get_I15(Z)
   I16= get_I16(Z)
   I17= get_I17(Z)
   I18= get_I18(Z)
   I19= get_I19(Z)
   I20= get_I20(Z)
   I21= get_I21(Z)
   I22= get_I22(Z)
   return(
     np.expand_dims(I1 + I2 + I3 + I4 + I5 + I6+I7+I8+I9+I10+I11+I12+I13+I14+I15+I16+I17+I18+I19+I20+I21+I22, axis=1)
        )

for t in t_values:
    supOUbv =get_supOU(t,tau1,tau2,tau3,tau4,tau5,tau6,tau7,tau8,tau9,tau10,tauj)
    supOUb[t] = supOUbv


# Iterate through each time value in t_values
for t in t_values:
    print(f"Vector at time {t}:\n{supOUb[t]}")

import matplotlib.pyplot as plt


# Iterate through each time value in t_values
for t in t_values:
    supOUbv = get_supOU(t,tau1,tau2,tau3,tau4,tau5,tau6,tau7,tau8,tau9,tau10,tauj)
    supOUb[t] = supOUbv

# Extract x and y components for all vectors at all time steps
component_1_values = []
component_2_values = []
component_3_values = []
component_4_values = []
component_5_values = []
component_6_values = []
component_7_values = []
component_8_values = []
component_9_values = []
component_10_values = []

# Assuming t_values is defined somewhere above
for t in t_values:

    # Extract x and y components for each vector at this time step
    vector = supOUb[t].flatten()
    component_1 = vector[0]
    component_2 = vector[1]
    component_3 = vector[2]
    component_4 = vector[3]
    component_5 = vector[4]
    component_6 = vector[5]
    component_7 = vector[6]
    component_8 = vector[7]
    component_9 = vector[8]
    component_10 = vector[9]

    component_1_values.append(component_1)
    component_2_values.append(component_2)
    component_3_values.append(component_3)
    component_4_values.append(component_4)
    component_5_values.append(component_5)
    component_6_values.append(component_6)
    component_7_values.append(component_7)
    component_8_values.append(component_8)
    component_9_values.append(component_9)
    component_10_values.append(component_10)

# Select data for time range t=25 to t=1000
start_time = 25
end_time = 1000
selected_t_values = t_values[start_time:end_time]
selected_1_components = component_1_values[start_time:end_time]
selected_2_components = component_2_values[start_time:end_time]
selected_3_components = component_3_values[start_time:end_time]
selected_4_components = component_4_values[start_time:end_time]
selected_5_components = component_5_values[start_time:end_time]
selected_6_components = component_6_values[start_time:end_time]
selected_7_components = component_7_values[start_time:end_time]
selected_8_components = component_8_values[start_time:end_time]
selected_9_components = component_9_values[start_time:end_time]
selected_10_components = component_10_values[start_time:end_time]

mean1 = np.mean(selected_1_components)
variance1 = np.var(selected_1_components)

print("Mean of 1 component:", mean1)
print("Variance of 1 component:", variance1)

mean2 = np.mean(selected_2_components)
variance2 = np.var(selected_2_components)

print("Mean of 2 component:", mean2)
print("Variance of 2 component:", variance2)

mean3 = np.mean(selected_3_components)
variance3 = np.var(selected_3_components)

print("Mean of 3 component:", mean3)
print("Variance of 3 component:", variance3)

mean4 = np.mean(selected_4_components)
variance4 = np.var(selected_4_components)

print("Mean of 4 component:", mean4)
print("Variance of 4 component:", variance4)

mean5 = np.mean(selected_5_components)
variance5 = np.var(selected_5_components)

print("Mean of 5 component:", mean5)
print("Variance of 5 component:", variance5)

mean6 = np.mean(selected_6_components)
variance6 = np.var(selected_6_components)

print("Mean of 6 component:", mean6)
print("Variance of 6 component:", variance6)

mean7 = np.mean(selected_7_components)
variance7 = np.var(selected_7_components)

print("Mean of 7 component:", mean7)
print("Variance of 7 component:", variance7)

mean8 = np.mean(selected_8_components)
variance8 = np.var(selected_8_components)

print("Mean of 8 component:", mean8)
print("Variance of 8 component:", variance8)

mean9 = np.mean(selected_9_components)
variance9 = np.var(selected_9_components)

print("Mean of 9 component:", mean9)
print("Variance of 9 component:", variance9)

mean10 = np.mean(selected_10_components)
variance10 = np.var(selected_10_components)

print("Mean of 10 component:", mean10)
print("Variance of 10 component:", variance10)

'''
def autocorr2(x, lags):
    mean = np.mean(x)
    var = np.var(x)
    xp = x - mean
    corr = [1. if l == 0 else np.sum(xp[l:] * xp[:-l]) / len(x) / var for l in lags]
    return np.array(corr)

lags = np.arange(0, 100)

autocorrx = autocorr2(selected_x_components, lags)
autocorry = autocorr2(selected_y_components, lags)

#theoretical
D1_values = []
D2_values = []
b=np.array( [[ 0.3419,-0.0526 ],
      [ -0.0526,0.3419]] )
lags = np.arange(0, 100)

for h in lags:
    a = np.array([[1 + h, 0.5 * h],
                  [0.5 * h, 1 + h]])
    C=np.dot(scipy.linalg.fractional_matrix_power(a, -0.95),b)
    D=np.diag(C)

    D1_values.append(D[0])
    D2_values.append(D[1])


fig, axes = plt.subplots(2, 1, figsize=(15, 10))
title = "Bivariate supOU "
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
plt.plot(lags, D1_values, color='black', label='Diagonal Element 1')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation of x component')
plt.title('Autocorrelation Function of first component')
plt.show(block=True)

plt.stem(lags, autocorry, use_line_collection=True)
plt.plot(lags, D2_values, color='black', label='Diagonal Element 2')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation of y component')
plt.title('Autocorrelation Function of second component')
plt.show(block=True)

np.savez('Bivariatesup_OU(nu=10,a=3,b=0.05,B=-0.1,an=1.95).npz', selected_x_components=selected_x_components, selected_y_components=selected_y_components)
'''
print("My program took", time.time() - time_taken, "to run")