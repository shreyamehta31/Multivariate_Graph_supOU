# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from typing import Optional

import numpy as np
from scipy.stats import gamma


nu=10 #rate
a=3     #shape for U
b=0.05     #scale for U
B= -0.1    #in A
an=1.95  #shape for A
mu=(nu)*(a/(1/b)) #mean of Levy
var= (nu)*((a*(a+1)/(1/b)**2))   #variance of Levy
#var1=nu*(a*(1+a)/(1/b)**2)


mean_theoretical=-mu/(B*(an-1))
vardem=2*B*(an-1)
variance_theoretical=-(var/vardem)



#Function for generating Ui rv
def get_gamma(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.gamma(a, b, s)


U = get_gamma(s=20000,random_state=42) #Ui for positive index
Un=get_gamma(s=20000,random_state=43) #Ui for negative index


#Function for generating Ai rv
def get_gammaP(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return B*np.random.gamma(an, 1, s)


A=get_gammaP(s=20000,random_state=45) #Ai for positive index
An=get_gammaP(s=20000,random_state=44) #Ai for negative index

meanA=np.mean(A)
#Function for generating Ti rv
def get_exp(s: int, random_state: Optional[int] = None) -> np.ndarray:
    np.random.seed(random_state)
    return np.random.exponential(1/nu, s) #scale=1/rate

T=get_exp(s=20000,random_state=46)#Ti for positive index
Tn=get_exp(s=20000,random_state=47)#Ti for negative index

meanT=np.mean(T)

#Function for generating tau_i's rv
def get_tau(arr):
    cumulative_array = []
    running_sum = 0
    for num in arr:
        running_sum += num
        cumulative_array.append(running_sum)
    return cumulative_array

tau=get_tau(T)#tau_i for positive index
taun=get_tau(Tn)#tau_i for negative index

#Function for first sum
def get_I1(Z:np.ndarray,k:int) -> np.ndarray:

   Sum=0
   for i in range(0,k):
       Sum=Sum+np.exp(A[i]*(Z-tau[i]))*U[i]
   return(
       Sum
        )


#Function for second sum
def get_I2(Z:np.ndarray) -> np.ndarray:

   Sum2=0
   for i in range(0,2000):
       Sum2=Sum2+np.exp(An[i]*(Z+taun[i]))*Un[i]
   return(
       Sum2
        )


#Function for supOU for 1 t
def get_supOU(Z:int,tau:np.ndarray) -> np.ndarray:
   # t= np.arange(Z, dtype=np.float128)
   index=0;
   for l in tau:
       if l<=Z:
           index=index+1
   k=index;

   I1=get_I1(Z, k)
   I2=get_I2(Z)
   return(
     I1+ I2
        )

# Generate array of function outputs for t values 0, 1, ..., 99
t_values = np.arange(1000)
sup_OU = np.zeros_like(t_values,dtype=float)  # Initialize array to store function outputs

for i, t in enumerate(t_values):
    sup_OU[i] = get_supOU(t,tau)

# Print or use the output array as needed
print(sup_OU)

mean=np.mean(sup_OU)
variance=np.var(sup_OU)


#Estimating least squares

meanLS=(mean_theoretical-mean)**2
varianceLS=(variance_theoretical-variance)**2

# Calculate autocorrelation function


def autocorr2(x,lags):
    '''manualy compute, non partial'''

    mean=np.mean(x)
    var=np.var(x)
    xp=x-mean
    corr=[1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]

    return np.array(corr)

lags = np.arange(0, 100)

autocorr=autocorr2(sup_OU,lags)
#autocorr = np.correlate(sup_OU, sup_OU, mode='full')

# Extract autocorrelation values for positive lags
#autocorr_positive = autocorr[len(sup_OU)-1:]
cov=np.cov(sup_OU,sup_OU, bias=True)
autocovdem=2*B*(an-1)
autocovariance_theoretical=-var*(1-B*lags)**(1-an)/autocovdem


 # plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 7))
title = "Univariate supOU "
#title += r"$\alpha=0.07$, $\gamma = 0$, $\beta = 0.001$"
plt.plot(t_values,sup_OU)
plt.gca().set_title(title, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

'''h=np.arange(100)
plt.plot(h,(1-B*h)**(1-an))
plt.xlabel('Lag')
plt.ylabel('Theoretical Autocorrelation')
plt.title('Theoretical Autocorrelation Function')
plt.show()'''
# Create lag values for x-axis


# Plot autocorrelation function
plt.stem(lags, autocorr, use_line_collection=True)
plt.plot(lags,(1-B*lags)**(1-an), color='black',label='Theoretical Autocorrelation Function')

plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function')
plt.show(block=True)