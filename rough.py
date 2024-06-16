import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad

A = np.array([[-1, -0.5], [-0.5, -1]])
W = np.array([[0.6, 0.225], [0.225, 0.6]])

def integrand(t):
    eAt = expm(A * t)
    eATt = expm(A.T * t)
    result = np.dot(np.dot(eAt, W), eATt)
    return result.flatten()

# Integrate the function from 0 to infinity
result, _ = quad(integrand, 0, np.inf)

print(result)
