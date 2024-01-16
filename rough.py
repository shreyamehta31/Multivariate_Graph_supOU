import numpy as np

# Parameters
rate = 10
alpha = 3
beta = 20
time = 1000  # Time horizon

# Simulate arrival times from an exponential distribution
arrival_times = np.random.exponential(scale=1 / rate, size=time)

# Simulate jump sizes from a Gamma distribution
num_jumps = np.random.poisson(rate * time)  # Number of jumps in the given time
jump_sizes = np.random.gamma(shape=alpha, scale=beta, size=num_jumps)

# Generate the compound Poisson process
compound_poisson = np.zeros(time)
arrival_sum = 0
for i, jump_time in enumerate(arrival_times):
    if arrival_sum >= time:
        break
    arrival_sum += jump_time
    jumps_until_time = np.sum(jump_sizes[np.cumsum(arrival_times) <= arrival_sum])
    compound_poisson[min(int(arrival_sum), time - 1)] = jumps_until_time

# Calculate mean and variance of the simulated compound Poisson process
mean_poisson = np.mean(compound_poisson)
variance_poisson = np.var(compound_poisson)

print("Mean of Compound Poisson Process:", mean_poisson)
print("Variance of Compound Poisson Process:", variance_poisson)
