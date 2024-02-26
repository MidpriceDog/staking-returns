from stochastic_process import const_drift_const_vol_model
import numpy as np
import scipy
from matplotlib import pyplot as plt


def stochastic_yield():
    """
    Simulate the yield from staking using a stochastic process
    """
    # Set the initial yield, vol, and drift
    initial_yield = 0.05
    drift_constant = -0.05
    vol = 0.1
    timesteps = 10
    # Simulate the yield path
    yield_path = const_drift_const_vol_model(
        initial_yield, drift_constant, vol, timesteps)
    return yield_path


if __name__ == "__main__":
    # Simulate the yield path
    yield_path = stochastic_yield()
    # Plot the yield path
    plt.plot(yield_path)
    plt.xlabel('Time')
    plt.ylabel('Yield')
    plt.title('Simulated Yield Path')
    plt.show()
