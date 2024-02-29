import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt


def const_drift_const_vol_model(initial_price, drift_constant, volatility, timesteps):
    """
    Simulate the price of ETH to follow Brownian motion with average drift and volatility
    """
    # Set the initial price
    price_path = [initial_price]
    # Set the time step
    dt = 1 / timesteps
    # Simulate the price path
    for t in range(1, timesteps):
        # Generate the random shock
        shock = np.random.normal(0, 1)
        # Calculate the new price
        new_price = price_path[t - 1] * \
            (1 + drift_constant * dt + volatility * np.sqrt(dt) * shock)
        # Append the new price to the price path
        price_path.append(new_price)
    return price_path


if __name__ == "__main__":
    # Set the initial price, vol, and drift
    initial_price = 3250
    drift_constant = 0.01
    vol = 2.19
    timesteps = 100
    # Simulate the price path
    for i in range(5):
        # Pick a color for the price path at random
        color = np.random.rand(3)
        price_path = const_drift_const_vol_model(
            initial_price, drift_constant, vol, timesteps)

        # Plot the price path
        plt.plot(price_path, alpha=0.9, color=color)

    plt.xlabel('Time')
    plt.ylabel('ETH Price')
    plt.title('Simulated ETH Price Path')
    plt.show()
