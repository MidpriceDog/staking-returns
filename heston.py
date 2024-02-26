import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt


def heston_model(S0, T, r, kappa, theta, xi, rho, N):
    """
    Simulate the price of a stock using the Heston model

    Parameters
    ----------
    S0 : float
        The initial price of the asset
    T : float
        The time to maturity
    r : float
        The risk-free interest rate
    kappa : float
        The mean-reversion rate
    theta : float
        The long-run average volatility
    xi : float
        The volatility of the volatility
    rho : float
        The correlation between the asset price and its volatility
    N : int
        The number of time steps


    Returns
    -------
    price_path : np.ndarray
        The simulated price path of the stock using the Heston model
    """
    # Set the initial price
    price_path = np.zeros(N)
    # Set the time step
    dt = T / N
    # Simulate the price path

    # Set the initial price
    price_path[0] = S0
    # Set the initial volatility
    v = theta
    # Simulate the price path
    for t in range(1, N):
        # Generate the random shocks
        z1 = np.random.normal(0, 1)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1)
        # Calculate the new volatility
        v = v + kappa * (theta - v) * dt + xi * np.sqrt(v * dt) * z2
        # Calculate the new price
        price_path[t] = price_path[t - 1] * \
            np.exp((r - 0.5 * v) * dt + np.sqrt(v * dt) * z1)
    return price_path


def plot_price_path(price_path):
    """
    Plot the price path
    """
    plt.plot(price_path)
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.title('Simulated Asset Price Path')
    plt.show()


if __name__ == "__main__":
    # Create some observed data
    # Define the parameters
    S0 = 3067
    T = 1
    r = 0.05
    kappa = 2
    theta = 0.05
    xi = 0.4
    rho = -0.5
    N = 365

    observed_data = heston_model(S0, T, r, kappa, theta, xi, rho, N)
    # Plot the price path
    plot_price_path(observed_data)
