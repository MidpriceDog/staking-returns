from stochastic_process import const_drift_const_vol_model
import numpy as np
import scipy
from matplotlib import pyplot as plt
from heston import heston_model
from heston import calculate_correlation_between_asset_price_and_volatility
from heston import calculate_vol_of_vol
import pandas as pd


def daily_proposal_probability():
    """See https://luckystaker.com/home/
    """
    return 0.0075


def k_proposal_within_n_days_probability(k, n):
    """The probability of exactly k proposals within n days

    Parameters
    ----------
    k : int
        The number of block proposals
    n : int
        The number of days
    """
    def k_successes_in_n_trials(k, n, p):
        """
        The probability of exactly k successes in n trials

        Parameters
        ----------
        k : int
            The number of successes
        n : int
            The number of trials
        p : float
            The probability of success
        """
        return scipy.stats.binom.pmf(k, n, p)
    return k_successes_in_n_trials(k, n, daily_proposal_probability())


def k_or_more_proposals_within_n_days(k, n, p=daily_proposal_probability()):
    """The probability of k or more proposals within n days

    Parameters
    ----------
    k : int
        The number of proposals
    n : int
        The number of days
    p : float
        The probability of a proposal
    """
    def k_or_more_successes_in_n_trials(k, n, p):
        """
        The probability of k or more successes in n trials

        Parameters
        ----------
        k : int
            The number of successes
        n : int
            The number of trials
        p : float
            The probability of success
        """
        return 1 - scipy.stats.binom.cdf(k - 1, n, p)
    return k_or_more_successes_in_n_trials(k, n, p)


def number_of_proposals_within_n_days(n, p=daily_proposal_probability()):
    """Number of proposals within n days given a daily probability of a proposal

    Parameters
    ----------
    n : int
        The number of days
    p : float, optional
        The daily probability, by default daily_proposal_probability()

    Returns
    -------
    int
        The number of proposals within n days
    """
    return np.random.binomial(n, p, 1)


def mev_tips(scaling_factor, mu, sigma, num_blocks):
    """Draw the priority fee and MEV received for one or more block proposals from a log-normal distribution

    Parameters
    ----------
    mu : float
        The mean value of the underlying normal distribution
    sigma : float
        The standard deviation of the underlying normal distribution
    num_blocks : int
        The number of blocks to simulate the priority fee and MEV for

    Returns
    -------
    np.ndarray
        The simulated priority fees and MEV received for each block proposal as a 1D array
    """
    # Create log-normal distribution and scale it
    s = np.random.lognormal(mu, sigma, num_blocks) * scaling_factor
    return s


def plot_mev_tips(scaling_factor, mu, sigma, num_blocks):
    """Plot the priority fee and MEV during block proposal across multiple blocks

    Parameters
    ----------
    mu : float
        The mean value of the underlying normal distribution
    sigma : float
        The standard deviation of the underlying normal distribution
    num_blocks : int
        The number of blocks to simulate the priority fee and MEV for
    """
    # Create log-normal distribution
    s = np.random.lognormal(mu, sigma, num_blocks) * scaling_factor
    # Plot the distribution
    _, bins, _ = plt.hist(s, 100, density=True, align='mid')
    x = np.linspace(min(bins), max(bins), 10000)
    pdf_with_scaling = scipy.stats.lognorm.pdf(x, sigma, scale=scaling_factor)
    plt.plot(x, pdf_with_scaling, linewidth=2, color='r')
    plt.show()


def stochastic_yield(initial_yield, drift_constant, vol, timesteps):
    """
    Simulate the yield from staking using a stochastic process

    Parameters
    ----------
    intial_yield : float
        The initial yield
    drift_constant : float
        The drift constant of the yield
    vol : float
        The volatility of the yield
    timesteps : int
        The number of time steps to simulate the yield over

    Returns
    -------
    np.ndarray
        The simulated yield path as a 1D array
    """
    # Simulate the yield path
    yield_path = const_drift_const_vol_model(
        initial_yield, drift_constant, vol, timesteps)
    return yield_path


def plot_stochastic_yield(initial_yield, drift_constant, vol, timesteps):
    """
    Plot the simulated yield as a percentage from staking using a stochastic process

    Parameters
    ----------
    intial_yield : float
        The initial yield
    drift_constant : float
        The drift constant of the yield
    vol : float
        The volatility of the yield
    timesteps : int
        The number of time steps to simulate the yield over
    """
    # Simulate the yield path
    yield_path = const_drift_const_vol_model(
        initial_yield, drift_constant, vol, timesteps)
    # Plot the yield path
    plt.plot(yield_path)
    plt.xlabel('Time')
    plt.ylabel('Yield')
    plt.title('Simulated Yield Path')
    plt.show()


def plot_stochastic_yield(yield_path):
    """
    Plot the simulated yield from staking using a stochastic process

    Parameters
    ----------
    yield_path : np.ndarray
        The simulated yield path as a 1D array
    """
    # Plot the yield path
    plt.plot(yield_path)
    plt.xlabel('Time')
    plt.ylabel('Yield')
    plt.title('Simulated Yield Path')
    plt.show()


def calculate_staking_return(initial_stake, yield_path):
    """
    Calculate the total return from staking

    Parameters
    ----------
    initial_stake : float
        The initial amount staked
    yield_path : np.ndarray
        The simulated yield path as a 1D array

    Returns
    -------
    float
        The total return from staking
    """
    total_eth = initial_stake
    timesteps = len(yield_path)
    # The yield path are the APR yields. We need to convert these to a daily yield
    yield_path = np.exp(np.array(yield_path) / 365) - 1
    # Staking rewards are paid out every 6.5 minutes when staking on Ethereum 2.0
    for t in range(timesteps):
        total_eth += initial_stake * yield_path[t]

    # Calculate the number of block proposals that occured in the time period
    num_block_proposals = number_of_proposals_within_n_days(timesteps)
    sum_tips = np.sum(np.array(mev_tips(0.1, 0, 1, num_block_proposals)))

    return total_eth + sum_tips


def time_varying_staking_return(initial_stake, yield_path):
    """
    Calculate the time-varying return from staking

    Parameters
    ----------
    initial_stake : float
        The initial amount staked
    yield_path : np.ndarray
        The simulated yield path as a 1D array

    Returns
    -------
    np.ndarray
        The time-varying return from staking as a 1D array
    """
    timesteps = len(yield_path)
    staking_return = [initial_stake]
    # The yield path are the APR yields. We need to convert these to a daily yield
    yield_path = np.exp(np.array(yield_path) / 365) - 1

    for t in range(1, timesteps):
        staking_return.append(initial_stake*yield_path[t])
        proposal = number_of_proposals_within_n_days(1)
        if proposal:
            tips = mev_tips(0.1, 0, 1, proposal)
            print(tips)
            staking_return[-1] += np.sum(tips)
    return np.cumsum(staking_return)


def plot_time_varying_staking_return(initial_stake, yield_path):
    """
    Plot the time-varying return from staking

    Parameters
    ----------
    initial_stake : float
        The initial amount staked
    yield_path : np.ndarray
        The simulated yield path as a 1D array
    """
    timesteps = len(yield_path)
    staking_return = [initial_stake]
    # The yield path are the APR yields. We need to convert these to a daily yield
    yield_path = np.exp(np.array(yield_path) / 365) - 1
    for t in range(1, timesteps):

        staking_return.append(initial_stake*yield_path[t])

        proposal = number_of_proposals_within_n_days(1)
        if proposal:
            tips = mev_tips(0.05, 0, 1, proposal)

            staking_return[-1] += np.sum(tips)

    plt.plot(staking_return)
    plt.xlabel('Time (days)')
    plt.ylabel('Total ETH')
    plt.show()


def calculate_max_percentage_return(return_in_usd_matrix):
    """
    Calculate the percentage return from staking across multiple simulations

    Parameters
    ----------
    return_in_usd_matrix : np.ndarray
        The return matrix from staking in USD as a 2D array

    Returns
    -------
    float
        The percentage return from staking
    """
    initial = return_in_usd_matrix[:, 0]
    end = return_in_usd_matrix[:, -1]
    return np.max((end - initial) / initial)


def calculate_min_percentage_return(return_in_usd_matrix):
    """
    Calculate the percentage return from staking across multiple simulations

    Parameters
    ----------
    return_in_usd_matrix : np.ndarray
        The return matrix from staking in USD as a 2D array

    Returns
    -------
    float
        The percentage return from staking
    """
    initial = return_in_usd_matrix[:, 0]
    end = return_in_usd_matrix[:, -1]
    return np.min((end - initial) / initial)


def calculate_mean_percentage_return(return_in_usd_matrix):
    """
    Calculate the percentage return from staking across multiple simulations

    Parameters
    ----------
    return_in_usd_matrix : np.ndarray
        The return matrix from staking in USD as a 2D array

    Returns
    -------
    float
        The percentage return from staking
    """
    initial = return_in_usd_matrix[:, 0]
    end = return_in_usd_matrix[:, -1]
    return np.mean((end - initial) / initial)


def calculate_std_dev_percentage_return(return_in_usd_matrix):
    """
    Calculate the percentage return from staking across multiple simulations

    Parameters
    ----------
    return_in_usd_matrix : np.ndarray
        The return matrix from staking in USD as a 2D array

    Returns
    -------
    float
        The percentage return from staking
    """
    initial = return_in_usd_matrix[:, 0]
    end = return_in_usd_matrix[:, -1]
    return np.std((end - initial) / initial)


if __name__ == "__main__":

    # Set the initial yield, vol, and drift
    initial_yield = 0.035
    yield_drift_constant = -0.01
    yield_vol = 1.0
    timesteps = 365*5

    # plot_mev_tips(0.1, 0, 1, 1000)
    num_block_proposals = number_of_proposals_within_n_days(365)

    # Create some observed data
    # Define the parameters
    S0 = 3250
    T = 1
    r = 0.05
    kappa = 2.5
    theta = 2.19
    xi = 0.2
    rho = 0.9
    N = timesteps
    num_simulations = 100
    return_in_usd_matrix = np.zeros((num_simulations, timesteps))
    # Read in historical data
    historical_data = pd.read_csv('ETHUSD_d.csv', header=1)

    historical_open_price = np.array(historical_data['open'])

    model = "const"

    if model == "heston":
        # Use historical data to get rho, xi
        rho = calculate_correlation_between_asset_price_and_volatility(
            historical_open_price)
        xi = calculate_vol_of_vol(historical_open_price)
        print("Rho:", rho)
        print("Xi:", xi)

    for i in range(num_simulations):
        # Staking Yield path
        yield_path = stochastic_yield(
            initial_yield, yield_drift_constant, yield_vol, timesteps)
        # plot_stochastic_yield(yield_path)
        # Staking return in ETH including block proposal tips + staking rewards
        return_in_eth = time_varying_staking_return(32, yield_path)

        if model == "heston":
            eth_price_in_usd = heston_model(S0, T, r, kappa, theta, xi, rho, N)
        elif model == "const_vol":
            eth_drift_const = 0.05
            vol = 2.19

            eth_price_in_usd = np.array(const_drift_const_vol_model(
                S0, eth_drift_const, vol, N))
        else:
            eth_price_in_usd = np.ones(timesteps)

        # Multiply the return in ETH by the price of ETH in USD
        return_in_usd = return_in_eth * eth_price_in_usd

        return_in_usd_matrix[i, :] = return_in_usd
        plt.plot(return_in_usd, alpha=0.5)

    # Plot the average at each time step
    plt.plot(np.mean(return_in_usd_matrix, axis=0),
             color='black', linewidth=2)

    print("mean percentage return:",
          calculate_mean_percentage_return(return_in_usd_matrix))
    print("max percentage return:",
          calculate_max_percentage_return(return_in_usd_matrix))
    print("min percentage return:",
          calculate_min_percentage_return(return_in_usd_matrix))
    print("std dev percentage return:",
          calculate_std_dev_percentage_return(return_in_usd_matrix))

    plt.xlabel('Time (days)')
    plt.ylabel('Total ETH')
    plt.show()
