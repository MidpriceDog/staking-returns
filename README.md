# staking-returns
Compare hypothetical returns from staking ETH by participating in validating
transactions on the Ethereum network

# Assumptions

- Staking rewards are compounded annually
- Risk of slashing is not considered
- Staking rewards are not taxed
- Daily proposal probability is a constant 0.75%
- Proposal reward is drawn from a log-normal distribution
- Proposal reward is not taxed
- Yield from staking follows Brownian motion with some average drift and volatility.
  Mean is equal to the historical average yield since switch from PoW to PoS. Vol
  is calculated from the historical yield data.

# How to use

1. Clone the repository
2. `cd` into the repository on your local machine
2. Run `python3 staking_returns.py`


