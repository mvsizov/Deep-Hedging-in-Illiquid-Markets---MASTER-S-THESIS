import numpy as np
import random

def simulate_heston(
        S0 = 1.0, # initial price  
        v0 = 0.07, # initial volaility
        rho = -0.7, # correlation between variance and price processes
        kappa = 8.0, # speed with which variance reverts to long-run
        theta = 0.07,# long run mean of variance
        sigma = 1.0,  # volatility of volatility
        r = 0.0,  # risk-free rate
        N = 30,  # num of trading periods
        days_in_year = 252, # number of days in year
        M = 200_000,  # number simulated paths
        seed = None
):
    """
    Heston model which simulates Heston paths.

    Returns:
    1. array of asset prices paths
    2. array of volatilities (square roots of variances)
    """
    # set seed to fix calculations if necessary
    if seed is not None:
        np.random.seed(seed)

    # set time step in years
    dt = 1.0 / days_in_year

    '''
    Below are parameters for non-central chi-squared
    which are used in CIR process
    '''
    # degrees of freedom non-center for chi-squared
    deg_freedom = 4.0 * kappa * theta / (sigma**2)
    # scale for non-center chi-squared
    scale = (sigma**2) * (1.0 - np.exp(-kappa * dt)) / (4.0 * kappa)

    # create empty arrays for prices and volatilities
    S = np.empty((N + 1, M), dtype=float)
    V = np.empty((N + 1, M), dtype=float)
    # set initial values of prices and volatility to S0,v0
    S[0, :] = S0
    V[0, :] = v0


    # loop over timesteps
    for t in range(1, N + 1):
        '''
        Broadie–Kaya exact method is applied  
        '''
        # computing lambda_nc (non-centrality parameter)
        lambda_nc = V[t - 1, :] * np.exp(-kappa * dt) / scale
        # randomly sample from chisquare
        chi_random_sample = np.random.noncentral_chisquare(deg_freedom, lambda_nc, size=M)
        # scaling to get next volatility
        V_next = scale * chi_random_sample

        '''
        approximating integrated variance by trapezoid - we need to take integral from t-1 to t,
        which is approximated by 0.5 * (v_t-1 + v_t)dt
        '''
        IV_approx = 0.5 * (V[t - 1, :] + V_next) * dt

        # draw random standard normals
        Z = np.random.randn(M)
        # calculate drift parameter
        drift = r * dt
        # correlation between Brownian processes of price and variance
        corr_term = (rho / sigma) * (V_next - V[t - 1, :] - kappa * (theta - V[t - 1, :]) * dt)
        # independent share of randomness in price process
        diffusion = np.sqrt((1.0 - rho ** 2) * IV_approx) * Z
        # log of next prices - we begin from yesterdays price and
        # add determenistic component + volatility shock part which is correlated with the price and independet shock
        # also added Ito correction which corrects the concavity of log 
        log_S_next = np.log(S[t - 1, :]) + drift - 0.5 * IV_approx + corr_term + diffusion
        # converting log of prices back to prices 
        S[t, :] = np.exp(log_S_next)
        V[t, :] = V_next

    # return price paths and volatility paths
    return S, np.sqrt(V) 