
import torch
import math

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    print("CUDA is available. You can use the GPU!")
else:
    print("CUDA is not available. Using the CPU instead.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_brownian(n_paths, n_steps):

    dW = torch.randn(n_paths, n_steps)
    dW[:, 0] = 0.0
    W = dW.cumsum(dim=-1)

    return W

def generate_geometric_brownian(n_paths=2, n_steps=5, sigma=0.2, dt=1/250):

    t = torch.arange(n_steps) * dt
    W = generate_brownian(n_paths, n_steps)

    return torch.exp((-0.5 * sigma**2) * t + sigma *  torch.sqrt(torch.tensor(dt)) * W)

spot = generate_geometric_brownian(2,10)
spot

def european_payoff(input, call=True, strike=1.0):

    if call:
        return torch.nn.functional.relu(input[..., -1] - strike)
    else:
        return torch.nn.functional.relu(strike - input[..., -1])

european_payoff(spot, call=False)

def lookback_payoff(input, call=True, strike=1.0):

    if call:
        return torch.nn.functional.relu(input.max(dim=-1).values - strike)
    else:
        return torch.nn.functional.relu(strike - input.min(dim=-1).values)

lookback_payoff(spot)

def american_binary_payoff(input, call=True, strike=1.0):

    if call:
        return (input.max(dim=-1).values >= strike).to(input)
    else:
        return (input.min(dim=-1).values <= strike).to(input)

american_binary_payoff(spot, strike=1.05)

def european_binary_payoff(input, call=True, strike=1.0):

    if call:
        return (input[..., -1] >= strike).to(input)
    else:
        return (input[..., -1] <= strike).to(input)

european_binary_payoff(spot)

def pl(spot, unit, cost=None, payoff=None):

    output = unit[..., :-1].mul(spot.diff(dim=-1)).sum(dim=(-2,-1))

    if payoff is not None:
        output -= payoff.squeeze(-1)

    if cost is not None:
        c = torch.tensor(cost).unsqueeze(0).unsqueeze(-1)
        output -= (spot[..., :-1].mul(unit.diff(dim=-1)).abs() * c).sum(dim=(-2, -1))

    return output

def compute_hedge(model, ds):
    outputs = []
    for i in ds:
        outputs.append(model(i))

    return torch.cat(outputs, dim=-1)

def compute_portfolio_2(model, ds, payoff):

    unit = compute_hedge(model, ds)

    return pl(spot.to(device), unit.to(device), payoff=payoff.to(device))

def entropic_risk_measure(x):
    return torch.logsumexp(-x, 0) - math.log(x.size(0))

spot = generate_geometric_brownian(2000,250).unsqueeze(1)
spot

def time_to_maturity(spot, dt):

    n_paths, _, n_steps = spot.size()
    t = torch.arange(n_steps) * dt

    return (t[-1] - t).unsqueeze(0).expand(n_paths, 1, -1)

def moneyness(spot, strike):
    return spot/strike

def volatility(spot, vol):
    return torch.ones_like(spot) * vol

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = torch.cat(data, dim=1)

    def __len__(self):
        return self.data.size(2)

    def __getitem__(self, index):
        return self.data[:, :, index].unsqueeze(1).to(device)

lm = moneyness(spot, 1.1)
t = time_to_maturity(spot, 0.004)
v = volatility(spot, 0.2)

ds = MyDataset([lm, t, v])

from torch import nn

class MyModel(nn.Module):

    def __init__(self, n_inputs):  # <- We take n_inputs
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        self.register_buffer("prev_hegde", None)

    def forward(self, x):
        if self.prev_hegde is None:
            self.register_buffer("prev_hedge", torch.zeros(x.size(0), x.size(1), 1).to(device))

        new_x = torch.cat([x, self.prev_hedge], dim=-1)
        out = self.model(new_x)
        self.prev_hedge = out.detach()

        return out

m = MyModel(4).to(device)

optimizer = torch.optim.Adam(m.parameters())

for i in range(100):
    optimizer.zero_grad()
    cash = compute_portfolio_2(m, ds, american_binary_payoff(spot, strike=1.1))
    loss = entropic_risk_measure(cash)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(loss)

import numpy as np

def simulate_gbm(S0, T, r, sigma, M, I):
    """ Simulate I paths with M time steps """
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        Z = np.random.standard_normal(I)
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return S

def american_binary_option_monte_carlo(S0, K, T, r, sigma, M, I, option_type='call', payout=1.0):
    """ Monte Carlo pricing for American binary options """
    S = simulate_gbm(S0, T, r, sigma, M, I)
    payoff = np.zeros(I)

    for i in range(I):
        if option_type == 'call':
            if np.any(S[:, i] > K):
                payoff[i] = payout
        elif option_type == 'put':
            if np.any(S[:, i] < K):
                payoff[i] = payout

    price = np.exp(-r * T) * np.mean(payoff)
    return price

# Parameters
S0 = 1    # Initial stock price
K = 1.1     # Strike price
T = 1.0     # Time to maturity in years
r = 0.00    # Risk-free interest rate
sigma = 0.2 # Volatility
M = 50      # Number of time steps
I = 10000   # Number of simulations
payout = 1.0 # Fixed payout for the binary option

# American Binary Call Option
binary_call_price = american_binary_option_monte_carlo(S0, K, T, r, sigma, M, I, option_type='call', payout=payout)
print(f"American Binary Call Option Price: {binary_call_price:.2f}")

# American Binary Put Option
binary_put_price = american_binary_option_monte_carlo(S0, K, T, r, sigma, M, I, option_type='put', payout=payout)
print(f"American Binary Put Option Price: {binary_put_price:.2f}")

compute_hedge(m, ds)

import matplotlib.pyplot as plt

compute_hedge(m, ds).size()

plt.plot(compute_hedge(m, ds).squeeze(1).cpu().detach().numpy().T)

plt.show()

import numpy as np
from scipy.stats import norm

d1 = (np.log(1 / 1.1) + (0 + 0.5 * 0.2 ** 2) * 1) / (0.2 * np.sqrt(1))
d2 = d1 - 0.2 * np.sqrt(1)

1 * norm.cdf(d1) - 1.1 * norm.cdf(d2)

import math
from scipy.stats import norm

def black_scholes_call(S, X, T, r, sigma):
    d1 = (math.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * norm.cdf(d1) - X * math.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, X, T, r, sigma):
    d1 = (math.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    put_price = X * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Example usage:
S0 = 1    # Current price of the underlying asset
X = 1.1     # Strike price of the option
T = 1       # Time to expiration in years
r = 0.00    # Risk-free interest rate
sigma = 0.2 # Volatility of the underlying asset

call_price = black_scholes_call(S0, X, T, r, sigma)
put_price = black_scholes_put(S0, X, T, r, sigma)

print(f"Call Option Price: {call_price:.2f}")
print(f"Put Option Price: {put_price:.2f}")

import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes option price for European call or put options.

    Parameters:
    S (float): Current stock price (underlying asset price)
    K (float): Option strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)
    option_type (str): Type of the option - "call" or "put"

    Returns:
    float: Theoretical price of the option
    """

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        # Call option price using Black-Scholes formula
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        # Put option price using Black-Scholes formula
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return option_price

# Example usage:
S = 1  # Current stock price
K = 1.1  # Option strike price
T = 1    # Time to maturity (1 year)
r = 0.00  # Risk-free interest rate (5%)
sigma = 0.2  # Volatility (20%)

# Calculate the price of a European call option
call_price = black_scholes(S, K, T, r, sigma, option_type="call")
print(f"Call Option Price: {call_price:.2f}")

# Calculate the price of a European put option
put_price = black_scholes(S, K, T, r, sigma, option_type="put")
print(f"Put Option Price: {put_price:.2f}")

import numpy as np

def simulate_gbm(S0, T, r, sigma, M, I):
    """ Simulate I paths with M time steps """
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        Z = np.random.standard_normal(I)
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return S

def lookback_option_monte_carlo(S0, K, T, r, sigma, M, I, option_type='call', floating_strike=True):
    """ Monte Carlo pricing for lookback options """
    S = simulate_gbm(S0, T, r, sigma, M, I)

    if floating_strike:
        if option_type == 'call':
            payoff = S[-1] - np.min(S, axis=0)
        elif option_type == 'put':
            payoff = np.max(S, axis=0) - S[-1]
    else:
        if option_type == 'call':
            payoff = np.max(S, axis=0) - K
        elif option_type == 'put':
            payoff = K - np.min(S, axis=0)

    payoff = np.maximum(payoff, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    return price

# Parameters
S0 = 1    # Initial stock price
K = 1.1     # Strike price (used for fixed strike options)
T = 1.0     # Time to maturity in years
r = 0.00    # Risk-free interest rate
sigma = 0.2 # Volatility
M = 50      # Number of time steps
I = 10000   # Number of simulations

# Floating Strike Lookback Call Option
lookback_call_price = lookback_option_monte_carlo(S0, K, T, r, sigma, M, I, option_type='call', floating_strike=True)
print(f"Floating Strike Lookback Call Option Price: {lookback_call_price:.2f}")

# Floating Strike Lookback Put Option
lookback_put_price = lookback_option_monte_carlo(S0, K, T, r, sigma, M, I, option_type='put', floating_strike=True)
print(f"Floating Strike Lookback Put Option Price: {lookback_put_price:.2f}")

# Fixed Strike Lookback Call Option
lookback_fixed_call_price = lookback_option_monte_carlo(S0, K, T, r, sigma, M, I, option_type='call', floating_strike=False)
print(f"Fixed Strike Lookback Call Option Price: {lookback_fixed_call_price:.2f}")

# Fixed Strike Lookback Put Option
lookback_fixed_put_price = lookback_option_monte_carlo(S0, K, T, r, sigma, M, I, option_type='put', floating_strike=False)
print(f"Fixed Strike Lookback Put Option Price: {lookback_fixed_put_price:.2f}")

import math
from scipy.stats import norm

def lookback_call_floating(S, sigma, r, T):
    """ Price of a floating strike lookback call option """
    d1 = (r + 0.5 * sigma**2) * T / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * (norm.cdf(d1) - sigma**2 / (2 * r) * (norm.cdf(-d1) - math.exp(-r * T) * norm.cdf(-d2)))
    return call_price

def lookback_put_floating(S, sigma, r, T):
    """ Price of a floating strike lookback put option """
    d1 = (r + 0.5 * sigma**2) * T / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    put_price = S * (norm.cdf(-d1) + sigma**2 / (2 * r) * (norm.cdf(d1) - math.exp(-r * T) * norm.cdf(d2)))
    return put_price

# Example usage:
S0 = 1    # Current price of the underlying asset
T = 1       # Time to expiration in years
r = 0.00    # Risk-free interest rate
sigma = 0.2 # Volatility of the underlying asset

lookback_call_price = lookback_call_floating(S0, sigma, r, T)
lookback_put_price = lookback_put_floating(S0, sigma, r, T)

print(f"Floating Strike Lookback Call Option Price: {lookback_call_price:.2f}")
print(f"Floating Strike Lookback Put Option Price: {lookback_put_price:.2f}")

import numpy as np

def monte_carlo_lookback(S0, K, T, r, sigma, n_simulations=10000, n_steps=252, option_type='call', fixed_strike=True):
    """
    Monte Carlo simulation for lookback options pricing.

    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price (only for fixed strike options)
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the stock (annualized)
    n_simulations (int): Number of simulations
    n_steps (int): Number of steps in each simulation
    option_type (str): 'call' or 'put'
    fixed_strike (bool): True for fixed strike, False for floating strike

    Returns:
    float: Theoretical price of the lookback option
    """

    dt = T / n_steps
    discount_factor = np.exp(-r * T)

    # Simulate asset price paths
    S = np.zeros((n_simulations, n_steps + 1))
    S[:, 0] = S0
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_simulations)
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    # Calculate the option payoff based on the option type and strike type
    if fixed_strike:
        if option_type == 'call':
            payoff = np.maximum(np.max(S, axis=1) - K, 0)
        elif option_type == 'put':
            payoff = np.maximum(K - np.min(S, axis=1), 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    else:
        if option_type == 'call':
            payoff = S[:, -1] - np.min(S, axis=1)
        elif option_type == 'put':
            payoff = np.max(S, axis=1) - S[:, -1]
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    # Calculate the average discounted payoff
    option_price = discount_factor * np.mean(payoff)

    return option_price

# Example usage:
S0 = 1.0  # Current stock price
K = 1.1  # Strike price
T = 1     # Time to maturity (1 year)
r = 0.0  # Risk-free interest rate (5%)
sigma = 0.2  # Volatility (20%)

# Pricing a fixed strike lookback call option
lookback_call_fixed = monte_carlo_lookback(S0, K, T, r, sigma, option_type='call', fixed_strike=True)
print(f"Fixed Strike Lookback Call Option Price: {lookback_call_fixed:.2f}")

# Pricing a floating strike lookback put option
lookback_put_floating = monte_carlo_lookback(S0, K, T, r, sigma, option_type='put', fixed_strike=False)
print(f"Floating Strike Lookback Put Option Price: {lookback_put_floating:.2f}")

import numpy as np
from scipy.stats import norm

def european_binary_option(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the price of a European binary option using the Black-Scholes formula.

    Parameters:
    S (float): Current stock price (underlying asset price)
    K (float): Option strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)
    option_type (str): Type of the option - "call" or "put"

    Returns:
    float: Theoretical price of the European binary option
    """

    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == 'call':
        # Binary call option price
        option_price = np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        # Binary put option price
        option_price = np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return option_price

# Example usage:
S = 100  # Current stock price
K = 100  # Option strike price
T = 1    # Time to expiration (1 year)
r = 0.05  # Risk-free interest rate (5%)
sigma = 0.2  # Volatility (20%)

# Price a European binary call option
binary_call_price = european_binary_option(S, K, T, r, sigma, option_type='call')
print(f"European Binary Call Option Price: {binary_call_price:.4f}")

# Price a European binary put option
binary_put_price = european_binary_option(S, K, T, r, sigma, option_type='put')
print(f"European Binary Put Option Price: {binary_put_price:.4f}")
