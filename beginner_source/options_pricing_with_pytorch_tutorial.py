"""
Option Pricing with PyTorch
===========================

This tutorial demonstrates how to use PyTorch tensors and automatic
differentiation (autograd) to implement option pricing models used in
quantitative finance. We will implement Black-Scholes pricing, Monte Carlo
simulation, and compute option Greeks automatically using torch.autograd —
without deriving a single derivative formula by hand.

**What you will learn:**

- How to implement the Black-Scholes formula using PyTorch tensors
- How to price options across a grid of strikes and expiries using vectorization
- How to simulate thousands of stock price paths using PyTorch for Monte Carlo pricing
- How to compute option Greeks (delta, gamma, vega) automatically using autograd
- How autograd Greeks compare to analytically derived formulas

**Prerequisites:**

- Basic understanding of Python and NumPy
- Familiarity with options concepts (strike, expiry, call/put)
- Basic calculus (derivatives)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available and use it if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

######################################################################
# Black-Scholes Pricing with PyTorch Tensors
# -------------------------------------------
#
# The Black-Scholes formula prices European options using five inputs:
# spot price (S), strike price (K), time to expiry (T), risk-free rate (r),
# and volatility (sigma). We implement this using PyTorch tensors instead
# of NumPy arrays for two reasons:
#
# 1. PyTorch tensors can run on GPU for massive parallelism
# 2. PyTorch tracks operations for automatic differentiation (autograd)
#
# The key difference from a standard NumPy implementation: when we mark
# a tensor with ``requires_grad=True``, PyTorch records every operation
# performed on it, building a computational graph we can differentiate
# through later to compute Greeks.
#

def normal_cdf(x):
    """Cumulative distribution function of the standard normal distribution.
    
    PyTorch does not have a built-in normal CDF, so we implement it using
    the error function (torch.erf), which is mathematically equivalent:
    N(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Compute Black-Scholes price for European options using PyTorch tensors.

    Args:
        S: Spot price (tensor)
        K: Strike price (tensor)
        T: Time to expiry in years (tensor)
        r: Risk-free rate (tensor)
        sigma: Volatility (tensor)
        option_type: 'call' or 'put'

    Returns:
        Option price as a PyTorch tensor
    """
    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)

    if option_type == "call":
        price = S * normal_cdf(d1) - K * torch.exp(-r * T) * normal_cdf(d2)
    else:
        price = K * torch.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)

    return price

######################################################################
# Let's price a single option and verify against our analytical formula.
#

# Define option parameters as PyTorch tensors
S = torch.tensor(100.0)   # spot price
K = torch.tensor(105.0)   # strike price
T = torch.tensor(0.5)     # 6 months to expiry
r = torch.tensor(0.05)    # 5% risk-free rate
sigma = torch.tensor(0.2) # 20% volatility

# Price the call and put options
call_price = black_scholes(S, K, T, r, sigma, option_type="call")
put_price = black_scholes(S, K, T, r, sigma, option_type="put")

print(f"PyTorch Call Price: ${call_price.item():.4f}")
print(f"PyTorch Put Price:  ${put_price.item():.4f}")

# Verify against SciPy analytical result
d1 = (np.log(100/105) + (0.05 + 0.5*0.04)*0.5) / (0.2*np.sqrt(0.5))
d2 = d1 - 0.2*np.sqrt(0.5)
scipy_call = 100*norm.cdf(d1) - 105*np.exp(-0.05*0.5)*norm.cdf(d2)
scipy_put = 105*np.exp(-0.05*0.5)*norm.cdf(-d2) - 100*norm.cdf(-d1)

print(f"\nSciPy Call Price:   ${scipy_call:.4f}")
print(f"SciPy Put Price:    ${scipy_put:.4f}")
print(f"\nDifference (call):  ${abs(call_price.item() - scipy_call):.8f}")
print(f"Difference (put):   ${abs(put_price.item() - scipy_put):.8f}")

# Verify put-call parity
pcp = call_price - put_price - (S - K * torch.exp(-r * T))
print(f"Put-Call Parity error: {pcp.item():.2e}")

######################################################################
# Vectorized Pricing Across a Strike Grid
# ----------------------------------------
#
# One of PyTorch's key strengths is vectorization — applying the same
# operation to an entire tensor of values simultaneously. Instead of
# looping over strikes one at a time, we can price options across a
# full grid in a single operation. This is identical to how a trading
# desk would reprice an entire options chain instantly.
#

# Create a range of strikes from 80% to 120% of spot
strikes = torch.linspace(80, 120, 41)  # 41 strikes from $80 to $120
S_scalar = torch.tensor(100.0)
T_scalar = torch.tensor(0.5)
r_scalar = torch.tensor(0.05)
sigma_scalar = torch.tensor(0.2)

# Price calls and puts across all strikes simultaneously — no loop needed
call_prices = black_scholes(S_scalar, strikes, T_scalar, r_scalar, sigma_scalar, "call")
put_prices = black_scholes(S_scalar, strikes, T_scalar, r_scalar, sigma_scalar, "put")

print(f"Priced {len(strikes)} options simultaneously")
print(f"Strike range: ${strikes[0].item():.0f} to ${strikes[-1].item():.0f}")
print(f"\nSample prices:")
print(f"  $80 strike call:  ${call_prices[0].item():.4f}")
print(f"  $100 strike call: ${call_prices[20].item():.4f}")
print(f"  $120 strike call: ${call_prices[-1].item():.4f}")

# Plot the option price curve
plt.figure(figsize=(10, 5))
plt.plot(strikes.numpy(), call_prices.detach().numpy(), 
         label="Call Price", color="#534AB7", linewidth=2)
plt.plot(strikes.numpy(), put_prices.detach().numpy(), 
         label="Put Price", color="#D85A30", linewidth=2)
plt.axvline(x=100, color="gray", linestyle="--", label="Spot Price ($100)")
plt.xlabel("Strike Price ($)")
plt.ylabel("Option Price ($)")
plt.title("Black-Scholes Option Prices Across Strike Range")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("option_prices.png", dpi=150)
plt.show()
print("Plot saved to option_prices.png")

######################################################################
# Computing Option Greeks with Autograd
# --------------------------------------
#
# Greeks measure how sensitive an option's price is to changes in each
# input. Traditionally these require deriving and implementing separate
# formulas for each Greek. PyTorch's autograd computes them automatically
# by differentiating through the pricing function itself.
#
# This is the same mechanism used to train neural networks — backpropagation
# is just automatic differentiation applied to a loss function. Here we
# apply it to an option pricing function instead.
#
# ``requires_grad=True`` tells PyTorch to track all operations on this
# tensor so gradients can be computed later.
#

def compute_greeks(S_val, K_val, T_val, r_val, sigma_val, option_type="call"):
    """Compute option price and Greeks using automatic differentiation.
    
    Args:
        S_val: Spot price (float)
        K_val: Strike price (float)
        T_val: Time to expiry (float)
        r_val: Risk-free rate (float)
        sigma_val: Volatility (float)
        option_type: 'call' or 'put'
    
    Returns:
        Dictionary containing price and three Greeks
    """
    # Create tensors with requires_grad=True for inputs we want to differentiate
    S = torch.tensor(S_val, requires_grad=True, dtype=torch.float64)
    K = torch.tensor(K_val, dtype=torch.float64)
    T = torch.tensor(T_val, dtype=torch.float64)
    r = torch.tensor(r_val, dtype=torch.float64)
    sigma = torch.tensor(sigma_val, requires_grad=True, dtype=torch.float64)

    # Compute price — PyTorch builds the computational graph here
    price = black_scholes(S, K, T, r, sigma, option_type)

    # Delta: dPrice/dS — first derivative with respect to spot price
    # create_graph=True allows us to differentiate again for gamma
    delta = torch.autograd.grad(price, S, create_graph=True)[0]

    # Gamma: d²Price/dS² — second derivative, derivative of delta
    gamma = torch.autograd.grad(delta, S, create_graph=True)[0]

    # Vega: dPrice/dsigma — first derivative with respect to volatility
    vega = torch.autograd.grad(price, sigma, create_graph=True)[0]

    return {
        "price": price.item(),
        "delta": delta.item(),
        "gamma": gamma.item(),
        "vega": vega.item() / 100,  # per 1 vol point convention
    }

######################################################################
# Let's verify autograd Greeks match our analytical formulas exactly.
#

# Compute Greeks using autograd
params = dict(S_val=100.0, K_val=105.0, T_val=0.5,
              r_val=0.05, sigma_val=0.2, option_type="call")
autograd_greeks = compute_greeks(**params)

# Compute analytical Greeks using scipy for comparison
S_n, K_n, T_n, r_n, sigma_n = 100.0, 105.0, 0.5, 0.05, 0.2
d1 = (np.log(S_n/K_n) + (r_n + 0.5*sigma_n**2)*T_n) / (sigma_n*np.sqrt(T_n))
d2 = d1 - sigma_n * np.sqrt(T_n)

analytical = {
    "price":  S_n*norm.cdf(d1) - K_n*np.exp(-r_n*T_n)*norm.cdf(d2),
    "delta":  norm.cdf(d1),
    "gamma":  norm.pdf(d1) / (S_n * sigma_n * np.sqrt(T_n)),
    "vega":   S_n * norm.pdf(d1) * np.sqrt(T_n) / 100,
}

# Print comparison table
print(f"{'Greek':<10} {'Autograd':>12} {'Analytical':>12} {'Difference':>14}")
print("-" * 52)
for greek in ["price", "delta", "gamma", "vega"]:
    auto = autograd_greeks[greek]
    anal = analytical[greek]
    diff = abs(auto - anal)
    print(f"{greek:<10} {auto:>12.6f} {anal:>12.6f} {diff:>14.2e}")

######################################################################
# Now let's plot how each Greek varies across a range of spot prices.
#

spot_range = np.linspace(70, 130, 60)
deltas, gammas, vegas = [], [], []

for s in spot_range:
    g = compute_greeks(s, K_val=105.0, T_val=0.5,
                       r_val=0.05, sigma_val=0.2, option_type="call")
    deltas.append(g["delta"])
    gammas.append(g["gamma"])
    vegas.append(g["vega"])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(spot_range, deltas, color="#534AB7", linewidth=2)
axes[0].axvline(x=105, color="orange", linestyle="--", label="Strike K")
axes[0].set_title("Delta vs Spot Price")
axes[0].set_xlabel("Spot Price ($)")
axes[0].set_ylabel("Delta")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(spot_range, gammas, color="#1D9E75", linewidth=2)
axes[1].axvline(x=105, color="orange", linestyle="--", label="Strike K")
axes[1].set_title("Gamma vs Spot Price")
axes[1].set_xlabel("Spot Price ($)")
axes[1].set_ylabel("Gamma")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(spot_range, vegas, color="#D85A30", linewidth=2)
axes[2].axvline(x=105, color="orange", linestyle="--", label="Strike K")
axes[2].set_title("Vega vs Spot Price")
axes[2].set_xlabel("Spot Price ($)")
axes[2].set_ylabel("Vega")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle("Option Greeks Computed via PyTorch Autograd", fontsize=14)
plt.tight_layout()
plt.savefig("autograd_greeks.png", dpi=150)
plt.show()
print("Greeks plot saved to autograd_greeks.png")

######################################################################
# Monte Carlo Option Pricing with PyTorch
# ----------------------------------------
#
# Monte Carlo simulation estimates option prices by simulating thousands
# of possible future stock price paths. PyTorch accelerates this with
# vectorized tensor operations — all paths are simulated simultaneously
# rather than in a loop. On a GPU, this can be orders of magnitude faster
# than a CPU-based NumPy implementation.
#
# We simulate terminal stock prices using the GBM closed-form solution:
#
# S_T = S_0 * exp((r - sigma^2/2) * T + sigma * sqrt(T) * Z)
#
# where Z ~ N(0,1) is a standard normal random variable.
#

def monte_carlo_price(S, K, T, r, sigma, option_type="call", n_sims=100000):
    """Price a European option using Monte Carlo simulation with PyTorch.

    Args:
        S: Spot price (float)
        K: Strike price (float)
        T: Time to expiry in years (float)
        r: Risk-free rate (float)
        sigma: Volatility (float)
        option_type: 'call' or 'put'
        n_sims: Number of simulations

    Returns:
        Dictionary with price estimate and confidence interval
    """
    # Generate random standard normal draws — all at once, vectorized
    torch.manual_seed(0)
    Z = torch.randn(n_sims, device=device)

    # Simulate terminal stock prices using GBM closed-form solution
    S_T = S * torch.exp((r - 0.5 * sigma**2) * T + sigma * torch.sqrt(torch.tensor(T)) * Z)

    # Compute payoffs
    if option_type == "call":
        payoffs = torch.maximum(S_T - K, torch.zeros_like(S_T))
    else:
        payoffs = torch.maximum(K - S_T, torch.zeros_like(S_T))

    # Discount average payoff to present value
    price = torch.exp(torch.tensor(-r * T)) * payoffs.mean()

    # Compute 95% confidence interval
    std_error = payoffs.std() / torch.sqrt(torch.tensor(float(n_sims)))
    confidence = 1.96 * std_error * torch.exp(torch.tensor(-r * T))

    return {
        "price": price.item(),
        "ci_lower": (price - confidence).item(),
        "ci_upper": (price + confidence).item(),
        "std_error": std_error.item(),
    }


# Run Monte Carlo and compare to Black-Scholes
bs_price = black_scholes(
    torch.tensor(100.0), torch.tensor(105.0),
    torch.tensor(0.5), torch.tensor(0.05),
    torch.tensor(0.2), "call"
).item()

# Set seed for reproducible Monte Carlo results
torch.manual_seed(42)
mc_result = monte_carlo_price(100.0, 105.0, 0.5, 0.05, 0.2, "call", n_sims=100000)

print(f"Black-Scholes price:  ${bs_price:.4f}")
print(f"Monte Carlo price:    ${mc_result['price']:.4f}")
print(f"95% CI:               [${mc_result['ci_lower']:.4f}, ${mc_result['ci_upper']:.4f}]")
print(f"Difference:           ${abs(mc_result['price'] - bs_price):.4f}")
print(f"BS inside CI:         {mc_result['ci_lower'] < bs_price < mc_result['ci_upper']}")

######################################################################
# Monte Carlo Convergence Analysis
# ----------------------------------
#
# Monte Carlo estimates improve as we increase the number of simulations.
# The error decreases at a rate of 1/sqrt(N) — known as the Monte Carlo
# convergence rate. Doubling precision requires quadrupling simulations.
# Here we visualize this convergence toward the Black-Scholes price.
#

sim_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
mc_prices = []
ci_lowers = []
ci_uppers = []

for n in sim_counts:
    result = monte_carlo_price(100.0, 105.0, 0.5, 0.05, 0.2, "call", n_sims=n)
    mc_prices.append(result["price"])
    ci_lowers.append(result["ci_lower"])
    ci_uppers.append(result["ci_upper"])

# Plot convergence
plt.figure(figsize=(10, 5))
plt.semilogx(sim_counts, mc_prices, "o-",
             color="#1D9E75", linewidth=2, label="MC Price")
plt.fill_between(sim_counts, ci_lowers, ci_uppers,
                 alpha=0.2, color="#1D9E75", label="95% CI")
plt.axhline(y=bs_price, color="#534AB7", linestyle="--",
            linewidth=2, label=f"BS Price (${bs_price:.4f})")
plt.xlabel("Number of Simulations (log scale)")
plt.ylabel("Option Price ($)")
plt.title("Monte Carlo Convergence to Black-Scholes Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mc_convergence.png", dpi=150)
plt.show()
print("Convergence plot saved to mc_convergence.png")

# Print convergence table
print(f"\n{'Simulations':<14} {'MC Price':>10} {'BS Price':>10} {'Error':>10} {'CI Width':>10}")
print("-" * 58)
for i, n in enumerate(sim_counts):
    error = abs(mc_prices[i] - bs_price)
    ci_width = ci_uppers[i] - ci_lowers[i]
    print(f"{n:<14} {mc_prices[i]:>10.4f} {bs_price:>10.4f} {error:>10.4f} {ci_width:>10.4f}")

######################################################################
# Conclusion
# -----------
#
# In this tutorial we demonstrated how PyTorch enables quantitative
# finance applications through three key capabilities:
#
# 1. **Vectorized pricing** — pricing an entire options chain across
#    41 strikes in a single tensor operation, with no Python loops.
#
# 2. **Automatic differentiation** — computing delta, gamma, and vega
#    automatically via ``torch.autograd.grad``, matching analytical
#    formulas to within floating point machine precision (< 1e-8).
#
# 3. **GPU-accelerated Monte Carlo** — simulating 100,000 GBM paths
#    simultaneously using PyTorch tensors, with the same code running
#    on CPU or GPU without modification.
#
# These same tools underpin modern quantitative research — neural network
# pricing models, differentiable portfolio optimization, and risk-neutral
# calibration all rely on automatic differentiation through financial
# models. The Greeks you computed here via autograd are mathematically
# identical to the gradients used to train a neural network pricer.
#
# **Further reading:**
#
# - `PyTorch Autograd documentation <https://pytorch.org/docs/stable/autograd.html>`_
# - `Black-Scholes model (Wikipedia) <https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model>`_
# - `John Hull, Options Futures and Other Derivatives <https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938>`_
#

print("\nTutorial complete.")
print(f"Files saved: option_prices.png, autograd_greeks.png, mc_convergence.png")