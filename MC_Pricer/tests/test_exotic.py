from exotic import asian_mc, geometric_asian_bs, price_lookback_mc, barrier_mc
from european import european_mc
import numpy as np
np.random.seed(42)

def test_geometric_asian_matches_closed_form(params):
    price, se = asian_mc(**params, n=252, n_paths=100_000, average_type="geometric", type="call")
    bs_price = geometric_asian_bs(**params, type="call")
    assert abs(price - bs_price) < 3 * se

def test_arithmetic_asian_exceeds_geometric(params):
    arith_price, _ = asian_mc(**params, n=252, n_paths=100_000, average_type="arithmetic", type="call")
    geo_price, _ = asian_mc(**params, n=252, n_paths=100_000, average_type="geometric", type="call")
    assert arith_price > geo_price

def test_lookback_exceeds_vanilla(params):
    lb_price, _ = price_lookback_mc(S0=params["S0"], r=params["r"], sigma=params["sigma"],
                                       T=params["T"], n=252, n_paths=100_000, type="call")
    vanilla_price, _ = european_mc(**params, n=252, n_paths=100_000, type="call")
    assert lb_price > vanilla_price

def test_barrier_in_out_parity(params):
    B = 130
    price_out, se_out = barrier_mc(**params, B=B, n=252, n_paths=100_000, type="call", barrier_type="up-and-out")
    price_in, se_in = barrier_mc(**params, B=B, n=252, n_paths=100_000, type="call", barrier_type="up-and-in")
    vanilla_price, se_vanilla = european_mc(**params, n=252, n_paths=100_000, type="call")
    combined_se = (se_out**2 + se_in**2 + se_vanilla**2) ** 0.5
    assert abs((price_out + price_in) - vanilla_price) < 4 * combined_se

def test_barrier_extreme_out_matches_vanilla(params):
    B_extreme = 1000
    price_out, _ = barrier_mc(**params, B=B_extreme, n=252, n_paths=100_000, type="call", barrier_type="up-and-out")
    vanilla_price, _ = european_mc(**params, n=252, n_paths=100_000, type="call")
    assert abs(price_out - vanilla_price) < 0.15

def test_barrier_extreme_in_near_zero(params):
    B_extreme = 1000
    price_in, _ = barrier_mc(**params, B=B_extreme, n=252, n_paths=100_000, type="call", barrier_type="up-and-in")
    assert price_in < 0.05
