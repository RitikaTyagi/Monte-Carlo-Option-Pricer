import numpy as np
np.random.seed(42)

from european import european_mc, european_mc_antithetic, european_mc_control_variate, european_sobol_rqmc_error

def test_naive_mc_converges_to_bs(params, bs_call_price):
    price, se = european_mc(**params, n=252, n_paths=200_000, type="call")
    assert abs(price - bs_call_price) < 3 * se

def test_antithetic_converges_to_bs(params, bs_call_price):
    price, se = european_mc_antithetic(**params, n=252, n_paths=200_000, type="call")
    assert abs(price - bs_call_price) < 3 * se

def test_control_variate_converges_to_bs(params, bs_call_price):
    price, se = european_mc_control_variate(**params, n=252, n_paths=200_000, type="call")
    assert abs(price - bs_call_price) < 3 * se

def test_sobol_converges_to_bs(params, bs_call_price):
    price, se = european_sobol_rqmc_error(S0=params["S0"], K=params["K"], r=params["r"],
                                            sigma=params["sigma"], T=params["T"],
                                            n_paths=10_000, type="call")
    assert abs(price - bs_call_price) < 3 * se

def test_antithetic_reduces_variance(params):
    _, se_naive = european_mc(**params, n=252, n_paths=50_000, type="call")
    _, se_anti = european_mc_antithetic(**params, n=252, n_paths=50_000, type="call")
    assert se_anti < se_naive
