from greeks import pathwise_delta, pathwise_vega, lr_delta, lr_vega
from black_scholes import bs_delta, bs_vega
import numpy as np
np.random.seed(42)

def test_pathwise_delta_matches_bs(params):
    delta, se = pathwise_delta(**params, n_paths=200_000, type="call")
    bs_d = bs_delta(**params, type="call")
    assert abs(delta - bs_d) < 3 * se

def test_lr_delta_matches_bs(params):
    delta, se = lr_delta(**params, n_paths=200_000, type="call")
    bs_d = bs_delta(**params, type="call")
    assert abs(delta - bs_d) < 3 * se

def test_pathwise_vega_matches_bs(params):
    vega, se = pathwise_vega(**params, n_paths=200_000, type="call")
    bs_v = bs_vega(S0=params["S0"], K=params["K"], r=params["r"], sigma=params["sigma"], T=params["T"])
    assert abs(vega - bs_v) < 3 * se

def test_pathwise_more_precise_than_lr(params):
    _, se_pw = pathwise_delta(**params, n_paths=100_000, type="call")
    _, se_lr = lr_delta(**params, n_paths=100_000, type="call")
    assert se_pw < se_lr
