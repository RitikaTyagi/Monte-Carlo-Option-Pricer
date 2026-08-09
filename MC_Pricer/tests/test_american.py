from american import american_longstaff_schwartz, american_binomial, american_longstaff_schwartz_fast
from european import european_mc
import numpy as np
np.random.seed(42)

def test_american_put_exceeds_european(params):
    am_price, _ = american_longstaff_schwartz(**params, n=252, n_paths=100_000, type="put")
    eu_price, _ = european_mc(**params, n=252, n_paths=100_000, type="put")
    assert am_price > eu_price

def test_american_call_equals_european(params):
    am_price, se_am = american_longstaff_schwartz(**params, n=252, n_paths=100_000, type="call")
    eu_price, se_eu = european_mc(**params, n=252, n_paths=100_000, type="call")
    combined_se = (se_am**2 + se_eu**2) ** 0.5
    assert abs(am_price - eu_price) < 5 * combined_se

def test_ls_matches_binomial(params):
    ls_price, se = american_longstaff_schwartz(**params, n=252, n_paths=100_000, type="put")
    binom_price = american_binomial(S0=params["S0"], K=params["K"], r=params["r"],
                                       sigma=params["sigma"], T=params["T"], n=1000, type="put")
    assert abs(ls_price - binom_price) < 3 * se

def test_ls_fast_matches_original(params):
    price_orig, se_orig = american_longstaff_schwartz(**params, n=252, n_paths=100_000, type="put")
    price_fast, se_fast = american_longstaff_schwartz_fast(**params, n=252, n_paths=100_000, type="put")
    assert abs(price_orig - price_fast) < 3 * max(se_orig, se_fast)
