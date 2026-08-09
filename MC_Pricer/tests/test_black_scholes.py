import numpy as np
from black_scholes import Black_Scholes
import numpy as np
np.random.seed(42)

def test_bs_call_atm(params, bs_call_price):
    price = Black_Scholes(**params, type="call")
    assert abs(price - bs_call_price) < 0.001

def test_bs_off_atm_regression():
    # this exact case caught a real precedence bug earlier in the project - keep permanently
    price = Black_Scholes(100, 110, 0.05, 0.2, 1.0, type="call")
    assert abs(price - 6.0401) < 0.001

def test_bs_put_call_parity(params):
    call = Black_Scholes(**params, type="call")
    put = Black_Scholes(**params, type="put")
    S0, K, r, T = params["S0"], params["K"], params["r"], params["T"]
    assert abs((call - put) - (S0 - K*np.exp(-r*T))) < 1e-6
