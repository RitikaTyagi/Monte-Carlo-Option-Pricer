import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
np.random.seed(42)

import import_ipynb
import pytest

@pytest.fixture
def params():
    return dict(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)

@pytest.fixture
def bs_call_price():
    return 10.4506

@pytest.fixture
def bs_put_price():
    return 5.5735
