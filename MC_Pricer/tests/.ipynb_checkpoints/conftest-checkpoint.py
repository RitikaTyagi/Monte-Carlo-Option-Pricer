import pytest
@pytest.fixture
def params():
    return dict(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)

@pytest.fixture
def bs_call_price():
    return 10.4506  # ground truth, cross-checked against an external BS calculator

@pytest.fixture
def bs_put_price():
    return 5.5735  # compute and verify this once by hand before hardcoding it
