from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(1234)


# Strict equivalence tests run in float64 so that a 1e-5 tolerance tests the
# algebra (cache vs recompute, GQA vs reference) rather than fp32 rounding.
ATOL = 1e-5
RTOL = 1e-5
