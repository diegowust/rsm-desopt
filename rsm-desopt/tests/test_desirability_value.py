import numpy as np
from rsm_desopt import Goal, composite_desirability

def test_composite_range():
    D = composite_desirability(np.array([200.0]), [Goal("minimize", 100, 300)])
    assert 0.0 < D < 1.0
