# Minimal smoke test for the API
import numpy as np
from rsm_desopt import Goal, composite_desirability

y = np.array([200.0])
goals = [Goal(goal="minimize", low=100, high=300)]
D = composite_desirability(y, goals)
print("Composite desirability:", D)
