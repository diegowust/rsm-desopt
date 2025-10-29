# rsm-desopt

Utilities for Response Surface Methodology (RSM), Derringer–Suich desirability optimization,  
and model diagnostics (LOF test, Pareto t-chart, residual plots).

---

## Installation

This package can be installed directly from GitHub using `pip`.  
A **Git installation is required** before running the installation commands.

### Requirements
- Python ≥ 3.11  
- Git installed and available in the system PATH (`git --version` should work from the terminal)

### Install command

Install the latest stable release (v0.1.0):
```bash
pip install "rsm-desopt @ git+https://github.com/diegowust/rsm-desopt.git@v0.1.0"
```

Or install the latest commit on the main branch:
```bash
pip install "rsm-desopt @ git+https://github.com/diegowust/rsm-desopt.git@main"
```

---

## Quickstart

```python
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from rsm_desopt import (
    Goal, OptimizeConfig, optimize_desirability,
    make_predict_fn_ols_realinputs
)

# Example (fit in coded units)
# model = ols("Result ~ A + B + C + D + E"
#             " + I(A**2) + I(B**2) + I(C**2) + I(D**2) + I(E**2)"
#             " + (A + B + C + D + E)**2",
#             data=df_coded).fit()

factor_names = ["A", "B", "C", "D", "E"]
# Real-unit bounds for each factor (example values)
x_bounds = [(0, 10), (-5, 5), (8, 12), (400, 900), (1, 2)]

# Build a real-input predictor from a coded-OLS model
# predict = make_predict_fn_ols_realinputs({"response": model}, factor_names, x_bounds)

# Define goal(s) and optimize (example: minimize a single response)
goals = [Goal(goal="minimize", low=100, high=300, weight=1.0)]

# cfg = OptimizeConfig(n_starts=24, random_state=42)
# x_best, D_best, info, corners = optimize_desirability(
#     predict, x_bounds, goals, cfg,
#     integer_indices=[]  # e.g., [2] if C must be integer
# )
```

---

## Diagnostics

```python
from rsm_desopt import (
    lack_of_fit_test_from_model,
    pareto_standardized_effects,
    diagnostic_plots
)

# Example usage with the fitted model in coded units:
# factors = ["A", "B", "C", "D", "E"]
# lof = lack_of_fit_test_from_model(model, df_coded, factors=factors, response_col="Result")
# pareto = pareto_standardized_effects(model, factors=factors, only_main=False, alpha=0.05)
# diagnostic_plots(model)
```
