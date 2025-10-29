# rsm-desopt

Utilities for Response Surface Methodology (RSM), Derringer-Suich desirability optimization,
and model diagnostics (LOF test, Pareto t-chart, residual plots).

## Install (editable dev)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

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
# model = ols("Result ~ D + TS + T + V + I(D**2) + I(TS**2) + I(T**2)", data=df_coded).fit()

factor_names = ["D","TS","T","V","GFR"]
x_bounds = [(-17.5,-15.5),(300,900),(1,3),(8,12),(40,80)]
# predict = make_predict_fn_ols_realinputs({"linewidth": model}, factor_names, x_bounds)

goals = [Goal(goal="minimize", low=100, high=300, weight=1.0)]
# x_best, D_best, info, corners = optimize_desirability(predict, x_bounds, goals, OptimizeConfig(), integer_indices=[2])
```

## Diagnostics
```python
from rsm_desopt import lack_of_fit_test_from_model, pareto_standardized_effects, diagnostic_plots
# lack_of_fit_test_from_model(model, df, factors=["D","TS","T","V","GFR"], response_col="Result")
# pareto_standardized_effects(model, factors=["D","TS","T","V","GFR"], only_main=False, alpha=0.05)
# diagnostic_plots(model)
```

## Testing
```bash
pytest -q
```
