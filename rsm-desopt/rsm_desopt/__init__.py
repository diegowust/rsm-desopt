# rsm_desopt: RSM + Derringer-Suich desirability optimization utilities.
from .desirability_opt import (
    Goal, OptimizeConfig, derringer_suich, composite_desirability,
    make_predict_fn_ols, make_predict_fn_ols_realinputs, latin_hypercube,
    optimize_desirability
)
from .diagnostics import (
    lack_of_fit_test_from_model, pareto_standardized_effects, diagnostic_plots
)

__all__ = [
    "Goal","OptimizeConfig","derringer_suich","composite_desirability",
    "make_predict_fn_ols","make_predict_fn_ols_realinputs","latin_hypercube",
    "optimize_desirability",
    "lack_of_fit_test_from_model","pareto_standardized_effects","diagnostic_plots"
]
