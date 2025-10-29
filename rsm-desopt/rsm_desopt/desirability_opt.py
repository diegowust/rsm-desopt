"""
desirability_opt.py

Implements Derringer Suich desirability functions and a multi-start optimizer
that mirrors the method used in Minitab's Response Optimizer.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
from scipy.optimize import minimize


#==========================================================================================
#===========================Data Containers (Goals and Configs)============================
#==========================================================================================

@dataclass
class Goal:
    """
    Defines optimization goal for one response.
    - goal: "maximize", "minimize", or "target".
    - low, high: acceptable ranges for the response to be considered in desirability calculation.
    - target: desired target value (only for goal="target").
    - s, t: Derringer Suich exponents (shape of desirability curve).
    - weight: relative importance for composite desirability.
    """
    goal: str
    low: float
    high: float
    target: Optional[float] = None
    s: float = 1.0
    t: float = 1.0
    weight: float = 1.0


@dataclass
class OptimizeConfig:
    """
    Settings for the optimizer.
    - n_starts: number of random starting points (for latin hypercube sampling).
    - random_state: RNG seed for reproducibility.
    - local_method: optimization method (e.g., "L-BFGS-B" or "Powell").
    - maxiter: max iterations per local optimization.
    """
    n_starts: int = 24
    random_state: Optional[int] = 42
    local_method: str = "L-BFGS-B"
    maxiter: int = 500


#==========================================================================================
#==================================Desirability Functions==================================
#==========================================================================================

def derringer_suich(y: float, g: Goal) -> float:
    """
    Calculates Derringer-Suich desirability for one response value y under Goal g.
    - y: response value.
    - g: Goal dataclass instance defining the desirability parameters.
    """
    if g.goal.lower() == "maximize":
        if y < g.low:   return 0.0
        if y > g.high:  return 1.0
        return ((y - g.low) / (g.high - g.low)) ** g.s

    if g.goal.lower() == "minimize":
        if y < g.low:   return 1.0
        if y > g.high:  return 0.0
        return ((g.high - y) / (g.high - g.low)) ** g.t

    if g.goal.lower() == "target":
        if g.target is None:
            raise ValueError("Goal 'target' requires a target value.")
        if y < g.low or y > g.high:
            return 0.0
        if y == g.target:
            return 1.0
        if y < g.target:
            return ((y - g.low) / (g.target - g.low)) ** g.s
        else:
            return ((g.high - y) / (g.high - g.target)) ** g.t

    raise ValueError("goal must be 'maximize', 'minimize', or 'target'.")


def composite_desirability(y: np.ndarray, goals: List[Goal], eps: float = 1e-12) -> float:
    """
    Calculates the weighted geometric mean of individual desirabilities. 
    Returns overall desirability D in [0,1].
    - y: np.array with desirability values for each response.
    - goals: list of Goal dataclass instances corresponding to each response.
    """
    d_vals = []
    w_vals = []
    for yi, gi in zip(y, goals):
        di = max(derringer_suich(float(yi), gi), eps) # here, eps (epsilon) is a safeguard, if desirability is lower than 1e-12 (i.e. 0) when plugged into log it is undefined (inf). To avoid this, in cases where di is too low, the eps value is used. 
        d_vals.append(di)
        w_vals.append(max(gi.weight, eps)) # eps is used as a safeguard here too, to avoid dividing by 0

    d_vals = np.array(d_vals)
    w_vals = np.array(w_vals)

    return float(np.exp(np.sum(w_vals * np.log(d_vals)) / np.sum(w_vals)))


#==========================================================================================
#==========================Prediction wrapper for statsmodels OLS==========================
#==========================================================================================

def make_predict_fn_ols(models: Dict[str, "statsmodels.regression.linear_model.RegressionResultsWrapper"],
                        factor_names: List[str]):

    def predict2(x: np.ndarray) -> np.ndarray:
        row = pd.DataFrame([dict(zip(factor_names, x))]) # zip() pairs factor names with coded values. dict() creates a dictionary from these pairs. pd.DataFrame() turns this dictionary into a single-row dataframe, where the column headers are the factor names, and the values are the coded values.

        ys = [float(mdl.predict(row)[0]) for _, mdl in models.items()] # .items() iterates over the input model dictionary. It ignores the key (using _), and for each model object (mdl), it calls mdl.predict(row)[0] (a statsmodel function that returns the predicted value for the given input row). The [0] grabs the single predicted value from the returned array or dataframe. Each predicted value is converted to float and appended to the list ys.

        return np.array(ys, dtype=float)

    return predict2 # the inner function is returned so it can be called with real-unit vectors to obtain model predictions.

def make_predict_fn_ols_realinputs(
    models: Dict[str, "statsmodels.regression.linear_model.RegressionResultsWrapper"],
    factor_names: List[str],
    x_bounds: List[Tuple[float, float]],
):
    """
    Build a predict(x_real) function for one or more OLS models fit in *coded* units,
    while allowing calls with *real-unit* factors.
    - models: dict {name: fitted OLS model}.
    - factor_names: list of factor names (order must match the x vector supplied to the models).
    - x_bounds: list of (low, high) real-unit bounds for each factor.
    """
    if len(factor_names) != len(x_bounds):
        raise ValueError(f"len(factor_names)={len(factor_names)} must equal len(x_bounds)={len(x_bounds)}")  # a ValueError is raised if the number of factor names doesn't match the number of bounds.
    lo = np.array([a for a, _ in x_bounds], dtype=float)  # a numpy array containing the lower bounds is created
    hi = np.array([b for _, b in x_bounds], dtype=float)  # a numpy array containing the upper bounds is created

    centers = (lo + hi) / 2.0  # centers are calculated.
    scales  = (hi - lo) / 2.0  # the scales that are used to go from real units to coded units for each factor is calculated.

    if np.any(scales <= 0):
        raise ValueError("All bounds must satisfy high > low to compute a valid coding scale.")  # a ValueError is raised if any upper bound is smaller than a lower bound.

    def predict(x_real: np.ndarray) -> np.ndarray:
        """
        Helper function that maps real-unit inputs to coded space, assembles the single-row DataFrame the OLS models expect, and returns the predicted responses as a float array.
        """
        x_real = np.asarray(x_real, dtype=float)  # input is converted to a numpy float array.

        if x_real.shape != centers.shape:
            raise ValueError(f"x_real must have shape {centers.shape}, got {x_real.shape}")  # a ValueError is raised if the length or order of the real-unit vector doesn't match the factors used by the model.
        x_real = np.clip(x_real, lo, hi)  # real-unit inputs are clipped to the provided bounds. This keeps evaluations in range.
        x_coded = (x_real - centers) / scales  # each of the real x inputs is transformed into the corresponding coded values.

        row = pd.DataFrame([dict(zip(factor_names, x_coded))])  # zip() pairs factor names with coded values. dict() creates a dictionary from these pairs. pd.DataFrame() turns this dictionary into a single-row dataframe, where the column headers are the factor names, and the values are the coded values.

        ys = [float(mdl.predict(row)[0]) for _, mdl in models.items()]  # .items() iterates over the input model dictionary. It ignores the key (using _), and for each model object (mdl), it calls mdl.predict(row)[0] (a statsmodel function that returns the predicted value for the given input row). The [0] grabs the single predicted value from the returned array or dataframe. Each predicted value is converted to float and appended to the list ys.

        return np.array(ys, dtype=float)

    return predict  # the inner function is returned so it can be called with real-unit vectors to obtain model predictions.


#==========================================================================================
#===================================Optimization Routine===================================
#==========================================================================================

def latin_hypercube(n_points: int, bounds: List[Tuple[float, float]], rng: np.random.Generator):
    """
    Gnerates a Latin Hypercube sample within bounds (random point in each bin).
    - n_points: number of samples to generate.
    - bounds: list of (low, high) tuples for each factor.
    - rng: numpy random generator instance for reproducibility.
    Returns an (n_points, k) array of samples in real units.
    """
    k = len(bounds) # counts how many factors (columns) there are from the list of (low, high) bounds.

    cut = np.linspace(0.0, 1.0, n_points + 1) # divides the interval [0,1] into n_points with equal distance. cut holds all the bin edges.

    X01 = np.zeros((n_points, k), dtype=float) # prepares an empty n×k matrix that will hold the samples (currently all zeros).
    for j in range(k): # loops over each factor (column).
        u = rng.uniform(low=cut[:-1], high=cut[1:]) # cut[:-1] are the left edges, cut[1:] are the right edges of each bin from where a random point will be drawn. By passing 2 arrays as low and high, rng.uniform makes pairs from these. For example low = [0.1, 0.2] and high = [0.2, 0.3] will lead to [(0.1, 0.2), (0.2, 0.3)], and rng.uniform will take a random value from these 2 intervals. 
        X01[:, j] = rng.permutation(u) # shuffles those n_points samples so their order is randomized in the column.

    lo = np.array([a for a, _ in bounds], dtype=float) # extracts the lower bounds of each factor into an array.
    hi = np.array([b for _, b in bounds], dtype=float) # extracts the upper bounds of each factor into an array.
    return lo + X01 * (hi - lo) # rescales the unit samples in X01 to the actual ranges given to the function.


#==========================================================================================
#=====================================Tie-Break Routine====================================
#==========================================================================================
def _tie_break_score(y: np.ndarray, goals: List[Goal]) -> float:
    """
    Lower is better. Helper function to help break ties in desirability optimization. Calculates a score based on raw responses:
    - maximize: prefer larger y  -> score = -y
    - minimize: prefer smaller y -> score = +y
    - target:   prefer close to target -> score = |y - target| normalized
    Weighted by Goal.weight; target is normalized by (high-low) to be scale-aware.
    """
    score = 0.0
    for yi, g in zip(y, goals):
        range = float(g.high - g.low)
        if g.goal.lower() == "maximize":
            score += -g.weight * float(yi) / range # since lower is better, when we want to maximize, the score is negative of the value (so that higher values give lower scores). It's divided by the range to normalize, making it scale-aware.
        elif g.goal.lower() == "minimize":
            score +=  g.weight * float(yi) / range # in this case, lower values give lower scores, so the score is just the value itself. It's divided by the range to normalize, making it scale-aware.
        elif g.goal.lower() == "target":
            tgt = float(g.target)
            score += g.weight * abs(float(yi) - tgt) / range # here, the score is the distance from the target, normalized by the range (high - low) so that the score is scale-aware. Closer to the target leads to a lower score, which is better.
        else:
            raise ValueError("Unknown goal type in tie-break.")
    return float(score)
# --------------------------------------------------------------------------------- #


def optimize_desirability(
    predict,
    x_bounds: List[Tuple[float, float]],
    goals: List[Goal],
    cfg: OptimizeConfig = OptimizeConfig(),
    integer_indices: Optional[List[int]] = None,
):
    """     
    Maximize Derringer-Suich composite desirability D(x) over given bounds.
    - predict: function that takes (x_1, x_2, ..., x_N) input factor value arrays and returns (y_1, y_2, ..., y_N) response value arrays.
    - x_bounds: list of (low, high) tuples for each factor. 
    - goals: list of Goal dataclass instances for each response.
    - cfg: OptimizeConfig dataclass instance with optimization settings (optional).
    - integer_indices: optional list of factor indices that should be treated as integers.
    Returns (x_best, D_best, info).
    """
    rng = np.random.default_rng(cfg.random_state) # creates a random generation seed using the number stored in random_state in the cfg dataclass
    k = len(x_bounds) # calculates number of factors

    def project_inbounds(x):
        for i, (lo, hi) in enumerate(x_bounds):
            x[i] = np.clip(x[i], lo, hi) # takes each x[i] value and checks if it's between lo and hi. If it's above, it is changed to hi, if it's below, it is changed to lo. 
        return x

    def _apply_integer_and_clip(x: np.ndarray) -> np.ndarray:
        x2 = x.copy()
        if integer_indices: # if any integers are supposed to be rounded, they are.
            for idx in integer_indices:
                x2[idx] = np.round(x2[idx])
        return project_inbounds(x2) # returns the x vector after rounding to integers and clipping to the provided bounds.

    def neg_D(x):
        x_eff = _apply_integer_and_clip(x)  # ensures consistency with integer handling
        y = predict(x_eff) # a prediction is calculated using the factor values x_eff.
        return -composite_desirability(y, goals) # the negative of the desirability is taken, as the function used further on to find an optimum point is the scipy.optimize.minimize function. Finding the minimium of -desirability gives the maximum desirability.  

    # Seeds: LHS + center + corners, these are starting points for the optimization algorithm. LHS is random, and analyzing both centers and corners as well is good practice. 
    lhs = latin_hypercube(cfg.n_starts, x_bounds, rng) # computes n_starts LHS points considering the amount of factors and the rng seed provided to get a random starting point for each factor. 
    center = np.array([[(lo + hi)/2.0 for lo, hi in x_bounds]]) # creates an array containing the center point.
    corners = np.array(np.meshgrid(*[(lo, hi) for lo, hi in x_bounds])).T.reshape(-1, k) # * unpacks the list of tuples, meshgrid builds grids of coordinates covering all combinations, the meshgrid is then turned into an array, transposed and then reshaped and flattened so that the final array contains all possible corners for the given factors. It has shape (2^k, k).
    seeds = np.vstack([corners, center, lhs]) # seeds are stacked, to eventually be used as starting points for local optimization. 

    # Tracking best solution with tie-breaking
    tol = 1e-12  # desirability tie tolerance
    D_best = -np.inf
    best_fun = np.inf
    best_x = None
    y_best = None
    trace = [] # starts an empty list that will be used to log runs.

    for x0 in seeds:
        res = minimize(
            fun=neg_D, x0=x0, method=cfg.local_method, bounds=x_bounds,
            options={"maxiter": cfg.maxiter}
        ) # this minimizes the response of the function neg_D, to find the minimum negative desirability (i.e. max desirability).
        trace.append((res.x.copy(), float(res.fun), res.success, res.nit)) # each iteration is logged and saved in the list "trace"

        # Evaluates current solution consistently with integer rounding
        x_curr = _apply_integer_and_clip(res.x)
        y_curr = predict(x_curr)
        D_curr = composite_desirability(y_curr, goals)

        if (D_curr > D_best + tol):
            # saves the desirability and corresponding x and y if it's the best found so far
            D_best = float(D_curr)
            best_fun = float(res.fun)
            best_x = x_curr.copy()
            y_best = y_curr.copy()
        elif (abs(D_curr - D_best) <= tol) and (y_best is not None):
            # compares the raw-response using _tie_break_score() function if desirabilities are tied
            if _tie_break_score(y_curr, goals) < _tie_break_score(y_best, goals):
                D_best = float(D_curr)
                best_fun = float(res.fun)
                best_x = x_curr.copy()
                y_best = y_curr.copy()

    # Final integer rounding if requested 
    if integer_indices and best_x is not None:
        for idx in integer_indices:
            best_x[idx] = np.round(best_x[idx])
            lo, hi = x_bounds[idx]
            best_x[idx] = np.clip(best_x[idx], lo, hi)

    # Recomputes y_best at finalized x_best
    y_best = predict(best_x)
    return best_x, float(D_best), {"y_best": y_best, "trace": trace}, corners
