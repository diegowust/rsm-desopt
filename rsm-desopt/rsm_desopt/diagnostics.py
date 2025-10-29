import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

#==========================================================================================
#=================================Lack of Fit Test=========================================
#==========================================================================================
def lack_of_fit_test_from_model(
    model,
    df: pd.DataFrame,
    factors: list,
    response_col: str = "result",
    print_summary: bool = True
):
    """
    Computes pure error and lack-of-fit (LOF) given an already fitted statsmodels model.

    Parameters
    ----------
    model : fitted OLS model (e.g., result of ols(...).fit()).
    df : dataframe with response and factor columns.
    factors : string list containing the names of all design factors.
    response_col : string that contains the name of the result column.
    print_summary : boolean, if True, prints a summary.

    Returns
    -------
    dictionary containing sums of squares, dfs, mean squares, F-test, and p-value.
    """

    # Residual SS and df from the model
    SSE_all = float(np.sum(model.resid ** 2)) # calculates the residual sum of squares 
    df_res = int(model.df_resid) # takes the degrees of freedom for the residuals

    # Pure error from exact replicates
    SS_within = 0.0
    df_pure = 0
    for _, g in df.groupby(factors, dropna=False): # groupby separates the df into sub-dataframes that have the same combination of values in the columns provided. This means that it will take the original df and create sub-df where each one contains the exact same combination of factor settings (A=–1, B=+1, … H=–1, for example).
        n = len(g)
        if n > 1: # if the sub-df 'g' has more than 1 row, it means that there are multiple runs with the same combination of factors. Contributing to variation among replicates (SS_within)
            y = g[response_col].to_numpy() # this takes the result column for all rows in the sub-df that share the same factor levels, and turns them into a numpy array
            SS_within += float(np.sum((y - y.mean()) ** 2)) # we calculate SS within 'g' and add it to the total
            df_pure += len(y) - 1 # we add the degrees of freedom to the total

    # Lack-of-fit components
    SS_lof = SSE_all - SS_within # when we remove the error within groups from the total, we get the systematic model miss (lack-of-fit)
    df_lof = df_res - df_pure # we also do the same to get lof degrees of freedom

    if df_pure <= 0 or df_lof <= 0: # this checks if any of the degrees of freedom are 0, which would lead to a division by 0
        MS_pure = np.nan
        MS_lof = np.nan
        F_lof = np.nan
        p_lof = np.nan
    else: # if no df is equal to 0, then mean square, F, and p-value are calculated
        MS_pure = SS_within / df_pure
        MS_lof = SS_lof / df_lof
        F_lof = MS_lof / MS_pure if MS_pure > 0 else np.inf
        p_lof = stats.f.sf(F_lof, df_lof, df_pure)

    if print_summary: # if indicated that a summary is necessary, it's printed
        print(f"Pure error:   SS={SS_within:.4g}, df={df_pure}, MS={MS_pure if np.isfinite(MS_pure) else np.nan:.4g}")
        print(f"Lack-of-fit:  SS={SS_lof:.4g}, df={df_lof}, MS={MS_lof if np.isfinite(MS_lof) else np.nan:.4g}")
        print(f"F(LOF) = {F_lof if np.isfinite(F_lof) else np.nan:.3f} "
              f"with df=({df_lof},{df_pure}), p-value={p_lof if np.isfinite(p_lof) else np.nan:.4g}")

    return {
        "SSE_all": SSE_all,
        "df_res": df_res,
        "SS_within": SS_within,
        "df_pure": df_pure,
        "SS_lof": SS_lof,
        "df_lof": df_lof,
        "MS_pure": MS_pure,
        "MS_lof": MS_lof,
        "F_lof": F_lof,
        "p_lof": p_lof,
    }

#==========================================================================================
#=================================Pareto Chart Code========================================
#==========================================================================================
def pareto_standardized_effects(model, factors=None, only_main=False, alpha=0.05, title=None, save_fig=False, file_name='pareto_chart'):
    """
    Creates and saves a png of a Pareto chart of standardized effects (considering absolute t-values) from a fitted statsmodels OLS.

    Parameters
    ----------
    model : fitted OLS model (result of ols(...).fit()).
    factors : string list containing the names of all design factors.
    only_main : boolean. If True, plots only main effects, if False plots interactions as well.
    alpha : two-sided significance level for reference line (default 0.05).
    title : figure title.
    save_fig : boolean, if True saves the figure as a png in a 'figures' folder.
    file_name : string, name of the saved figure file (without extension).
    """
    # Get parameters and standard errors
    params = model.params.copy()
    ses    = model.bse.copy()
    for key in ['Intercept', 'const']: # drops the intercept if present
        if key in params.index:
            params = params.drop(key)
            ses    = ses.drop(key)

    # Decide which terms to keep
    idx = params.index.tolist()
    if only_main:
        if factors is None:
            keep = [name for name in idx if ':' not in name] # only interactions and second order terms have ':' in their names
        else:
            keep = [name for name in idx if name in factors] # only keeps main effects that are in the factors list
    else:
        keep = idx # keeps all terms

    # Compute absolute standardized effects (|t|) and sort
    tvals = (params[keep] / ses[keep]).abs().dropna()
    tvals = tvals.sort_values(ascending=False)

    # Reference lines
    dof = int(model.df_resid)
    if dof <= 0:
        raise ValueError(f"Non-positive residual dof ({dof}). Check your model fit.")
    t_crit = stats.t.ppf(1 - alpha/2, dof)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(tvals)), tvals.values, edgecolor='k')
    plt.xticks(range(len(tvals)), tvals.index, rotation=90)
    plt.ylabel('Absolute Standardized Effect (|t|)')

    if title is None:
        base = 'Main Effects' if only_main else 'Effects (Main + Interactions)'
        title = f'Pareto Chart of Standardized {base}'
    plt.title(title)

    # Add significance lines
    plt.axhline(y=t_crit, linestyle='--', color='red', label=f'α={alpha:.2f} (t={t_crit:.2f})')

    plt.legend()
    plt.tight_layout()

    # Save figure if requested
    if save_fig:
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/{file_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

#==========================================================================================
#=============================Diagnostic Residual Plots====================================
#==========================================================================================
def diagnostic_plots(model, save_fig=False, file_name='residuals'):
    # Extract residuals and predicted values from the model
    residuals = model.resid
    predicted = model.fittedvalues
    
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    
    # Residuals vs Predicted
    axs[0].scatter(predicted, residuals, edgecolors='k', facecolors='none')
    axs[0].axhline(y=0, color='k', linestyle='dashed', linewidth=1)
    axs[0].set_title('Residuals vs. Predicted')
    axs[0].set_xlabel('Predicted values')
    axs[0].set_ylabel('Residuals')
    
    # Residuals vs. Runs (order of data collection)
    n_runs = len(residuals)

    axs[1].scatter(range(n_runs), residuals, edgecolors='k', facecolors='none')
    axs[1].axhline(y=0, color='k', linestyle='dashed', linewidth=1)
    axs[1].set_title('Residuals vs. Run')
    axs[1].set_xlabel('Run')
    axs[1].set_ylabel('Residuals')

    # Add grid divisions equal to number of runs
    axs[1].set_xticks(range(n_runs))
    axs[1].set_xticklabels([str(i) if i % 5 == 0 else "" for i in range(n_runs)])
    axs[1].grid(True, axis='x', linestyle=':', linewidth=0.7)  # vertical gridlines
    axs[1].grid(True, axis='y', linestyle=':', linewidth=0.7)  # horizontal gridlines (optional)
    
    # Q-Q plot
    sm.qqplot(residuals, line='45', fit=True, ax=axs[2])
    axs[2].set_title('Q-Q Plot')
    stat, p_value = stats.shapiro(residuals)
    axs[2].set_title(f"Q-Q Plot\nShapiro-Wilk p={p_value:.3f}")

    plt.tight_layout()

    # Save figure if requested
    if save_fig:
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/{file_name}.png", dpi=300, bbox_inches="tight")
    
    plt.show()