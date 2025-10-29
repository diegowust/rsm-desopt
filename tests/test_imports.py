def test_public_api():
    import rsm_desopt as R
    names = [
        "Goal","OptimizeConfig","derringer_suich","composite_desirability",
        "make_predict_fn_ols","make_predict_fn_ols_realinputs","latin_hypercube",
        "optimize_desirability",
        "lack_of_fit_test_from_model","pareto_standardized_effects","diagnostic_plots"
    ]
    for n in names:
        assert hasattr(R, n), f"Missing {n} in public API"
