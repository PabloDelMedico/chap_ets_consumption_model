# ETS consumption

CHAP model for the `health_supply` domain. Exponential smoothing (ETS) per location, choosing by AIC between level-only, damped trend and, with at least two years of data, additive seasonal models. Probabilistic forecasts are simulated sample paths. Missing entries and stock-out months are interpolated before fitting.

Declares `target: consumption`, `domain: health_supply` and the required covariate
`stock_on_hand` in `MLproject`. Monthly data only.

Compare against the AMC baseline:
https://github.com/PabloDelMedico/chap_naive_consumption_model
