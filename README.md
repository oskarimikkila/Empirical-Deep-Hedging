# Empirical Deep Hedging

Code used in the article Empirical Deep Hedging (Mikkilä & Kanniainen, 2021)

Prepared settings files:

Constant volatility:
- GBM_kappa1 (risk factor = 1)
- GBM_kappa2 (risk factor = 2)
- GBM_kappa3 (risk factor = 3)

Constant volatility:
- Heston_kappa1 (risk factor = 1)
- Heston_kappa2 (risk factor = 2)
- Heston_kappa3 (risk factor = 3)

Empirical data:
- Empirical_kappa1 (risk factor = 1)
- Empirical_kappa2 (risk factor = 2)
- Empirical_kappa3 (risk factor = 3)

These files can be used to replicate the results in the article. The codebase has been tested on Windows with an environment created from the requirements.txt file. Python 3.8.

### Commands

Training: `main.py --settings Heston_kappa1` (parameter "settings" is optional. If not provided, uses the settings in settings.json)

Validation: `testing.py --validate --model Heston_kappa1` (reads validation result files and returns the best state of the model)

Testing: `testing.py --test --model Heston_kappa1` 
