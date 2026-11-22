# Context
This project builds both tabular and sequential machine learning models for predicting fantasy football points scored in future years.

# Project Structure
- data: Contains actual data used in modelling. Follows medallion architecture where bronze contains raw data tables from a data source, silver contains cleaned data from the bronze tables, and gold contains silver data properly assembled together with new columns for features. Gold data is intended to be the superset of all training data.
- mlruns: Tracking directory for mlflow, accessed by running the mlflow ui server locally
- notebooks: Jupyter notebooks for interative work like viewing feature importance, making ad hoc models or predictions, etc.
- src/analysis: Source code for data analysis
- src/data: Source code for raw data scraping from various data sources
- src/modeling: Source code for training, evaluating, and making predictions with models.
- src/processing: Source code for turning raw data into training quality data

Dependencies are managed with uv

# Naming
- Refer to sample weights as `sample_weights`, not generic `weights`, to distinguish them from model weights. Use similarly explicit names in code, tests, comments, and documentation whenever different domain concepts could otherwise be confused.
- Reserve `pipeline` for the complete model/inference pipeline; call imputation/scaling components `preprocessor`.

# Python Style
- Use blank lines within methods to separate logical phases, such as validation, setup, transformation, and reporting. Keep related statements together rather than adding a blank line after every statement.
- Add a blank line before a return statement when it follows other work. Short guard clauses and methods containing only a return can remain compact.
- Use clear, concise names that express the actual domain concept and responsibility. Avoid ambiguous abbreviations, unnecessary verbosity, and names that suggest behavior the code does not perform. Apply this to test names too.
- Put `__init__` first in a class, then group related methods in an intuitive order. Use static methods for stateless helpers unless class/subclass-aware behavior is actually needed.
- Use keyword arguments when they clarify a call, especially with several related inputs or `None` placeholders, e.g. `predict(context=None, model_input=inputs)`.
- Avoid local variables that merely alias an attribute/property when direct access is equally clear. Keep locals when they capture a snapshot, avoid repeated computation, or make the logic easier to understand.

# Testing
- Tests of DataFrame transformations should generally construct the full expected DataFrame and compare it with `pd.testing.assert_frame_equal(actual, expected)`. This checks values, columns, row counts, indexes, and dtypes together.
- Use focused assertions instead when the test intentionally targets a specific DataFrame detail, such as one column, a dtype, missing-value behavior, or an error condition.

# Plans
Current project plans can be found in .plan/. Typically work goes in before the football season to refresh and improve the models
before the next football season.

Read the current year's plan to get an idea of the work completed, the work remaining, and what the next session will likely be about.
