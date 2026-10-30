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

# Plans
Current project plans can be found in .plan/. Typically work goes in before the football season to refresh and improve the models
before the next football season.

Read the current year's plan to get an idea of the work completed, the work remaining, and what the next session will likely be about.
