# Fantasy Football Prediction Model

A machine learning project to predict top fantasy football players for the 2025 season based on historical NFL data.

## Project Overview

This project collects historical NFL player and team data from Pro Football Reference and other sources, processes it, and builds machine learning models to predict fantasy performance for the upcoming season.

## Features
scraper.py
- Scrapes passing, receiving, rushing, and team offense stats from pro football reference year by year and saves it down
- Built to allow the addition of more data sources

processor.py
- Takes the raw data files from the scraper and combines them all into one file per major stat
   - Performs any transformations of cleaning needed
- Creates ratio columns of certain stats i.e. targets_per_game
- Creates rollup columns that are rolling averages of a stat up to 3 years. i.e. pass_yards_3_yr_avg
- Builds the training data by joining each years fantasy point total to the previous years stats
- Builds the live data set by joining this years top players with their stats from last year

analysis.py
- Performs some sanity checks on the training and live data
- Computes pearsons correlation between features to identify redundant features
- Computes pearsons correlation between features and targets
- Computes mutual information between features and targets
- The feature information is purely informational, the modelling step does feature selection if useful.

modelling.py
- Has flows for building models that can predict the following targets:
   - ppr_points, ppr_points_per_game, standard_points, standard_points_per_game, value_over_replacement
- Splits the training data into a training and test set.
- Has a flow to evaluate different scikit-learn regression models via grid search
   - Logs the results to mlflow
- Has a flow to tune parameters of a particular model via grid search
   - Logs the results and best model to mlflow
- Loads a saved model and makes predictions on the test set, logs the results to mlflow and some of the predictions
- Loads a saved model and makes predictions on the live set, saves the predictions.

rankings.py
- Loads predictions, takes a league format and calculates per position rankings and overall rankings


## Thoughts
- The first model build from this project was for the 2025 season and was a ridge regression model to predict ppr_points_per_game.
- The model had:
   - An avg r^2 of 0.59 and an avg RMSE of 3.66 during training.
   - An r^2 of 0.64 on the test set and an rmse of 3.48 on the test set

- I didn't attempt to predict rookies in this model, so they wont show up in projections
- Traditionally a players environment is considered important in fantasy which was why I included team stats, however
the team stats all fared poorly in the pearsons and mutual information analysis. 

## Next Up
- Are there more valuable features out there
   - Some ideas: NFL draft position, Fantasy ADP
- Experiment with more models and targets
- A rookie model using college stats

## License

MIT 