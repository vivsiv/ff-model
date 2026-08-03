# Fantasy Football Prediction Model

A machine learning project to predict top fantasy football players for the 2025 season based on historical NFL data.

## Project Overview

This project collects historical NFL player and team data from Pro Football Reference and other sources, processes it, and builds machine learning models to predict fantasy performance for the upcoming season.

## Features

src/data/ -- one file per data source, each with a `*Scraper` class that saves raw data to `data/bronze/{source}/`
- nflverse.py (`NflverseScraper`): pulls player and team season stats for all available seasons via nflreadpy
- pro_football_reference.py (`ProFootballReferenceScraper`): legacy scraper, pro-football-reference now blocks bot traffic so this is no longer actively used

src/processing/ -- one file per data source, each with a `*Processor` class that reads bronze data and saves cleaned data to `data/silver/{source}/`
- nflverse.py (`NflverseProcessor`): filters player stats to fantasy-relevant positions (QB/RB/WR/TE)
- pro_football_reference.py (`ProFootballReferenceProcessor`): legacy silver-layer processing (name/team standardization, rolling averages, ratio stats) for the old pro-football-reference pipeline
- column_registry.py / column_registry.yaml: registry of which raw columns from each silver table are identity columns, candidate stat features, prediction targets, or excluded entirely -- the single source of truth `gold.py` reads from, rather than hardcoding column lists
- gold.py (`TrainingSetBuilder`): builds the gold training set from silver data -- computes a positional baseline (trailing 5-season league average per position, to avoid an all-time average going stale as league-wide offensive output drifts), expanding career-to-date features per player (average/std/max/min/trend, and a shrinkage-adjusted average that pulls short careers toward the positional baseline so a 1-year outlier season isn't treated as equally reliable as a decade-long track record), then joins each player's next-season target onto their most recent prior season's features (bridging gap seasons, e.g. a player who missed a year to injury, rather than dropping them)

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