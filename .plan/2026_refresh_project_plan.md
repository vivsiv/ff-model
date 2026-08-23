# 2026 Refresh Project Plan

## Background

This project builds a model that predicts fantasy football points for the upcoming NFL season.
At a high level the project:
- Collects data from relevant sources.
- Processes the data into data sets that can be used to train and evaluate a model.
- Trains various model architectures on various prediction targets and evaluates them.
- Uses the champion model to predict the desired target (i.e. ppr_fantasy_points_per_game).
- Uses the predictions to create rankings.

## 2025 model issues/areas for improvements.
The 2025 model needs to be retrained with 2025 data to make 2026 predictions however a number of factors
have made the 2026 refresh require a more holistic rework of the pipeline.
1. The data source used for the 2025 model (pro football reference) now blocks bot based data scraping solutions
2. The 2025 training set had a bug where it split data at random but certain features rolled up stats over multiple 
years resulting in leakage where a training row was using data present in the validation set.
3. The 2025 model valued single season performance exactly the same as a proven track record. This resulted in 
overvaluing shorter careers and undervaluing longer careers.

## 2026 Proposed Solutions
1. Use (nflverse)[https://github.com/nflverse/nflreadpy] as the new data source.
2. Split training and validation data by year, have a proper test set
3. There are multiple possible solutions here:
    - For a tabular model compute each positions average value for a stat then weight a given players value for the stat toward the mean. Weighting is more aggressive for players with shorter careers and less aggressive for players with longer careers. Also add full career averages, standard deviations, min, and max for all stats. Add additional features to capture the recent "trend" of a stat relative to the player's average.
    - Create a sequential model (like an lstm or a transformer), build true sequences of data for each player so the length and trajectory of their careers is naturally represented in the data.

## Project plan

PHASE 1. First Model (Total PPR Points) [DONE]
1. Create a data scraper for nflverse data [DONE]
2. Refactor data collection to support multiple data sources. [DONE]
    - Model: move raw data collection into src/data each file within it should have a `Scraper` class for a particular data source.
        - src/data/nflverse.py contains NflverseScraper
        - src/scraper.py is ostensibly src/data/pro_football_reference.py and should be moved as such.
    - Data should now be saved under data/bronze/{data_source}
3. Refactor data processing to account for multiple data sources [DONE]
    - Model: similar to what was done for scrapers. Processing should exist in src/processing, each file within it should have a `Processor` class for a particular data source
        - src/processing/nflverse.py contains NflVerseProcessor
    - Data should now be saved under data/silver/{data_source}
4. Create the training set builder. [DONE]
    - Only uses nflverse's "player_stats" table for now.
    - Pull out relevant columns from the silver layer.
    - Compute the positional baseline stat values and the career stat features.
    - Join the stat features with the target to be predicted (ppr_points).
    - NOTE: src/processor.py has the code both for the pro_football_reference.py processor and training set assembler in this new model.
5. Create the first model [DONE]
    - Correctly split the training and eval data.
        - Open question: Should we do 2024 and 2025 as eval or have 2024 be eval and 2025 be a test set.
    - Create a random forest regressor, fit it to the training data and see how it does on the eval/test data.
    - Move the making of predictions to its own file modeling/tabular_predictions.py
    - Add build_prediction set to gold.py (this should have season == last season (2025) and target season == 2026 with the target_column blank)

PHASE 2. Improve the first model [DONE]
1. Add Out of Bag error as a metric (what does the oob_score parameter do?) [DONE].
2. Update the notebook to be able to load a saved model and inspect it [DONE]
3. View which features are contributing the most to decisions. Feature importance, partial depdendance. [DONE]
4. Add more features:
    - team_stats [DONE]
    - games should be a feature!! [DONE]
    - snap counts (pfr_id) [DONE]
    - draft picks (pfr_id) (has all_pro, allpro, probowls) [DONE]
    - fantasy_stats (regular id) [DEFERRED]
5. Reduce shrinkage constant. [DONE]
    - Tried [0.75, 1.125, 1.5, 1.75, 2, 3 (orig)], 1.5 yielded best eval results.
    - shrinkage_k default in src/processing/gold.py changed from 3.0 -> 1.5.
6. Grid search hyperparameters/auto ml. [DONE]
    - Try min_samples_leaf 4,8,16. Are any other hyperparams worth adjusting?
7. Try different model architectures (Gradient Boosting, Ridge, Lasso, Linear, etc.) [DONE]

PHASE 3. Second Model (PPR POINTS PER GAME) [IN PROGRESS]
1. Create new per game target column (ppr_points_per_game), see how the existing feature set does predicting new target. [DONE]
2. Add per game variants of existing features. [DONE]
3. Sample Data so that higher scoring players are more represented [DONE]
4. Make positional models [IN PROGRESS]

PHASE 4. Predictions & Rankings [TODO]
1. Add functionality to eval on the test set. [DONE]
2. Generate list of players and teams to predict for 2026.
3. Generate rankings process based on current code.
4. Re-evaluate rankings process

PHASE 5. Sequential Model [TODO]
1. Create a new training set builder that builds a sequential version of the data that can be used to feed an lstm or transformer
2. Build an lstm model and evaluate it
3. Build a trasformer model or finetune an existing one and evaluate it
