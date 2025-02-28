# Fantasy Football Predictor 2025

A machine learning project to predict top fantasy football players for the 2025 season based on historical NFL data.

## Project Overview

This project collects historical NFL player and team data from Pro Football Reference and other sources, processes it, and builds machine learning models to predict fantasy performance for the upcoming season.

## Features

- Web scraping of NFL player statistics (season totals and game logs)
- Collection of team offensive efficiency metrics
- Data preprocessing and feature engineering
- Machine learning models for fantasy point prediction
- Analysis of prediction accuracy and player rankings

## Project Structure

```
ff-predictor-2025/
├── data/               # Raw and processed data
├── models/             # Trained ML models
├── notebooks/          # Jupyter notebooks for analysis
├── src/                # Source code
│   ├── scraper.py      # Web scraping utilities
│   ├── preprocess.py   # Data preprocessing
│   ├── features.py     # Feature engineering
│   ├── train.py        # Model training
│   └── predict.py      # Making predictions
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the data collection scripts:
   ```
   python src/scraper.py
   ```

## Data Sources

- Pro Football Reference (player stats, game logs)
- Football Outsiders (team efficiency metrics)
- Additional sources to be added

## License

MIT 