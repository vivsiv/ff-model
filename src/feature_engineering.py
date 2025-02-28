#!/usr/bin/env python3
"""
NFL Fantasy Football Feature Engineering

This module combines player and team statistics and creates advanced
features for fantasy football prediction.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FantasyFeatureEngineer:
    """Engineer features for fantasy football prediction."""
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the feature engineer.
        
        Args:
            data_dir: Directory containing the processed data
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.final_dir = os.path.join(data_dir, "final")
        self.output_dir = os.path.join(data_dir, "features")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_processed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all processed data.
        
        Returns:
            Dictionary of DataFrames
        """
        data = {}
        
        # Load fantasy stats
        fantasy_path = os.path.join(self.final_dir, "fantasy_stats.csv")
        if os.path.exists(fantasy_path):
            data['fantasy_stats'] = pd.read_csv(fantasy_path)
            logger.info(f"Loaded fantasy stats: {len(data['fantasy_stats'])} rows")
        
        # Load game logs
        game_logs_path = os.path.join(self.final_dir, "game_logs.csv")
        if os.path.exists(game_logs_path):
            data['game_logs'] = pd.read_csv(game_logs_path)
            logger.info(f"Loaded game logs: {len(data['game_logs'])} rows")
        
        # Load team stats
        team_stats_path = os.path.join(self.final_dir, "team_stats.csv")
        if os.path.exists(team_stats_path):
            data['team_stats'] = pd.read_csv(team_stats_path)
            logger.info(f"Loaded team stats: {len(data['team_stats'])} rows")
        
        # Load advanced team stats
        adv_team_stats_path = os.path.join(self.final_dir, "advanced_team_stats.csv")
        if os.path.exists(adv_team_stats_path):
            data['advanced_team_stats'] = pd.read_csv(adv_team_stats_path)
            logger.info(f"Loaded advanced team stats: {len(data['advanced_team_stats'])} rows")
        
        return data
    
    def create_player_season_stats(self, game_logs: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate game logs to create season stats for each player.
        
        Args:
            game_logs: DataFrame with player game logs
            
        Returns:
            DataFrame with aggregated season stats
        """
        if game_logs.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = game_logs.copy()
        
        # Ensure we have the necessary columns
        required_cols = ['Player_ID', 'Player_Name', 'Season', 'Team']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns in game logs: {[col for col in required_cols if col not in df.columns]}")
            return pd.DataFrame()
        
        # Group by player, season, and team
        grouped = df.groupby(['Player_ID', 'Player_Name', 'Season', 'Team'])
        
        # Aggregate stats
        agg_dict = {
            # Games played
            'Date': 'count',
        }
        
        # Add numeric columns to aggregate
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col not in ['Player_ID', 'Season']:
                agg_dict[col] = 'sum'
        
        # Perform aggregation
        season_stats = grouped.agg(agg_dict).reset_index()
        
        # Rename the Date column to Games_Played
        season_stats = season_stats.rename(columns={'Date': 'Games_Played'})
        
        logger.info(f"Created season stats for {len(season_stats)} player-seasons")
        
        return season_stats
    
    def calculate_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate efficiency metrics for players.
        
        Args:
            df: DataFrame with player stats
            
        Returns:
            DataFrame with added efficiency metrics
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate QB efficiency metrics
        if 'Passing_Att' in df.columns and 'Passing_Cmp' in df.columns:
            df['Completion_Pct'] = (df['Passing_Cmp'] / df['Passing_Att'] * 100).round(1)
        
        if 'Passing_Yds' in df.columns and 'Passing_Att' in df.columns:
            df['Yards_Per_Attempt'] = (df['Passing_Yds'] / df['Passing_Att']).round(1)
        
        if 'Passing_TD' in df.columns and 'Passing_Att' in df.columns:
            df['TD_Rate'] = (df['Passing_TD'] / df['Passing_Att'] * 100).round(1)
        
        if 'Int' in df.columns and 'Passing_Att' in df.columns:
            df['Int_Rate'] = (df['Int'] / df['Passing_Att'] * 100).round(1)
        
        # Calculate RB efficiency metrics
        if 'Rushing_Yds' in df.columns and 'Rushing_Att' in df.columns:
            df['Yards_Per_Carry'] = (df['Rushing_Yds'] / df['Rushing_Att']).round(1)
        
        if 'Rushing_TD' in df.columns and 'Rushing_Att' in df.columns:
            df['Rush_TD_Rate'] = (df['Rushing_TD'] / df['Rushing_Att'] * 100).round(1)
        
        # Calculate WR/TE efficiency metrics
        if 'Receiving_Yds' in df.columns and 'Rec' in df.columns:
            df['Yards_Per_Reception'] = (df['Receiving_Yds'] / df['Rec']).round(1)
        
        if 'Rec' in df.columns and 'Tgt' in df.columns:
            df['Catch_Rate'] = (df['Rec'] / df['Tgt'] * 100).round(1)
        
        if 'Receiving_TD' in df.columns and 'Rec' in df.columns:
            df['Rec_TD_Rate'] = (df['Receiving_TD'] / df['Rec'] * 100).round(1)
        
        if 'Receiving_Yds' in df.columns and 'Tgt' in df.columns:
            df['Yards_Per_Target'] = (df['Receiving_Yds'] / df['Tgt']).round(1)
        
        # Calculate fantasy points per game
        if 'STANDARD_Fantasy_Points' in df.columns and 'Games_Played' in df.columns:
            df['STANDARD_Fantasy_Points_Per_Game'] = (df['STANDARD_Fantasy_Points'] / df['Games_Played']).round(1)
        
        if 'PPR_Fantasy_Points' in df.columns and 'Games_Played' in df.columns:
            df['PPR_Fantasy_Points_Per_Game'] = (df['PPR_Fantasy_Points'] / df['Games_Played']).round(1)
        
        if 'HALF_PPR_Fantasy_Points' in df.columns and 'Games_Played' in df.columns:
            df['HALF_PPR_Fantasy_Points_Per_Game'] = (df['HALF_PPR_Fantasy_Points'] / df['Games_Played']).round(1)
        
        logger.info("Calculated efficiency metrics")
        
        return df
    
    def add_team_context(self, player_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add team context to player stats.
        
        Args:
            player_df: DataFrame with player stats
            team_df: DataFrame with team stats
            
        Returns:
            DataFrame with added team context
        """
        if player_df.empty or team_df.empty:
            return player_df
        
        # Make copies to avoid modifying the originals
        player_df = player_df.copy()
        team_df = team_df.copy()
        
        # Ensure we have the necessary columns
        if 'Team' not in player_df.columns or 'Season' not in player_df.columns:
            logger.error("Missing Team or Season column in player stats")
            return player_df
        
        if 'Team' not in team_df.columns or 'Season' not in team_df.columns:
            logger.error("Missing Team or Season column in team stats")
            return player_df
        
        # Standardize team names if needed
        # This would need to be customized based on the actual team names in the data
        
        # Merge player and team stats
        merged_df = pd.merge(
            player_df,
            team_df,
            on=['Team', 'Season'],
            how='left',
            suffixes=('', '_Team')
        )
        
        logger.info(f"Added team context to {len(merged_df)} player-seasons")
        
        return merged_df
    
    def add_previous_season_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add previous season stats for each player.
        
        Args:
            df: DataFrame with player stats
            
        Returns:
            DataFrame with added previous season stats
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have the necessary columns
        if 'Player_ID' not in df.columns or 'Season' not in df.columns:
            logger.error("Missing Player_ID or Season column")
            return df
        
        # Sort by player and season
        df = df.sort_values(['Player_ID', 'Season'])
        
        # Get list of columns to shift
        stat_cols = [col for col in df.columns if col not in ['Player_ID', 'Player_Name', 'Team', 'Season', 'Position']]
        
        # Create a dictionary to map columns to their previous season versions
        prev_cols = {col: f'Prev_{col}' for col in stat_cols}
        
        # Group by player and shift stats
        for col in stat_cols:
            df[prev_cols[col]] = df.groupby('Player_ID')[col].shift(1)
        
        # Add a flag for rookies (no previous season data)
        df['Is_Rookie'] = df['Prev_Games_Played'].isna().astype(int)
        
        logger.info("Added previous season stats")
        
        return df
    
    def add_position_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add position group flags.
        
        Args:
            df: DataFrame with player stats
            
        Returns:
            DataFrame with added position group flags
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have the Position column
        if 'Position' not in df.columns:
            logger.error("Missing Position column")
            return df
        
        # Create position group flags
        df['Is_QB'] = (df['Position'] == 'QB').astype(int)
        df['Is_RB'] = (df['Position'] == 'RB').astype(int)
        df['Is_WR'] = (df['Position'] == 'WR').astype(int)
        df['Is_TE'] = (df['Position'] == 'TE').astype(int)
        
        logger.info("Added position group flags")
        
        return df
    
    def create_combined_dataset(self) -> pd.DataFrame:
        """
        Create a combined dataset with player and team stats.
        
        Returns:
            Combined DataFrame
        """
        # Load all processed data
        data = self.load_processed_data()
        
        if not data:
            logger.error("No data loaded")
            return pd.DataFrame()
        
        # Create player season stats from game logs if available
        if 'game_logs' in data and not data['game_logs'].empty:
            player_stats = self.create_player_season_stats(data['game_logs'])
        elif 'fantasy_stats' in data and not data['fantasy_stats'].empty:
            player_stats = data['fantasy_stats']
        else:
            logger.error("No player stats available")
            return pd.DataFrame()
        
        # Calculate efficiency metrics
        player_stats = self.calculate_efficiency_metrics(player_stats)
        
        # Add team context if available
        if 'team_stats' in data and not data['team_stats'].empty:
            player_stats = self.add_team_context(player_stats, data['team_stats'])
        
        # Add advanced team context if available
        if 'advanced_team_stats' in data and not data['advanced_team_stats'].empty:
            player_stats = self.add_team_context(player_stats, data['advanced_team_stats'])
        
        # Add previous season stats
        player_stats = self.add_previous_season_stats(player_stats)
        
        # Add position groups
        player_stats = self.add_position_groups(player_stats)
        
        # Save the combined dataset
        output_path = os.path.join(self.output_dir, "combined_stats.csv")
        player_stats.to_csv(output_path, index=False)
        logger.info(f"Saved combined dataset to {output_path}")
        
        return player_stats


def main():
    """Main function to run the feature engineer."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Set data directory relative to script directory
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    # Create feature engineer
    engineer = FantasyFeatureEngineer(data_dir=data_dir)
    
    # Create combined dataset
    combined_data = engineer.create_combined_dataset()
    
    if not combined_data.empty:
        logger.info(f"Created combined dataset with {len(combined_data)} rows and {len(combined_data.columns)} columns")
    else:
        logger.error("Failed to create combined dataset")


if __name__ == "__main__":
    main() 