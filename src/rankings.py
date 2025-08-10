import pandas as pd
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LeagueFormat:
    def __init__(
        self,
        teams: int,
        qbs_per_team: int,
        rbs_per_team: int,
        wrs_per_team: int,
        tes_per_team: int,
        flex_per_team: int,
    ):
        self.total_picks = (qbs_per_team + rbs_per_team + wrs_per_team + tes_per_team + flex_per_team) * teams
        self.qbs_picked = qbs_per_team * teams
        self.rbs_picked = (rbs_per_team + (0.5 * flex_per_team)) * teams
        self.wrs_picked = (wrs_per_team + (0.5 * flex_per_team)) * teams
        self.tes_picked = tes_per_team * teams


class Rankings:
    def __init__(
        self,
        year: int,
        target_col: str,
        data_dir: str = "../data",
        league_format: LeagueFormat = LeagueFormat(
            teams=12, qbs_per_team=1, rbs_per_team=2,
            wrs_per_team=2, tes_per_team=1, flex_per_team=1,
        )
    ):
        self.data_dir = data_dir
        self.year = year
        self.target_col = target_col
        self.league_format = league_format

        try:
            self.predictions_dir = os.path.join(self.data_dir, "predictions")
            self.predictions = self.load_predictions()
        except Exception as e:
            logger.error(
                f"Error loading predictions at: {self.predictions_dir}: {e}"
            )
            raise e

        self.rankings_dir = os.path.join(self.data_dir, "rankings")
        os.makedirs(self.rankings_dir, exist_ok=True)

    def load_predictions(self) -> pd.DataFrame:
        predictions_df = pd.read_csv(os.path.join(self.predictions_dir, f"{self.target_col}_live_predictions.csv"))
        return predictions_df

    def build_rankings(self) -> pd.DataFrame:
        rankings_df = self.predictions.copy()

        position_picks = {
            'qb': self.league_format.qbs_picked,
            'rb': self.league_format.rbs_picked,
            'wr': self.league_format.wrs_picked,
            'te': self.league_format.tes_picked
        }

        processed_positions = []

        for position, picks_needed in position_picks.items():
            position_rankings = rankings_df[rankings_df['position'] == position].copy()
            position_rankings = position_rankings.sort_values(by=self.target_col, ascending=False)

            position_rankings['position_rank'] = range(1, len(position_rankings) + 1)

            replacement_rank = int(picks_needed) + 1
            replacement_player = position_rankings[position_rankings['position_rank'] == replacement_rank]

            if len(replacement_player) > 0:
                replacement_value = replacement_player[self.target_col].iloc[0]
            else:
                # If not enough players, use the last player
                replacement_value = (
                    position_rankings[self.target_col].iloc[-1]
                    if len(position_rankings) > 0 else 0
                )

            position_rankings['points_over_bench'] = position_rankings[self.target_col] - replacement_value
            position_rankings['points_over_bench'] = position_rankings['points_over_bench'].round(2)
            processed_positions.append(position_rankings)

        final_rankings = pd.concat(processed_positions, ignore_index=True)
        final_rankings = final_rankings.sort_values(by='points_over_bench', ascending=False)

        final_rankings['overall_rank'] = range(1, len(final_rankings) + 1)

        rankings_path = os.path.join(self.rankings_dir, f"{self.target_col}_{self.year}_rankings.csv")
        final_rankings[["player", "position", "overall_rank", 'position_rank', self.target_col, 'points_over_bench']].to_csv(rankings_path, index=False)
        logger.info(f"Rankings saved to {rankings_path}")

        return final_rankings

    def save_position_rankings(self, overall_rankings: pd.DataFrame) -> None:
        """Save rankings split by position to separate CSV files."""

        for position in overall_rankings['position'].unique():
            position_df = overall_rankings[overall_rankings['position'] == position].copy()

            position_df = position_df.sort_values('position_rank')

            position_file = os.path.join(self.rankings_dir, f"{self.target_col}_{self.year}_{position}_rankings.csv")
            position_df[["player", "position", "overall_rank", 'position_rank', self.target_col, 'points_over_bench']].to_csv(position_file, index=False)

            logger.info(f"Saved {position.upper()} rankings to {position_file} ({len(position_df)} players)")

        logger.info(f"All position rankings saved to {self.rankings_dir}")


def main():
    rankings = Rankings(year=2025, target_col="ppr_fantasy_points_per_game")
    ppr_ppg_2025_rankings = rankings.build_rankings()

    rankings.save_position_rankings(ppr_ppg_2025_rankings)

    print(f"Top 10 {rankings.target_col} rankings for {rankings.year}:")
    print(ppr_ppg_2025_rankings.head(10))


if __name__ == "__main__":
    main()
