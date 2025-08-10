import logging
import os
import pandas as pd
import matplotlib.pyplot as plt

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
        qbs: int,
        rbs: int,
        wrs: int,
        tes: int,
        flex: int,
        bench: int
    ):
        self.total_players = (qbs + rbs + wrs + tes + flex + bench) * teams
        self.qbs_started = qbs * teams
        self.rbs_started = (rbs + (0.5 * flex)) * teams
        self.wrs_started = (wrs + (0.5 * flex)) * teams
        self.tes_started = tes * teams


class Rankings:
    def __init__(
        self,
        year: int,
        target_col: str,
        data_dir: str = "../data",
        league_format: LeagueFormat = LeagueFormat(
            teams=12, qbs=1, rbs=2, wrs=2, tes=1, flex=1, bench=5
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

    def calculate_position_rankings(self, position_predictions: pd.DataFrame, position: str, position_starters: int) -> pd.DataFrame:
        position_rankings = position_predictions.sort_values(by=self.target_col, ascending=False)
        position_rankings['position_rank'] = range(1, len(position_rankings) + 1)

        replacement_rank = int(position_starters) + 1
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

        return position_rankings

    def save_rankings(self, rankings: dict[str, pd.DataFrame]) -> None:
        rankings_dir = os.path.join(self.rankings_dir, f"{self.year}/{self.target_col}")
        os.makedirs(rankings_dir, exist_ok=True)

        for rank_type, rankings_df in rankings.items():
            rankings_path = os.path.join(rankings_dir, f"{rank_type}.csv")
            selected_columns = ["player", "position", 'overall_rank', 'position_rank', self.target_col, 'points_over_bench'] if rank_type == 'overall' else ["player", "position", 'position_rank', self.target_col]
            rankings_df[selected_columns].to_csv(rankings_path, index=False)
            logger.info(f"{rank_type} rankings saved to {rankings_path}")

    def build_rankings(self) -> dict[str, pd.DataFrame]:
        predictions_df = self.predictions.copy()

        starters_by_position = {
            'qb': self.league_format.qbs_started,
            'rb': self.league_format.rbs_started,
            'wr': self.league_format.wrs_started,
            'te': self.league_format.tes_started
        }

        rankings = {}

        for position, position_starters in starters_by_position.items():
            position_predictions = predictions_df[predictions_df['position'] == position].copy()
            position_rankings = self.calculate_position_rankings(position_predictions, position, position_starters)

            rankings[position] = position_rankings

        overall_rankings = pd.concat(rankings.values(), ignore_index=True)
        overall_rankings = overall_rankings.sort_values(by='points_over_bench', ascending=False)
        overall_rankings['overall_rank'] = range(1, len(overall_rankings) + 1)

        rankings['overall'] = overall_rankings

        self.save_rankings(rankings)

        return rankings

    def plot_rankings(self, rankings_dict: dict[str, pd.DataFrame]) -> None:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

        position_config = {
            'qb': ('#FF6B6B', fig.add_subplot(gs[0, 0]), self.league_format.qbs_started * 2),
            'rb': ('#8B4513', fig.add_subplot(gs[0, 1]), self.league_format.rbs_started * 2.5),
            'wr': ('#45B7D1', fig.add_subplot(gs[1, 0]), self.league_format.wrs_started * 2.5),
            'te': ('#9370DB', fig.add_subplot(gs[1, 1]), self.league_format.tes_started * 2),
            'overall': ('#FFEAA7', fig.add_subplot(gs[2, 0]), None)
        }

        overall_bench_ax = fig.add_subplot(gs[2, 1])

        for position, rankings in rankings_dict.items():
            color, ax, max_rank = position_config[position]

            if position == 'overall':
                filtered_rankings = rankings[rankings['overall_rank'] <= self.league_format.total_players]
                for pos in ['qb', 'rb', 'wr', 'te']:
                    pos_data = filtered_rankings[filtered_rankings['position'] == pos]
                    if len(pos_data) > 0:
                        pos_color = position_config[pos][0]
                        ax.scatter(pos_data['overall_rank'], pos_data[self.target_col],
                                 alpha=0.6, color=pos_color, label=pos.upper())
                        overall_bench_ax.scatter(pos_data['overall_rank'], pos_data['points_over_bench'],
                                               alpha=0.6, color=pos_color, label=pos.upper())
                ax.legend()
                overall_bench_ax.legend()
            else:
                filtered_rankings = rankings[rankings['position_rank'] <= max_rank]
                ax.scatter(filtered_rankings['position_rank'], filtered_rankings[self.target_col],
                          alpha=0.7, color=color)

                bench_player_rank = max_rank / 2
                ax.axvline(x=bench_player_rank, color='black', linestyle='--', alpha=0.7, label='Bench Player')
                ax.legend()

            ax.set_xlabel('Rank')
            if position == 'overall':
                ax.set_ylabel(self.target_col)
                ax.set_title(f'{position.upper()} Rankings')
                overall_bench_ax.set_xlabel('Rank')
                overall_bench_ax.set_ylabel('Points Over Bench')
                overall_bench_ax.set_title('Overall Points Over Bench Rankings')
                overall_bench_ax.tick_params(axis='x', rotation=45)
            else:
                ax.set_ylabel(self.target_col)
                ax.set_title(f'{position.upper()} Rankings')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


def main():
    rankings = Rankings(year=2025, target_col="ppr_fantasy_points_per_game")
    ppr_ppg_2025_rankings_dict = rankings.build_rankings()

    print(f"Top 10 {rankings.target_col} rankings for {rankings.year}:")
    print(ppr_ppg_2025_rankings_dict['overall'].head(10))


if __name__ == "__main__":
    main()
