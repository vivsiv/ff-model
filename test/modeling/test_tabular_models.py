import pandas as pd
import os
import shutil
import tempfile

from src.modeling.tabular_models import FantasyModel
from src.processing.column_registry import get_identity_columns


class TestFantasyModel:
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.gold_dir = os.path.join(cls.test_dir, "gold")

        os.makedirs(cls.gold_dir)

        n = 10  # need 10 players so an 80/20 split has 8/2

        # Must include every registry identity column, since FantasyModel selects
        # self.training_data[self.identity_cols] and would KeyError on anything missing.
        identity_data = {col: [f"{col}_{i}" for i in range(n)] for col in get_identity_columns("nflverse", "player_stats")}
        identity_data["target_season"] = [2024] * n

        training_data = pd.DataFrame({
            **identity_data,
            'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'f2': [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
            'f3': [12, 0, 8, 12, 0, 8, 12, 0, 8, 12],
            'target': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        })
        training_data.to_csv(os.path.join(cls.gold_dir, "target_1__training_set.csv"), index=False)

        cls.model = FantasyModel(data_dir=cls.test_dir, target="target_1")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_initial_datasets(self):
        model = FantasyModel(data_dir=self.test_dir, target="target_1")

        expected_train_features = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'f2': [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
            'f3': [12, 0, 8, 12, 0, 8, 12, 0, 8, 12]
        })
        expected_train_target = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], name="target")

        assert model.train_features.equals(expected_train_features)
        assert model.train_target.equals(expected_train_target)
        assert list(model.train_identity.columns) == get_identity_columns("nflverse", "player_stats") + ["target_season"]
        assert len(model.train_identity) == 10
        assert model.target == "target_1"

    def test_split_data_has_correct_shape(self):
        data = self.model.split_data()

        identity_col_count = len(get_identity_columns("nflverse", "player_stats")) + 1  # + target_season

        assert data['X_train'].shape == (8, 3)
        assert data['X_test'].shape == (2, 3)
        assert data['y_train'].shape == (8,)
        assert data['y_test'].shape == (2,)
        assert data['identity_train'].shape == (8, identity_col_count)
        assert data['identity_test'].shape == (2, identity_col_count)

    def test_split_data_is_deterministic(self):
        data1 = self.model.split_data()
        data2 = self.model.split_data()

        pd.testing.assert_frame_equal(data1['X_train'], data2['X_train'])
        pd.testing.assert_frame_equal(data1['X_test'], data2['X_test'])
        pd.testing.assert_series_equal(data1['y_train'], data2['y_train'])
        pd.testing.assert_series_equal(data1['y_test'], data2['y_test'])
        pd.testing.assert_frame_equal(data1['identity_train'], data2['identity_train'])
        pd.testing.assert_frame_equal(data1['identity_test'], data2['identity_test'])
