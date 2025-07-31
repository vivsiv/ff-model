import pytest
import pandas as pd
import os
import shutil
import tempfile

from src.modelling import FantasyModel


class TestFantasyModel:
    @classmethod
    def setup_class(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.gold_dir = os.path.join(cls.test_dir, "gold")

        os.makedirs(cls.gold_dir)

        training_data = pd.DataFrame({
            # Need 10 players so i can do an 80/20 split
            'id': ['p1_2024', 'p2_2024', 'p3_2024', 'p4_2024', 'p5_2024', 'p6_2024', 'p7_2024', 'p8_2024', 'p9_2024', 'p10_2024'],
            'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'f2': [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
            'f3': [12, 0, 8, 12, 0, 8, 12, 0, 8, 12],
            'target_1': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'target_2': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        })
        training_data.to_csv(os.path.join(cls.gold_dir, "training_set.csv"), index=False)

        live_data = pd.DataFrame({
            'id': ['p100_QB', 'p101_RB', 'p102_WR'],
            'f1': [10, 11, 12],
            'f2': [0, 50, 90],
            'f3': [100, 101, 102],
        })
        live_data.to_csv(os.path.join(cls.gold_dir, "live_set.csv"), index=False)

        cls.model = FantasyModel(data_dir=cls.test_dir, target_col="target_1", possible_targets=["target_1", "target_2"])

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_initial_datasets(self):
        model_t1 = FantasyModel(data_dir=self.test_dir, target_col="target_1", possible_targets=["target_1", "target_2"])
        model_t2 = FantasyModel(data_dir=self.test_dir, target_col="target_2", possible_targets=["target_1", "target_2"])

        expected_train_ids = pd.Series(['p1_2024', 'p2_2024', 'p3_2024', 'p4_2024', 'p5_2024', 'p6_2024', 'p7_2024', 'p8_2024', 'p9_2024', 'p10_2024'])
        expected_train_features = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'f2': [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
            'f3': [12, 0, 8, 12, 0, 8, 12, 0, 8, 12]
        })
        expected_train_target = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        expected_train_target_2 = pd.Series([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        expected_live_ids = pd.Series(['p100_QB', 'p101_RB', 'p102_WR'])
        expected_live_features = pd.DataFrame({
            'f1': [10, 11, 12],
            'f2': [0, 50, 90],
            'f3': [100, 101, 102],
        })

        assert model_t1.train_features.equals(expected_train_features)
        assert model_t1.train_target.equals(expected_train_target)
        assert model_t1.train_ids.equals(expected_train_ids)
        assert model_t1.live_ids.equals(expected_live_ids)
        assert model_t1.live_features.equals(expected_live_features)
        assert model_t1.target_col == "target_1"

        assert model_t2.train_features.equals(expected_train_features)
        assert model_t2.train_target.equals(expected_train_target_2)
        assert model_t2.train_ids.equals(expected_train_ids)
        assert model_t2.live_ids.equals(expected_live_ids)
        assert model_t2.live_features.equals(expected_live_features)
        assert model_t2.target_col == "target_2"

    def test_split_data_has_correct_shape(self):
        data = self.model.split_data()

        assert data['X_train'].shape == (8, 3)
        assert data['X_test'].shape == (2, 3)
        assert data['y_train'].shape == (8,)
        assert data['y_test'].shape == (2,)
        assert data['Id_train'].shape == (8,)
        assert data['Id_test'].shape == (2,)

    def test_split_data_is_deterministic(self):
        data1 = self.model.split_data()
        data2 = self.model.split_data()

        pd.testing.assert_frame_equal(data1['X_train'], data2['X_train'])
        pd.testing.assert_frame_equal(data1['X_test'], data2['X_test'])
        pd.testing.assert_series_equal(data1['y_train'], data2['y_train'])
        pd.testing.assert_series_equal(data1['y_test'], data2['y_test'])
        pd.testing.assert_series_equal(data1['Id_train'], data2['Id_train'])
        pd.testing.assert_series_equal(data1['Id_test'], data2['Id_test'])
