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

        gold_data = pd.DataFrame({
            # Need 10 players so i can do an 80/20 split
            'id': ['p1_2024', 'p2_2024', 'p3_2024', 'p4_2024', 'p5_2024', 'p6_2024', 'p7_2024', 'p8_2024', 'p9_2024', 'p10_2024'],
            'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'f2': [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
            'f3': [12, 0, 8, 12, 0, 8, 12, 0, 8, 12],
            'target_1': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'target_2': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        })
        gold_data.to_csv(os.path.join(cls.gold_dir, "final_stats.csv"), index=False)

        cls.model = FantasyModel(data_dir=cls.test_dir, target_col="target_1", possible_targets=["target_1", "target_2"])

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.test_dir)

    def test_initial_datasets(self):
        model_t1 = FantasyModel(data_dir=self.test_dir, target_col="target_1", possible_targets=["target_1", "target_2"])
        model_t2 = FantasyModel(data_dir=self.test_dir, target_col="target_2", possible_targets=["target_1", "target_2"])

        expected_Id = pd.Series(['p1_2024', 'p2_2024', 'p3_2024', 'p4_2024', 'p5_2024', 'p6_2024', 'p7_2024', 'p8_2024', 'p9_2024', 'p10_2024'])
        expected_X = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'f2': [100, 50, 0, 100, 50, 0, 100, 50, 0, 100],
            'f3': [12, 0, 8, 12, 0, 8, 12, 0, 8, 12]
        })
        expected_Y_t1 = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        expected_Y_t2 = pd.Series([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        assert model_t1.X.equals(expected_X)
        assert model_t1.Y.equals(expected_Y_t1)
        assert model_t1.Id.equals(expected_Id)
        assert model_t1.target_col == "target_1"

        assert model_t2.X.equals(expected_X)
        assert model_t2.Y.equals(expected_Y_t2)
        assert model_t2.Id.equals(expected_Id)
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
