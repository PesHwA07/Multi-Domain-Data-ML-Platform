from energy_model import preprocess_energy_data
import unittest
from unittest.mock import patch
import pandas as pd

import sys
import os
# Adjust path to import the DAG module properly
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../dags')))


class TestEnergyModel(unittest.TestCase):

    @patch('energy_model.pd.read_sql')
    @patch('energy_model.create_engine')
    def test_preprocess_energy_data(self, mock_create_engine, mock_read_sql):
        """
        Tests the time-series preprocessing logic:
        1. Ensures missing hours are exposed and interpolated correctly.
        2. Ensures the 80/20 train/test split is strictly sequential.
        """
        # Create a dummy dataframe spanning 5 hours
        dates = pd.date_range('2023-01-01 00:00:00',
                              '2023-01-01 04:00:00', freq='h')

        # Introduce a missing hour by dropping index 2 (02:00:00)
        dates_missing = dates.drop(dates[2])

        # Values: 10, 20, (missing 30), 40, 50
        df = pd.DataFrame({
            'timestamp': dates_missing,
            'consumption': [10.0, 20.0, 40.0, 50.0]
        })
        mock_read_sql.return_value = df

        # Run the target function
        train, test = preprocess_energy_data()

        # Assertion 1: Check if the missing hour was interpolated back in
        self.assertEqual(len(train) + len(test), 5,
                         "Total length should be 5 after resampling")

        # Check combined train + test to verify interpolation accuracy
        combined = pd.concat([train, test])
        interpolated_val = combined.loc['2023-01-01 02:00:00', 'consumption']
        self.assertEqual(interpolated_val, 30.0,
                         "Missing value was not interpolated correctly")

        # Assertion 2: Check 80/20 sequential split
        # 5 rows total -> 80% is 4 rows train, 20% is 1 row test
        self.assertEqual(
            len(train), 4, "Train set should have exactly 4 rows (80% of 5)")
        self.assertEqual(
            len(test), 1, "Test set should have exactly 1 row (20% of 5)")


if __name__ == '__main__':
    unittest.main()
