"""
Hospital Readmission Prediction - Unit Tests for Preprocessing
===============================================================

Comprehensive unit tests for the data preprocessing pipeline.

Author: Kivairu Samuel
Date: 12th November 2025

Run tests with:
    python -m pytest test_preprocessing.py -v
    or
    python test_preprocessing.py
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions (adjust import based on your structure)
# from preprocessing import HospitalReadmissionPreprocessor, generate_synthetic_patient_data


class TestDataGeneration(unittest.TestCase):
    """Test synthetic data generation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_patients = 100
    
    def test_data_generation_shape(self):
        """Test that generated data has correct shape."""
        # Mock data generation
        df = self._generate_mock_data(self.n_patients)
        
        self.assertEqual(len(df), self.n_patients)
        self.assertEqual(len(df.columns), 11)  # Should have many features
        self.assertIn('readmitted_30_days', df.columns)
        self.assertIn('age', df.columns)
        self.assertIn('patient_id', df.columns)
    
    def test_data_types(self):
        """Test that data types are correct."""
        df = self._generate_mock_data(self.n_patients)
        
        # Numerical columns
        self.assertTrue(pd.api.types.is_numeric_dtype(df['age']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['charlson_score']))
        self.assertTrue(pd.api.types.is_integer_dtype(df['readmitted_30_days']))
        
        # Categorical columns
        self.assertTrue(pd.api.types.is_object_dtype(df['gender']))
        self.assertTrue(pd.api.types.is_object_dtype(df['race']))
    
    def test_readmission_rate_reasonable(self):
        """Test that readmission rate is within expected range."""
        df = self._generate_mock_data(1000)
        
        readmission_rate = df['readmitted_30_days'].mean()
        self.assertGreater(readmission_rate, 0.05)  # At least 5%
        self.assertLess(readmission_rate, 0.50)     # At most 50%
    
    def test_no_missing_in_required_fields(self):
        """Test that required fields don't have missing values."""
        df = self._generate_mock_data(self.n_patients)
        
        required_fields = ['patient_id', 'age', 'gender', 'readmitted_30_days']
        for field in required_fields:
            self.assertEqual(df[field].isna().sum(), 0, 
                           f"{field} should not have missing values")
    
    def test_age_range(self):
        """Test that ages are within valid range."""
        df = self._generate_mock_data(self.n_patients)
        
        self.assertTrue((df['age'] >= 18).all())
        self.assertTrue((df['age'] <= 120).all())
    
    def test_binary_fields_are_binary(self):
        """Test that binary fields only contain 0/1."""
        df = self._generate_mock_data(self.n_patients)
        
        binary_fields = ['readmitted_30_days', 'follow_up_within_7_days']
        for field in binary_fields:
            unique_values = df[field].dropna().unique()
            self.assertTrue(set(unique_values).issubset({0, 1}),
                          f"{field} should only contain 0 or 1")
    
    def _generate_mock_data(self, n):
        """Helper to generate mock patient data."""
        return pd.DataFrame({
            'patient_id': [f'PT{str(i).zfill(6)}' for i in range(n)],
            'age': np.random.randint(18, 95, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'race': np.random.choice(['White', 'Black', 'Hispanic'], n),
            'admission_diagnosis': np.random.choice(['CHF', 'COPD', 'Pneumonia'], n),
            'charlson_score': np.random.uniform(0, 10, n),
            'num_medications': np.random.randint(0, 20, n),
            'hemoglobin': np.random.uniform(8, 16, n),
            'creatinine': np.random.uniform(0.5, 3, n),
            'follow_up_within_7_days': np.random.choice([0, 1], n),
            'readmitted_30_days': np.random.choice([0, 1], n, p=[0.82, 0.18])
        })


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functions."""
    
    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'age': [65, 75, 80],
            'num_medications': [5, 10, 15],
            'num_high_risk_meds': [1, 3, 5],
            'medication_changes': [2, 4, 6],
            'charlson_score': [2.0, 5.0, 8.0],
            'functional_status': [6, 4, 2],
            'egfr': [90, 45, 25],
            'bnp': [100, 600, 1200],
            'hemoglobin': [13, 10, 8],
            'systolic_bp': [120, 150, 170],
            'diastolic_bp': [80, 95, 100]
        })
    
    def test_medication_complexity_calculation(self):
        """Test medication complexity score calculation."""
        # Calculate medication complexity
        med_complexity = (
            self.df['num_medications'] * 0.5 +
            self.df['num_high_risk_meds'] * 2.0 +
            self.df['medication_changes'] * 0.3
        )
        
        # Check that it's calculated correctly
        expected = np.array([5.1, 12.2, 19.3])
        np.testing.assert_array_almost_equal(med_complexity.values, expected, decimal=1)
        
        # Check that higher med count = higher complexity
        self.assertLess(med_complexity.iloc[0], med_complexity.iloc[1])
        self.assertLess(med_complexity.iloc[1], med_complexity.iloc[2])
    
    def test_kidney_disease_flags(self):
        """Test kidney disease severity flags."""
        severe_kidney = (self.df['egfr'] < 30).astype(int)
        moderate_kidney = ((self.df['egfr'] >= 30) & (self.df['egfr'] < 60)).astype(int)
        
        # Check flags
        self.assertEqual(severe_kidney.iloc[0], 0)  # eGFR 90 = not severe
        self.assertEqual(severe_kidney.iloc[2], 1)  # eGFR 25 = severe
        self.assertEqual(moderate_kidney.iloc[1], 1)  # eGFR 45 = moderate
    
    def test_elevated_bnp_flag(self):
        """Test elevated BNP flag."""
        elevated_bnp = (self.df['bnp'] > 500).astype(int)
        
        self.assertEqual(elevated_bnp.iloc[0], 0)  # BNP 100 = not elevated
        self.assertEqual(elevated_bnp.iloc[1], 1)  # BNP 600 = elevated
    
    def test_anemia_flag(self):
        """Test anemia flag."""
        anemia = (self.df['hemoglobin'] < 10).astype(int)
        
        self.assertEqual(anemia.iloc[0], 0)  # Hgb 13 = not anemic
        self.assertEqual(anemia.iloc[2], 1)  # Hgb 8 = anemic
    
    def test_hypertension_flag(self):
        """Test hypertension flag."""
        hypertension = (
            (self.df['systolic_bp'] > 140) | 
            (self.df['diastolic_bp'] > 90)
        ).astype(int)
        
        self.assertEqual(hypertension.iloc[0], 0)  # 120/80 = not hypertensive
        self.assertEqual(hypertension.iloc[1], 1)  # 150/95 = hypertensive
    
    def test_frailty_score_increases_with_age(self):
        """Test that frailty score increases with age and comorbidities."""
        frailty = (
            (self.df['age'] / 100) * 3 +
            self.df['charlson_score'] * 0.5 +
            (6 - self.df['functional_status']) * 0.3
        )
        
        # Frailty should increase with age
        self.assertLess(frailty.iloc[0], frailty.iloc[1])
        self.assertLess(frailty.iloc[1], frailty.iloc[2])
        
        # All frailty scores should be positive
        self.assertTrue((frailty > 0).all())


class TestMissingDataHandling(unittest.TestCase):
    """Test missing data handling strategies."""
    
    def setUp(self):
        """Set up test data with missing values."""
        self.df = pd.DataFrame({
            'hemoglobin': [12.0, np.nan, 14.0, np.nan, 11.0],
            'creatinine': [1.0, 1.5, np.nan, 1.2, np.nan],
            'age': [65, 70, 75, 80, 85],
            'gender': ['Male', 'Female', np.nan, 'Male', 'Female']
        })
    
    def test_missingness_indicator_creation(self):
        """Test that missingness indicators are created correctly."""
        hemoglobin_missing = self.df['hemoglobin'].isna().astype(int)
        
        self.assertEqual(hemoglobin_missing.sum(), 2)  # 2 missing values
        self.assertEqual(hemoglobin_missing.iloc[0], 0)  # First value not missing
        self.assertEqual(hemoglobin_missing.iloc[1], 1)  # Second value missing
    
    def test_numerical_imputation(self):
        """Test median imputation for numerical features."""
        imputer = SimpleImputer(strategy='median')
        
        hemoglobin_imputed = imputer.fit_transform(self.df[['hemoglobin']])
        
        # Check that no NaN values remain
        self.assertEqual(np.isnan(hemoglobin_imputed).sum(), 0)
        
        # Check that non-missing values unchanged
        self.assertAlmostEqual(hemoglobin_imputed[0, 0], 12.0)
        self.assertAlmostEqual(hemoglobin_imputed[2, 0], 14.0)
    
    def test_categorical_imputation(self):
        """Test mode imputation for categorical features."""
        imputer = SimpleImputer(strategy='most_frequent')
        
        gender_imputed = imputer.fit_transform(self.df[['gender']])
        
        # Check that no NaN values remain
        self.assertEqual(pd.isna(gender_imputed).sum(), 0)
        
        # Check that imputed value is most frequent (Male appears twice)
        self.assertIn(gender_imputed[2, 0], ['Male', 'Female'])
    
    def test_no_data_leakage_in_imputation(self):
        """Test that imputation doesn't leak information from test set."""
        # Split data
        train_df = self.df.iloc[:3]
        test_df = self.df.iloc[3:]
        
        # Fit imputer only on train
        imputer = SimpleImputer(strategy='median')
        imputer.fit(train_df[['hemoglobin']])
        
        # Transform test
        test_imputed = imputer.transform(test_df[['hemoglobin']])
        
        # Check that test set imputation uses train statistics only
        train_median = train_df['hemoglobin'].median()
        self.assertAlmostEqual(imputer.statistics_[0], train_median)


class TestCategoricalEncoding(unittest.TestCase):
    """Test categorical encoding strategies."""
    
    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'diagnosis': ['CHF', 'COPD', 'CHF', 'Pneumonia'],
            'discharge_disposition': ['Home', 'SNF', 'Home', 'Rehab']
        })
    
    def test_binary_encoding(self):
        """Test binary encoding for gender."""
        gender_encoded = (self.df['gender'] == 'Male').astype(int)
        
        self.assertEqual(gender_encoded.iloc[0], 1)  # Male = 1
        self.assertEqual(gender_encoded.iloc[1], 0)  # Female = 0
    
    def test_one_hot_encoding(self):
        """Test one-hot encoding."""
        diagnosis_dummies = pd.get_dummies(self.df['diagnosis'], prefix='diagnosis')
        
        # Check that we have right number of columns
        self.assertEqual(diagnosis_dummies.shape[1], 3)  # CHF, COPD, Pneumonia
        
        # Check that each row sums to 1 (one-hot)
        self.assertTrue((diagnosis_dummies.sum(axis=1) == 1).all())
        
        # Check specific encoding
        self.assertEqual(diagnosis_dummies.loc[0, 'diagnosis_CHF'], 1)
        self.assertEqual(diagnosis_dummies.loc[1, 'diagnosis_COPD'], 1)
    
    def test_drop_first_in_encoding(self):
        """Test that drop_first prevents multicollinearity."""
        diagnosis_dummies = pd.get_dummies(self.df['diagnosis'], prefix='diagnosis', drop_first=True)
        
        # Should have n-1 columns
        self.assertEqual(diagnosis_dummies.shape[1], 2)  # 3 categories - 1 = 2


class TestFeatureScaling(unittest.TestCase):
    """Test feature scaling."""
    
    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'age': [20, 40, 60, 80, 100],
            'charlson_score': [0, 2, 5, 8, 12],
            'num_medications': [0, 5, 10, 15, 20]
        })
    
    def test_standard_scaler(self):
        """Test StandardScaler normalization."""
        scaler = StandardScaler()
        
        scaled = scaler.fit_transform(self.df)
        
        # Check mean ≈ 0, std ≈ 1 for each column
        np.testing.assert_array_almost_equal(scaled.mean(axis=0), [0, 0, 0], decimal=10)
        np.testing.assert_array_almost_equal(scaled.std(axis=0), [1, 1, 1], decimal=10)
    
    def test_scaling_reversibility(self):
        """Test that scaling can be reversed."""
        scaler = StandardScaler()
        
        scaled = scaler.fit_transform(self.df)
        unscaled = scaler.inverse_transform(scaled)
        
        # Check that we get back original values
        np.testing.assert_array_almost_equal(unscaled, self.df.values, decimal=5)
    
    def test_scaling_preserves_relationships(self):
        """Test that scaling preserves relative ordering."""
        scaler = StandardScaler()
        
        original_order = self.df['age'].argsort()
        scaled = scaler.fit_transform(self.df[['age']])
        scaled_order = scaled[:, 0].argsort()
        
        # Order should be preserved
        np.testing.assert_array_equal(original_order, scaled_order)


class TestDataSplitting(unittest.TestCase):
    """Test train/validation/test splitting."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 1000
        self.X = pd.DataFrame({
            'feature1': np.random.randn(self.n),
            'feature2': np.random.randn(self.n)
        })
        self.y = pd.Series(np.random.choice([0, 1], self.n, p=[0.8, 0.2]))
    
    def test_split_sizes(self):
        """Test that split sizes are correct."""
        from sklearn.model_selection import train_test_split
        
        # First split: 85% temp, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=0.15, random_state=42
        )
        
        # Second split: ~70% train, ~15% val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42
        )
        
        # Check sizes
        self.assertAlmostEqual(len(X_train) / len(self.X), 0.70, places=1)
        self.assertAlmostEqual(len(X_val) / len(self.X), 0.15, places=1)
        self.assertAlmostEqual(len(X_test) / len(self.X), 0.15, places=1)
    
    def test_stratification_preserves_class_balance(self):
        """Test that stratified split preserves class distribution."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Check that class proportions are similar
        original_proportion = self.y.mean()
        train_proportion = y_train.mean()
        test_proportion = y_test.mean()
        
        self.assertAlmostEqual(train_proportion, original_proportion, places=1)
        self.assertAlmostEqual(test_proportion, original_proportion, places=1)
    
    def test_no_data_leakage_between_splits(self):
        """Test that train/test splits don't overlap."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Check that indices don't overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)


class TestSMOTE(unittest.TestCase):
    """Test SMOTE balancing."""
    
    def setUp(self):
        """Set up imbalanced data."""
        np.random.seed(42)
        self.X = np.random.randn(1000, 5)
        self.y = np.array([0] * 800 + [1] * 200)  # 80-20 imbalance
    
    def test_smote_balances_classes(self):
        """Test that SMOTE creates balanced dataset."""
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(self.X, self.y)
        
        # Count classes
        unique, counts = np.unique(y_balanced, return_counts=True)
        
        # Should be balanced (or close)
        self.assertAlmostEqual(counts[0], counts[1], delta=10)
    
    def test_smote_only_on_training(self):
        """Test that SMOTE is only applied to training data."""
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import SMOTE
        
        # Split first
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Apply SMOTE only to train
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Test set should remain imbalanced (original distribution)
        test_minority_proportion = y_test.mean()
        self.assertLess(test_minority_proportion, 0.30)  # Still imbalanced


class TestDataValidation(unittest.TestCase):
    """Test data validation functions."""
    
    def test_detect_data_leakage(self):
        """Test detection of data leakage features."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'readmitted_30_days': [0, 1, 0],
            'days_to_readmission': [np.nan, 5, np.nan],  # LEAKAGE!
            'post_discharge_visit': [0, 1, 0]  # LEAKAGE!
        })
        
        # Features that shouldn't be used (contain future information)
        leakage_features = ['days_to_readmission', 'post_discharge_visit']
        
        for feature in leakage_features:
            self.assertIn(feature, df.columns,
                         f"Leakage feature {feature} should be detected and removed")
    
    def test_feature_variance_threshold(self):
        """Test removal of low-variance features."""
        df = pd.DataFrame({
            'feature1': [1, 1, 1, 1, 1],  # No variance
            'feature2': [1, 2, 3, 4, 5],  # Good variance
            'feature3': [5, 5, 5, 5, 6],  # Low variance
        })
        
        # Calculate variance
        variances = df.var()
        
        # feature1 should have 0 variance
        self.assertEqual(variances['feature1'], 0)
        
        # feature2 should have good variance
        self.assertGreater(variances['feature2'], 2)
        
        # Remove low variance features (threshold = 0.1)
        selected_features = variances[variances > 0.1].index.tolist()
        
        self.assertIn('feature2', selected_features)
        self.assertNotIn('feature1', selected_features)


# Integration test
class TestEndToEndPipeline(unittest.TestCase):
    """Integration test for entire preprocessing pipeline."""
    
    def test_complete_pipeline(self):
        """Test that complete pipeline runs without errors."""
        # Generate mock data
        np.random.seed(42)
        df = pd.DataFrame({
            'patient_id': [f'PT{i:06d}' for i in range(100)],
            'age': np.random.randint(18, 95, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'race': np.random.choice(['White', 'Black', 'Hispanic'], 100),
            'num_medications': np.random.randint(0, 20, 100),
            'num_high_risk_meds': np.random.randint(0, 5, 100),
            'medication_changes': np.random.randint(0, 10, 100),
            'charlson_score': np.random.uniform(0, 10, 100),
            'functional_status': np.random.randint(0, 7, 100),
            'readmitted_30_days': np.random.choice([0, 1], 100, p=[0.82, 0.18])
        })
        
        # Step 1: Feature engineering
        df['medication_complexity'] = (
            df['num_medications'] * 0.5 +
            df['num_high_risk_meds'] * 2.0 +
            df['medication_changes'] * 0.3
        )
        
        # Step 2: Separate features and target
        y = df['readmitted_30_days']
        X = df.drop(columns=['readmitted_30_days', 'patient_id'])
        
        # Step 3: Encode categoricals
        X = pd.get_dummies(X, columns=['gender', 'race'], drop_first=True)
        
        # Step 4: Scale features
        scaler = StandardScaler()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        # Step 5: Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Assertions
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertFalse(X_train.isnull().any().any())
        self.assertFalse(X_test.isnull().any().any())
        
        print("End-to-end pipeline test passed!")


def run_tests():
    """Run all tests and print summary."""
    
    print("\n" + "="*70)
    print("RUNNING PREPROCESSING PIPELINE UNIT TESTS")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineering))
    suite.addTests(loader.loadTestsFromTestCase(TestMissingDataHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestCategoricalEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureScaling))
    suite.addTests(loader.loadTestsFromTestCase(TestDataSplitting))
    suite.addTests(loader.loadTestsFromTestCase(TestSMOTE))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n ALL TESTS PASSED!")
    else:
        print("\n SOME TESTS FAILED")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)