"""
Hospital Readmission Prediction - Data Generation & Preprocessing Pipeline
============================================================================

This script generates synthetic patient data that mimics real EHR characteristics
and implements the comprehensive preprocessing pipeline discussed in the assignment.

Author: Kivairu Samuel
Date: 12th November 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# SECTION 1: SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_patient_data(n_patients=10000):
    """
    Generate synthetic hospital patient data with realistic characteristics.
    
    Parameters:
    -----------
    n_patients : int
        Number of patient records to generate
    
    Returns:
    --------
    pd.DataFrame
        Synthetic patient dataset with features and target variable
    """
    
    print(f"Generating synthetic data for {n_patients} patients...")
    
    # --- Demographics ---
    age = np.random.normal(65, 15, n_patients).clip(18, 95)
    gender = np.random.choice(['Male', 'Female'], n_patients, p=[0.48, 0.52])
    race = np.random.choice(
        ['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
        n_patients, 
        p=[0.60, 0.20, 0.12, 0.05, 0.03]
    )
    
    # --- Clinical Features ---
    # Primary admission diagnosis (ICD-10 categories)
    diagnoses = ['CHF', 'COPD', 'Pneumonia', 'Sepsis', 'MI', 'Stroke', 'Diabetes', 'Renal_Failure']
    admission_diagnosis = np.random.choice(diagnoses, n_patients, p=[0.18, 0.15, 0.12, 0.10, 0.10, 0.08, 0.15, 0.12])
    
    # Number of comorbidities (0-10)
    num_comorbidities = np.random.poisson(2.5, n_patients).clip(0, 10)
    
    # Charlson Comorbidity Index (0-15, higher = more severe)
    charlson_score = (num_comorbidities * 1.2 + np.random.normal(0, 1, n_patients)).clip(0, 15)
    
    # Length of stay (1-30 days)
    length_of_stay = np.random.gamma(3, 2, n_patients).clip(1, 30).astype(int)
    
    # Prior admissions in last 12 months
    prior_admissions_12mo = np.random.poisson(1.2, n_patients).clip(0, 10)
    
    # --- Laboratory Values ---
    # Hemoglobin (g/dL) - normal range 12-16
    hemoglobin = np.random.normal(12.5, 2.5, n_patients).clip(7, 18)
    
    # Creatinine (mg/dL) - normal range 0.7-1.3, higher indicates kidney problems
    creatinine = np.random.gamma(2, 0.8, n_patients).clip(0.5, 8)
    
    # eGFR (estimated glomerular filtration rate) - kidney function
    egfr = 175 * (creatinine ** -1.154) * (age ** -0.203)
    egfr = egfr.clip(10, 120)
    
    # Sodium (mEq/L) - normal 135-145
    sodium = np.random.normal(138, 4, n_patients).clip(120, 155)
    
    # BNP (brain natriuretic peptide) - heart failure marker
    # Higher in CHF patients
    bnp = np.where(
        admission_diagnosis == 'CHF',
        np.random.gamma(8, 100, n_patients),  # Higher for CHF
        np.random.gamma(2, 50, n_patients)    # Lower for non-CHF
    ).clip(0, 5000)
    
    # --- Vital Signs (averaged across admission) ---
    systolic_bp = np.random.normal(130, 20, n_patients).clip(80, 200)
    diastolic_bp = np.random.normal(80, 12, n_patients).clip(50, 120)
    heart_rate = np.random.normal(80, 15, n_patients).clip(50, 140)
    respiratory_rate = np.random.normal(18, 4, n_patients).clip(10, 35)
    
    # Vital sign stability (lower = more stable)
    vital_instability = np.random.gamma(2, 1.5, n_patients).clip(0, 10)
    
    # --- Medications ---
    num_medications = np.random.gamma(3, 2.5, n_patients).clip(0, 25).astype(int)
    
    # High-risk medications (anticoagulants, insulin, etc.)
    num_high_risk_meds = (num_medications * np.random.uniform(0.1, 0.3, n_patients)).astype(int)
    
    # Medication changes during admission
    medication_changes = np.random.poisson(2, n_patients).clip(0, 10)
    
    # --- Social Determinants of Health ---
    lives_alone = np.random.choice([0, 1], n_patients, p=[0.65, 0.35])
    has_caregiver = np.random.choice([0, 1], n_patients, p=[0.40, 0.60])
    transportation_access = np.random.choice([0, 1], n_patients, p=[0.25, 0.75])
    
    # Insurance type
    insurance = np.random.choice(
        ['Medicare', 'Medicaid', 'Private', 'Uninsured'], 
        n_patients, 
        p=[0.50, 0.20, 0.25, 0.05]
    )
    
    # --- Discharge Planning ---
    follow_up_within_7_days = np.random.choice([0, 1], n_patients, p=[0.30, 0.70])
    
    discharge_disposition = np.random.choice(
        ['Home', 'Home_with_Services', 'SNF', 'Rehabilitation', 'AMA'],
        n_patients,
        p=[0.60, 0.20, 0.12, 0.06, 0.02]
    )
    
    # Functional status (ADL independence score 0-6, higher = more independent)
    functional_status = np.random.binomial(6, 0.7, n_patients)
    
    # --- Create Base Risk Score ---
    # Used to generate realistic target variable
    base_risk = (
        0.01 * age +
        0.05 * charlson_score +
        0.03 * num_comorbidities +
        0.04 * prior_admissions_12mo +
        0.02 * num_medications +
        0.03 * num_high_risk_meds +
        0.15 * (discharge_disposition == 'AMA').astype(int) +
        0.10 * lives_alone +
        -0.08 * has_caregiver +
        -0.10 * follow_up_within_7_days +
        -0.05 * transportation_access +
        0.05 * vital_instability +
        0.01 * (egfr < 30).astype(int) * 5 +  # Severe kidney disease
        0.01 * (bnp > 500).astype(int) * 3 +   # Elevated heart failure marker
        np.random.normal(0, 0.3, n_patients)    # Random variation
    )
    
    # Convert to probability using logistic function
    readmission_prob = 1 / (1 + np.exp(-base_risk))
    
    # Generate binary target (1 = readmitted, 0 = not readmitted)
    readmitted_30_days = (np.random.random(n_patients) < readmission_prob).astype(int)
    
    # --- Introduce Missing Data (realistic patterns) ---
    # Labs: 15-30% missing (not all tests ordered for all patients)
    hemoglobin = introduce_missing(hemoglobin, missing_rate=0.20)
    creatinine = introduce_missing(creatinine, missing_rate=0.15)
    sodium = introduce_missing(sodium, missing_rate=0.18)
    bnp = introduce_missing(bnp, missing_rate=0.35)  # BNP only ordered for cardiac patients
    
    # SDOH: 30-50% missing (not routinely collected)
    lives_alone = introduce_missing(lives_alone, missing_rate=0.40)
    has_caregiver = introduce_missing(has_caregiver, missing_rate=0.35)
    transportation_access = introduce_missing(transportation_access, missing_rate=0.45)
    
    # Functional status: 25% missing
    functional_status = introduce_missing(functional_status, missing_rate=0.25)
    
    # --- Create DataFrame ---
    data = pd.DataFrame({
        # Demographics
        'patient_id': [f'PT{str(i).zfill(6)}' for i in range(n_patients)],
        'age': age.round(0).astype(int),
        'gender': gender,
        'race': race,
        
        # Clinical
        'admission_diagnosis': admission_diagnosis,
        'num_comorbidities': num_comorbidities,
        'charlson_score': charlson_score.round(1),
        'length_of_stay': length_of_stay,
        'prior_admissions_12mo': prior_admissions_12mo,
        
        # Labs
        'hemoglobin': hemoglobin.round(1),
        'creatinine': creatinine.round(2),
        'egfr': egfr.round(1),
        'sodium': sodium.round(1),
        'bnp': bnp.round(0),
        
        # Vitals
        'systolic_bp': systolic_bp.round(0).astype(int),
        'diastolic_bp': diastolic_bp.round(0).astype(int),
        'heart_rate': heart_rate.round(0).astype(int),
        'respiratory_rate': respiratory_rate.round(0).astype(int),
        'vital_instability': vital_instability.round(2),
        
        # Medications
        'num_medications': num_medications,
        'num_high_risk_meds': num_high_risk_meds,
        'medication_changes': medication_changes,
        
        # SDOH
        'lives_alone': lives_alone,
        'has_caregiver': has_caregiver,
        'transportation_access': transportation_access,
        'insurance': insurance,
        
        # Discharge
        'follow_up_within_7_days': follow_up_within_7_days,
        'discharge_disposition': discharge_disposition,
        'functional_status': functional_status,
        
        # Target variable
        'readmitted_30_days': readmitted_30_days
    })
    
    print(f"✓ Generated {len(data)} patient records")
    print(f"✓ Readmission rate: {readmitted_30_days.mean():.1%}")
    print(f"✓ Missing data introduced in key features")
    
    return data


def introduce_missing(array, missing_rate=0.2):
    """Randomly introduce missing values (NaN) into array."""
    array_copy = array.astype(float).copy()  # Ensure array is float to allow NaN
    missing_mask = np.random.random(len(array)) < missing_rate
    array_copy[missing_mask] = np.nan
    return array_copy


# ============================================================================
# SECTION 2: DATA PREPROCESSING PIPELINE
# ============================================================================

class HospitalReadmissionPreprocessor:
    """
    Comprehensive preprocessing pipeline for hospital readmission prediction.
    
    Implements all strategies discussed in the assignment:
    - Missing data handling
    - Feature engineering
    - Encoding categorical variables
    - Scaling numerical features
    - Class imbalance handling
    """
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_names = []
        
    def fit_transform(self, df, target_col='readmitted_30_days'):
        """
        Fit preprocessing pipeline and transform training data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw patient data
        target_col : str
            Name of target variable column
            
        Returns:
        --------
        X : pd.DataFrame
            Preprocessed features
        y : pd.Series
            Target variable
        """
        
        print("\n" + "="*70)
        print("PREPROCESSING PIPELINE")
        print("="*70)
        
        df = df.copy()
        
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col, 'patient_id'])
        
        # Step 1: Feature Engineering
        print("\n[1/6] Feature Engineering...")
        X = self._engineer_features(X)
        
        # Step 2: Handle Missing Data
        print("[2/6] Handling Missing Data...")
        X = self._handle_missing_data(X, fit=True)
        
        # Step 3: Encode Categorical Variables
        print("[3/6] Encoding Categorical Variables...")
        X = self._encode_categorical(X, fit=True)
        
        # Step 4: Scale Numerical Features
        print("[4/6] Scaling Numerical Features...")
        X = self._scale_features(X, fit=True)
        
        # Step 5: Feature Selection (remove low variance)
        print("[5/6] Feature Selection...")
        X = self._select_features(X, fit=True)
        
        # Step 6: Handle Class Imbalance (only on training data)
        print("[6/6] Class Imbalance Handling...")
        print(f"   Original class distribution: {y.value_counts().to_dict()}")
        
        self.feature_names = X.columns.tolist()
        
        print(f"\n✓ Preprocessing complete!")
        print(f"✓ Final feature count: {X.shape[1]}")
        print(f"✓ Sample size: {X.shape[0]}")
        
        return X, y
    
    def transform(self, df):
        """Transform new data using fitted preprocessors."""
        df = df.copy()
        X = df.drop(columns=['readmitted_30_days', 'patient_id'], errors='ignore')
        
        X = self._engineer_features(X)
        X = self._handle_missing_data(X, fit=False)
        X = self._encode_categorical(X, fit=False)
        X = self._scale_features(X, fit=False)
        X = self._select_features(X, fit=False)
        
        return X
    
    def _engineer_features(self, X):
        """Create derived clinical features."""
        
        # Medication complexity score
        X['medication_complexity'] = (
            X['num_medications'] * 0.5 +
            X['num_high_risk_meds'] * 2.0 +
            X['medication_changes'] * 0.3
        )
        
        # Kidney disease severity (based on eGFR)
        X['severe_kidney_disease'] = (X['egfr'] < 30).astype(int)
        X['moderate_kidney_disease'] = ((X['egfr'] >= 30) & (X['egfr'] < 60)).astype(int)
        
        # Heart failure risk (elevated BNP)
        X['elevated_bnp'] = (X['bnp'] > 500).astype(int)
        
        # Hypertension (high blood pressure)
        X['hypertension'] = ((X['systolic_bp'] > 140) | (X['diastolic_bp'] > 90)).astype(int)
        
        # Anemia (low hemoglobin)
        X['anemia'] = (X['hemoglobin'] < 10).astype(int)
        
        # Frailty score (combination of age, comorbidities, functional status)
        X['frailty_score'] = (
            (X['age'] / 100) * 3 +
            X['charlson_score'] * 0.5 +
            (6 - X['functional_status'].fillna(3)) * 0.3
        )
        
        # Social isolation risk
        X['social_isolation'] = (
            X['lives_alone'].fillna(0) * 1 +
            (1 - X['has_caregiver'].fillna(0.5)) * 1 +
            (1 - X['transportation_access'].fillna(0.5)) * 0.5
        )
        
        # High utilization (frequent admissions)
        X['high_utilizer'] = (X['prior_admissions_12mo'] >= 2).astype(int)
        
        # Age groups (categorical)
        X['age_group'] = pd.cut(
            X['age'], 
            bins=[0, 50, 65, 80, 100], 
            labels=['<50', '50-65', '65-80', '80+']
        )
        
        print(f"   ✓ Created {9} engineered features")
        return X
    
    def _handle_missing_data(self, X, fit=False):
        """Handle missing values with appropriate strategies."""
        
        # Strategy 1: Create missingness indicators for labs
        lab_features = ['hemoglobin', 'creatinine', 'sodium', 'bnp']
        for lab in lab_features:
            if lab in X.columns:
                X[f'{lab}_missing'] = X[lab].isna().astype(int)
        
        # Strategy 2: Median imputation for numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if fit:
            self.imputers['numerical'] = SimpleImputer(strategy='median')
            X[numerical_cols] = self.imputers['numerical'].fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.imputers['numerical'].transform(X[numerical_cols])
        
        # Strategy 3: Mode imputation for categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if fit:
            self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = self.imputers['categorical'].fit_transform(X[categorical_cols])
        else:
            X[categorical_cols] = self.imputers['categorical'].transform(X[categorical_cols])
        
        missing_counts = X.isna().sum().sum()
        print(f"   ✓ Handled missing data (remaining NaNs: {missing_counts})")
        
        return X
    
    def _encode_categorical(self, X, fit=False):
        """Encode categorical variables."""
        
        # Binary encoding for gender
        if 'gender' in X.columns:
            X['gender_male'] = (X['gender'] == 'Male').astype(int)
            X = X.drop(columns=['gender'])
        
        # One-hot encoding for low-cardinality categoricals
        categorical_cols = ['admission_diagnosis', 'discharge_disposition', 
                           'insurance', 'age_group', 'race']
        
        for col in categorical_cols:
            if col in X.columns:
                if fit:
                    # Store categories for consistency
                    self.encoders[col] = X[col].unique()
                
                # One-hot encode
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(columns=[col])
        
        print(f"   ✓ Encoded categorical variables")
        return X
    
    def _scale_features(self, X, fit=False):
        """Normalize numerical features."""
        
        # Features to scale (continuous variables)
        scale_features = ['age', 'charlson_score', 'length_of_stay', 
                         'hemoglobin', 'creatinine', 'egfr', 'sodium', 'bnp',
                         'systolic_bp', 'diastolic_bp', 'heart_rate',
                         'medication_complexity', 'frailty_score', 'social_isolation']
        
        scale_features = [f for f in scale_features if f in X.columns]
        
        if fit:
            self.scalers['standard'] = StandardScaler()
            X[scale_features] = self.scalers['standard'].fit_transform(X[scale_features])
        else:
            X[scale_features] = self.scalers['standard'].transform(X[scale_features])
        
        print(f"   ✓ Scaled {len(scale_features)} numerical features")
        return X
    
    def _select_features(self, X, fit=False):
        """Remove low-variance features."""
        
        if fit:
            # Calculate variance for all numerical columns
            variances = X.var()
            self.selected_features = variances[variances > 0.01].index.tolist()
            removed = len(X.columns) - len(self.selected_features)
            if removed > 0:
                print(f"   ✓ Removed {removed} low-variance features")
        
        X = X[self.selected_features]
        return X


# ============================================================================
# SECTION 3: DATA SPLITTING & SMOTE
# ============================================================================

def prepare_train_val_test_split(X, y, test_size=0.15, val_size=0.15, 
                                 apply_smote=True, random_state=42):
    """
    Split data into train/validation/test sets and apply SMOTE to training data.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Preprocessed features
    y : pd.Series
        Target variable
    test_size : float
        Proportion of data for test set
    val_size : float
        Proportion of remaining data for validation set
    apply_smote : bool
        Whether to apply SMOTE to balance training data
    random_state : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary containing train/val/test splits
    """
    
    print("\n" + "="*70)
    print("DATA SPLITTING & BALANCING")
    print("="*70)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"\nOriginal split sizes:")
    print(f"  Training:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\nClass distribution before SMOTE:")
    print(f"  Training:   {dict(y_train.value_counts())}")
    print(f"  Validation: {dict(y_val.value_counts())}")
    print(f"  Test:       {dict(y_test.value_counts())}")
    
    # Apply SMOTE only to training data
    if apply_smote:
        print(f"\nApplying SMOTE to training data...")
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"  After SMOTE: {len(X_train_balanced):,} samples")
        print(f"  Class distribution: {dict(pd.Series(y_train_balanced).value_counts())}")
    else:
        X_train_balanced = X_train
        y_train_balanced = y_train
    
    return {
        'X_train': X_train_balanced,
        'y_train': y_train_balanced,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("HOSPITAL READMISSION PREDICTION - DATA PIPELINE")
    print("="*70)
    
    # Generate synthetic data
    df = generate_synthetic_patient_data(n_patients=10000)
    
    # Save raw data
    df.to_csv('synthetic_patient_data.csv', index=False)
    print(f"\n✓ Raw data saved to 'synthetic_patient_data.csv'")
    
    # Display sample
    print("\nSample of raw data:")
    print(df.head())
    
    # Initialize preprocessor
    preprocessor = HospitalReadmissionPreprocessor()
    
    # Preprocess data
    X, y = preprocessor.fit_transform(df)
    
    # Split data
    data_splits = prepare_train_val_test_split(X, y, apply_smote=True)
    
    # Save preprocessed data
    pd.DataFrame(data_splits['X_train']).to_csv('X_train.csv', index=False)
    pd.DataFrame(data_splits['y_train']).to_csv('y_train.csv', index=False)
    pd.DataFrame(data_splits['X_val']).to_csv('X_val.csv', index=False)
    pd.DataFrame(data_splits['y_val']).to_csv('y_val.csv', index=False)
    pd.DataFrame(data_splits['X_test']).to_csv('X_test.csv', index=False)
    pd.DataFrame(data_splits['y_test']).to_csv('y_test.csv', index=False)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  - synthetic_patient_data.csv (raw data)")
    print(f"  - X_train.csv, y_train.csv (training set)")
    print(f"  - X_val.csv, y_val.csv (validation set)")
    print(f"  - X_test.csv, y_test.csv (test set)")
    print(f"\nReady for model training!")