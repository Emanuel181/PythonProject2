# src/config.py

import os # Imports the 'os' module, which provides functions for interacting with the operating system, such as path manipulation and directory creation.

# --- Project Paths ---
# Defines core directory paths for the project. These paths are crucial for organizing raw data, processed data, trained models, reports, and logs.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Determines the absolute path to the project's root directory. It goes up two levels from the current file's location.
DATA_DIR = os.path.join(PROJECT_ROOT, "data") # Defines the path to the 'data' directory within the project root.
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "MachineLearningCVE") # Defines the path to the directory containing raw (unprocessed) data, specifically for the CIC-IDS2017 dataset.
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed") # Defines the path to the directory where processed data will be stored.
MODELS_DIR = os.path.join(PROJECT_ROOT, "models") # Defines the path to the directory where trained machine learning models will be saved.
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports") # Defines the path to the directory where generated reports (e.g., evaluation summaries, plots) will be stored.
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs") # Defines the path to the directory where application logs will be stored.

# Creates all necessary directories if they don't already exist. `exist_ok=True` prevents an error if the directory already exists.
os.makedirs(RAW_DATA_DIR, exist_ok=True) # Creates the raw data directory.
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) # Creates the processed data directory.
os.makedirs(MODELS_DIR, exist_ok=True) # Creates the models directory.
os.makedirs(REPORTS_DIR, exist_ok=True) # Creates the reports directory.
os.makedirs(LOGS_DIR, exist_ok=True) # Creates the logs directory.


# --- Global Environment Mode ---
# This section defines a global variable to control the application's environment mode ('test' or 'prod').
# This variable will be set based on command-line arguments in main.py.
# Default to 'test' for development convenience.
ENV_MODE = "test"  # 'test' or 'prod' # Initializes the environment mode to 'test' by default.

# --- Base Configuration ---
# Defines fundamental configuration parameters that are generally common across environments unless explicitly overridden.
TARGET_COLUMN = "Label" # Specifies the name of the target column in the dataset, which contains the labels (e.g., attack types).
BENIGN_LABEL = "BENIGN" # Defines the specific label used for benign (non-attack) network traffic.
SEED = 42 # Sets a random seed for reproducibility across various operations (e.g., data splitting, model training).
USE_RANDOMIZED_SEARCH = True  # Always use RandomizedSearchCV for efficiency # A boolean flag to indicate whether RandomizedSearchCV (True) or GridSearchCV (False) should be used for hyperparameter tuning. RandomizedSearchCV is generally more efficient for large search spaces.
CONFUSION_MATRIX_NORMALIZE = True # A boolean flag to determine if confusion matrices should be normalized during evaluation.

# --- Dataset Specifics (STATIC - does not change with ENV_MODE) ---
# Lists the names of the CSV files that constitute the CIC-IDS2017 dataset. These are static and do not change with the environment mode.
CSV_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", # First CSV file name.
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", # Second CSV file name.
    "Friday-WorkingHours-Morning.pcap_ISCX.csv", # Third CSV file name.
    "Monday-WorkingHours.pcap_ISCX.csv", # Fourth CSV file name.
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", # Fifth CSV file name.
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv", # Sixth CSV file name.
    "Wednesday-WorkingHours.pcap_ISCX.csv", # Seventh CSV file name.
]

# --- Data Validation Schemas (STATIC - does not change with ENV_MODE) ---
# Defines the expected schema (column names and their approximate data types) for the dataset. This is used for early data validation.
EXPECTED_COLUMNS_AND_DTYPES = {
    "Destination Port": "int", # Expected data type for 'Destination Port' column.
    "Flow Duration": "int", # Expected data type for 'Flow Duration' column.
    "Total Fwd Packets": "int", # Expected data type for 'Total Fwd Packets' column.
    "Total Backward Packets": "int", # Expected data type for 'Total Backward Packets' column.
    "Total Length of Fwd Packets": "int", # Expected data type for 'Total Length of Fwd Packets' column.
    "Total Length of Bwd Packets": "int", # Expected data type for 'Total Length of Bwd Packets' column.
    "Fwd Packet Length Max": "int", # Expected data type for 'Fwd Packet Length Max' column.
    "Fwd Packet Length Min": "int", # Expected data type for 'Fwd Packet Length Min' column.
    "Fwd Packet Length Mean": "float", # Expected data type for 'Fwd Packet Length Mean' column.
    "Fwd Packet Length Std": "float", # Expected data type for 'Fwd Packet Length Std' column.
    "Bwd Packet Length Max": "int", # Expected data type for 'Bwd Packet Length Max' column.
    "Bwd Packet Length Min": "int", # Expected data type for 'Bwd Packet Length Min' column.
    "Bwd Packet Length Mean": "float", # Expected data type for 'Bwd Packet Length Mean' column.
    "Bwd Packet Length Std": "float", # Expected data type for 'Bwd Packet Length Std' column.
    "Flow Bytes/s": "float", # Expected data type for 'Flow Bytes/s' column.
    "Flow Packets/s": "float", # Expected data type for 'Flow Packets/s' column.
    "Flow IAT Mean": "float", # Expected data type for 'Flow IAT Mean' column.
    "Flow IAT Std": "float", # Expected data type for 'Flow IAT Std' column.
    "Flow IAT Max": "int", # Expected data type for 'Flow IAT Max' column.
    "Flow IAT Min": "int", # Expected data type for 'Flow IAT Min' column.
    "Fwd IAT Total": "int", # Expected data type for 'Fwd IAT Total' column.
    "Fwd IAT Mean": "float", # Expected data type for 'Fwd IAT Mean' column.
    "Fwd IAT Std": "float", # Expected data type for 'Fwd IAT Std' column.
    "Fwd IAT Max": "int", # Expected data type for 'Fwd IAT Max' column.
    "Fwd IAT Min": "int", # Expected data type for 'Fwd IAT Min' column.
    "Bwd IAT Total": "int", # Expected data type for 'Bwd IAT Total' column.
    "Bwd IAT Mean": "float", # Expected data type for 'Bwd IAT Mean' column.
    "Bwd IAT Std": "float", # Expected data type for 'Bwd IAT Std' column.
    "Bwd IAT Max": "int", # Expected data type for 'Bwd IAT Max' column.
    "Bwd IAT Min": "int", # Expected data type for 'Bwd IAT Min' column.
    "Fwd PSH Flags": "int", # Expected data type for 'Fwd PSH Flags' column.
    "Bwd PSH Flags": "int", # Expected data type for 'Bwd PSH Flags' column.
    "Fwd URG Flags": "int", # Expected data type for 'Fwd URG Flags' column.
    "Bwd URG Flags": "int", # Expected data type for 'Bwd URG Flags' column.
    "Fwd Header Length": "int", # Expected data type for 'Fwd Header Length' column.
    "Bwd Header Length": "int", # Expected data type for 'Bwd Header Length' column.
    "Fwd Packets/s": "float", # Expected data type for 'Fwd Packets/s' column.
    "Bwd Packets/s": "float", # Expected data type for 'Bwd Packets/s' column.
    "Min Packet Length": "int", # Expected data type for 'Min Packet Length' column.
    "Max Packet Length": "int", # Expected data type for 'Max Packet Length' column.
    "Packet Length Mean": "float", # Expected data type for 'Packet Length Mean' column.
    "Packet Length Std": "float", # Expected data type for 'Packet Length Std' column.
    "Packet Length Variance": "float", # Expected data type for 'Packet Length Variance' column.
    "FIN Flag Count": "int", # Expected data type for 'FIN Flag Count' column.
    "SYN Flag Count": "int", # Expected data type for 'SYN Flag Count' column.
    "RST Flag Count": "int", # Expected data type for 'RST Flag Count' column.
    "PSH Flag Count": "int", # Expected data type for 'PSH Flag Count' column.
    "ACK Flag Count": "int", # Expected data type for 'ACK Flag Count' column.
    "URG Flag Count": "int", # Expected data type for 'URG Flag Count' column.
    "CWE Flag Count": "int", # Expected data type for 'CWE Flag Count' column.
    "ECE Flag Count": "int", # Expected data type for 'ECE Flag Count' column.
    "Down/Up Ratio": "int", # Expected data type for 'Down/Up Ratio' column.
    "Average Packet Size": "float", # Expected data type for 'Average Packet Size' column.
    "Avg Fwd Segment Size": "float", # Expected data type for 'Avg Fwd Segment Size' column.
    "Avg Bwd Segment Size": "float", # Expected data type for 'Avg Bwd Segment Size' column.
    "Fwd Header Length.1": "int", # Expected data type for 'Fwd Header Length.1' column (note the '.1' which might indicate a duplicated column name in original data).
    "Fwd Avg Bytes/Bulk": "int", # Expected data type for 'Fwd Avg Bytes/Bulk' column.
    "Fwd Avg Packets/Bulk": "int", # Expected data type for 'Fwd Avg Packets/Bulk' column.
    "Fwd Avg Bulk Rate": "int", # Expected data type for 'Fwd Avg Bulk Rate' column.
    "Bwd Avg Bytes/Bulk": "int", # Expected data type for 'Bwd Avg Bytes/Bulk' column.
    "Bwd Avg Packets/Bulk": "int", # Expected data type for 'Bwd Avg Packets/Bulk' column.
    "Bwd Avg Bulk Rate": "int", # Expected data type for 'Bwd Avg Bulk Rate' column.
    "Subflow Fwd Packets": "int", # Expected data type for 'Subflow Fwd Packets' column.
    "Subflow Fwd Bytes": "int", # Expected data type for 'Subflow Fwd Bytes' column.
    "Subflow Bwd Packets": "int", # Expected data type for 'Subflow Bwd Packets' column.
    "Subflow Bwd Bytes": "int", # Expected data type for 'Subflow Bwd Bytes' column.
    "Init_Win_bytes_forward": "int", # Expected data type for 'Init_Win_bytes_forward' column.
    "Init_Win_bytes_backward": "int", # Expected data type for 'Init_Win_bytes_backward' column.
    "act_data_pkt_fwd": "int", # Expected data type for 'act_data_pkt_fwd' column.
    "min_seg_size_forward": "int", # Expected data type for 'min_seg_size_forward' column.
    "Active Mean": "float", # Expected data type for 'Active Mean' column.
    "Active Std": "float", # Expected data type for 'Active Std' column.
    "Active Max": "int", # Expected data type for 'Active Max' column.
    "Active Min": "int", # Expected data type for 'Active Min' column.
    "Idle Mean": "float", # Expected data type for 'Idle Mean' column.
    "Idle Std": "float", # Expected data type for 'Idle Std' column.
    "Idle Max": "int", # Expected data type for 'Idle Max' column.
    "Idle Min": "int", # Expected data type for 'Idle Min' column.
    "Label": "object", # Expected data type for 'Label' column (the target variable).
}


# --- Environment-Specific Settings ---
# Defines different configurations for 'test' and 'prod' modes, allowing for flexible execution environments.
# These will override the general parameters defined above.

# TEST MODE SETTINGS (for fast iteration and debugging)
# This dictionary holds configuration parameters specific to the 'test' environment.
TEST_SETTINGS = {
    "TEST_MODE_ACTIVE": True, # Boolean flag indicating if test mode is active.
    "TEST_MODE_TRAIN_SAMPLE_RATIO": 0.005,  # Use a very small sample of training data # Specifies the ratio of training data to use in test mode (0.5% in this case) for faster runs.
    "CV_FOLDS": 2,  # Fewer CV folds for faster tuning # Sets the number of cross-validation folds for hyperparameter tuning.
    "N_ITER_SEARCH_MODELS": 1,  # Minimal iterations for hyperparameter search # Sets the number of iterations for RandomizedSearchCV, keeping it minimal for speed.
    "RF_N_ESTIMATORS": 20,  # Smaller ensemble size for RF # Number of estimators for RandomForest in test mode.
    "XGB_N_ESTIMATORS": 20,  # Smaller ensemble size for XGB # Number of estimators for XGBoost in test mode.
    "LGBM_N_ESTIMATORS": 20,  # Smaller ensemble size for LGBM # Number of estimators for LightGBM in test mode.
    "RF_MAX_DEPTH_TUNE": [3, 5],  # Limited max_depth for faster RF # Range of max_depth for RandomForest hyperparameter tuning.
    "XGB_MAX_DEPTH_TUNE": [3, 5],  # Limited max_depth for faster XGB # Range of max_depth for XGBoost hyperparameter tuning.
    "LGBM_MAX_DEPTH_TUNE": [3, 5],  # Limited max_depth for faster LGBM # Range of max_depth for LightGBM hyperparameter tuning.
    "LOG_LEVEL": "INFO",  # Could be 'DEBUG' for more verbosity # Sets the logging level for the test environment.
}

# PRODUCTION MODE SETTINGS (for final training and deployment)
# This dictionary holds configuration parameters specific to the 'prod' (production) environment.
PRODUCTION_SETTINGS = {
    "TEST_MODE_ACTIVE": False, # Boolean flag indicating if test mode is active (False for prod).
    "TEST_MODE_TRAIN_SAMPLE_RATIO": 1.0,  # Use full training data # Specifies to use the full training data in production mode.
    "CV_FOLDS": 5,  # More robust CV folds # Sets a more robust number of cross-validation folds for production.
    "N_ITER_SEARCH_MODELS": 50,  # More extensive hyperparameter search # Sets a more extensive number of iterations for hyperparameter search.
    "RF_N_ESTIMATORS": 100,  # Larger ensemble size for RF # Larger number of estimators for RandomForest in production.
    "XGB_N_ESTIMATORS": 100,  # Larger ensemble size for XGB # Larger number of estimators for XGBoost in production.
    "LGBM_N_ESTIMATORS": 100,  # Larger ensemble size for LGBM # Larger number of estimators for LightGBM in production.
    "RF_MAX_DEPTH_TUNE": [5, 10, 20],  # Broader max_depth range for RF # Broader range of max_depth for RandomForest hyperparameter tuning.
    "XGB_MAX_DEPTH_TUNE": [5, 10, 15],  # Broader max_depth range for XGB # Broader range of max_depth for XGBoost hyperparameter tuning.
    "LGBM_MAX_DEPTH_TUNE": [5, 10, 20],  # Broader max_depth range for LGBM # Broader range of max_depth for LightGBM hyperparameter tuning.
    "LOG_LEVEL": "INFO",  # Standard info level # Sets the standard logging level for the production environment.
}

# --- Apply Settings Based on ENV_MODE (default or overridden by main.py) ---
CURRENT_SETTINGS = TEST_SETTINGS  # Default to test settings # Initializes CURRENT_SETTINGS to TEST_SETTINGS by default.

# This function will be called by main.py to set the environment mode dynamically.
# It uses a global variable to ensure settings are updated only once.
_settings_applied = False # A flag to ensure `set_env_mode` is applied only once.


def set_env_mode(mode: str): # Defines a function to set the environment mode.
    global ENV_MODE, CURRENT_SETTINGS, _settings_applied # Declares global variables that this function will modify.
    if not _settings_applied:  # Only apply settings once to avoid re-initializing # Checks if settings have already been applied to prevent redundant execution.
        if mode == "prod": # If the requested mode is 'prod'.
            ENV_MODE = "prod" # Sets the global environment mode to 'prod'.
            CURRENT_SETTINGS = PRODUCTION_SETTINGS # Assigns production settings to CURRENT_SETTINGS.
        else:  # Default to 'test' # If the requested mode is not 'prod' (defaults to 'test').
            ENV_MODE = "test" # Sets the global environment mode to 'test'.
            CURRENT_SETTINGS = TEST_SETTINGS # Assigns test settings to CURRENT_SETTINGS.
        _settings_applied = True # Sets the flag to True, indicating settings have been applied.
        # Using print here, as logger might not be fully configured at initial import time.
        print(
            f"INFO: Project running in '{ENV_MODE}' mode. Settings applied dynamically."
        ) # Prints an informational message to the console about the applied environment mode.


# Initialize settings with default mode, can be overridden by main.py later.
set_env_mode(ENV_MODE) # Calls `set_env_mode` with the default `ENV_MODE` ('test') upon import. This can be overridden later by `main.py`.

# --- Data Preparation Parameters ---
# Defines parameters specifically for the data preparation phase.
MISSING_VALUE_DROP_THRESHOLD = 0.7 # Threshold for dropping columns with a high percentage of missing values (70%).
HIGH_CARDINALITY_THRESHOLD = 50 # Threshold for identifying high-cardinality categorical features (more than 50 unique values).
TARGET_SAMPLES_PER_MINORITY_CLASS = 100000 # The target number of samples for each minority class after oversampling (e.g., using SMOTE).
SMOTE_MIN_SAMPLES_THRESHOLD = 2 # Minimum number of samples required for a class to be considered for SMOTE oversampling.

# --- Feature Engineering Parameters ---
# Defines parameters for the feature engineering phase.
FE_MOMENT_FEATURES = [ # List of numerical features for which statistical moments (skewness, kurtosis) will be calculated.
    "Flow Duration",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

FE_POLYNOMIAL_FEATURES = [ # List of numerical features for which polynomial features will be generated.
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
]
FE_POLYNOMIAL_DEGREE = 2 # The degree of polynomial features to generate (e.g., degree 2 will include original features, squared terms, and interaction terms).

# --- Feature Selection Parameters ---
# Defines parameters for the model-based feature selection process.
FEATURE_SELECTION_MODEL_PARAMS = { # Parameters for the RandomForestClassifier used within SelectFromModel for feature selection.
    "n_estimators": 50, # Number of trees in the forest.
    "random_state": SEED, # Random seed for reproducibility.
    "n_jobs": -1, # Uses all available CPU cores for parallel processing.
    "class_weight": "balanced", # Adjusts weights inversely proportional to class frequencies, useful for imbalanced data.
    "max_depth": 10, # Maximum depth of the trees.
}
FEATURE_SELECTION_THRESHOLD = "0.005*mean" # The threshold for feature importance. Features with importance values below this threshold will be removed. "mean" refers to the mean importance of all features.

# --- Model Specifics (Dynamically configured based on ENV_MODE) ---
# Defines the machine learning classifiers to be trained, along with their base parameters and hyperparameter tuning ranges.
# These settings are dynamically updated based on the `CURRENT_SETTINGS` (derived from ENV_MODE).
CLASSIFIERS_TO_TRAIN = { # Dictionary holding configurations for different classifier models.
    "RandomForest": { # Configuration for the RandomForestClassifier.
        "model": "RandomForestClassifier", # String name of the model class.
        "apply_calibration": True, # Whether to apply probability calibration using CalibratedClassifierCV.
        "params": { # Base parameters for the RandomForestClassifier.
            "n_estimators": CURRENT_SETTINGS["RF_N_ESTIMATORS"], # Number of trees, from current environment settings.
            "random_state": SEED, # Random seed for reproducibility.
            "n_jobs": -1, # Uses all available CPU cores.
            "class_weight": "balanced", # Adjusts class weights.
        },
        "tune_params": { # Hyperparameter tuning grid/distribution for RandomizedSearchCV.
            "calibrator__estimator__n_estimators": [ # Tuning parameter for the number of estimators, accessed via pipeline step 'calibrator' and its 'estimator'.
                CURRENT_SETTINGS["RF_N_ESTIMATORS"]
            ],
            "calibrator__estimator__max_depth": CURRENT_SETTINGS["RF_MAX_DEPTH_TUNE"], # Tuning parameter for max_depth.
            "calibrator__estimator__min_samples_split": [2, 5, 10], # Tuning parameter for min_samples_split.
        },
        "n_iter_search": CURRENT_SETTINGS["N_ITER_SEARCH_MODELS"], # Number of iterations for RandomizedSearchCV, from current environment settings.
    },
    "XGBoost": { # Configuration for the XGBoost Classifier.
        "model": "XGBClassifier", # String name of the model class.
        "apply_calibration": False, # Whether to apply probability calibration (XGBoost typically has good inherent calibration).
        "params": { # Base parameters for the XGBClassifier.
            "objective": "multi:softprob", # Objective function for multi-class classification, outputs probabilities.
            "eval_metric": "mlogloss", # Evaluation metric for monitoring training.
            "n_estimators": CURRENT_SETTINGS["XGB_N_ESTIMATORS"], # Number of boosting rounds/estimators.
            "random_state": SEED, # Random seed.
            "n_jobs": 1, # Number of parallel jobs. Set to 1 as GPU acceleration often handles parallelism.
            "tree_method": "hist", # Histogram-based tree construction, efficient for large datasets.
            "device": "cuda", # Specifies to use CUDA-enabled GPU if available.
            "grow_policy": "depthwise", # Strategy for growing trees.
            "max_bin": 256, # Maximum number of bins for histogram creation.
            "learning_rate": 0.1, # Step size shrinkage to prevent overfitting.
        },
        "tune_params": { # Hyperparameter tuning grid/distribution for RandomizedSearchCV.
            "classifier__n_estimators": [CURRENT_SETTINGS["XGB_N_ESTIMATORS"]], # Tuning parameter for n_estimators.
            "classifier__max_depth": CURRENT_SETTINGS["XGB_MAX_DEPTH_TUNE"], # Tuning parameter for max_depth.
            "classifier__learning_rate": [0.05, 0.1, 0.2], # Tuning parameter for learning_rate.
            "classifier__subsample": [0.7, 0.85, 1.0], # Tuning parameter for subsample ratio of the training instance.
            "classifier__colsample_bytree": [0.7, 0.85, 1.0], # Tuning parameter for subsample ratio of columns when constructing each tree.
        },
        "n_iter_search": CURRENT_SETTINGS["N_ITER_SEARCH_MODELS"], # Number of iterations for RandomizedSearchCV.
    },
    "LightGBM": { # Configuration for the LightGBM Classifier.
        "model": "LGBMClassifier", # String name of the model class.
        "apply_calibration": True, # Whether to apply probability calibration.
        "params": { # Base parameters for the LGBMClassifier.
            "objective": "multiclass", # Objective function for multi-class classification.
            "num_class": 12,  # Will be overridden by actual unique classes in modeling.py # Placeholder for number of classes; will be dynamically set.
            "n_estimators": CURRENT_SETTINGS["LGBM_N_ESTIMATORS"], # Number of boosting rounds/estimators.
            "random_state": SEED, # Random seed.
            "n_jobs": -1, # Uses all available CPU cores.
            "boosting_type": "gbdt", # Traditional Gradient Boosting Decision Tree.
            "class_weight": "balanced", # Adjusts class weights.
        },
        "tune_params": { # Hyperparameter tuning grid/distribution for RandomizedSearchCV.
            "calibrator__estimator__n_estimators": [ # Tuning parameter for n_estimators, accessed via pipeline step 'calibrator' and its 'estimator'.
                CURRENT_SETTINGS["LGBM_N_ESTIMATORS"]
            ],
            "calibrator__estimator__max_depth": CURRENT_SETTINGS["LGBM_MAX_DEPTH_TUNE"], # Tuning parameter for max_depth.
            "calibrator__estimator__learning_rate": [0.05, 0.1, 0.2], # Tuning parameter for learning_rate.
            "calibrator__estimator__subsample": [0.7, 0.85, 1.0], # Tuning parameter for subsample ratio.
            "calibrator__estimator__colsample_bytree": [0.7, 0.85, 1.0], # Tuning parameter for subsample ratio of columns.
        },
        "n_iter_search": CURRENT_SETTINGS["N_ITER_SEARCH_MODELS"], # Number of iterations for RandomizedSearchCV.
    },
}

# --- Dynamic Parameters (Accessed by other modules) ---
# These variables directly expose the currently active settings, making them easy to access from other modules.
TEST_MODE = CURRENT_SETTINGS["TEST_MODE_ACTIVE"] # Boolean indicating if test mode is active, reflecting the current environment settings.
TEST_MODE_TRAIN_SAMPLE_RATIO = CURRENT_SETTINGS["TEST_MODE_TRAIN_SAMPLE_RATIO"] # Training data sample ratio for test mode, reflecting current environment settings.
CV_FOLDS = CURRENT_SETTINGS["CV_FOLDS"] # Number of cross-validation folds, reflecting current environment settings.

# --- Logging Configuration (Dynamically set based on mode) ---
# Defines logging-specific parameters that can change based on the environment mode.
LOG_FILE = os.path.join(LOGS_DIR, "project_execution.log") # Full path for the project's log file.
LOG_LEVEL = CURRENT_SETTINGS["LOG_LEVEL"]  # Set overall log level # The overall logging level (e.g., 'INFO', 'DEBUG'), reflecting current environment settings.