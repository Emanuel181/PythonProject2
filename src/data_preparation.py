# src/data_preparation.py

import os # Imports the 'os' module for interacting with the operating system, like creating directories or managing file paths.
from typing import Any, List, Optional, Tuple # Imports specific type hints for better code readability and maintainability.

import joblib # Imports joblib for efficient serialization and deserialization of Python objects (e.g., trained models, transformers).
import numpy as np # Imports NumPy for numerical operations, especially array manipulations.
import pandas as pd # Imports Pandas for data manipulation and analysis using DataFrames.
from imblearn.combine import SMOTEENN, SMOTETomek # Imports specific combination over- and under-sampling techniques from imblearn.
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE # Imports various oversampling techniques from imblearn for handling class imbalance.
from sklearn.ensemble import IsolationForest, RandomForestClassifier # Imports IsolationForest for outlier detection and RandomForestClassifier for feature selection.
from sklearn.experimental import enable_iterative_imputer # Enables experimental features in scikit-learn, specifically the IterativeImputer.
from sklearn.feature_selection import SelectFromModel # Imports SelectFromModel for model-based feature selection.
from sklearn.impute import IterativeImputer, SimpleImputer # Imports IterativeImputer for sophisticated missing value imputation and SimpleImputer for basic imputation.
from sklearn.model_selection import train_test_split # Imports train_test_split for splitting data into training and testing sets.
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler # Imports LabelEncoder for encoding categorical labels, PolynomialFeatures for creating polynomial interaction features, and StandardScaler for feature scaling.

import src.config as config  # Import config to access dynamic settings # Imports the project's configuration file to access various settings and parameters.
from src.utils import ( # Imports utility functions from the project's utils module.
    load_dataframe, # Function to load a DataFrame from a CSV file.
    plot_and_save, # Function to plot and save figures.
    save_dataframe, # Function to save a DataFrame to a CSV file.
    save_object, # Function to save a Python object using joblib.
    setup_logging, # Function to set up the logging configuration.
)

logger = setup_logging() # Initializes the logger for this module, inheriting the configuration set up in `main.py`.


def data_preparation_phase( # Defines the main data preparation function.
    df_raw: pd.DataFrame, # Accepts the raw DataFrame as input.
) -> Tuple[ # Specifies the return type as a tuple of optional DataFrames, Series, and a NumPy array.
    Optional[pd.DataFrame], # X_train
    Optional[pd.DataFrame], # X_test
    Optional[pd.Series], # y_train
    Optional[pd.Series], # y_test
    Optional[np.ndarray], # label_encoder_classes
]:

    """
    Performs comprehensive data cleaning, preprocessing, and feature engineering,
    and prepares the data for machine learning modeling. This includes:
    - Handling missing values and infinities.
    - Dropping duplicate rows.
    - **Advanced Feature Engineering**: Creates new features like statistical moments (skewness, kurtosis)
      and polynomial features.
    - Removing constant and high-cardinality features.
    - Outlier detection and capping.
    - Encoding categorical target variable.
    - **Sophisticated Feature Selection**: Sets up a model-based feature selector for use in the ML pipeline.
    - Handling class imbalance using advanced oversampling/undersampling techniques.
    - Splitting data into training and testing sets.
    - Applies a sampling strategy for 'TEST_MODE' to enable faster iteration during development.

    Args:
        df_raw (pd.DataFrame): The raw DataFrame from the data understanding phase.
                               This DataFrame is expected to have already undergone basic
                               column name cleaning and dtype optimization.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[np.ndarray]]:
        X_train, X_test, y_train, y_test (processed for modeling, unscaled, features selected if applicable),
        and the classes from LabelEncoder.
        Returns None for any component if data preparation fails at a critical step.
    """
    logger.info("--- CRISP-DM Phase 3: Data Preparation ---") # Logs the start of the Data Preparation phase.

    if df_raw is None or df_raw.empty: # Checks if the input DataFrame is None or empty.
        logger.error("No data provided for Data Preparation. Aborting.") # Logs an error if no data is provided.
        return None, None, None, None, None # Returns None for all outputs, indicating failure.

    df = df_raw.copy() # Creates a copy of the raw DataFrame to avoid modifying the original data.

    # --- 1. Drop Duplicate Rows ---
    initial_rows = len(df) # Stores the initial number of rows before dropping duplicates.
    df.drop_duplicates(inplace=True) # Removes duplicate rows directly from the DataFrame.
    rows_dropped_duplicates = initial_rows - len(df) # Calculates the number of rows dropped.
    logger.info(f"Dropped {rows_dropped_duplicates} duplicated rows.") # Logs the number of duplicate rows dropped.

    # --- 2. Handle Missing Values and Infinities (Re-check and Impute) ---
    logger.info("Handling missing values and infinities...") # Logs the start of missing value and infinity handling.
    df.replace([np.inf, -np.inf], np.nan, inplace=True) # Replaces all infinite values (positive and negative) with NaN.

    object_or_category_cols = df.select_dtypes( # Selects columns with 'object' or 'category' data types.
        include=["object", "category"]
    ).columns.tolist() # Converts the selected column index to a list.
    if config.TARGET_COLUMN in object_or_category_cols: # Checks if the target column is in the list of object/category columns.
        object_or_category_cols.remove(config.TARGET_COLUMN) # Removes the target column from the list if it's present.

    converted_cols = [] # Initializes an empty list to store names of columns successfully converted to numeric.
    for col in object_or_category_cols: # Iterates through each object or category column.
        original_dtype = df[col].dtype # Stores the original data type of the column.
        df[col] = pd.to_numeric(df[col], errors="coerce") # Attempts to convert the column to numeric, coercing errors to NaN.
        if original_dtype != df[col].dtype: # Checks if the data type of the column has changed (indicating successful conversion).
            converted_cols.append(col) # Adds the column name to the list of converted columns.
            logger.info(
                f"Attempted conversion of '{col}' from {original_dtype} to numeric. New dtype: {df[col].dtype}."
            ) # Logs the attempted conversion and new data type.

    if converted_cols: # Checks if any columns were converted.
        logger.info(
            f"Successfully attempted conversion for {len(converted_cols)} object/category columns to numeric: {converted_cols}"
        ) # Logs the successful conversion of object/category columns.
    else:
        logger.info(
            "No object/category columns identified for numeric conversion or already numeric."
        ) # Logs if no object/category columns needed conversion.

    missing_cols = df.isnull().sum() # Calculates the count of null values for each column.
    missing_cols = missing_cols[missing_cols > 0].index.tolist() # Filters to get only columns with missing values and converts their names to a list.

    if missing_cols: # Checks if there are any columns with missing values.
        logger.info(
            f"Columns with missing values (including those from conversion): {missing_cols}"
        ) # Logs the list of columns that still have missing values.

        cols_to_drop_high_missing = [] # Initializes an empty list to store columns to be dropped due to high missing values.
        for col in missing_cols: # Iterates through columns with missing values.
            if df[col].isnull().sum() / len(df) > config.MISSING_VALUE_DROP_THRESHOLD: # Checks if the percentage of missing values exceeds the defined threshold.
                cols_to_drop_high_missing.append(col) # Adds the column to the list if it meets the drop criteria.

        if cols_to_drop_high_missing: # Checks if there are columns to drop.
            df.drop(columns=cols_to_drop_high_missing, inplace=True) # Drops the identified columns from the DataFrame.
            logger.warning(
                f"Dropped columns due to high percentage of missing values (>{config.MISSING_VALUE_DROP_THRESHOLD * 100}%): {cols_to_drop_high_missing}"
            ) # Logs a warning about the dropped columns.
            missing_cols = [ # Updates the list of missing columns by removing those that were dropped.
                c for c in missing_cols if c not in cols_to_drop_high_missing
            ]

        if missing_cols: # After dropping high-missing columns, checks if any missing values remain.
            numerical_cols_to_impute = [ # Identifies numerical columns that still have missing values and need imputation.
                col
                for col in missing_cols
                if df[col].dtype in ["float32", "float64", "int16", "int32", "int64"]
            ]
            if numerical_cols_to_impute: # Checks if there are numerical columns to impute.
                logger.info(
                    f"Imputing numerical missing values using IterativeImputer for: {numerical_cols_to_impute}"
                ) # Logs the start of numerical imputation.

                for col in numerical_cols_to_impute: # Ensures numerical columns are float64 before imputation.
                    if df[col].dtype not in ["float32", "float64"]:
                        df[col] = df[col].astype(np.float64)

                imputer = IterativeImputer(random_state=config.SEED, max_iter=10) # Initializes IterativeImputer with a random state and max iterations.
                df[numerical_cols_to_impute] = imputer.fit_transform( # Fits the imputer and transforms the specified numerical columns.
                    df[numerical_cols_to_impute].copy() # Uses a copy to avoid modifying the original DataFrame in place during transformation.
                )

                for col in numerical_cols_to_impute: # Iterates through imputed numerical columns.
                    # Attempt to downcast imputed floats back to integers if appropriate
                    # Check if original column was integer type (from raw_df) AND all non-null current values are integers
                    if ( # Checks if the original column's dtype was integer or convertible to integer, and if all current non-null values are integers.
                        df_raw[col].dtype in ["int16", "int32", "int64"] # Check if original was int.
                        or df_raw[col].dtype == np.dtype("object") # Or if original was object and convertible to int.
                        and pd.api.types.is_integer_dtype( # Check if object column values are integer-like after coercing to numeric.
                            pd.to_numeric(df_raw[col], errors="coerce").dropna()
                        )
                    ) and (df[col].dropna() % 1 == 0).all(): # Check if current non-null values are whole numbers.
                        df[col] = pd.to_numeric( # Downcasts the column to integer type.
                            df[col], downcast="integer", errors="coerce"
                        )
                        logger.debug(
                            f"Downcasted imputed column '{col}' to integer type."
                        ) # Logs the downcasting.

                save_object(imputer, config.MODELS_DIR, "numerical_imputer.pkl") # Saves the trained numerical imputer object.

            categorical_cols_to_impute = [ # Identifies categorical columns that still have missing values.
                col
                for col in missing_cols
                if col not in numerical_cols_to_impute and col != config.TARGET_COLUMN
            ]
            if categorical_cols_to_impute: # Checks if there are categorical columns to impute.
                logger.info(
                    f"Imputing categorical missing values using SimpleImputer (most frequent) for: {categorical_cols_to_impute}"
                ) # Logs the start of categorical imputation.
                mode_imputer = SimpleImputer(strategy="most_frequent") # Initializes SimpleImputer with a 'most_frequent' strategy.
                df[categorical_cols_to_impute] = mode_imputer.fit_transform( # Fits the imputer and transforms the specified categorical columns.
                    df[categorical_cols_to_impute]
                )
                save_object(mode_imputer, config.MODELS_DIR, "categorical_imputer.pkl") # Saves the trained categorical imputer object.

        missing_after_imputation = df.isnull().sum().sum() # Calculates the total number of missing values after all imputation attempts.
        if missing_after_imputation == 0: # Checks if all missing values have been handled.
            logger.info("All missing values successfully handled after imputation.") # Logs success if no missing values remain.
        else:
            logger.error(
                f"WARNING: Still {missing_after_imputation} missing values after imputation. Review strategy."
            ) # Logs a warning if missing values still exist.
    else:
        logger.info("No missing values to handle.") # Logs if there were no missing values to begin with.

    # --- 3. Advanced Feature Engineering ---
    logger.info("Performing advanced feature engineering...") # Logs the start of advanced feature engineering.

    # A. Statistical Moments (Skewness, Kurtosis)
    moment_features = [ # Selects numerical features for moment calculation based on configuration.
        col
        for col in config.FE_MOMENT_FEATURES
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    for col in moment_features: # Iterates through selected features for moment calculation.
        if df[col].count() > 2: # Checks if there are enough non-null values to calculate skewness and kurtosis.
            df[f"{col}_Skew"] = df[col].skew() # Calculates and adds a new column for skewness.
            df[f"{col}_Kurtosis"] = df[col].kurtosis() # Calculates and adds a new column for kurtosis.
        else:
            df[f"{col}_Skew"] = 0.0 # If not enough data, sets skewness to 0.0.
            df[f"{col}_Kurtosis"] = 0.0 # If not enough data, sets kurtosis to 0.0.
            if col not in df.columns: # Checks if the column was originally missing from the DataFrame.
                logger.debug(
                    f"Feature '{col}' not found for moment calculation. Adding empty skew/kurtosis columns."
                ) # Logs if the feature was not found.
            else:
                logger.debug(f"Not enough data for skew/kurtosis in {col}. Set to 0.") # Logs if there was insufficient data for calculation.

    # B. Domain-Specific Ratios
    if ( # Checks if necessary columns are present for calculating ratio features.
        "Total Length of Fwd Packets" in df.columns
        and "Total Fwd Packets" in df.columns
        and "Total Backward Packets" in df.columns
    ):
        df["Flow_Bytes_Per_Packet"] = df["Total Length of Fwd Packets"] / ( # Calculates Flow_Bytes_Per_Packet ratio.
            df["Total Fwd Packets"] + df["Total Backward Packets"] + 1e-6 # Adds a small epsilon to denominator to prevent division by zero.
        )
        df["Fwd_Bwd_Packet_Ratio"] = df["Total Fwd Packets"] / ( # Calculates Fwd_Bwd_Packet_Ratio.
            df["Total Backward Packets"] + 1e-6 # Adds a small epsilon to denominator.
        )
    else:
        logger.warning(
            "Could not create 'Flow_Bytes_Per_Packet' or 'Fwd_Bwd_Packet_Ratio' due to missing columns."
        ) # Logs a warning if columns for ratio calculation are missing.

    # C. Polynomial Features (Interaction terms, e.g., A*B, A^2)
    polynomial_features_cols = [ # Selects numerical features for polynomial transformation.
        col
        for col in config.FE_POLYNOMIAL_FEATURES
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(polynomial_features_cols) > 1 and config.FE_POLYNOMIAL_DEGREE > 1: # Checks if there are enough features and if polynomial degree is greater than 1.
        logger.info(
            f"Generating polynomial features (degree {config.FE_POLYNOMIAL_DEGREE}) for: {polynomial_features_cols}"
        ) # Logs the start of polynomial feature generation.
        poly = PolynomialFeatures( # Initializes PolynomialFeatures transformer.
            degree=config.FE_POLYNOMIAL_DEGREE, include_bias=False # Sets the polynomial degree and excludes bias term.
        )
        poly_data = df[polynomial_features_cols].astype(np.float32) # Extracts and converts relevant columns to float32 for processing.
        poly_transformed = poly.fit_transform(poly_data) # Fits the transformer and transforms the data.

        poly_feature_names = poly.get_feature_names_out(polynomial_features_cols) # Gets the names of the newly generated polynomial features.

        new_poly_features_only = [ # Filters for newly generated polynomial features that are not already in the DataFrame.
            name
            for name in poly_feature_names
            if name not in df.columns and " " not in name and "^1" not in name
        ]

        df_poly_features = pd.DataFrame( # Creates a DataFrame from the transformed polynomial features.
            poly_transformed, columns=poly_feature_names, index=df.index
        )

        for col_name in new_poly_features_only: # Iterates through the new polynomial feature names.
            if col_name in df_poly_features.columns: # Ensures the column exists in the transformed features DataFrame.
                df[col_name] = df_poly_features[col_name] # Adds the new polynomial feature columns to the main DataFrame.

        logger.info(f"Added {len(new_poly_features_only)} new polynomial features.") # Logs the number of new polynomial features added.
    elif len(polynomial_features_cols) <= 1: # Checks if there are not enough numerical columns for polynomial features.
        logger.warning(
            "Not enough numerical columns to generate polynomial features. Skipping."
        ) # Logs a warning if polynomial features cannot be generated.
    else:
        logger.info(
            "Polynomial feature degree not greater than 1. Skipping polynomial feature generation."
        ) # Logs if the polynomial degree is not set to generate new features.

    logger.info(
        "Completed advanced feature engineering (statistical moments, ratios, and polynomial features)."
    ) # Logs the completion of advanced feature engineering.

    # --- 4. Remove Columns with Zero Variance (Constant Columns) ---
    constant_cols = [ # Identifies columns that have only one unique value (constant columns), excluding the target column.
        col
        for col in df.columns
        if df[col].nunique(dropna=False) == 1 and col != config.TARGET_COLUMN
    ]
    if constant_cols: # Checks if any constant columns were found.
        df.drop(columns=constant_cols, inplace=True) # Drops the constant columns from the DataFrame.
        logger.info(f"Dropped constant columns: {constant_cols}") # Logs the names of the dropped constant columns.
    else:
        logger.info("No constant columns to drop.") # Logs if no constant columns were found.

    # --- 5. Handle High Cardinality Features (if any) ---
    high_cardinality_cols = [ # Identifies categorical columns with cardinality exceeding the defined threshold, excluding the target column.
        col
        for col in df.select_dtypes(include=["object", "category"]).columns
        if col != config.TARGET_COLUMN
        and df[col].nunique() > config.HIGH_CARDINALITY_THRESHOLD
    ]
    if high_cardinality_cols: # Checks if any high cardinality columns were found.
        logger.warning(
            f"High cardinality columns detected: {high_cardinality_cols}. "
            "These are typically dropped for tree-based models or require advanced encoding. Dropping for now."
        ) # Logs a warning about high cardinality columns and the decision to drop them.
        df.drop(columns=high_cardinality_cols, inplace=True) # Drops the high cardinality columns.
    else:
        logger.info("No high cardinality categorical columns detected to drop.") # Logs if no high cardinality columns were found.

    # --- 6. Outlier Detection and Treatment (Capping) ---
    logger.info("Detecting and treating outliers using IsolationForest...") # Logs the start of outlier detection and treatment.
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist() # Selects all numerical columns.
    if config.TARGET_COLUMN in numerical_cols: # Checks if the target column is in the numerical columns list.
        numerical_cols.remove(config.TARGET_COLUMN) # Removes the target column if it's numerical.

    if numerical_cols: # Checks if there are numerical columns to process.
        # FIX: Removed n_jobs=-1 from IsolationForest initialization
        iso_forest = IsolationForest(contamination=0.01, random_state=config.SEED) # Initializes IsolationForest with a contamination factor and random state.
        if len(df) > 500000: # If the DataFrame is very large, sample for fitting IsolationForest to improve performance.
            sample_df = df[numerical_cols].sample(n=500000, random_state=config.SEED) # Takes a sample of 500,000 rows for fitting.
            iso_forest.fit(sample_df) # Fits IsolationForest on the sampled numerical data.
        else:
            iso_forest.fit(df[numerical_cols]) # Fits IsolationForest on the full numerical data if it's not too large.

        outliers = iso_forest.predict(df[numerical_cols]) # Predicts outliers (-1) or inliers (1) for all numerical data.

        outlier_count = np.sum(outliers == -1) # Counts the number of detected outliers.
        logger.info(
            f"Detected {outlier_count} outliers (approx. {outlier_count / len(df) * 100:.2f}% of data)."
        ) # Logs the number and percentage of detected outliers.

        for col in numerical_cols: # Iterates through each numerical column to cap outliers.
            if df[col].dtype != np.float32: # Ensures the column is float32 for consistent processing.
                df[col] = df[col].astype(np.float32)

            Q1 = df[col].quantile(0.01) # Calculates the 1st percentile.
            Q99 = df[col].quantile(0.99) # Calculates the 99th percentile.
            df[col] = np.where(df[col] < Q1, Q1, df[col]) # Caps values below Q1 to Q1.
            df[col] = np.where(df[col] > Q99, Q99, df[col]) # Caps values above Q99 to Q99.
        logger.info(
            "Outliers capped at 1st and 99th percentiles for numerical features."
        ) # Logs that outliers have been capped.
        save_object( # Saves the trained IsolationForest model.
            iso_forest, config.MODELS_DIR, "isolation_forest_outlier_detector.pkl"
        )
    else:
        logger.info("No numerical columns found for outlier detection.") # Logs if no numerical columns are available for outlier detection.

    # --- 7. Encode Target Variable ---
    if config.TARGET_COLUMN not in df.columns: # Checks if the target column exists in the DataFrame.
        logger.error(
            f"Target column '{config.TARGET_COLUMN}' not found. Aborting data preparation."
        ) # Logs an error if the target column is missing.
        return None, None, None, None, None # Returns None for all outputs, indicating failure.

    le = LabelEncoder() # Initializes LabelEncoder.
    df["Label_Encoded"] = le.fit_transform(df[config.TARGET_COLUMN]) # Fits the encoder to the target column and transforms it to numerical labels.
    logger.info(f"Encoded '{config.TARGET_COLUMN}' column.") # Logs that the target column has been encoded.
    for label, encoded_val in zip(le.classes_, le.transform(le.classes_)): # Iterates through original and encoded labels.
        logger.info(f"  {label}: {encoded_val}") # Logs the mapping of original to encoded labels.

    label_encoder_classes = le.classes_ # Stores the classes learned by the LabelEncoder.
    save_object(le, config.MODELS_DIR, "label_encoder.pkl") # Saves the trained LabelEncoder object.
    save_dataframe( # Saves a DataFrame containing the original label classes.
        pd.DataFrame(label_encoder_classes, columns=["Original_Label"]),
        config.PROCESSED_DATA_DIR,
        "label_encoder_classes.csv",
    )
    logger.info(
        f"Saved label encoder and classes to {config.MODELS_DIR} and {config.PROCESSED_DATA_DIR}."
    ) # Logs where the label encoder and classes were saved.

    df_processed = df.drop(config.TARGET_COLUMN, axis=1) # Creates a new DataFrame `df_processed` by dropping the original target column.

    # --- 8. Separate Features (X) and Target (y) ---
    X = df_processed.drop("Label_Encoded", axis=1) # Separates features (X) by dropping the encoded target column.
    y = df_processed["Label_Encoded"] # Extracts the encoded target variable (y).
    logger.info(
        f"Separated features (X) with shape {X.shape} and target (y) with shape {y.shape}."
    ) # Logs the shapes of the separated features and target.

    # --- 9. Feature Selection (Sophisticated - NEW) ---
    logger.info("Performing model-based feature selection setup...") # Logs the start of feature selection setup.
    fs_estimator = RandomForestClassifier(**config.FEATURE_SELECTION_MODEL_PARAMS) # Initializes a RandomForestClassifier as the estimator for feature selection, using parameters from config.
    selector_for_pipeline = SelectFromModel( # Initializes SelectFromModel, which will select features based on the importance of `fs_estimator`.
        fs_estimator, threshold=config.FEATURE_SELECTION_THRESHOLD, prefit=False # Sets the importance threshold and `prefit=False` for use in a pipeline.
    )

    try: # Begins a try block to handle potential errors during dummy fitting for feature selection logging.
        if len(X) > 500000: # If the dataset is very large, sample for dummy fitting.
            sample_X_fs = X.sample(n=500000, random_state=config.SEED) # Takes a sample of X.
            sample_y_fs = y.loc[sample_X_fs.index] # Gets corresponding y labels for the sample.
            dummy_fs_estimator_for_logging = RandomForestClassifier( # Initializes a dummy estimator for logging selected features.
                **config.FEATURE_SELECTION_MODEL_PARAMS
            )
            dummy_fs_estimator_for_logging.fit(sample_X_fs, sample_y_fs) # Fits the dummy estimator on the sample.
            dummy_selector_for_logging = SelectFromModel( # Initializes a dummy selector, `prefit=True` because the estimator is already fitted.
                dummy_fs_estimator_for_logging,
                threshold=config.FEATURE_SELECTION_THRESHOLD,
                prefit=True,
            )
        else: # If the dataset is not too large, fit on the full data.
            dummy_fs_estimator_for_logging = RandomForestClassifier( # Initializes a dummy estimator.
                **config.FEATURE_SELECTION_MODEL_PARAMS
            )
            dummy_fs_estimator_for_logging.fit(X, y) # Fits the dummy estimator on the full data.
            dummy_selector_for_logging = SelectFromModel( # Initializes a dummy selector.
                dummy_fs_estimator_for_logging,
                threshold=config.FEATURE_SELECTION_THRESHOLD,
                prefit=True,
            )

        selected_features_mask = dummy_selector_for_logging.get_support() # Gets a boolean mask indicating selected features.
        selected_feature_names = X.columns[selected_features_mask].tolist() # Extracts the names of the selected features.

        logger.info(
            f"Selected {len(selected_feature_names)} features out of {X.shape[1]}: {selected_feature_names[:10]}... (and {len(selected_feature_names) - 10} more)"
        ) # Logs the number of selected features and a preview of their names.
        save_object(selector_for_pipeline, config.MODELS_DIR, "feature_selector.pkl") # Saves the unfitted feature selector for use in the modeling pipeline.
        logger.info(
            f"Feature selector configured and saved (unfitted instance). Actual feature transformation will occur within the modeling pipeline."
        ) # Logs that the feature selector has been saved.

    except Exception as e: # Catches any exceptions during the feature selection setup.
        logger.error(
            f"Error during feature selection setup (dummy fit): {e}. Skipping feature selection for pipelines.",
            exc_info=True,
        ) # Logs the error with traceback.
        save_object(None, config.MODELS_DIR, "feature_selector.pkl") # Saves `None` as the feature selector if an error occurs.

    # --- 10. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split( # Splits the data into training and testing sets.
        X, y, test_size=0.3, random_state=config.SEED, stratify=y # Uses 30% for test, a fixed random state, and stratifies by y to preserve class distribution.
    )
    logger.info(
        f"Initial data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets."
    ) # Logs the sizes of the training and testing sets.
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}") # Logs the shapes of X_train and y_train.
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}") # Logs the shapes of X_test and y_test.

    # --- 11. Class Imbalance Handling on Training Data (SMOTE/Advanced Resampling) ---
    logger.info(
        "Handling class imbalance using advanced resampling on training data..."
    ) # Logs the start of class imbalance handling.
    logger.info(
        f"Original training label distribution:\n{y_train.value_counts().to_string()}"
    ) # Logs the original distribution of labels in the training set.

    value_counts_initial_train = y_train.value_counts() # Gets the counts of each class in the training set.
    majority_class_label_encoded = value_counts_initial_train.idxmax() # Identifies the encoded label of the majority class.

    # Define sampling_strategy_dict here, so it's always available within this scope
    sampling_strategy_dict = {} # Initializes an empty dictionary for SMOTE's sampling strategy.

    minority_classes_eligible = value_counts_initial_train[ # Identifies minority classes that are eligible for SMOTE (not majority and meet minimum sample threshold).
        (
            value_counts_initial_train # Checks if class count is less than majority class count.
            < value_counts_initial_train[majority_class_label_encoded]
        )
        & (value_counts_initial_train >= config.SMOTE_MIN_SAMPLES_THRESHOLD) # Checks if class count is greater than or equal to the minimum SMOTE threshold.
    ].index.tolist() # Converts eligible minority class indices to a list.

    if minority_classes_eligible: # Checks if there are any eligible minority classes for oversampling.
        # Populate sampling_strategy_dict only if there are eligible minority classes
        sampling_strategy_dict = { # Populates the sampling strategy dictionary.
            cls: max(count, config.TARGET_SAMPLES_PER_MINORITY_CLASS) # For each eligible minority class, set its target samples to at least TARGET_SAMPLES_PER_MINORITY_CLASS or its current count if larger.
            for cls, count in value_counts_initial_train.items()
            if cls in minority_classes_eligible
        }

        # Choose the resampler based on your preference; BorderlineSMOTE is a good general choice.
        # resampler = SMOTE(random_state=config.SEED, sampling_strategy=sampling_strategy_dict)
        # resampler = ADASYN(random_state=config.SEED, sampling_strategy=sampling_strategy_dict)
        # FIX: Removed n_jobs=-1 from BorderlineSMOTE initialization
        resampler = BorderlineSMOTE( # Initializes BorderlineSMOTE for oversampling minority classes.
            random_state=config.SEED, # Sets random state for reproducibility.
            sampling_strategy=sampling_strategy_dict, # Applies the defined sampling strategy.
            kind="borderline-1", # Specifies the kind of borderline samples to generate.
        )  # Removed n_jobs=-1 # Comment indicating removal of n_jobs=-1 to avoid potential issues in some environments.

        logger.info(f"Using resampler: {resampler.__class__.__name__}") # Logs the name of the resampler being used.

        X_train_resampling = X_train.astype(np.float32) # Converts X_train to float32 before resampling to ensure compatibility.

        X_train_resampled, y_train_resampled = resampler.fit_resample( # Performs resampling on the training data.
            X_train_resampling, y_train
        )

        logger.info(f"Training data resampled. New shape: {X_train_resampled.shape}") # Logs the new shape of the resampled training data.
        logger.info(
            f"New training label distribution:\n{y_train_resampled.value_counts().to_string()}"
        ) # Logs the new label distribution after resampling.
        save_object( # Saves the trained resampler object.
            resampler,
            config.MODELS_DIR,
            f"{resampler.__class__.__name__.lower()}_transformer.pkl",
        )

        X_train, y_train = X_train_resampled, y_train_resampled # Updates X_train and y_train with the resampled data.
    else:
        # Now, sampling_strategy_dict is guaranteed to be defined (as an empty dict) if no eligible minorities,
        # so this logic won't raise an UnresolvedReferenceError.
        skipped_classes_for_smote_reason = [] # Initializes a list to store reasons for skipping SMOTE.
        for label, count in value_counts_initial_train.items(): # Iterates through each class in the original training label distribution.
            if ( # Checks if a class was not targeted by the sampling strategy.
                label not in sampling_strategy_dict
            ):  # Check against potentially empty sampling_strategy_dict
                if label == majority_class_label_encoded: # If it's the majority class.
                    skipped_classes_for_smote_reason.append(
                        f"Class {label} (majority class)."
                    ) # Adds reason for skipping majority class.
                elif count < config.SMOTE_MIN_SAMPLES_THRESHOLD: # If the class count is below the minimum threshold for SMOTE.
                    skipped_classes_for_smote_reason.append(
                        f"Class {label} (fewer than {config.SMOTE_MIN_SAMPLES_THRESHOLD} samples)."
                    ) # Adds reason for skipping due to insufficient samples.
            elif sampling_strategy_dict[label] <= count: # If the target samples for a class are already met or exceeded.
                skipped_classes_for_smote_reason.append(
                    f"Class {label} (already met or exceeded target samples)."
                ) # Adds reason for skipping because target was already reached.

        logger.warning(
            "SMOTE skipped for some or all minority classes. Reasons: "
            + ", ".join(skipped_classes_for_smote_reason)
            + " Overall insufficient eligible classes for oversampling."
        ) # Logs a warning if SMOTE was skipped for certain classes and the reasons.

    # --- Test Mode Sampling (Dynamically controlled by config.TEST_MODE_TRAIN_SAMPLE_RATIO) ---
    if config.TEST_MODE and config.TEST_MODE_TRAIN_SAMPLE_RATIO < 1.0: # Checks if TEST_MODE is active and if a sampling ratio less than 1.0 is specified.
        logger.info(
            f"TEST_MODE is ON: Sampling {config.TEST_MODE_TRAIN_SAMPLE_RATIO * 100:.2f}% of the training data. This reduces 'Volume' for quick tests."
        ) # Logs that test mode sampling is active and its purpose.
        if ( # Checks if stratified sampling is possible (multiple unique classes, sufficient samples per class).
            len(y_train.unique()) > 1 # More than one unique class.
            and (y_train.value_counts() >= 1).all() # At least one sample per class.
            and (len(y_train) * config.TEST_MODE_TRAIN_SAMPLE_RATIO) # Checks if the target sample size is at least the number of unique classes.
            >= len(y_train.unique())
        ):
            _, X_train_sampled, _, y_train_sampled = train_test_split( # Performs stratified sampling on the training data.
                X_train,
                y_train,
                test_size=config.TEST_MODE_TRAIN_SAMPLE_RATIO, # Uses the specified test sample ratio.
                random_state=config.SEED, # Uses a fixed random state.
                stratify=y_train, # Ensures stratification.
            )
            X_train = X_train_sampled # Updates X_train with the sampled data.
            y_train = y_train_sampled # Updates y_train with the sampled data.
            logger.info(
                f"Training data size reduced for TEST_MODE. New X_train shape: {X_train.shape}"
            ) # Logs the new shape of the training data.
            logger.info(
                f"New training label distribution (TEST_MODE):\n{y_train.value_counts().to_string()}"
            ) # Logs the new label distribution in test mode.
        else:
            logger.warning(
                "Cannot perform stratified sample for TEST_MODE with current data distribution. Using non-stratified sampling or skipping sample if too small."
            ) # Logs a warning if stratified sampling is not possible.
            if len(y_train) * config.TEST_MODE_TRAIN_SAMPLE_RATIO > 0: # Checks if non-stratified sampling would result in more than 0 rows.
                X_train = X_train.sample( # Performs non-stratified sampling.
                    frac=config.TEST_MODE_TRAIN_SAMPLE_RATIO, random_state=config.SEED
                )
                y_train = y_train.loc[X_train.index] # Selects corresponding y labels.
                logger.info(
                    f"Training data size reduced for TEST_MODE (non-stratified). New X_train shape: {X_train.shape}"
                ) # Logs the new shape after non-stratified sampling.
                logger.info(
                    f"New training label distribution (TEST_MODE):\n{y_train.value_counts().to_string()}"
                ) # Logs the new label distribution.
            else:
                logger.warning(
                    "Skipping TEST_MODE sampling as the target size is too small (0 rows)."
                ) # Logs a warning if the target sample size is zero.

    # Save processed data.
    X_train = X_train.astype(np.float32) # Converts X_train to float32 before saving.
    X_test = X_test.astype(np.float32) # Converts X_test to float32 before saving.

    save_dataframe(X_train, config.PROCESSED_DATA_DIR, "X_train.csv") # Saves X_train to a CSV file.
    save_dataframe(X_test, config.PROCESSED_DATA_DIR, "X_test.csv") # Saves X_test to a CSV file.
    save_dataframe( # Saves y_train (as a DataFrame) to a CSV file.
        pd.DataFrame(y_train, columns=["Label_Encoded"]),
        config.PROCESSED_DATA_DIR,
        "y_train.csv",
    )
    save_dataframe( # Saves y_test (as a DataFrame) to a CSV file.
        pd.DataFrame(y_test, columns=["Label_Encoded"]),
        config.PROCESSED_DATA_DIR,
        "y_test.csv",
    )

    logger.info("--- Data Preparation Phase Complete ---\n") # Logs the completion of the Data Preparation phase.

    return X_train, X_test, y_train, y_test, label_encoder_classes # Returns the processed data and label encoder classes.


if __name__ == "__main__": # Checks if the script is being run directly.
    # When running standalone, explicitly set the environment mode for testing purposes.
    # Otherwise, it would default to 'test' from config.py's initial load.
    import sys # Imports the 'sys' module to access command-line arguments.

    # Check for --env argument if running standalone
    if "--env" in sys.argv: # Checks if the '--env' argument is provided in the command line.
        env_idx = sys.argv.index("--env") + 1 # Gets the index of the environment mode argument.
        if env_idx < len(sys.argv): # Ensures there is a value after '--env'.
            config.set_env_mode(sys.argv[env_idx]) # Sets the environment mode based on the command-line argument.
            logger = setup_logging( # Re-sets up the logger with the determined log level.
                log_level=config.LOG_LEVEL
            )  # Re-setup logger with determined level

    df_understood = load_dataframe(config.PROCESSED_DATA_DIR, "data_understood.csv") # Loads the DataFrame from the data understanding phase.
    if df_understood is not None: # Checks if the DataFrame was loaded successfully.
        X_train, X_test, y_train, y_test, label_classes = data_preparation_phase( # Calls the data_preparation_phase with the loaded DataFrame.
            df_understood
        )
        if X_train is not None: # Checks if data preparation completed successfully.
            logger.info(
                "Data preparation completed and data split successfully for standalone test."
            ) # Logs successful completion for standalone test.
    else:
        logger.error(
            "Could not load 'data_understood.csv'. Please run data_understanding_phase first."
        ) # Logs an error if the initial DataFrame could not be loaded.