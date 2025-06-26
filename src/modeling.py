# src/modeling.py

import os # Imports the 'os' module for interacting with the operating system, like managing file paths.
from typing import Any, Dict, Optional, Tuple # Imports specific type hints for better code readability and maintainability.

import joblib # Imports joblib for efficient serialization/deserialization of Python objects.
import numpy as np # Imports NumPy for numerical operations.
import pandas as pd # Imports Pandas for data manipulation and analysis.
from lightgbm import LGBMClassifier # Imports the LightGBM classifier.
from sklearn.calibration import CalibratedClassifierCV # Imports CalibratedClassifierCV for probability calibration.
from sklearn.ensemble import RandomForestClassifier, VotingClassifier # Imports RandomForestClassifier and VotingClassifier for ensemble methods.
from sklearn.feature_selection import SelectFromModel # Imports SelectFromModel for model-based feature selection.
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score # Imports specific metrics and `make_scorer` for custom scoring.
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # Imports GridSearchCV and RandomizedSearchCV for hyperparameter optimization.
from sklearn.pipeline import Pipeline # Imports Pipeline for creating a sequence of data processing and modeling steps.
from sklearn.preprocessing import StandardScaler # Imports StandardScaler for feature scaling.
from xgboost import XGBClassifier # Imports the XGBoost classifier.

import src.config as config  # Import config to access dynamic settings # Imports the project's configuration file to access various settings and parameters.
from src.utils import load_dataframe, load_object, save_object, setup_logging # Imports utility functions for loading/saving data/objects and logging setup.

logger = setup_logging() # Initializes the logger for this module, inheriting the configuration set up in `main.py`.



def modeling_phase( # Defines the main modeling phase function.
        X_train: pd.DataFrame, y_train: pd.Series, label_encoder_classes: np.ndarray # Accepts training features, target, and label encoder classes.
) -> Dict[str, Any]: # Returns a dictionary mapping model names to their trained objects (pipelines).
    """
    Trains and tunes multiple machine learning models (RandomForest, XGBoost, LightGBM)
    using either GridSearchCV or RandomizedSearchCV for hyperparameter optimization.
    It integrates preprocessing steps (scaling, feature selection, and optionally calibration)
    into a scikit-learn Pipeline. Finally, an ensemble model (VotingClassifier) is trained.
    This phase addresses the 'Variety' (different model types, pipelines) and
    'Velocity' (GPU acceleration, efficient tuning) aspects of Big Data.

    Args:
        X_train (pd.DataFrame): Training features, expected to be unscaled.
                                This is because scaling is now handled within the pipeline.
        y_train (pd.Series): Training target variable, encoded numerically.
        label_encoder_classes (np.ndarray): Array of original label names from LabelEncoder.classes_.
                                            Used to correctly set num_class for multi-class classifiers.

    Returns:
        Dict[str, Any]: A dictionary where keys are model names (e.g., 'RandomForest', 'XGBoost', 'LightGBM', 'Ensemble')
                        and values are the trained best models (or pipelines).
                        Returns an empty dictionary if training data is not provided or
                        if critical errors occur during model training/tuning.
    """
    logger.info("--- CRISP-DM Phase 4: Modeling ---") # Logs the start of the Modeling phase.

    if X_train is None or y_train is None or X_train.empty or y_train.empty: # Checks if training data is missing or empty.
        logger.error("Training data not provided. Exiting Modeling phase.") # Logs an error.
        return {} # Returns an empty dictionary.

    if label_encoder_classes is None: # Checks if label encoder classes are provided.
        logger.error( # Logs an error if label encoder classes are missing.
            "Label encoder classes not provided. Cannot proceed with modeling. Exiting."
        )
        return {} # Returns an empty dictionary.

    trained_models = {} # Initializes an empty dictionary to store trained models.

    # --- Data Safety Checks Before Modeling ---
    logger.info("Applying data safety checks (astype, fillna) before modeling...") # Logs the start of data safety checks.
    numerical_cols = X_train.select_dtypes(include=np.number).columns # Selects numerical columns in X_train.
    X_train[numerical_cols] = X_train[numerical_cols].replace([np.inf, -np.inf], np.nan) # Replaces infinite values with NaN.
    X_train[numerical_cols] = X_train[numerical_cols].fillna(0).astype(np.float32) # Fills NaNs with 0 and converts to float32.
    logger.info( # Logs the shape of X_train after safety checks.
        f"X_train shape after safety checks and dtype conversion: {X_train.shape}"
    )

    # Load the feature selector object.
    feature_selector_pipeline_step = load_object( # Loads the feature selector object from disk.
        config.MODELS_DIR, "feature_selector.pkl"
    )

    # If feature selector loading failed or it's None, create a passthrough transformer.
    if feature_selector_pipeline_step is None: # Checks if the feature selector was not loaded.
        logger.warning( # Logs a warning that features will be passed through without selection.
            "Feature selector not loaded or found. All features will be passed through."
        )

        class PassthroughTransformer: # Defines a dummy transformer that does nothing.
            def fit(self, X, y=None): # Fit method.
                return self # Returns self.

            def transform(self, X): # Transform method.
                return X # Returns X as is.

            def get_params(self, deep=True): # Get parameters method (required for pipelines).
                return {}

            def _get_tags(self): # Internal method for scikit-learn tags.
                return {"preserves_sample_order": True}

        feature_selector_pipeline_step = PassthroughTransformer() # Assigns the passthrough transformer.

    # Define custom scoring metrics for hyperparameter optimization.
    scorers = { # Defines a dictionary of custom scoring functions for GridSearchCV/RandomizedSearchCV.
        "f1_macro": make_scorer(f1_score, average="macro"), # Macro F1-score.
        "precision_macro": make_scorer( # Macro precision score, with zero_division handled.
            precision_score, average="macro", zero_division=0
        ),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0), # Macro recall score, with zero_division handled.
    }

    # Iterate through the defined classifiers in config.py to train and tune each one.
    for ( # Loops through each classifier defined in `config.CLASSIFIERS_TO_TRAIN`.
            model_name, # The name of the model (e.g., "RandomForest").
            model_info, # A dictionary containing model class string, params, tune_params, etc.
    ) in config.CLASSIFIERS_TO_TRAIN.items():  # Use config.CLASSIFIERS_TO_TRAIN
        logger.info(f"\n--- Training {model_name} ---") # Logs the start of training for the current model.
        model_class_str = model_info["model"] # Gets the string name of the model class (e.g., "RandomForestClassifier").
        current_model_params = model_info["params"].copy() # Gets a copy of the base parameters for the current model.
        current_tune_params = model_info["tune_params"].copy() # Gets a copy of the tuning parameters for the current model.

        apply_calibration = model_info.get("apply_calibration", False) # Checks if probability calibration should be applied for this model.

        classifier = None # Initializes classifier to None.
        if model_class_str == "RandomForestClassifier": # If the model is RandomForest.
            classifier = RandomForestClassifier(**current_model_params) # Instantiates RandomForestClassifier with its parameters.
        elif model_class_str == "XGBClassifier": # If the model is XGBoost.
            # FIX: Use the total number of classes from label_encoder_classes
            current_model_params["num_class"] = len(label_encoder_classes) # Sets `num_class` for multi-class XGBoost based on actual labels.

            # SOLUTION: Explicitly set the device to 'cpu' to resolve the mismatch warning
            # during prediction in the Streamlit app. This overrides any 'device' setting
            # that might be present in the config file, ensuring a CPU-compatible model.
            current_model_params['device'] = 'cpu' # Forces XGBoost to use CPU to prevent Streamlit prediction issues with GPU models.

            classifier = XGBClassifier(**current_model_params) # Instantiates XGBClassifier with its parameters.
        elif model_class_str == "LGBMClassifier": # If the model is LightGBM.
            # FIX: Use the total number of classes from label_encoder_classes
            current_model_params["num_class"] = len(label_encoder_classes) # Sets `num_class` for multi-class LightGBM.
            classifier = LGBMClassifier(**current_model_params) # Instantiates LGBMClassifier with its parameters.
        else:
            logger.warning(f"Skipping unknown model type: {model_class_str}") # Logs a warning if the model type is unknown.
            continue # Skips to the next model in the loop.

        # --- Build the Scikit-learn Pipeline ---
        pipeline_steps = [ # Defines the initial steps for the scikit-learn pipeline.
            ("scaler", StandardScaler()), # First step: feature scaling using StandardScaler.
            ("selector", feature_selector_pipeline_step), # Second step: feature selection (could be a passthrough).
        ]

        # Dynamically adjust tune_params keys based on whether calibration is applied
        adjusted_tune_params = {} # Initializes an empty dictionary for adjusted tuning parameters.
        if apply_calibration: # If calibration is enabled for the current model.
            calibrator = CalibratedClassifierCV( # Instantiates CalibratedClassifierCV.
                estimator=classifier, # Sets the base classifier to be calibrated.
                method="isotonic", # Uses isotonic calibration (non-parametric).
                cv=config.CV_FOLDS, # Uses configured cross-validation folds for calibration.
                n_jobs=1,  # Use config.CV_FOLDS # Sets `n_jobs` to 1 for calibrator to avoid issues with some versions/configs.
            )
            pipeline_steps.append(("calibrator", calibrator)) # Adds the calibrator as the final step in the pipeline.

            for key, value in current_tune_params.items(): # Iterates through original tuning parameters.
                if not key.startswith("calibrator__estimator__"): # If key doesn't already have calibrator prefix.
                    new_key = f"calibrator__estimator__{key}" # Adds `calibrator__estimator__` prefix for pipeline tuning.
                else:
                    new_key = key # Uses existing key if prefix is already there.
                adjusted_tune_params[new_key] = value # Stores the adjusted tuning parameter.
        else: # If calibration is not applied.
            pipeline_steps.append(("classifier", classifier)) # Adds the classifier directly as the final step.
            for key, value in current_tune_params.items(): # Iterates through original tuning parameters.
                if not key.startswith("classifier__"): # If key doesn't already have classifier prefix.
                    new_key = f"classifier__{key}" # Adds `classifier__` prefix for pipeline tuning.
                else:
                    new_key = key # Uses existing key if prefix is already there.
                adjusted_tune_params[new_key] = value # Stores the adjusted tuning parameter.

        pipeline = Pipeline(steps=pipeline_steps) # Creates the scikit-learn Pipeline with the defined steps.
        logger.info( # Logs the created pipeline's steps.
            f"Pipeline created for {model_name}: {[step[0] for step in pipeline.steps]}"
        )

        search_cv = None # Initializes `search_cv` to None.
        # Since we forced XGBoost to CPU, we can safely use n_jobs=-1 for all models.
        search_cv_n_jobs = -1 # Sets `n_jobs` to -1 to use all available CPU cores for hyperparameter search.

        if config.USE_RANDOMIZED_SEARCH:  # Use config.USE_RANDOMIZED_SEARCH # Checks if RandomizedSearchCV is enabled in config.
            logger.info( # Logs the start of RandomizedSearchCV.
                f"Starting RandomizedSearchCV for {model_name} pipeline (n_jobs={search_cv_n_jobs})..."
            )
            search_cv = RandomizedSearchCV( # Instantiates RandomizedSearchCV.
                estimator=pipeline, # The pipeline to tune.
                param_distributions=adjusted_tune_params, # The hyperparameter distributions to sample from.
                n_iter=model_info.get( # Number of parameter settings that are sampled.
                    "n_iter_search", config.CURRENT_SETTINGS["N_ITER_SEARCH_MODELS"]
                ),
                # Use dynamic N_ITER_SEARCH_MODELS
                scoring=scorers, # The scoring metrics to use.
                refit="f1_macro", # The metric to use for refitting the best estimator.
                cv=config.CV_FOLDS,  # Use config.CV_FOLDS # Number of cross-validation folds.
                verbose=2, # Verbosity level.
                random_state=config.SEED,  # Use config.SEED # Random state for reproducibility.
                n_jobs=search_cv_n_jobs, # Number of jobs to run in parallel.
            )
        else: # If RandomizedSearchCV is not enabled, use GridSearchCV.
            logger.info( # Logs the start of GridSearchCV.
                f"Starting GridSearchCV for {model_name} pipeline (n_jobs={search_cv_n_jobs})..."
            )
            search_cv = GridSearchCV( # Instantiates GridSearchCV.
                estimator=pipeline, # The pipeline to tune.
                param_grid=adjusted_tune_params, # The hyperparameter grid to search.
                scoring=scorers, # The scoring metrics.
                refit="f1_macro", # The metric to use for refitting.
                cv=config.CV_FOLDS,  # Use config.CV_FOLDS # Number of cross-validation folds.
                verbose=2, # Verbosity level.
                n_jobs=search_cv_n_jobs, # Number of jobs to run in parallel.
            )

        if search_cv: # If a search CV object was successfully initialized.
            try: # Begins a try block for fitting the search CV.
                search_cv.fit(X_train, y_train) # Fits the GridSearchCV/RandomizedSearchCV to the training data.
                best_pipeline = search_cv.best_estimator_ # Retrieves the best estimator (pipeline) found during the search.

                logger.info( # Logs the best parameters found.
                    f"Best parameters for {model_name}: {search_cv.best_params_}"
                )
                logger.info( # Logs the best cross-validation score.
                    f"Best cross-validation F1-macro score for {model_name}: {search_cv.best_score_:.4f}"
                )

                trained_models[model_name] = best_pipeline # Stores the best trained pipeline in the `trained_models` dictionary.

                model_path = os.path.join( # Constructs the file path for saving the best pipeline.
                    config.MODELS_DIR, f"{model_name.lower()}_best_pipeline.pkl"
                )  # Use config.MODELS_DIR
                save_object( # Saves the best pipeline object to disk.
                    best_pipeline,
                    config.MODELS_DIR,
                    f"{model_name.lower()}_best_pipeline.pkl",
                )  # Use config.MODELS_DIR
                logger.info(f"Best {model_name} pipeline saved to {model_path}") # Logs where the pipeline was saved.

                results_df = pd.DataFrame(search_cv.cv_results_) # Creates a DataFrame from the cross-validation results.
                results_df = results_df.sort_values(by="rank_test_f1_macro") # Sorts results by F1-macro rank.
                results_path = os.path.join( # Constructs the file path for saving the search results.
                    config.REPORTS_DIR, f"{model_name.lower()}_search_results.csv"
                )  # Use config.REPORTS_DIR
                results_df.to_csv(results_path, index=False) # Saves the search results DataFrame to CSV.
                logger.info(f"Search results for {model_name} saved to {results_path}") # Logs where the search results were saved.

            except Exception as e: # Catches any exception during model training/tuning.
                logger.error(f"Error during {model_name} modeling: {e}", exc_info=True) # Logs the error with traceback.
                continue # Continues to the next model.
        else:
            logger.error( # Logs an error if search CV could not be initialized.
                f"Could not initialize GridSearchCV/RandomizedSearchCV for {model_name}. Skipping."
            )

    # --- Ensemble Model (VotingClassifier) ---
    if len(trained_models) >= 2: # Checks if at least two individual models were trained successfully to form an ensemble.
        logger.info("\n--- Training Ensemble Model (VotingClassifier) ---") # Logs the start of ensemble model training.
        ensemble_estimators = [] # Initializes an empty list to hold estimators for the VotingClassifier.
        for model_name, model_pipeline in trained_models.items(): # Iterates through the successfully trained models.
            if model_pipeline is not None: # Checks if the model pipeline is not None.
                ensemble_estimators.append((model_name.lower(), model_pipeline)) # Adds the model name and pipeline to the ensemble estimators list.

        if len(ensemble_estimators) >= 2: # Re-checks if there are at least two estimators for the ensemble.
            try: # Begins a try block for fitting the ensemble model.
                ensemble_model = VotingClassifier( # Instantiates the VotingClassifier.
                    estimators=ensemble_estimators, # Provides the list of individual estimators.
                    voting="soft", # Uses soft voting (averages predicted probabilities).
                    n_jobs=-1, # Uses all available CPU cores for fitting.
                    verbose=True, # Enables verbose output during fitting.
                )
                logger.info("Fitting Ensemble model...") # Logs that the ensemble model is being fitted.
                ensemble_model.fit(X_train, y_train) # Fits the ensemble model to the training data.
                trained_models["Ensemble"] = ensemble_model # Adds the trained ensemble model to the `trained_models` dictionary.
                save_object( # Saves the trained ensemble model.
                    ensemble_model, config.MODELS_DIR, "ensemble_voting_classifier.pkl"
                )  # Use config.MODELS_DIR
                logger.info("Ensemble model (VotingClassifier) trained and saved.") # Logs that the ensemble model was trained and saved.
            except Exception as e: # Catches any exception during ensemble modeling.
                logger.error(f"Error during Ensemble modeling: {e}", exc_info=True) # Logs the error with traceback.
        else:
            logger.warning( # Logs a warning if not enough models were trained for the ensemble.
                "Skipping Ensemble modeling: Not enough individual models trained successfully to form an ensemble (need at least 2)."
            )
    else:
        logger.warning( # Logs a warning if there aren't enough trained models to form an ensemble at all.
            "Skipping Ensemble modeling: Not enough individual models trained to form an ensemble (need at least 2)."
        )

    logger.info("--- Modeling Phase Complete ---\n") # Logs the completion of the Modeling phase.
    return trained_models # Returns the dictionary of all trained models.


if __name__ == "__main__": # Checks if the script is being run directly.
    X_train_df = load_dataframe( # Loads the training features DataFrame.
        config.PROCESSED_DATA_DIR, "X_train.csv"
    )  # Use config.PROCESSED_DATA_DIR
    y_train_df = load_dataframe( # Loads the training labels DataFrame.
        config.PROCESSED_DATA_DIR, "y_train.csv"
    )  # Use config.PROCESSED_DATA_DIR
    label_encoder_obj = load_object(config.MODELS_DIR, "label_encoder.pkl") # Loads the label encoder object.
    label_encoder_classes_loaded = ( # Extracts the classes from the loaded label encoder.
        label_encoder_obj.classes_ if label_encoder_obj else None
    )

    if ( # Checks if all necessary components for standalone testing are loaded.
            X_train_df is not None
            and y_train_df is not None
            and label_encoder_classes_loaded is not None
    ):
        y_train_series = y_train_df["Label_Encoded"] # Extracts the encoded labels as a Series.
        trained_models_dict = modeling_phase( # Calls the modeling phase.
            X_train_df, y_train_series, label_encoder_classes_loaded
        )
        if trained_models_dict: # Checks if models were successfully trained.
            logger.info( # Logs successful training for standalone test.
                f"Successfully trained {len(trained_models_dict)} models for standalone test."
            )
    else:
        logger.error( # Logs an error if essential components could not be loaded.
            "Could not load training data or label encoder. Please ensure data_preparation_phase ran successfully and saved the data."
        )