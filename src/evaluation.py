# src/evaluation.py

import os # Imports the 'os' module for interacting with the operating system, like managing file paths.
from typing import Any, Callable, Dict, List, Optional, Tuple # Imports specific type hints for better code readability and maintainability.

import joblib # Imports joblib for efficient serialization/deserialization of Python objects.
import matplotlib.pyplot as plt # Imports `matplotlib.pyplot` for creating plots.
import numpy as np # Imports `numpy` for numerical operations.
import pandas as pd # Imports `pandas` for data manipulation and analysis.
import seaborn as sns # Imports `seaborn` for creating statistical graphics.
import shap # Imports the SHAP library for explainable AI (model interpretability).
from lightgbm import LGBMClassifier # Imports the LightGBM classifier.
from sklearn.calibration import CalibratedClassifierCV # Imports CalibratedClassifierCV for probability calibration.
from sklearn.ensemble import RandomForestClassifier, VotingClassifier # Imports RandomForestClassifier and VotingClassifier for ensemble methods.
from sklearn.feature_selection import SelectFromModel # Imports SelectFromModel for model-based feature selection.
from sklearn.metrics import ( # Imports various metrics from scikit-learn for model evaluation.
    accuracy_score, # For calculating overall accuracy.
    auc, # For calculating Area Under the Curve (AUC).
    classification_report, # For generating a text report showing main classification metrics.
    cohen_kappa_score, # For calculating Cohen's Kappa score.
    confusion_matrix, # For creating a confusion matrix.
    log_loss, # For calculating log loss (cross-entropy loss).
    precision_recall_curve, # For calculating precision-recall pairs for different probability thresholds.
    precision_recall_fscore_support, # For calculating precision, recall, F-score, and support for each class.
    roc_auc_score, # For calculating Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    roc_curve, # For computing Receiver Operating Characteristic (ROC) curve.
)
from sklearn.pipeline import Pipeline # Imports Pipeline for creating a sequence of data processing and modeling steps.
from sklearn.preprocessing import label_binarize # Imports label_binarize for one-hot encoding labels in a one-vs-rest fashion.
from xgboost import XGBClassifier # Imports the XGBoost classifier.

import src.config as config # Imports the project's configuration file to access various settings and parameters.
from src.utils import load_dataframe, load_object, plot_and_save, setup_logging # Imports utility functions for loading data/objects, plotting, and logging setup.

logger = setup_logging() # Initializes the logger for this module, inheriting the configuration set up in `main.py`.


def evaluate_model( # Defines the function to evaluate a single machine learning model.
    model: Any, # The trained machine learning model or pipeline object.
    X_test: pd.DataFrame, # Test features (expected to be unscaled, as pipelines handle preprocessing internally).
    y_test: pd.Series, # True target values for the test set.
    label_encoder_classes: np.ndarray, # Array of original label names (from LabelEncoder.classes_).
    model_name: str, # The name of the model being evaluated (e.g., 'RandomForest').
) -> Dict[str, Any]: # Returns a dictionary containing key performance metrics for the model.
    """
    Evaluates a single trained machine learning model (or pipeline) on the test set and generates
    comprehensive performance reports and visualizations (classification report,
    confusion matrix - raw and normalized, ROC curves, feature importances, Precision-Recall curves).
    Includes optional SHAP explanations for tree-based models (if not in test mode).


    Args:
        model (Any): The trained machine learning model or pipeline object.
        X_test (pd.DataFrame): Test features (expected to be unscaled, as pipelines handle preprocessing internally).
        y_test (pd.Series): True target values for the test set.
        label_encoder_classes (np.ndarray): Array of original label names
                                            (from LabelEncoder.classes_).
        model_name (str): The name of the model being evaluated (e.g., 'RandomForest').

    Returns:
        Dict[str, Any]: A dictionary containing key performance metrics for the model.
    """
    logger.info(f"\n--- Evaluating {model_name} ---") # Logs the start of the evaluation for the current model.

    if ( # Checks if the model or test data is not provided or empty.
        model is None
        or X_test is None
        or y_test is None
        or X_test.empty
        or y_test.empty
    ):
        logger.error("Model or test data not provided for evaluation. Skipping.") # Logs an error if essential data is missing.
        return {} # Returns an empty dictionary, indicating skipping evaluation.

    X_test_processed = X_test.select_dtypes(include=np.number).copy() # Selects only numerical columns from X_test and creates a copy.
    X_test_processed = ( # Replaces infinite values with NaN, fills any NaNs with 0, and casts to float32.
        X_test_processed.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    )

    logger.info("Making predictions on the test set...") # Logs that predictions are being made.
    y_pred = model.predict(X_test_processed) # Makes class predictions on the processed test data.

    y_pred_proba = None # Initializes y_pred_proba to None.
    target_names = label_encoder_classes.tolist() # Converts the numpy array of label encoder classes to a list of target names.
    n_classes = len( # Calculates the total number of unique classes.
        target_names
    )  # Total number of possible classes (from 0 to 11 in your case)

    if hasattr(model, "predict_proba"): # Checks if the model has a `predict_proba` method.
        try: # Begins a try block to handle potential errors during probability prediction.
            raw_y_pred_proba = model.predict_proba(X_test_processed) # Gets raw probability predictions from the model.

            # Determine the order of classes from the model if available,
            # otherwise assume it's ordered 0 to N-1
            model_classes_order = None # Initializes `model_classes_order` to None.
            if hasattr(model, "classes_"): # Checks if the model itself has a `classes_` attribute (direct classifier).
                model_classes_order = model.classes_ # Assigns classes from the model directly.
            elif isinstance(model, Pipeline): # If the model is a scikit-learn Pipeline.
                # Try to get classes_ from the final estimator in the pipeline
                final_estimator = model.named_steps.get( # Tries to get the final estimator, which might be a calibrator or a classifier.
                    "calibrator", model.named_steps.get("classifier")
                )
                if hasattr(final_estimator, "classes_"): # Checks if the final estimator has a `classes_` attribute.
                    model_classes_order = final_estimator.classes_ # Assigns classes from the final estimator.
                elif hasattr(getattr(final_estimator, "estimator", None), "classes_"): # If it's a CalibratedClassifierCV, check its base estimator.
                    # For CalibratedClassifierCV, get from its base estimator
                    model_classes_order = final_estimator.estimator.classes_ # Assigns classes from the base estimator of CalibratedClassifierCV.

            if model_classes_order is None: # If class order could not be determined from the model.
                # Fallback: Assume model outputs probabilities for classes 0 to N-1 in order
                # This is a less safe assumption, but might be needed if `classes_` is unavailable.
                logger.warning( # Logs a warning about assuming class order.
                    f"Could not retrieve class order from model {model_name}. Assuming 0 to N-1 order."
                )
                model_classes_order = np.arange(raw_y_pred_proba.shape[1]) # Assumes classes are ordered from 0 to N-1.

            # Create a padded y_pred_proba array for all 'n_classes'
            # Initialize with small values to avoid log(0)
            padded_y_pred_proba = np.full( # Creates a new array for probabilities, padded to include all `n_classes`.
                (raw_y_pred_proba.shape[0], n_classes), 1e-9, dtype=np.float32 # Initializes with a very small value to avoid log(0).
            )

            # Map probabilities from raw_y_pred_proba to correct columns in padded_y_pred_proba
            for i, class_label in enumerate(model_classes_order): # Iterates through the model's predicted class labels and their original indices.
                if ( # Ensures the class label is within the expected total number of classes.
                    class_label < n_classes
                ):  # Ensure the class_label is within expected bounds
                    padded_y_pred_proba[:, class_label] = raw_y_pred_proba[:, i] # Maps the raw probability to the correct column in the padded array.
                else:
                    logger.warning( # Logs a warning if the model predicted an unexpected class label.
                        f"Model {model_name} predicted for an unexpected class label: {class_label}. Ignoring this probability column."
                    )

            y_pred_proba = padded_y_pred_proba # Assigns the padded probabilities.

            # Normalize probabilities to sum to 1 after padding,
            # especially important with the 1e-9 padding.
            sum_of_probs = y_pred_proba.sum(axis=1, keepdims=True) # Calculates the sum of probabilities for each row.
            # Avoid division by zero for rows that might sum to exactly 0 (though unlikely with 1e-9 padding)
            sum_of_probs[sum_of_probs == 0] = 1.0  # Prevent division by zero # Replaces zero sums with 1.0 to prevent division by zero.
            y_pred_proba = y_pred_proba / sum_of_probs # Normalizes probabilities so they sum to 1.

            logger.info( # Logs that probability predictions are available.
                "Model supports probability predictions for ROC AUC, Log Loss, and PR Curves."
            )
        except Exception as e: # Catches any exception during probability prediction.
            logger.warning( # Logs a warning if probability prediction fails.
                f"Model {model_name} has 'predict_proba' but failed to produce probabilities: {e}",
                exc_info=True, # Includes traceback in the log.
            )
            y_pred_proba = None # Sets y_pred_proba to None if an error occurs.
    else:
        logger.warning( # Logs a warning if the model does not have a `predict_proba` method.
            f"Model {model_name} does not have 'predict_proba'. ROC AUC, Log Loss, and PR Curves will not be computed."
        )

    # --- 1. Overall Accuracy ---
    accuracy = accuracy_score(y_test, y_pred) # Calculates the overall accuracy score.
    metrics = {"accuracy": accuracy} # Initializes a dictionary to store metrics, starting with accuracy.
    logger.info(f"Overall Accuracy: {accuracy:.4f}") # Logs the overall accuracy.

    # --- 2. Classification Report ---
    logger.info("\nClassification Report:") # Logs a header for the classification report.
    # Pass target_names to classification_report to ensure full class labels are used
    report_str = classification_report( # Generates a classification report as a string.
        y_test, y_pred, target_names=target_names, digits=4, zero_division=0 # Provides true labels, predicted labels, target names, digits for formatting, and handles zero division.
    )
    logger.info(report_str) # Logs the classification report.

    report_path = os.path.join( # Constructs the file path for saving the classification report.
        config.REPORTS_DIR, f"{model_name.lower()}_classification_report.txt"
    )
    with open(report_path, "w") as f: # Opens the file in write mode.
        f.write(report_str) # Writes the classification report string to the file.
    logger.info(f"Classification report saved to {report_path}") # Logs where the report was saved.

    report_dict = classification_report( # Generates the classification report as a dictionary.
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    metrics.update( # Updates the metrics dictionary with macro and weighted average scores.
        {
            "F1-Score (macro)": report_dict["macro avg"]["f1-score"], # Macro average F1-score.
            "Precision (macro)": report_dict["macro avg"]["precision"], # Macro average precision.
            "Recall (macro)": report_dict["macro avg"]["recall"], # Macro average recall.
            "F1-Score (weighted)": report_dict["weighted avg"]["f1-score"], # Weighted average F1-score.
            "Precision (weighted)": report_dict["weighted avg"]["precision"], # Weighted average precision.
            "Recall (weighted)": report_dict["weighted avg"]["recall"], # Weighted average recall.
        }
    )
    logger.info(f"Extracted macro F1-score: {metrics['F1-Score (macro)']:.4f}") # Logs the extracted macro F1-score.

    # --- 3. Confusion Matrix (Raw and Normalized) ---
    cm = confusion_matrix(y_test, y_pred) # Computes the raw confusion matrix.
    logger.info("\nConfusion Matrix (Raw Counts):") # Logs a header for the raw confusion matrix.
    logger.info( # Logs the raw confusion matrix as a string representation of a DataFrame.
        f"\n{pd.DataFrame(cm, index=target_names, columns=target_names).to_string()}"
    )

    plot_and_save( # Calls the utility function to plot and save the raw confusion matrix.
        plot_func=sns.heatmap, # Uses seaborn's heatmap for plotting.
        filename=f"{model_name.lower()}_confusion_matrix_raw.png", # Specifies the filename.
        title=f"Confusion Matrix (Raw Counts) for {model_name}", # Sets the plot title.
        data=cm, # Passes the raw confusion matrix data.
        annot=True, # Annotates the heatmap with cell values.
        fmt="d", # Formats annotation as integers.
        cmap="Blues", # Sets the colormap.
        xticklabels=target_names, # Sets x-axis tick labels to target names.
        yticklabels=target_names, # Sets y-axis tick labels to target names.
        cbar=True, # Displays a color bar.
        linewidths=0.5, # Sets linewidths between cells.
        linecolor="black", # Sets line color.
        figsize=(max(10, n_classes * 0.8), max(8, n_classes * 0.7)), # Dynamically sets figure size.
    )

    if config.CONFUSION_MATRIX_NORMALIZE: # Checks if normalization of confusion matrix is enabled in config.
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # Normalizes the confusion matrix by row (true labels), representing recall.
        cm_normalized = np.nan_to_num(cm_normalized) # Replaces any NaNs (resulting from division by zero for classes with no true instances) with zeros.

        logger.info("\nConfusion Matrix (Normalized by True Labels - Recall):") # Logs a header for the normalized confusion matrix.
        logger.info( # Logs the normalized confusion matrix as a string representation of a DataFrame, rounded to 2 decimal places.
            f"\n{pd.DataFrame(cm_normalized, index=target_names, columns=target_names).round(2).to_string()}"
        )

        plot_and_save( # Calls the utility function to plot and save the normalized confusion matrix.
            plot_func=sns.heatmap, # Uses seaborn's heatmap.
            filename=f"{model_name.lower()}_confusion_matrix_normalized.png", # Specifies the filename.
            title=f"Confusion Matrix (Normalized) for {model_name}", # Sets the plot title.
            data=cm_normalized, # Passes the normalized confusion matrix data.
            annot=True, # Annotates the heatmap with cell values.
            fmt=".2f", # Formats annotation as floats with 2 decimal places.
            cmap="Blues", # Sets the colormap.
            xticklabels=target_names, # Sets x-axis tick labels.
            yticklabels=target_names, # Sets y-axis tick labels.
            cbar=True, # Displays a color bar.
            linewidths=0.5, # Sets linewidths.
            linecolor="black", # Sets line color.
            figsize=(max(10, n_classes * 0.8), max(8, n_classes * 0.7)), # Dynamically sets figure size.
        )

    # --- 4. Cohen's Kappa Score ---
    kappa = cohen_kappa_score(y_test, y_pred) # Calculates Cohen's Kappa score.
    logger.info( # Logs Cohen's Kappa score.
        f"Cohen's Kappa Score: {kappa:.4f} (measures agreement between predictions and true labels)"
    )
    metrics["Cohen's Kappa"] = kappa # Adds Cohen's Kappa to the metrics dictionary.

    # --- 5. Log Loss ---
    if y_pred_proba is not None: # Checks if probability predictions are available.
        try: # Begins a try block to handle potential errors during log loss calculation.
            # Pass the list of all possible classes explicitly using np.arange(n_classes)
            ll = log_loss(y_test, y_pred_proba, labels=np.arange(n_classes)) # Calculates log loss, explicitly providing all possible class labels.
            logger.info(f"Log Loss: {ll:.4f} (penalizes confident wrong predictions)") # Logs the log loss.
            metrics["Log Loss"] = ll # Adds log loss to the metrics dictionary.
        except Exception as e: # Catches any exception during log loss calculation.
            logger.warning( # Logs a warning if log loss calculation fails.
                f"Could not calculate Log Loss for {model_name}: {e}", exc_info=True
            )
    else:
        metrics["Log Loss"] = np.nan # Sets log loss to NaN if probabilities are not available.

    # --- 6. ROC AUC ---
    if y_pred_proba is not None: # Checks if probability predictions are available.
        try: # Begins a try block to handle potential errors during ROC AUC calculation and plotting.
            # Pass the list of all possible classes explicitly using np.arange(n_classes)
            roc_auc_ovr = roc_auc_score( # Calculates the One-vs-Rest (OvR) ROC AUC score.
                y_test,
                y_pred_proba,
                multi_class="ovr", # Specifies OvR strategy for multi-class.
                average="macro", # Calculates macro-average AUC.
                labels=np.arange(n_classes), # Provides all possible class labels.
            )
            logger.info(f"ROC AUC (One-vs-Rest, Macro Average): {roc_auc_ovr:.4f}") # Logs the macro-average ROC AUC.
            metrics["ROC AUC (Macro)"] = roc_auc_ovr # Adds macro ROC AUC to the metrics dictionary.

            plt.figure(figsize=(12, 8)) # Creates a new figure for the ROC curve plot.
            for i, class_name in enumerate(target_names): # Iterates through each class.
                if (y_test == i).any(): # Checks if the current class is present in the true test labels.
                    # For ROC curve, use probabilities for the specific class directly
                    fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i]) # Computes FPR and TPR for the current class (one-vs-rest).
                    roc_auc = auc(fpr, tpr) # Calculates AUC for the current class.
                    plt.plot( # Plots the ROC curve for the current class.
                        fpr, tpr, label=f"Class {class_name} (AUC = {roc_auc:.2f})"
                    )
                else:
                    logger.debug( # Logs if a class is not present in y_test for plotting.
                        f"Class {class_name} not present in y_test for ROC curve plotting."
                    )

            plt.plot([0, 1], [0, 1], "k--", label="No Skill") # Plots the diagonal "no-skill" line.
            plt.xlabel("False Positive Rate") # Sets x-axis label.
            plt.ylabel("True Positive Rate") # Sets y-axis label.
            plt.title(f"ROC Curve (One-vs-Rest) for {model_name}") # Sets plot title.
            plt.legend(loc="lower right") # Displays legend.
            plt.grid(True) # Adds a grid.
            plt.tight_layout() # Adjusts plot parameters for a tight layout.
            roc_plot_path = os.path.join( # Constructs file path for saving the ROC plot.
                config.REPORTS_DIR, f"{model_name.lower()}_roc_curve_ovr.png"
            )
            plt.savefig(roc_plot_path) # Saves the ROC plot.
            logger.info(f"ROC curve plot saved to {roc_plot_path}") # Logs where the plot was saved.
            plt.close() # Closes the plot to free memory.

        except ValueError as e: # Catches ValueError (e.g., if a class has no positive samples).
            logger.warning( # Logs a warning if ROC AUC calculation/plotting fails.
                f"Could not calculate or plot ROC AUC for {model_name}: {e}. "
                "This can happen if a class is not present in y_test or y_pred_proba after sampling.",
                exc_info=True,
            )
            metrics["ROC AUC (Macro)"] = np.nan # Sets macro ROC AUC to NaN if calculation fails.
    else:
        metrics["ROC AUC (Macro)"] = np.nan # Sets macro ROC AUC to NaN if probabilities are not available.

    # --- 7. Precision-Recall Curve (NEW) ---
    if y_pred_proba is not None and n_classes > 1: # Checks if probability predictions are available and there is more than one class.
        logger.info("\nGenerating Precision-Recall curves...") # Logs that PR curves are being generated.
        try: # Begins a try block to handle potential errors during PR curve calculation and plotting.
            # Binarize y_test based on ALL possible classes (0 to n_classes-1)
            y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes)) # Binarizes true labels into a one-vs-rest format for all possible classes.

            # Iterate over all possible classes (0 to n_classes-1) for PR curve plotting
            # Check if class is present in y_test and has positive predictions for meaningful curve.
            plt.figure(figsize=(12, 8)) # Creates a new figure for the PR curve plot.
            pr_aucs = [] # Initializes a list to store AUCs for individual PR curves.
            for i in np.arange(n_classes):  # Iterate through all possible class indices # Iterates through each class index.
                # Check if the class is actually present in y_test for this specific subset
                # and if its corresponding probability column is non-empty.
                if ( # Checks if the class has positive samples in y_test and a valid probability column.
                    (y_test_binarized[:, i].sum() > 0) # Checks if there are actual positive samples for this class.
                    and (y_pred_proba.shape[1] > i) # Checks if the probability array has a column for this class.
                    and (y_pred_proba[:, i].sum() > 0) # Checks if there are any non-zero probabilities for this class.
                ):
                    precision, recall, _ = precision_recall_curve( # Computes precision-recall pairs for the current class.
                        y_test_binarized[:, i], y_pred_proba[:, i]
                    )
                    # Handle cases where precision/recall might be all 0/1 due to single class in data
                    if len(precision) > 1 and len(recall) > 1: # Ensures there are enough points to form a curve.
                        pr_auc = auc(recall, precision) # Calculates AUC for the precision-recall curve.
                        plt.plot( # Plots the PR curve for the current class.
                            recall,
                            precision,
                            label=f"Class {target_names[i]} (AUC = {pr_auc:.2f})",
                        )
                        pr_aucs.append(pr_auc) # Adds the PR AUC to the list.
                    else:
                        logger.debug( # Logs if PR curve is skipped due to insufficient data points.
                            f"Skipping PR curve for class {target_names[i]}: Insufficient data points for curve plotting (e.g., only one value of precision/recall)."
                        )
                else:
                    logger.debug( # Logs if PR curve is skipped because the class is not present or has no predictions.
                        f"Class {target_names[i]} not present in y_test or no predictions for it for PR curve plotting."
                    )

            # Compute micro-average PR curve for all classes if more than one is present
            if n_classes > 1: # Checks if there is more than one class for micro-average.
                precision_micro, recall_micro, _ = precision_recall_curve( # Computes micro-average precision-recall.
                    y_test_binarized.ravel(), y_pred_proba.ravel() # Flattens binarized true labels and probabilities.
                )
                # Only plot micro-average if it yields more than a single point
                if len(precision_micro) > 1 and len(recall_micro) > 1: # Ensures enough points for micro-average curve.
                    pr_auc_micro = auc(recall_micro, precision_micro) # Calculates micro-average PR AUC.
                    plt.plot( # Plots the micro-average PR curve.
                        recall_micro,
                        precision_micro,
                        label=f"Micro-average (AUC = {pr_auc_micro:.2f})",
                        linestyle=":",
                        linewidth=4,
                    )
                    metrics["PR AUC (Micro)"] = pr_auc_micro # Adds micro PR AUC to metrics.
                else:
                    logger.debug( # Logs if micro-average PR curve is skipped.
                        "Skipping Micro-average PR curve: Insufficient data points for curve plotting."
                    )
                    metrics["PR AUC (Micro)"] = np.nan # Sets micro PR AUC to NaN.
            else:
                metrics["PR AUC (Micro)"] = np.nan # Sets micro PR AUC to NaN if only one class.

            metrics["PR AUC (Macro)"] = np.mean(pr_aucs) if pr_aucs else np.nan # Calculates macro PR AUC from individual class AUCs.

            plt.xlabel("Recall") # Sets x-axis label.
            plt.ylabel("Precision") # Sets y-axis label.
            plt.title(f"Precision-Recall Curve (One-vs-Rest) for {model_name}") # Sets plot title.
            plt.legend(loc="lower left") # Displays legend.
            plt.grid(True) # Adds a grid.
            plt.tight_layout() # Adjusts layout.
            pr_plot_path = os.path.join( # Constructs file path for saving PR plot.
                config.REPORTS_DIR, f"{model_name.lower()}_pr_curve_ovr.png"
            )
            plt.savefig(pr_plot_path) # Saves the PR plot.
            logger.info(f"Precision-Recall curve plot saved to {pr_plot_path}") # Logs where the plot was saved.
            plt.close() # Closes the plot.

        except ValueError as e: # Catches ValueError during PR curve generation.
            logger.warning( # Logs a warning if PR AUC calculation/plotting fails due to ValueError.
                f"Could not calculate or plot PR AUC for {model_name}: {e}. "
                "This can happen if a class is not present in y_test.",
                exc_info=True,
            )
            metrics["PR AUC (Macro)"] = np.nan # Sets macro PR AUC to NaN.
            metrics["PR AUC (Micro)"] = np.nan # Sets micro PR AUC to NaN.
        except Exception as e:  # Catch other potential errors # Catches any other unexpected errors during PR curve generation.
            logger.warning( # Logs a warning for unexpected errors.
                f"An unexpected error occurred during PR curve generation for {model_name}: {e}",
                exc_info=True,
            )
            metrics["PR AUC (Macro)"] = np.nan # Sets macro PR AUC to NaN.
            metrics["PR AUC (Micro)"] = np.nan # Sets micro PR AUC to NaN.

    else:
        logger.warning( # Logs a warning if PR curves are skipped due to binary classification or missing probabilities.
            f"Skipping Precision-Recall curves for {model_name}: Not supported for binary classification "
            "or probability predictions not available."
        )
        metrics["PR AUC (Macro)"] = np.nan # Sets macro PR AUC to NaN.
        metrics["PR AUC (Micro)"] = np.nan # Sets micro PR AUC to NaN.

    # --- 8. Feature Importance & SHAP Explanations ---
    actual_classifier = None # Initializes actual_classifier to None.
    X_test_for_shap_transform = None # Initializes X_test_for_shap_transform to None.

    if isinstance(model, Pipeline): # Checks if the model is a scikit-learn Pipeline.
        final_estimator_name, final_estimator_obj = model.steps[-1] # Gets the name and object of the last step in the pipeline.

        if isinstance(final_estimator_obj, CalibratedClassifierCV): # If the final estimator is CalibratedClassifierCV.
            actual_classifier = final_estimator_obj.estimator # The actual classifier is its base estimator.
        else:
            actual_classifier = final_estimator_obj # Otherwise, the final estimator is the actual classifier.

        try: # Begins a try block for transforming data for SHAP.
            X_test_scaled = model.named_steps["scaler"].transform(X_test_processed) # Scales the processed test data using the pipeline's scaler.
            selector_step = model.named_steps["selector"] # Gets the feature selector step from the pipeline.

            if isinstance(selector_step, SelectFromModel) and hasattr( # Checks if the selector is SelectFromModel and has `get_support` method.
                selector_step, "get_support"
            ):
                selected_features_mask = selector_step.get_support(indices=False) # Gets the boolean mask of selected features.
                feature_names_after_selection = X_test_processed.columns[ # Gets the names of features after selection.
                    selected_features_mask
                ].tolist()
                X_test_for_shap_transform = pd.DataFrame( # Creates a DataFrame with only the selected features for SHAP.
                    selector_step.transform(X_test_scaled), # Transforms the scaled data using the selector.
                    columns=feature_names_after_selection, # Assigns selected feature names.
                    index=X_test_processed.index, # Preserves original index.
                )
            else:
                if ( # Handles cases where the scaled data is a numpy array and matches original column count.
                    isinstance(X_test_scaled, np.ndarray)
                    and len(X_test_scaled.shape) == 2
                    and X_test_scaled.shape[1] == X_test_processed.shape[1]
                ):
                    X_test_for_shap_transform = pd.DataFrame( # Creates DataFrame with original column names.
                        X_test_scaled,
                        columns=X_test_processed.columns,
                        index=X_test_processed.index,
                    )
                else:
                    X_test_for_shap_transform = pd.DataFrame( # Creates DataFrame with generic feature names if names are not easily obtainable.
                        X_test_scaled,
                        columns=[f"feature_{i}" for i in range(X_test_scaled.shape[1])],
                        index=X_test_processed.index,
                    )
                logger.warning( # Logs a warning if feature names could not be precisely mapped for SHAP.
                    "Feature selector did not provide specific feature names or was a passthrough; "
                    "SHAP explanation will use transformed feature names or generic ones."
                )

        except Exception as e: # Catches any exception during data transformation for SHAP.
            logger.warning( # Logs a warning if transformation fails.
                f"Error transforming X_test for SHAP in pipeline: {e}. Skipping feature importances and SHAP.",
                exc_info=True,
            )
            actual_classifier = None # Sets actual_classifier to None to skip SHAP.
    elif isinstance(model, VotingClassifier): # If the model is a VotingClassifier.
        logger.info( # Logs that feature importances and SHAP are skipped for VotingClassifier.
            f"Skipping feature importances and SHAP for {model_name} (VotingClassifier does not have direct importances or TreeExplainer support)."
        )
        actual_classifier = None # Sets actual_classifier to None.
    else:
        actual_classifier = model # If it's a direct classifier (not a pipeline), use it directly.
        X_test_for_shap_transform = X_test_processed # Use the processed test data directly for SHAP.

    if ( # Checks conditions for generating feature importances and SHAP plots.
        actual_classifier is not None # An actual classifier exists.
        and hasattr(actual_classifier, "feature_importances_") # The classifier has `feature_importances_` attribute.
        and X_test_for_shap_transform is not None # Transformed data for SHAP is available.
        and not X_test_for_shap_transform.empty # Transformed data is not empty.
        and len(X_test_for_shap_transform.columns) # Number of features matches importances.
        == len(actual_classifier.feature_importances_)
    ):

        feature_importances_available = True # Sets flag indicating feature importances are available.
        logger.info("\nFeature Importances (Top 20):") # Logs header for feature importances.
        feature_importances = pd.Series( # Creates a Pandas Series of feature importances with feature names as index.
            actual_classifier.feature_importances_,
            index=X_test_for_shap_transform.columns,
        )
        top_features = feature_importances.nlargest(20) # Gets the top 20 most important features.
        logger.info(top_features.to_string()) # Logs the top 20 features and their importances.

        plot_and_save( # Calls utility function to plot and save feature importances.
            plot_func=sns.barplot, # Uses seaborn's barplot.
            filename=f"{model_name.lower()}_feature_importances.png", # Specifies filename.
            title=f"Top 20 Feature Importances for {model_name}", # Sets plot title.
            x=top_features.values, # Values for x-axis.
            y=top_features.index, # Index (feature names) for y-axis.
            palette="viridis", # Color palette.
            figsize=(10, 8), # Figure size.
        )

        # SHAP only runs if NOT in TEST_MODE
        if not config.TEST_MODE and len(X_test_for_shap_transform) > 1000: # Checks if not in test mode and enough samples for SHAP.
            logger.info( # Logs SHAP explanation generation.
                f"Generating SHAP explanations for {model_name} (sample of 1000 test instances)..."
            )
            try: # Begins try block for SHAP calculation.
                X_sample_for_shap = X_test_for_shap_transform.sample( # Samples data for SHAP explanation (max 1000 instances).
                    n=min(1000, len(X_test_for_shap_transform)),
                    random_state=config.SEED,
                )

                if isinstance( # Checks if the classifier is a tree-based model compatible with TreeExplainer.
                    actual_classifier,
                    (RandomForestClassifier, XGBClassifier, LGBMClassifier),
                ):
                    if len(X_test_for_shap_transform) > 100: # If enough data for background data for SHAP.
                        background_data = X_test_for_shap_transform.sample( # Samples background data for SHAP.
                            n=min(100, len(X_test_for_shap_transform)),
                            random_state=config.SEED,
                        ).astype(np.float32)
                        explainer = shap.TreeExplainer( # Initializes SHAP TreeExplainer with background data.
                            actual_classifier, data=background_data
                        )
                    else:
                        explainer = shap.TreeExplainer(actual_classifier) # Initializes SHAP TreeExplainer without background data (less accurate but handles small data).

                    shap_values = explainer.shap_values( # Calculates SHAP values.
                        X_sample_for_shap.astype(np.float32)
                    )

                    if isinstance(shap_values, list): # Checks if SHAP values are a list (for multi-output models).
                        if len(shap_values) > 0 and len(target_names) > 0: # Ensures SHAP values and target names are not empty.
                            class_to_plot_idx = 0 # Default class to plot (usually benign).
                            if ( # Adjusts class to plot if BENIGN is the first class and other classes exist.
                                config.BENIGN_LABEL in target_names
                                and target_names[0] == config.BENIGN_LABEL
                                and len(target_names) > 1
                            ):
                                class_to_plot_idx = 1 # Set to plot the first non-benign class.

                            if class_to_plot_idx < len(shap_values): # Ensures the class index is valid for SHAP values.
                                shap.summary_plot( # Generates a SHAP summary plot for a specific class.
                                    shap_values[class_to_plot_idx],
                                    X_sample_for_shap,
                                    plot_type="bar", # Specifies bar plot type.
                                    show=False, # Prevents immediate display of plot.
                                )
                                plot_and_save( # Saves the SHAP summary plot.
                                    plot_func=plt.show, # Uses plt.show as a dummy plot_func to trigger save.
                                    filename=f"{model_name.lower()}_shap_summary_class_{target_names[class_to_plot_idx]}.png", # Specifies filename.
                                    title=f"SHAP Summary Plot for {model_name} (Class: {target_names[class_to_plot_idx]})", # Sets title.
                                    figsize=(10, 8), # Figure size.
                                )

                            if X_sample_for_shap.shape[1] < 50: # If number of features is not too large, plot all-class summary.
                                shap.summary_plot( # Generates an all-class SHAP summary plot (scatter plot).
                                    shap_values,
                                    X_sample_for_shap,
                                    class_names=target_names, # Provides class names for the plot.
                                    show=False,
                                )
                                plot_and_save( # Saves the all-class SHAP summary plot.
                                    plot_func=plt.show,
                                    filename=f"{model_name.lower()}_shap_summary_all_classes.png",
                                    title=f"SHAP Summary Plot for {model_name} (All Classes)",
                                    figsize=(12, 10),
                                )
                            else:
                                logger.info( # Logs if multi-class SHAP summary is skipped due to too many features.
                                    f"Skipping multi-class SHAP summary for {model_name} due to too many features ({X_sample_for_shap.shape[1]})."
                                )
                        else:
                            logger.warning( # Logs a warning if SHAP values list is empty or target names are missing.
                                f"SHAP: shap_values is a list but empty or target_names is empty for {model_name}. Skipping plot."
                            )
                    else: # If SHAP values are not a list (e.g., binary classification).
                        shap.summary_plot( # Generates a SHAP summary plot (scatter plot) for binary case.
                            shap_values, X_sample_for_shap, plot_type="bar", show=False
                        )
                        plot_and_save( # Saves the SHAP summary plot for binary case.
                            plot_func=plt.show,
                            filename=f"{model_name.lower()}_shap_summary.png",
                            title=f"SHAP Summary Plot for {model_name}",
                            figsize=(10, 8),
                        )
                    logger.info(f"SHAP explanation plots saved for {model_name}.") # Logs that SHAP plots were saved.
                else:
                    logger.info( # Logs if SHAP is skipped because TreeExplainer is not suitable.
                        f"Skipping SHAP for {model_name}: TreeExplainer not suitable for this classifier type."
                    )

            except Exception as e: # Catches any exception during SHAP explanation generation.
                logger.warning( # Logs a warning if SHAP generation fails.
                    f"Could not generate SHAP explanations for {model_name}: {e}. Skipping SHAP plots.",
                    exc_info=True,
                )
        elif not config.TEST_MODE and ( # Checks if not in test mode but data for SHAP is too small.
            X_test_for_shap_transform is None or len(X_test_for_shap_transform) <= 1000
        ):
            logger.info( # Logs that SHAP is skipped due to insufficient data.
                f"Skipping SHAP explanations for {model_name}: Test data for SHAP too small (current: {len(X_test_for_shap_transform) if X_test_for_shap_transform is not None else 'N/A'} samples). Need >1000 samples for meaningful SHAP plots when not in TEST_MODE."
            )
        elif config.TEST_MODE: # Checks if in test mode.
            logger.info( # Logs that SHAP is skipped because TEST_MODE is active (computationally intensive).
                f"Skipping SHAP explanations for {model_name}: TEST_MODE is ON. SHAP is computationally intensive and skipped for quick tests."
            )

    else:
        logger.info( # Logs if feature importances or SHAP support is not available for the model.
            f"Model {model_name} does not have 'feature_importances_' attribute or is not directly supported by TreeExplainer for SHAP or the transformed data is not valid."
        )

    logger.info(f"--- Evaluation for {model_name} Complete ---\n") # Logs the completion of evaluation for the current model.
    return metrics # Returns the dictionary of performance metrics.


def evaluation_phase( # Defines the main evaluation phase function.
    trained_models: Dict[str, Any], # Dictionary of trained model objects (from modeling_phase).
    X_test: pd.DataFrame, # Test features (unscaled, as pipelines handle scaling internally).
    y_test: pd.Series, # True target values for the test set.
    label_encoder_classes: np.ndarray, # Array of original label names.
) -> pd.DataFrame: # Returns a DataFrame summarizing key metrics for all models.
    """
    Orchestrates the evaluation of all trained models (including ensemble) and provides a comparative summary.
    This phase critically assesses the models' performance against project objectives.

    Args:
        trained_models (Dict[str, Any]): Dictionary of trained model objects (from modeling_phase).
                                        These models might be Pipelines or raw classifiers.
        X_test (pd.DataFrame): Test features (unscaled, as pipelines handle scaling internally).
        y_test (pd.Series): True target values for the test set.
        label_encoder_classes (np.ndarray): Array of original label names.

    Returns:
        pd.DataFrame: A DataFrame summarizing the key metrics for all models,
                      facilitating comparative analysis.
    """
    logger.info("--- CRISP-DM Phase 5: Evaluation ---") # Logs the start of the Evaluation phase.

    if not trained_models: # Checks if no trained models were provided.
        logger.error("No trained models provided for evaluation. Aborting.") # Logs an error.
        return pd.DataFrame() # Returns an empty DataFrame.
    if X_test is None or y_test is None or X_test.empty or y_test.empty: # Checks if test data is missing or empty.
        logger.error("Test data not provided. Aborting Evaluation phase.") # Logs an error.
        return pd.DataFrame() # Returns an empty DataFrame.

    all_models_metrics = [] # Initializes a list to store metrics for all models.

    for model_name, model in trained_models.items(): # Iterates through each trained model.
        metrics = evaluate_model( # Calls `evaluate_model` for the current model.
            model, X_test, y_test, label_encoder_classes, model_name
        )
        if metrics: # If metrics were successfully generated.
            metrics["Model"] = model_name # Adds the model name to the metrics dictionary. Changed to 'Model'
            all_models_metrics.append(metrics) # Appends the metrics to the list.

    if not all_models_metrics: # Checks if no models were successfully evaluated.
        logger.warning( # Logs a warning.
            "No models successfully evaluated. Cannot create comparative summary."
        )
        return pd.DataFrame() # Returns an empty DataFrame.

    metrics_df = pd.DataFrame(all_models_metrics) # Creates a DataFrame from the collected metrics.
    # Removed: metrics_df = metrics_df.set_index("Model") # REMOVED: No longer setting 'Model' as index
    logger.info("\n--- Comparative Model Performance Summary ---") # Logs a header for the summary.
    logger.info(metrics_df.round(4).to_string()) # Logs the comparative metrics summary, rounded to 4 decimal places.

    summary_path = os.path.join(config.REPORTS_DIR, "model_performance_summary.csv") # Constructs the file path for saving the summary.
    metrics_df.to_csv(summary_path, index=False) # Saves the metrics DataFrame to a CSV file, ensuring no index is written.
    logger.info(f"Model performance summary saved to {summary_path}") # Logs where the summary was saved.

    # Also updating the column names here for consistency with the Streamlit app's expected names
    # (e.g., 'f1_macro' to 'F1-Score (macro)')
    # Ensure these match how Streamlit app expects them for 'metrics_to_score'
    metrics_df = metrics_df.rename(columns={
        'accuracy': 'Accuracy',
        'f1_macro': 'F1-Score (macro)',
        'precision_macro': 'Precision (macro)',
        'recall_macro': 'Recall (macro)',
        'f1_weighted': 'F1-Score (weighted)',
        'precision_weighted': 'Precision (weighted)',
        'recall_weighted': 'Recall (weighted)',
        'cohen_kappa': 'Cohen\'s Kappa',
        'log_loss': 'Log Loss',
        'roc_auc_macro': 'ROC AUC (Macro)',
        'pr_auc_macro': 'PR AUC (Macro)',
        'pr_auc_micro': 'PR AUC (Micro)'
    })


    if not metrics_df.empty and "F1-Score (macro)" in metrics_df.columns: # Checks if the metrics DataFrame is not empty and contains 'F1-Score (macro)'.
        plot_and_save( # Calls utility function to plot and save the F1-score comparison.
            plot_func=metrics_df.set_index('Model')["F1-Score (macro)"].sort_values(ascending=False).plot, # Uses the plot method of the sorted F1-macro series.
            filename="f1_score_comparison.png", # Specifies the filename.
            title="Macro Avg F1-score Comparison Across Models", # Sets the plot title.
            kind="bar", # Specifies bar chart kind.
            ylabel="F1-score (Macro Avg)", # Sets y-axis label.
            xlabel="Model", # Sets x-axis label.
            rot=45, # Rotates x-axis labels.
            figsize=(10, 6), # Figure size.
        )
    else:
        logger.warning( # Logs a warning if F1-macro column is missing or DataFrame is empty.
            "No 'F1-Score (macro)' column or empty DataFrame in metrics for comparison plot."
        )

    logger.info("--- Evaluation Phase Complete ---\n") # Logs the completion of the Evaluation phase.
    return metrics_df # Returns the metrics DataFrame.


if __name__ == "__main__": # Checks if the script is being run directly.
    X_test_df = load_dataframe(config.PROCESSED_DATA_DIR, "X_test.csv") # Loads the test features.
    y_test_df = load_dataframe(config.PROCESSED_DATA_DIR, "y_test.csv") # Loads the test labels.

    label_encoder_obj = load_object(config.MODELS_DIR, "label_encoder.pkl") # Loads the label encoder object.
    label_encoder_classes = label_encoder_obj.classes_ if label_encoder_obj else None # Extracts classes from the label encoder if loaded.

    trained_models_dict = {} # Initializes an empty dictionary for trained models.
    for model_name_key in config.CLASSIFIERS_TO_TRAIN.keys(): # Iterates through model names defined in config.
        model = load_object( # Loads each best pipeline model.
            config.MODELS_DIR, f"{model_name_key.lower()}_best_pipeline.pkl"
        )
        if model: # If the model was loaded successfully.
            trained_models_dict[model_name_key] = model # Adds it to the dictionary.

    ensemble_model = load_object(config.MODELS_DIR, "ensemble_voting_classifier.pkl") # Loads the ensemble model.
    if ensemble_model: # If the ensemble model was loaded.
        trained_models_dict["Ensemble"] = ensemble_model # Adds it to the dictionary.

    if ( # Checks if all necessary components for standalone evaluation are loaded.
        X_test_df is not None
        and y_test_df is not None
        and label_encoder_classes is not None
        and trained_models_dict
    ):
        y_test_series = y_test_df["Label_Encoded"] # Extracts the encoded labels as a Series.
        evaluation_phase( # Calls the evaluation phase.
            trained_models_dict, X_test_df, y_test_series, label_encoder_classes
        )
        logger.info("Evaluation phase completed successfully for standalone test.") # Logs successful completion.
    else:
        logger.error( # Logs an error if components are missing.
            "Could not load all necessary components for standalone evaluation. "
            "Ensure modeling and data preparation phases completed successfully and saved artifacts."
        )
