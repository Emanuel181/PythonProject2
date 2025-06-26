# src/report_generator.py

import logging # Imports the standard Python logging library for logging messages.
import os # Imports the 'os' module for interacting with the operating system, like managing file paths.
from typing import List # Imports `List` from `typing` for type hints.

import pandas as pd # Imports `pandas` for data manipulation and analysis, used here for timestamp and DataFrame operations.

import src.config as config  # Import config to access dynamic settings # Imports the project's configuration file to access various settings and parameters.
from src.utils import load_dataframe, load_object, setup_logging # Imports utility functions for loading DataFrames/objects and logging setup.

logger = setup_logging() # Initializes the logger for this module, inheriting the configuration set up in `main.py`.



def generate_full_report(output_filename: str = "project_summary_report.md") -> None: # Defines the function to generate a comprehensive project report.
    """
    Generates a comprehensive Markdown-formatted summary report of the entire ML project.
    It consolidates key information, model performance, and insights from various
    generated artifacts (logs, CSVs, plots). This addresses the 'Auto Report Generator'
    improvement.

    Args:
        output_filename (str): The name of the Markdown file to generate. Defaults to 'project_summary_report.md'.
    """
    logger.info("--- Generating Comprehensive Project Report ---") # Logs the start of the report generation process.
    report_filepath = os.path.join(config.REPORTS_DIR, output_filename) # Constructs the full file path for the report in the reports directory.

    report_content = [] # Initializes an empty list to store lines of the report content.

    report_content.append("# DDoS and Botnet Attack Detection Project Report\n") # Appends the main title of the report.
    report_content.append( # Appends a subtitle for the automated ML pipeline.
        "## Automated Machine Learning Pipeline for Network Security\n"
    )
    report_content.append( # Appends the generation timestamp to the report.
        f"**Generated On:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    report_content.append(f"**Environment Mode:** `{config.ENV_MODE.upper()}`\n") # Appends the environment mode (e.g., TEST or PROD) from config.
    report_content.append("---\n") # Appends a horizontal rule for separation.

    # --- 1. Project Overview ---
    report_content.append( # Appends a section header for Project Overview.
        "## 1. Project Overview (CRISP-DM Phase 1: Business Understanding)\n"
    )
    report_content.append( # Appends a description of the project's main objective.
        "This project aims to develop a robust machine learning model for identifying and classifying various Distributed Denial of Service (DDoS) and Botnet attacks within network traffic from the CIC-IDS2017 dataset. The ultimate goal is to enhance network security and enable proactive threat response.\n"
    )
    report_content.append("### Big Data Challenges Addressed:\n") # Appends a subsection header for Big Data challenges.
    report_content.append( # Describes how the 'Volume' challenge is addressed.
        "- **Volume:** Handled large dataset sizes through efficient loading, memory optimization (dtype conversion), and controlled oversampling (SMOTE), and strategic data sampling during development."
    )
    report_content.append( # Describes how the 'Variety' challenge is addressed.
        "- **Variety:** Addressed diverse feature types (numerical, categorical) and multiple attack labels through comprehensive preprocessing, advanced feature engineering, and multi-class classification models."
    )
    report_content.append( # Describes how the 'Velocity' challenge is conceptually addressed.
        "- **Velocity (conceptual/acceleration):** Focused on efficient model training using optimized algorithms (LightGBM, XGBoost with GPU) and fast iteration loops (Test Mode) to lay groundwork for rapid model updates or real-time inference.\n"
    )

    # --- 2. Data Understanding Summary ---
    report_content.append("## 2. Data Understanding Summary (CRISP-DM Phase 2)\n") # Appends a section header for Data Understanding.
    report_content.append( # Describes the dataset used.
        "The CIC-IDS2017 dataset, a benchmark for intrusion detection, was used. It comprises simulated network traffic with labeled flows.\n"
    )

    df_initial = load_dataframe(config.PROCESSED_DATA_DIR, "data_understood.csv") # Attempts to load the intermediate DataFrame from the Data Understanding phase.
    if df_initial is not None: # Checks if the DataFrame was successfully loaded.
        report_content.append( # Reports the initial volume of data.
            f"- **Initial Data Volume:** Concatenated data consisted of {len(df_initial):,} rows and {len(df_initial.columns)} columns.\n"
        )

        label_counts = df_initial[config.TARGET_COLUMN].value_counts(normalize=True) # Calculates the normalized value counts for the target column.
        report_content.append( # Reports on class distribution and imbalance.
            f"- **Class Distribution (Top 5):** Highlights severe class imbalance.\n"
        )
        report_content.append("```\n") # Starts a Markdown code block for the data.
        report_content.append(label_counts.head().to_string()) # Appends the string representation of the top 5 label counts.
        report_content.append("\n```\n") # Ends the Markdown code block.
        report_content.append( # Provides context on the majority class percentage.
            f"  (The majority '{config.BENIGN_LABEL}' class represents {label_counts.iloc[0] * 100:.2f}% of traffic, necessitating imbalance handling.)\n"
        )

        report_content.append( # Reports on initial memory usage after dtype optimization.
            "- **Initial Memory Usage (after dtype optimization):** Approximately {:.2f} MB.\n".format(
                df_initial.memory_usage(deep=True).sum() / (1024**2)
            )
        )

        report_content.append( # Reports on data quality issues identified.
            "- **Data Quality Issues Identified:** Missing values in 'Flow Bytes/s' and 'Flow Packets/s', constant columns (removed), and highly correlated features were initially identified.\n"
        )
        report_content.append( # Refers to specific plot files for visual details.
            "  (Refer to `reports/missing_values_percentage.png`, `reports/highly_correlated_features_heatmap.png`, and `reports/label_distribution_raw.png` for visual details.)\n"
        )
    else:
        report_content.append( # Reports if the data understanding summary is not fully available.
            "- *Data Understanding summary not fully available (intermediate `data_understood.csv` could not be loaded).*\n"
        )

    # --- 3. Data Preparation Summary ---
    report_content.append("## 3. Data Preparation Summary (CRISP-DM Phase 3)\n") # Appends a section header for Data Preparation.
    report_content.append( # Introduces the scope of data preprocessing.
        "The data underwent comprehensive preprocessing, validation, and feature engineering:\n"
    )
    report_content.append( # Describes automated data validation.
        "- **Automated Data Validation:** Ensured incoming data adhered to expected schema and types for robustness.\n"
    )
    report_content.append( # Describes duplicate removal.
        "- **Duplicate Removal:** Identical records were removed to prevent model overfitting.\n"
    )
    report_content.append( # Describes missing value imputation and infinity handling.
        "- **Missing Value Imputation:** Missing numerical values imputed using `IterativeImputer` (model-based), and categorical using `SimpleImputer` (most frequent). Infinite values were also handled. Imputed integer-like floats were carefully downcasted back to integers to preserve data type integrity.\n"
    )
    report_content.append("- **Advanced Feature Engineering:**\n") # Sub-header for advanced feature engineering.
    report_content.append( # Describes the computation of statistical moments.
        f"  - Statistical moments (skew, kurtosis) were computed for {len(config.FE_MOMENT_FEATURES)} key numerical features, providing insights into their distribution shapes.\n"
    )
    report_content.append( # Describes the creation of domain-specific ratios.
        "  - Domain-specific ratios (e.g., `Flow_Bytes_Per_Packet`, `Fwd_Bwd_Packet_Ratio`) were created to capture more nuanced network flow patterns.\n"
    )
    if len(config.FE_POLYNOMIAL_FEATURES) > 1 and config.FE_POLYNOMIAL_DEGREE > 1: # Checks if polynomial features were generated.
        report_content.append( # Describes the generation of polynomial features.
            f"  - Polynomial features (up to degree {config.FE_POLYNOMIAL_DEGREE}) were generated from important base features ({', '.join(config.FE_POLYNOMIAL_FEATURES[:3])}...), creating new interaction terms to potentially capture non-linear relationships. This can significantly increase feature dimensionality, but offers richer model inputs.\n"
        )
    else:
        report_content.append( # Reports if polynomial feature generation was skipped.
            "  - Polynomial feature generation was skipped (either not enough base features or degree not > 1).\n"
        )

    report_content.append( # Describes outlier detection and treatment.
        "- **Outlier Treatment:** `IsolationForest` was used for detecting outliers, which were then capped at the 1st/99th percentiles for mitigation.\n"
    )
    report_content.append( # Describes target variable encoding.
        "- **Target Encoding:** Attack labels (`Label` column) were converted to numerical format using `LabelEncoder`.\n"
    )
    report_content.append( # Describes sophisticated feature selection.
        f"- **Sophisticated Feature Selection:** A `SelectFromModel` (using a RandomForest estimator) was employed to automatically select the most impactful features. The threshold used was `{config.FEATURE_SELECTION_THRESHOLD}`. This step is crucial for managing 'Volume' and focusing on relevant features.\n"
    )

    report_content.append( # Describes advanced class imbalance handling using BorderlineSMOTE.
        f"- **Advanced Class Imbalance Handling:** Used `BorderlineSMOTE` (an advanced variant of SMOTE) to oversample minority classes in the training data, synthesizing samples from near the decision boundary to address severe class imbalance. The `TARGET_SAMPLES_PER_MINORITY_CLASS` was set to {config.TARGET_SAMPLES_PER_MINORITY_CLASS:,} samples per eligible minority class, allowing control over the synthetic data volume. This directly addresses 'Variety' and 'Volume' challenges by balancing class representation for robust model training.\n"
    )

    if config.ENV_MODE == "test": # Checks if the environment mode is 'test'.
        report_content.append( # Describes data sampling for test mode.
            f"- **Test Mode Data Sampling:** For rapid development and debugging, the resampled training data was further sampled down to {config.TEST_MODE_TRAIN_SAMPLE_RATIO * 100:.2f}% of its size. This significantly reduced training times, directly managing 'Volume' in the inner loop of experimentation. **(Important: Final training should be conducted in `prod` mode for optimal model performance).**\n"
        )
    else:  # ENV_MODE == 'prod'
        report_content.append( # Describes full data training for production mode.
            "- **Full Data Training:** The models were trained on the full, processed, and balanced dataset, leveraging the complete information available for optimal production performance.\n"
        )

    # --- 4. Modeling Summary ---
    report_content.append("## 4. Modeling Summary (CRISP-DM Phase 4)\n") # Appends a section header for Modeling.
    report_content.append( # Introduces the types of models trained.
        "Multiple advanced ensemble-based classifiers were trained and tuned within a robust scikit-learn pipeline structure:\n"
    )
    report_content.append( # Lists the selected models and their rationale.
        "- **Models Selected:** RandomForestClassifier, XGBClassifier, LightGBMClassifier. These were chosen for their strong performance on tabular data and ability to handle complex patterns.\n"
    )
    report_content.append( # Describes hyperparameter tuning methodology.
        f"- **Hyperparameter Tuning:** `RandomizedSearchCV` was used for efficient exploration of the hyperparameter space across {config.CV_FOLDS} cross-validation folds. The `n_iter_search` was set to {config.CURRENT_SETTINGS['N_ITER_SEARCH_MODELS']} for each model, exploring broader parameter ranges (as configured in `config.py` for `{config.ENV_MODE}` mode). This method balances thoroughness with computational efficiency.\n"
    )
    report_content.append( # Describes probability calibration.
        "- **Probability Calibration:** `CalibratedClassifierCV` was applied for models that support it, ensuring that the predicted probabilities are reliable and interpretable. This is crucial for applications requiring confidence scores.\n"
    )
    report_content.append( # Describes GPU acceleration.
        "- **GPU Acceleration:** XGBoost (and LightGBM, if compiled with GPU support) leveraged the available CUDA-enabled GPU for significantly faster training times, directly addressing the 'Velocity' challenge in model building. XGBoost specifically used the `'hist'` tree method with `'depthwise'` grow policy for efficient GPU usage.\n"
    )
    report_content.append( # Describes ensemble modeling.
        "- **Ensemble Modeling (VotingClassifier):** A `VotingClassifier` (using soft voting to average probabilities) was trained to combine the strengths of the individual best models, aiming for superior and more robust generalization.\n"
    )

    # --- 5. Evaluation Summary ---
    report_content.append("## 5. Evaluation Summary (CRISP-DM Phase 5)\n") # Appends a section header for Evaluation.
    report_content.append( # Introduces the evaluation process and states that performance metrics will be summarized.
        "Models were rigorously evaluated on a dedicated test set (30% of original data), which was not seen during training or hyperparameter tuning. The following table summarizes key performance metrics:\n"
    )

    summary_file = os.path.join(config.REPORTS_DIR, "model_performance_summary.csv") # Constructs the path to the model performance summary CSV.
    if os.path.exists(summary_file): # Checks if the summary file exists.
        try: # Begins a try block to load and append the summary.
            metrics_df = pd.read_csv(summary_file, index_col="model") # Reads the summary CSV into a DataFrame, setting 'model' as index.
            report_content.append("### Comparative Model Performance:\n") # Appends a sub-header for comparative performance.
            report_content.append("```\n") # Starts a Markdown code block.
            report_content.append(metrics_df.round(4).to_string()) # Appends the string representation of the metrics DataFrame, rounded.
            report_content.append("\n```\n") # Ends the Markdown code block.
            report_content.append( # Refers to a visual comparison plot.
                "\n(See `reports/f1_score_comparison.png` for a visual comparison of Macro F1-scores.)\n"
            )

            # Detailed analysis points (driven by log output and performance metrics)
            report_content.append("### Detailed Analysis:\n") # Appends a sub-header for detailed analysis.

            best_model_name_overall = metrics_df["f1_macro"].idxmax() # Identifies the model with the highest macro F1-score.
            best_f1_macro_overall = metrics_df["f1_macro"].max() # Gets the maximum macro F1-score.
            report_content.append( # Reports the overall best performing model.
                f"- **Overall Best Performer:** The **{best_model_name_overall}** model achieved the highest Macro F1-score of **{best_f1_macro_overall:.4f}**, demonstrating its strong capability in detecting various attack types while maintaining balance across classes.\n"
            )

            if "XGBoost" in metrics_df.index: # Checks if XGBoost metrics are available.
                xgboost_f1_macro = metrics_df.loc["XGBoost", "f1_macro"] # Gets XGBoost's macro F1-score.
                xgboost_accuracy = metrics_df.loc["XGBoost", "accuracy"] # Gets XGBoost's accuracy.
                report_content.append( # Reports on XGBoost's performance.
                    f"- **XGBoost:** Maintained strong performance (Macro F1: {xgboost_f1_macro:.4f}, Accuracy: {xgboost_accuracy:.4f}). Its inherent strengths in gradient boosting and successful GPU acceleration contributed to its efficiency and high scores.\n"
                )
                report_content.append( # Refers to detailed XGBoost reports.
                    "  (Detailed report: `reports/xgboost_classification_report.txt`; Confusion matrices: `reports/xgboost_confusion_matrix_raw.png`, `reports/xgboost_confusion_matrix_normalized.png`; Feature importances: `reports/xgboost_feature_importances.png`.)\n"
                )

            if "LightGBM" in metrics_df.index: # Checks if LightGBM metrics are available.
                lightgbm_f1_macro = metrics_df.loc["LightGBM", "f1_macro"] # Gets LightGBM's macro F1-score.
                lightgbm_accuracy = metrics_df.loc["LightGBM", "accuracy"] # Gets LightGBM's accuracy.
                report_content.append( # Reports on LightGBM's performance.
                    f"- **LightGBM:** Also showed competitive performance (Macro F1: {lightgbm_f1_macro:.4f}, Accuracy: {lightgbm_accuracy:.4f}), often being faster than XGBoost on larger datasets. Its histogram-based algorithms are efficient for 'Volume' challenges.\n"
                )
                report_content.append( # Refers to detailed LightGBM reports.
                    "  (Detailed report: `reports/lightgbm_classification_report.txt`; Confusion matrices: `reports/lightgbm_confusion_matrix_raw.png`, `reports/lightgbm_confusion_matrix_normalized.png`; Feature importances: `reports/lightgbm_feature_importances.png`.)\n"
                )

            if "Ensemble" in metrics_df.index: # Checks if Ensemble metrics are available.
                ensemble_f1_macro = metrics_df.loc["Ensemble", "f1_macro"] # Gets Ensemble's macro F1-score.
                report_content.append( # Reports on Ensemble model's performance.
                    f"- **Ensemble (VotingClassifier):** Achieved robust performance (Macro F1: {ensemble_f1_macro:.4f}). While not always surpassing the individual best model in this test run, ensembles generally enhance generalization and stability. The combination of diverse models often smooths out individual prediction errors.\n"
                )
                report_content.append( # Refers to detailed Ensemble reports.
                    "  (Detailed report: `reports/ensemble_classification_report.txt`; Confusion matrices: `reports/ensemble_confusion_matrix_raw.png`, `reports/ensemble_confusion_matrix_normalized.png`.)\n"
                )

            report_content.append( # Appends a sub-header for challenging classes.
                "### Handling Challenging Classes (Extremely Rare Classes):\n"
            )
            report_content.append( # Discusses the challenges with extremely rare classes like 'Infiltration' and 'Heartbleed'.
                "Classes like 'Infiltration' and 'Heartbleed' remained challenging for all models due to their extremely low sample counts in the original dataset. While advanced resampling techniques like BorderlineSMOTE help during training by creating synthetic samples, their sparse representation in the test set (e.g., 11 for Infiltration, 3 for Heartbleed) makes robust evaluation and achieving high precision for these specific categories inherently difficult. This highlights a persistent challenge in detecting ultra-rare, potentially critical events in real-world network security scenarios.\n"
            )

            report_content.append( # Appends a sub-header for model interpretability.
                "### Model Interpretability & Explainable AI (XAI):\n"
            )
            report_content.append( # Discusses SHAP plots for interpretability.
                f"SHAP (SHapley Additive exPlanations) plots were configured to provide local and global interpretability for tree-based models (when running in `{config.ENV_MODE}` mode and sufficient data is available). These plots visually explain how individual features contribute positively or negatively to a specific prediction, which is crucial for security analysts to understand and trust the model's decisions. (See `reports/*shap_summary*.png`).\n"
            )
            report_content.append( # Discusses Precision-Recall curves.
                "Precision-Recall (PR) curves were also generated. For highly imbalanced datasets like network intrusion, PR curves are often more informative than ROC curves, as they highlight the trade-off between precision and recall specifically for the positive (attack) classes. This allows focusing on minimizing false negatives for critical events. (See `reports/*_pr_curve_ovr.png` for visual details).\n"
            )

            report_content.append("### Confusion Matrix Normalization:\n") # Appends a sub-header for confusion matrix normalization.
            report_content.append( # Explains the utility of normalized confusion matrices.
                "Both raw and row-normalized confusion matrices were generated for each model. Normalized matrices (e.g., `reports/*_confusion_matrix_normalized.png`) are particularly useful for understanding classifier performance per class by showing recall rates (percentage of actual instances correctly predicted), regardless of class imbalance in the test set. This provides a clearer picture of how well each class is truly detected.\n"
            )

        except Exception as e: # Catches any exception during loading/processing of summary file.
            logger.error( # Logs an error if the summary CSV cannot be loaded.
                f"Error loading model_performance_summary.csv for report: {e}",
                exc_info=True,
            )
            report_content.append( # Appends a message indicating an error in loading summary.
                "- *Error loading model performance summary for detailed analysis.*\n"
            )
    else:
        report_content.append( # Reports if the model performance summary was not found.
            "- *Model performance summary not found (ensure project runs to completion).* \n"
        )

    # --- 6. Deployment Considerations (CRISP-DM Phase 6) ---
    report_content.append("## 6. Deployment Considerations (CRISP-DM Phase 6)\n") # Appends a section header for Deployment Considerations.
    report_content.append( # Introduces crucial aspects for production deployment.
        "For a production deployment, the following aspects would be crucial:\n"
    )
    report_content.append( # Discusses Real-time Inference API.
        "- **Real-time Inference API:** Implement a dedicated service (e.g., using FastAPI) to expose the trained model as an API endpoint, allowing seamless, low-latency integration with live network monitoring tools. This directly addresses 'Velocity' requirements.\n"
    )
    report_content.append( # Discusses containerization using Docker.
        "- **Containerization (Docker):** Package the entire application (code, dependencies, environment) into a portable Docker container. This ensures consistent and reproducible deployment across various environments (development, staging, production).\n"
    )
    report_content.append( # Discusses Model Registry & Versioning.
        "- **Model Registry & Versioning (e.g., MLflow):** Use tools like MLflow to track all trained model versions, their associated metrics, parameters, and data sources. This is vital for managing the model lifecycle, enabling quick rollbacks, A/B testing in production, and continuous improvement.\n"
    )
    report_content.append( # Discusses Continuous Monitoring.
        "- **Continuous Monitoring:** Implement robust monitoring for model performance (accuracy, latency, data/concept drift) in production. Alerts would trigger automatic retraining or manual intervention when degradation is detected.\n"
    )
    report_content.append( # Discusses Scalable Data Processing & Inference.
        "- **Scalable Data Processing & Inference:** For truly massive, high-velocity data streams (terabytes/petabytes), integration with distributed computing frameworks (e.g., Apache Spark, Dask) would be necessary for both training and high-throughput inference.\n"
    )

    # --- 7. Conclusion and Future Work ---
    report_content.append("## 7. Conclusion and Future Work\n") # Appends a section header for Conclusion and Future Work.
    report_content.append( # Summarizes the project's success.
        "This project successfully established a robust and professional machine learning pipeline for DDoS and Botnet attack detection on the CIC-IDS2017 dataset, effectively addressing key Big Data challenges. The advanced feature engineering, sophisticated imbalance handling, and optimized model training contributed to strong performance. The pipeline is designed for modularity, reproducibility, and extensibility.\n"
    )
    report_content.append("### Future Work Enhancements:\n") # Sub-header for future work.
    report_content.append( # Discusses Full-Scale Training.
        f"- **Full-Scale Training:** The immediate next step for a production-ready model is to execute the final pipeline in `prod` mode, utilizing full data and expanded hyperparameter search ranges for optimal performance.\n"
    )
    report_content.append( # Discusses Advanced Imbalance Strategies.
        "- **Advanced Imbalance Strategies:** Further explore techniques beyond the current resampling methods, or implement custom cost-sensitive learning integrated into the model's loss function to explicitly favor detection of critical attacks.\n"
    )
    report_content.append( # Discusses Deep Learning Architectures.
        "- **Deep Learning Architectures:** Investigate Recurrent Neural Networks (RNNs) like LSTMs or Transformer models to explicitly capture temporal dependencies and sequential patterns in network flows, potentially leading to even more sophisticated attack recognition.\n"
    )
    report_content.append( # Discusses Automated Retraining & MLOps Pipeline.
        "- **Automated Retraining & MLOps Pipeline:** Implement a full MLOps CI/CD pipeline for automatic model retraining, testing, and deployment upon performance degradation, data drift, or new attack signature availability.\n"
    )
    report_content.append( # Discusses Explainable AI Refinement.
        "- **Explainable AI Refinement:** Further enhance SHAP integration for interactive dashboards or more targeted explanations for specific attack types, making the model's decisions even more transparent to security analysts. This includes exploring other XAI techniques beyond SHAP, suchs as LIME or ELI5.\n"
    )
    report_content.append( # Discusses Robustness Testing.
        "- **Robustness Testing:** Conduct rigorous testing of the deployed model's inference speed and resource consumption under high-throughput live network conditions. Also, investigate adversarial attacks and model resistance to small, malicious perturbations in input features. Implement data and concept drift detection mechanisms to ensure model relevance over time.\n"
    )
    report_content.append( # Discusses Unsupervised Anomaly Detection.
        "- **Unsupervised Anomaly Detection:** Develop a complementary unsupervised anomaly detection module to identify entirely novel (zero-day) attack patterns that doesn't fit known classifications, crucial for handling evolving threats.\n"
    )

    # Save the report
    with open(report_filepath, "w") as f: # Opens the report file in write mode.
        f.writelines(report_content) # Writes all lines of content to the file.
    logger.info(f"Project report generated and saved to {report_filepath}") # Logs that the report was generated and saved.