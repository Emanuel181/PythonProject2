# main.py

import argparse
import os

import src.config as config  # Import config as 'config' to access set_env_mode

# Import setup_logging first to ensure it's defined
from src.utils import load_dataframe, load_object, setup_logging

# --- CRITICAL: Environment Mode and Initial Logger Setup ---
# This section MUST run before any other project modules are imported
# or before other loggers (like those in business_understanding.py etc.) are initialized.

# 1. Temporarily parse only the --env argument to determine the mode.
#    add_help=False prevents argparse from exiting if other args are missing at this stage.
parser_temp = argparse.ArgumentParser(add_help=False)
parser_temp.add_argument("--env", default="test", choices=["test", "prod"])
temp_args, _ = (
    parser_temp.parse_known_args()
)  # parse_known_args allows parsing a subset of arguments

# 2. Set the global environment mode in config.py based on the parsed argument.
#    This updates all dynamic settings in the config module.
config.set_env_mode(temp_args.env)

# 3. Now that config.LOG_LEVEL is correctly set for the chosen environment,
#    initialize the root logger. This ensures all subsequent loggers across modules
#    will inherit the correct level.
logger = setup_logging(log_level=config.LOG_LEVEL)

# --- END CRITICAL SECTION ---

# Now, import other project modules. Their module-level loggers will now
# correctly inherit the configuration from the root logger established above.
from src.business_understanding import business_understanding_phase
from src.data_preparation import data_preparation_phase
from src.data_understanding import data_understanding_phase
from src.evaluation import evaluation_phase
from src.modeling import modeling_phase
from src.report_generator import generate_full_report


def main():
    """
    Main function to orchestrate the entire CRISP-DM lifecycle for the
    DDoS/Botnet detection project. It executes each phase sequentially,
    passing necessary data between them and handling potential failures
    to ensure a robust pipeline. Command-line arguments provide flexibility
    in running specific phases and controlling development/test mode.
    """
    # Re-parse all arguments here for completeness, including phases and skip_report.
    # The --env argument will be parsed again, but config.set_env_mode is idempotent
    # regarding the *first* setting of the mode.
    parser = argparse.ArgumentParser(
        description="Run the DDoS/Botnet detection ML project phases."
    )
    parser.add_argument(
        "--phases",
        nargs="*",
        default=["all"],
        help='Specify CRISP-DM phases to run (e.g., "data_prep modeling evaluation"). '
        'Use "all" to run all phases. Available phases: business_understanding, '
        "data_understanding, data_preparation, modeling, evaluation, report. "
        "Phases will attempt to load prior outputs if skipped.",
    )
    parser.add_argument(
        "--env",
        default="test",
        choices=["test", "prod"],
        help='Set environment mode: "test" for fast runs (small samples, fewer iterations), '
        '"prod" for full training (full data, more iterations).',
    )
    parser.add_argument(
        "--skip_report",
        action="store_true",
        help="Skip automated project report generation at the end.",
    )

    args = parser.parse_args()  # Parse arguments again (they are the same)

    # Logging messages to confirm current settings, which are now correctly set.
    logger.info(f"Project running in '{config.ENV_MODE}' environment mode.")
    logger.info(f"Current TEST_MODE_ACTIVE: {config.TEST_MODE}")
    logger.info(f"Current CV_FOLDS: {config.CV_FOLDS}")
    logger.info(
        f"Current N_ITER_SEARCH_MODELS: {config.CURRENT_SETTINGS['N_ITER_SEARCH_MODELS']}"
    )

    run_all_phases = "all" in args.phases

    logger.info("--- Starting CRISP-DM Project Lifecycle ---")

    # --- Phase 1: Business Understanding ---
    if run_all_phases or "business_understanding" in args.phases:
        business_understanding_phase()

    # --- Phase 2: Data Understanding ---
    df_raw = None
    if run_all_phases or "data_understanding" in args.phases:
        logger.info("\n--- Initiating Data Understanding Phase ---")
        df_raw = data_understanding_phase()
        if df_raw is None or df_raw.empty:
            logger.error(
                "Data Understanding phase failed or returned empty DataFrame. Aborting project."
            )
            return
    else:
        logger.info(
            "\n--- Skipping Data Understanding Phase, attempting to load 'data_understood.csv' ---"
        )
        df_raw = load_dataframe(config.PROCESSED_DATA_DIR, "data_understood.csv")
        if df_raw is None:
            logger.error(
                "Skipping Data Understanding, but 'data_understood.csv' not found. Cannot proceed. Aborting."
            )
            return

    # --- Phase 3: Data Preparation ---
    X_train, X_test, y_train, y_test, label_encoder_classes = (
        None,
        None,
        None,
        None,
        None,
    )
    if run_all_phases or "data_preparation" in args.phases:
        logger.info("\n--- Initiating Data Preparation Phase ---")
        X_train, X_test, y_train, y_test, label_encoder_classes = (
            data_preparation_phase(df_raw)
        )
        if X_train is None:
            logger.error("Data Preparation phase failed. Aborting project.")
            return
    else:
        logger.info(
            "\n--- Skipping Data Preparation Phase, attempting to load processed data ---"
        )
        X_train = load_dataframe(config.PROCESSED_DATA_DIR, "X_train.csv")
        y_train_df = load_dataframe(config.PROCESSED_DATA_DIR, "y_train.csv")
        y_train = y_train_df["Label_Encoded"] if y_train_df is not None else None
        X_test = load_dataframe(config.PROCESSED_DATA_DIR, "X_test.csv")
        y_test_df = load_dataframe(config.PROCESSED_DATA_DIR, "y_test.csv")
        y_test = y_test_df["Label_Encoded"] if y_test_df is not None else None
        label_encoder_obj = load_object(config.MODELS_DIR, "label_encoder.pkl")
        label_encoder_classes = (
            label_encoder_obj.classes_ if label_encoder_obj else None
        )

        if any(
            item is None
            for item in [X_train, y_train, X_test, y_test, label_encoder_classes]
        ):
            logger.error(
                "Skipping Data Preparation, but necessary processed data not found. Cannot proceed. Aborting."
            )
            return

    # --- Phase 4: Modeling ---
    trained_models = {}
    if run_all_phases or "modeling" in args.phases:
        logger.info("\n--- Initiating Modeling Phase ---")
        trained_models = modeling_phase(X_train, y_train)
        if not trained_models:
            logger.error(
                "Modeling phase failed. No models were trained. Aborting project."
            )
            return
    else:
        logger.info(
            "\n--- Skipping Modeling Phase, attempting to load pre-trained models ---"
        )
        for model_name_key in config.CLASSIFIERS_TO_TRAIN.keys():
            model = load_object(
                config.MODELS_DIR, f"{model_name_key.lower()}_best_pipeline.pkl"
            )
            if model:
                trained_models[model_name_key] = model
        ensemble_model = load_object(
            config.MODELS_DIR, "ensemble_voting_classifier.pkl"
        )
        if ensemble_model:
            trained_models["Ensemble"] = ensemble_model

        if not trained_models:
            logger.error(
                "Skipping Modeling, but no pre-trained models found. Cannot proceed. Aborting."
            )
            return

    # --- Phase 5: Evaluation ---
    if run_all_phases or "evaluation" in args.phases:
        logger.info("\n--- Initiating Evaluation Phase ---")
        evaluation_phase(trained_models, X_test, y_test, label_encoder_classes)

    # --- Automated Report Generation ---
    if run_all_phases or "report" in args.phases:
        if not args.skip_report:
            logger.info("\n--- Initiating Automated Report Generation ---")
            generate_full_report()
        else:
            logger.info(
                "Skipping automated report generation as requested via --skip_report."
            )
    else:
        logger.info(
            "Skipping automated report generation (phase 'report' not requested)."
        )

    logger.info("--- CRISP-DM Project Lifecycle Completed Successfully ---")


if __name__ == "__main__":
    main()

