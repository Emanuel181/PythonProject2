import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import logging
import io
from contextlib import redirect_stdout, redirect_stderr
import typing
import glob
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import types # Added this import statement

# --- Initial Imports and Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress initial logs from libraries (e.g., from sklearn.experimental)
_stdout_buffer = io.StringIO()
_stderr_buffer = io.StringIO()
with redirect_stdout(_stdout_buffer), redirect_stderr(_stderr_buffer):
    # These imports might produce logs, so we redirect them during import
    try:
        from sklearn.experimental import enable_iterative_imputer
        import src.config as config
        from src.utils import setup_logging, load_dataframe, load_object # Ensure load_dataframe and load_object are imported
        from src.business_understanding import business_understanding_phase
        from src.data_understanding import data_understanding_phase
        from src.data_preparation import data_preparation_phase
        from src.modeling import modeling_phase
        from src.evaluation import evaluation_phase
        from src.report_generator import generate_full_report
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer, IterativeImputer
        from sklearn.feature_selection import SelectFromModel
        from sklearn.pipeline import Pipeline
    except ImportError as e:
        st.error(f"Error importing necessary modules. Please ensure your 'src' directory is correctly set up and all dependencies are installed. Error: {e}")
        st.stop()


# --- Streamlit Setup ---
st.set_page_config(
    layout="wide",
    page_title="Intrusion Detection System Analytics Platform",
    page_icon="🛡️"
)


# --- High-Contrast CSS Styling ---
def load_css():
    """Loads custom CSS for a high-contrast, dark theme."""
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        /* Main Theme */
        .stApp {
            background-color: #0E1117; /* Dark background */
            color: #FAFAFA; /* Light text */
        }
        .main .block-container {
            padding-top: 2rem; padding-bottom: 2rem;
            padding-left: 3rem; padding-right: 3rem;
        }
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0E1117;
            border-right: 1px solid #334155; /* Subtle border */
        }
        /* High-Contrast Card Effect */
        .card {
            background: #1E293B; /* Slightly lighter dark background */
            border-radius: 12px;
            border: 1px solid #334155;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow */
        }
        /* Metric Styling */
        .stMetric {
            background-color: #334155; /* Darker blue-gray */
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            color: #FAFAFA;
        }
        /* Title and Header Styling */
        h1, h2, h3, h4, h5, h6 { color: #FAFAFA; font-weight: bold; }
        /* Insight Card */
        .insight-card {
            background-color: #1E293B;
            border-left: 5px solid #636EFA; /* A subtle accent color */
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .insight-card p { font-size: 16px; line-height: 1.6; }
        /* Legend Styling */
        .legend-container {
            background-color: #1E293B;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px; /* Spacing between legend items */
            align-items: center;
        }
        .legend-item {
            display: flex;
            align-items: center;
            font-size: 14px;
        }
        .legend-color-box {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 8px;
            border: 1px solid #334155; /* Add a subtle border to color boxes */
        }
    </style>
    """, unsafe_allow_html=True)


load_css()

# --- Logging and Phase Execution Setup ---
# Clear existing handlers to prevent duplicate logs if rerunning
for handler in list(logging.getLogger().handlers):
    logging.getLogger().removeHandler(handler)

def setup_standard_logging(log_level: typing.Optional[str] = None) -> logging.Logger:
    """Configures a standard logging setup for the application."""
    return setup_logging(log_level)

# This is a robust way to ensure our custom setup_logging is used
if 'src.utils' in sys.modules:
    sys.modules['src.utils'].setup_logging = setup_standard_logging
logger = logging.getLogger(__name__)


# --- Caching Functions for ML Pipeline Phases ---
@st.cache_resource(show_spinner="Running Business Understanding...")
def run_business_understanding():
    """Executes the Business Understanding phase."""
    business_understanding_phase()

@st.cache_resource(show_spinner="Running Data Understanding...")
def run_data_understanding():
    """Executes the Data Understanding phase and returns raw dataframe."""
    return data_understanding_phase()

@st.cache_resource(show_spinner="Running Data Preparation...")
def run_data_preparation(df_raw):
    """
    Executes the Data Preparation phase.
    Args:
        df_raw (pd.DataFrame): Raw dataframe from data understanding.
    Returns:
        tuple: X_train, X_test, y_train, y_test, le_classes
    """
    return data_preparation_phase(df_raw)

@st.cache_resource(show_spinner="Running Modeling...")
def run_modeling(X_train, y_train, le_classes):
    """
    Executes the Modeling phase.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        le_classes (list): Classes from label encoder.
    Returns:
        dict: Trained models.
    """
    return modeling_phase(X_train, y_train, le_classes)

@st.cache_resource(show_spinner="Running Evaluation...")
def run_evaluation(_m, X_test, y_test, le_classes):
    """
    Executes the Evaluation phase.
    Args:
        _m (dict): Trained models.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        le_classes (list): Classes from label encoder.
    Returns:
        pd.DataFrame: Evaluation metrics.
    """
    return evaluation_phase(_m, X_test, y_test, le_classes)

@st.cache_resource(show_spinner="Running Report Generation...")
def run_report_generation():
    """Generates the full project report."""
    generate_full_report()


# --- Centralized Artifact and Results Loading ---
@st.cache_resource(show_spinner="Loading trained models and artifacts...")
def load_all_artifacts_and_results():
    """
    Loads all necessary trained models and data artifacts from disk.
    Returns:
        dict: A dictionary of loaded artifacts, or None if critical ones are missing.
    """
    artifacts = {}
    try:
        # Load core prediction artifacts using load_object from src.utils
        artifacts['label_encoder'] = load_object(config.MODELS_DIR, 'label_encoder.pkl')
        artifacts['numerical_imputer'] = load_object(config.MODELS_DIR, 'numerical_imputer.pkl')

        # Load available trained models using load_object from src.utils
        available_models = {name: load_object(config.MODELS_DIR, f'{name.lower()}_best_pipeline.pkl')
                            for name in config.CLASSIFIERS_TO_TRAIN.keys()}
        # Attempt to load the ensemble model using load_object from src.utils
        ensemble_model = load_object(config.MODELS_DIR, 'ensemble_voting_classifier.pkl')
        if ensemble_model:
            available_models['Ensemble'] = ensemble_model

        artifacts['available_models'] = {k: v for k, v in available_models.items() if v is not None}

        # Load test data for advanced evaluation visualizations using load_dataframe from src.utils
        artifacts['X_test'] = load_dataframe(config.PROCESSED_DATA_DIR, 'X_test.csv')
        # Check if y_test dataframe was loaded successfully before slicing
        y_test_df = load_dataframe(config.PROCESSED_DATA_DIR, 'y_test.csv')
        artifacts['y_test'] = y_test_df.iloc[:, 0] if y_test_df is not None else None


        # Load cached evaluation metrics if they exist
        metrics_path = os.path.join(config.REPORTS_DIR, 'evaluation_metrics.csv')
        if os.path.exists(metrics_path):
            try:
                temp_metrics_df = pd.read_csv(metrics_path)
                if 'Model' in temp_metrics_df.columns:
                    st.session_state.metrics_df = temp_metrics_df
                else:
                    # Specific message for missing 'Model' column
                    st.error(f"Error: '{metrics_path}' found, but missing the crucial 'Model' column. "
                             f"This indicates an issue with the data generated by the `evaluation_phase` in `src/evaluation.py`. "
                             f"Please ensure `evaluation_phase` correctly produces a DataFrame with a 'Model' column. "
                             f"Reinitializing metrics_df for current session.")
                    st.session_state.metrics_df = pd.DataFrame() # Initialize empty if malformed
            except pd.errors.EmptyDataError:
                st.warning(f"'{metrics_path}' is empty. Reinitializing metrics_df.")
                st.session_state.metrics_df = pd.DataFrame()
            except Exception as read_err:
                st.error(f"Error reading '{metrics_path}': {read_err}. Reinitializing metrics_df.")
                st.session_state.metrics_df = pd.DataFrame()
        else:
            st.session_state.metrics_df = pd.DataFrame() # Initialize empty if not found

        # Load cached prediction results if they exist
        prediction_path = os.path.join(config.REPORTS_DIR, 'last_prediction_results.csv')
        if os.path.exists(prediction_path):
            st.session_state.prediction_results_cache = pd.read_csv(prediction_path)
        else:
            st.session_state.prediction_results_cache = pd.DataFrame() # Initialize empty if not found

        # Check for minimum required artifacts
        if not all(artifacts.get(key) for key in ['label_encoder', 'available_models']) or not artifacts['available_models']:
            st.warning("Not all core artifacts (label encoder, available models) could be loaded. Some features may be disabled.")
            return None # Return None to indicate partial success or failure to load core artifacts
        return artifacts
    except FileNotFoundError:
        st.warning("One or more required files for artifacts could not be found. Please ensure the training pipeline has been run successfully.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading artifacts: {e}")
        logger.exception("Error loading all artifacts.")
        return None


def preprocess_for_prediction(df: pd.DataFrame, numerical_imputer):
    """
    Preprocesses a DataFrame for prediction, handling inf values and imputation.
    Args:
        df (pd.DataFrame): DataFrame to preprocess.
        numerical_imputer: Trained numerical imputer object.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if numerical_imputer:
        # Identify numerical columns that the imputer was trained on and are present in the current DF
        cols_to_impute = [c for c in numerical_imputer.feature_names_in_ if c in df.columns]
        if cols_to_impute:
            # Apply imputation only to relevant numerical columns
            df[cols_to_impute] = numerical_imputer.transform(df[cols_to_impute])
    return df


# --- UI Layout ---
st.title("🛡️ State-Aware Intrusion Detection System Analytics Platform")

# Initialize artifacts in session state if not already loaded
if 'artifacts' not in st.session_state:
    st.session_state.artifacts = load_all_artifacts_and_results()

# Define tabs for navigation
tab_pipeline, tab_prediction, tab_ranking = st.tabs(
    ["🚀 ML Pipeline Execution", "🔍 Attack Prediction & Analysis", "📊 Model Ranking & Evaluation"])

# --- ML Pipeline Execution Tab ---
with tab_pipeline:
    st.header("Project Execution Status")

    with st.sidebar:
        st.header("⚙️ Pipeline Configuration")
        # Environment mode selection (test/prod)
        env_mode = st.radio("Select Environment Mode", ('test', 'prod'), index=0, key="env_mode_radio")

        # Check if environment mode has changed to trigger re-run
        if st.session_state.get('current_env_mode') != env_mode:
            config.set_env_mode(env_mode) # Update config based on selection
            st.session_state.current_env_mode = env_mode
            st.cache_resource.clear() # Clear all cached resources for a fresh start
            st.rerun() # Rerun the app to apply changes

        st.subheader("Controls")

        # Button to run the full ML pipeline
        if st.button("▶️ Run Full Training Pipeline", type="primary", key="run_full_pipeline_btn"):
            st.session_state.run_full_pipeline = True
            # Clear previous run flags and cached data to ensure a fresh start
            for key in ['df_raw', 'X_train', 'X_test', 'y_train', 'y_test', 'le_classes', 'trained_models',
                        'metrics_df', 'prediction_results_cache', 'prediction_results', 'deploy_df']:
                if key in st.session_state:
                    del st.session_state[key]
            st.cache_resource.clear()  # Clear all resource caches
            st.rerun() # Rerun to start pipeline execution

        st.markdown("---")
        # Button to clear all cached results and files
        if st.button("🗑️ Clear All Cached Results", key="clear_cache_btn"):
            # Delete physical report files
            for file_name in ['evaluation_metrics.csv', 'last_prediction_results.csv']:
                file_path = os.path.join(config.REPORTS_DIR, file_name)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        st.success(f"Removed: {file_name}")
                    except OSError as e:
                        st.error(f"Error removing {file_name}: {e}")

            # Clear relevant session state variables
            for key in ['metrics_df', 'prediction_results_cache', 'prediction_results', 'best_model_name',
                        'run_full_pipeline', 'df_raw', 'X_train', 'X_test', 'y_train', 'y_test',
                        'le_classes', 'trained_models', 'artifacts', 'deploy_df']:
                if key in st.session_state:
                    del st.session_state[key]
            st.cache_resource.clear()  # Clear all caches
            st.rerun() # Rerun the app to reflect changes


    # Execute full pipeline based on button click flag
    if st.session_state.get('run_full_pipeline'):
        # Reset the flag immediately to avoid re-running on subsequent refreshes
        st.session_state.run_full_pipeline = False

        # Initialize variables to None
        df_raw, X_train, X_test, y_train, y_test, le_classes, trained_models = [None] * 7
        pipeline_successful = True # Flag to track pipeline success

        # Use st.status for a visible progress indicator
        with st.status("Executing CRISP-DM Pipeline...", expanded=True) as status:
            try:
                # Business Understanding Phase
                status.write("Starting Business Understanding phase...")
                run_business_understanding()
                status.write("✅ Business Understanding complete. Project scope and objectives defined.")

                # Data Understanding Phase
                status.write("Starting Data Understanding phase...")
                df_raw = run_data_understanding()
                st.session_state.df_raw = df_raw  # Store raw data for next steps
                status.write("✅ Data Understanding complete. Raw data loaded and explored.")

                # Data Preparation Phase
                status.write("Starting Data Preparation phase...")
                if df_raw is not None:
                    X_train, X_test, y_train, y_test, le_classes = run_data_preparation(df_raw)
                    # Store processed data and label encoder classes in session state
                    st.session_state.X_train, st.session_state.X_test = X_train, X_test
                    st.session_state.y_train, st.session_state.y_test = y_train, y_test
                    st.session_state.le_classes = le_classes
                    status.write("✅ Data Preparation complete. Data cleaned, transformed, and split.")
                else:
                    raise ValueError("Raw data not available after Data Understanding. Cannot proceed to Data Preparation.")

                # Modeling Phase
                status.write("Starting Modeling phase...")
                if all(k is not None for k in [X_train, y_train, le_classes]):
                    trained_models = run_modeling(X_train, y_train, le_classes)
                    st.session_state.trained_models = trained_models
                    status.write(f"✅ Modeling complete. {len(trained_models)} models trained.")
                else:
                    raise ValueError("Prepared data (X_train, y_train) or label encoder classes not available for Modeling.")

                # Evaluation Phase
                status.write("Starting Evaluation phase...")
                if all(k is not None for k in [trained_models, X_test, y_test, le_classes]):
                    evaluation_metrics_df = run_evaluation(trained_models, X_test, y_test, le_classes)
                    st.session_state.metrics_df = evaluation_metrics_df
                    # Explicitly save metrics to disk for persistence
                    if evaluation_metrics_df is not None and not evaluation_metrics_df.empty:
                        # Ensure the reports directory exists
                        os.makedirs(config.REPORTS_DIR, exist_ok=True)
                        metrics_path = os.path.join(config.REPORTS_DIR, 'evaluation_metrics.csv')
                        evaluation_metrics_df.to_csv(metrics_path, index=False)
                        status.write(f"📊 Evaluation metrics saved to {metrics_path}")
                    else:
                        status.warning("No evaluation metrics were generated or loaded.")

                    status.write("✅ Evaluation complete. Model performance metrics calculated.")
                else:
                    raise ValueError("Trained models or test data not available for Evaluation.")

                # Report Generation Phase
                status.write("Starting Report Generation phase...")
                run_report_generation()
                status.write("✅ Full report generated.")

                # Final status update for success
                status.update(label="Pipeline Execution Complete!", state="complete", expanded=False)
                st.balloons() # Confetti animation for success
                # Reload artifacts after successful pipeline run to ensure prediction tab has latest models
                st.session_state.artifacts = load_all_artifacts_and_results()
                st.success("The full CRISP-DM pipeline has been executed successfully!")

            except Exception as e:
                pipeline_successful = False # Set flag to False on error
                status.update(label="Pipeline Failed!", state="error", expanded=True) # Update status to error
                st.error(f"An error occurred during pipeline execution: {e}")
                logger.exception("CRISP-DM Pipeline failed during execution.") # Log the full exception traceback

    else:
        st.info("Click '▶️ Run Full Training Pipeline' in the sidebar to start the process.")

# --- Attack Prediction & Analysis Tab ---
with tab_prediction:
    st.header("Live Prediction & Anomaly Analysis")
    # Check if artifacts are loaded before allowing prediction functionality
    if not st.session_state.get('artifacts') or not st.session_state.artifacts.get('available_models'):
        st.warning("Could not load models or essential artifacts. Please run the training pipeline first to generate them.");
        st.stop() # Stop execution of this tab if artifacts are missing

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("1. Upload or Load Data")
        col1, col2 = st.columns([2, 1]) # Layout for upload and 'view last' button

        with col1:
            uploaded_file = st.file_uploader("Upload new data for prediction (CSV)", type=["csv"], key="deploy_upload")
        with col2:
            st.write("") # Add some vertical space
            # Button to load previously saved prediction results
            if 'prediction_results_cache' in st.session_state and not st.session_state.prediction_results_cache.empty:
                if st.button("👁️ View Last Prediction Analysis", key="view_last_prediction_btn"):
                    st.session_state.prediction_results = st.session_state.prediction_results_cache
                    st.success("Loaded the last saved analysis.")
            else:
                st.info("No previous prediction analysis found to load.")


        # Handle file upload
        if uploaded_file:
            try:
                original_df = pd.read_csv(uploaded_file)
                original_df.columns = original_df.columns.str.strip() # Strip whitespace from column names
                # Drop unnamed index column if present
                if 'Unnamed: 0' in original_df.columns:
                    original_df = original_df.drop(columns=['Unnamed: 0'])
                st.session_state.deploy_df = original_df # Store original uploaded data
                # Clear previous prediction results if new data is uploaded
                if 'prediction_results' in st.session_state:
                    del st.session_state['prediction_results']
                st.success(f"Successfully loaded {uploaded_file.name} with {len(original_df)} records.")
            except Exception as e:
                st.error(f"Could not read file. Please ensure it's a valid CSV. Error: {e}")
                st.stop() # Stop if file cannot be read

        # Run prediction if new data is uploaded and no results are currently displayed
        if 'deploy_df' in st.session_state and 'prediction_results' not in st.session_state:
            original_df = st.session_state.deploy_df
            st.markdown("---")
            st.subheader("2. Run Prediction on Uploaded Data")
            artifacts = st.session_state.artifacts # Get loaded artifacts

            # Ensure there are models available
            model_options = list(artifacts['available_models'].keys())
            if not model_options:
                st.warning("No trained models found to perform prediction. Please run the ML Pipeline first.")
            else:
                # Select best model by default, or the first available
                best_model_name = st.session_state.get('best_model_name', model_options[0])
                selected_model_name = st.selectbox("Choose a trained model:", model_options,
                                                   index=model_options.index(best_model_name)
                                                   if best_model_name in model_options else 0,
                                                   key="selected_pred_model")

                if st.button("Analyze & Predict Uploaded Data", type="primary", key="run_prediction_btn"):
                    with st.spinner(f"Analyzing traffic with {selected_model_name}..."):
                        model_to_use = artifacts['available_models'][selected_model_name]
                        # Drop 'Label' column if it exists in the uploaded data before prediction
                        df_for_pred = original_df.drop(columns=['Label'], errors='ignore')

                        # Preprocess the DataFrame
                        processed_df = preprocess_for_prediction(df_for_pred.copy(), artifacts['numerical_imputer'])

                        # Ensure columns match training data for consistent prediction
                        # This part of the code expects X_train.csv to be loaded locally or downloaded via utils.load_dataframe
                        # Ensure X_train.csv is also accessible via your server for robust deployment
                        X_train_cols_df = load_dataframe(config.PROCESSED_DATA_DIR, 'X_train.csv')
                        if X_train_cols_df is not None:
                            X_train_cols = X_train_cols_df.columns
                            # Reindex to match training columns, filling missing with 0 and dropping extra
                            processed_df = processed_df.reindex(columns=X_train_cols, fill_value=0)
                            # Ensure numerical columns are of float type as expected by models
                            for col in processed_df.columns:
                                if pd.api.types.is_numeric_dtype(processed_df[col]):
                                    processed_df[col] = processed_df[col].astype(float)
                        else:
                            st.warning("X_train.csv not found or could not be downloaded. Prediction might fail due to column mismatch. Please run the full pipeline or ensure server is configured correctly.")
                            # Attempt to proceed but warn the user

                        try:
                            # Predict probabilities
                            pred_probabilities = model_to_use.predict_proba(processed_df)
                            results_df = original_df.copy() # Start with the original uploaded data
                            # Add predicted attack type using inverse transform from label encoder
                            results_df['Predicted_Attack_Type'] = artifacts['label_encoder'].inverse_transform(
                                np.argmax(pred_probabilities, axis=1))
                            # Add prediction confidence
                            results_df['Confidence'] = np.max(pred_probabilities, axis=1)

                            st.session_state.prediction_results = results_df # Store results in session state
                            # Save results to CSV for future sessions
                            # Ensure the reports directory exists
                            os.makedirs(config.REPORTS_DIR, exist_ok=True)
                            results_df.to_csv(os.path.join(config.REPORTS_DIR, 'last_prediction_results.csv'), index=False)
                            st.success("Analysis complete and results have been cached for future sessions.")
                        except Exception as pred_err:
                            st.error(f"Error during prediction. This could be due to data format issues or model problems. Error: {pred_err}")
                            logger.exception("Prediction failed.")

        st.markdown('</div>', unsafe_allow_html=True) # Close the card div

    # Display analysis dashboard if prediction results are available
    if 'prediction_results' in st.session_state and not st.session_state.prediction_results.empty:
        results_df_full = st.session_state.prediction_results # Use a new variable name to avoid confusion with filtered df

        st.header("3. Analysis Dashboard")

        # Interactive Confidence Threshold Filter
        min_confidence_threshold = st.slider(
            "Minimum Prediction Confidence for Attack Instances",
            min_value=0.0,
            max_value=1.0,
            value=0.7, # Default threshold
            step=0.05,
            help="Adjust to filter attack instances displayed in charts and tables based on prediction confidence."
        )

        # Apply confidence filter
        filtered_results_df = results_df_full[results_df_full['Confidence'] >= min_confidence_threshold].copy() # Added .copy() to avoid SettingWithCopyWarning
        attacks_df = filtered_results_df[filtered_results_df['Predicted_Attack_Type'] != 'BENIGN'].copy()

        # Define color map dynamically based on all classes from label encoder
        artifacts = st.session_state.get('artifacts')
        all_labels = artifacts['label_encoder'].classes_ if artifacts and 'label_encoder' in artifacts else []

        dynamic_color_map = {'BENIGN': '#28a745'}  # Green for BENIGN traffic
        # Use a qualitative color palette for attack types
        attack_colors = px.colors.qualitative.Dark24
        if len(all_labels) > len(attack_colors):
            attack_colors = px.colors.qualitative.Alphabet # Fallback for many labels

        color_idx = 0
        for label in sorted(all_labels): # Sort labels for consistent color assignment
            if label != 'BENIGN':
                dynamic_color_map[label] = attack_colors[color_idx % len(attack_colors)]
                color_idx += 1
        color_map = dynamic_color_map

        # --- Overall Traffic Summary (Prominent Metrics) ---
        st.markdown("---")
        st.subheader("🌐 Network Traffic Overview")
        total_records = len(results_df_full) # Use full data for overall counts
        benign_count = len(results_df_full[results_df_full['Predicted_Attack_Type'] == 'BENIGN'])
        attack_count_unfiltered = total_records - benign_count # Attacks before confidence filter
        attack_count_filtered = len(attacks_df) # Attacks after confidence filter

        col_metrics_1, col_metrics_2, col_metrics_3 = st.columns(3)
        with col_metrics_1:
            st.metric("Total Records Analyzed", f"{total_records:,}")
        with col_metrics_2:
            st.metric("Benign Traffic Instances", f"{benign_count:,}")
        with col_metrics_3:
            st.metric("Detected Attack Instances (Filtered)", f"{attack_count_filtered:,}")
            if attack_count_filtered != attack_count_unfiltered:
                st.markdown(f"<small><i>({attack_count_unfiltered:,} unfiltered attacks)</i></small>", unsafe_allow_html=True)


        st.markdown(f"""
        <div class="insight-card">
            <p><strong>Overall Analysis Summary:</strong></p>
            <p>A total of <strong>{total_records:,}</strong> network traffic records were analyzed. Of these, 
            <strong>{benign_count:,}</strong> ({benign_count / total_records:.2%}) were classified as benign, 
            while <strong>{attack_count_filtered:,}</strong> ({attack_count_filtered / total_records:.2%}) 
            instances were identified as potential attacks (with confidence $\\ge$ {min_confidence_threshold:.2f}).</p>
        </div>
        """, unsafe_allow_html=True)

        # Top Protocols in Total Traffic (in an expander)
        with st.expander("🔍 Detailed Protocol Overview (All Traffic)", expanded=False):
            protocol_cols = ['Protocol', 'proto', 'protocol']
            found_protocol_col_all = next((col for col in protocol_cols if col in results_df_full.columns), None)

            if found_protocol_col_all:
                st.markdown("##### Top Protocols in Total Traffic")
                protocol_counts = results_df_full[found_protocol_col_all].value_counts().head(5).reset_index()
                protocol_counts.columns = ['Protocol', 'Count']
                fig_all_proto = px.bar(protocol_counts, x='Protocol', y='Count',
                                       title="Top Protocols (Overall Traffic)", template="plotly_dark",
                                       color='Count', color_continuous_scale=px.colors.sequential.Plasma)
                st.plotly_chart(fig_all_proto, use_container_width=True)
            else:
                st.info(
                    "Protocol column not found for overall traffic analysis (looked for 'Protocol', 'proto', 'protocol').")

        if attacks_df.empty:
            st.success("✅ No attacks were detected based on the current confidence threshold.")
        else:
            # --- Attack Prediction Insights Expander ---
            with st.expander("📊 Attack Prediction Insights", expanded=True):
                st.markdown("#### Key Metrics and Distributions for Detected Attacks")

                # Insight: Total and Unique Attacks (moved inside)
                total_attacks_filtered = len(attacks_df)
                unique_attack_types_filtered = attacks_df['Predicted_Attack_Type'].nunique()
                st.markdown(f"""
                <div class="insight-card">
                    <p><strong>Total Detected Attack Instances (Filtered):</strong> {total_attacks_filtered}</p>
                    <p><strong>Unique Attack Types Detected (Filtered):</strong> {unique_attack_types_filtered}</p>
                </div>
                """, unsafe_allow_html=True)

                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    # Insight: Attack Type Distribution
                    st.markdown("##### Distribution of Detected Attack Types")
                    attack_counts = attacks_df['Predicted_Attack_Type'].value_counts().reset_index()
                    attack_counts.columns = ['Attack Type', 'Count']
                    fig_attack_dist = px.bar(attack_counts, x='Attack Type', y='Count',
                                             title="Count of Each Detected Attack Type",
                                             color='Attack Type',
                                             color_discrete_map={k: v for k, v in color_map.items() if k != 'BENIGN'},
                                             template="plotly_dark")
                    fig_attack_dist.update_layout(showlegend=False)
                    st.plotly_chart(fig_attack_dist, use_container_width=True)

                with col_chart2:
                    # Insight: Confidence Distribution for Attacks
                    st.markdown("##### Confidence Score Distribution for Attacks")
                    fig_confidence_dist = px.histogram(attacks_df, x='Confidence', nbins=20,
                                                       title="Distribution of Prediction Confidence for Attacks",
                                                       labels={'Confidence': 'Prediction Confidence'},
                                                       color_discrete_sequence=['#636EFA'],
                                                       template="plotly_dark")
                    st.plotly_chart(fig_confidence_dist, use_container_width=True)

                col_chart3, col_chart4 = st.columns(2)
                with col_chart3:
                    # Insight: Average Confidence by Attack Type
                    st.markdown("##### Average Confidence by Attack Type")
                    avg_confidence = attacks_df.groupby('Predicted_Attack_Type')['Confidence'].mean().reset_index()
                    avg_confidence.columns = ['Attack Type', 'Average Confidence']
                    fig_avg_conf = px.bar(avg_confidence, x='Attack Type', y='Average Confidence',
                                          title="Average Confidence for Each Detected Attack Type",
                                          color='Attack Type',
                                          color_discrete_map={k: v for k, v in color_map.items() if k != 'BENIGN'},
                                          template="plotly_dark")
                    fig_avg_conf.update_layout(showlegend=False)
                    st.plotly_chart(fig_avg_conf, use_container_width=True)

                with col_chart4:
                    # Insight: Statistical Summary of Confidence
                    st.markdown("##### Confidence Statistical Summary")
                    conf_summary = attacks_df['Confidence'].describe().to_frame()
                    st.dataframe(conf_summary, use_container_width=True)

                # New Insight: Severity based on Confidence
                st.markdown("##### Attack Severity by Confidence Level")
                if not attacks_df.empty:
                    def get_severity(confidence):
                        if confidence >= 0.9:
                            return 'High'
                        elif confidence >= 0.7:
                            return 'Medium'
                        else:
                            return 'Low'

                    attacks_df['Severity'] = attacks_df['Confidence'].apply(get_severity)
                    severity_counts = attacks_df['Severity'].value_counts().reset_index()
                    severity_counts.columns = ['Severity', 'Count']
                    fig_severity = px.pie(severity_counts, values='Count', names='Severity',
                                          title="Attack Severity by Confidence",
                                          color_discrete_map={'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#17a2b8'},
                                          template="plotly_dark")
                    st.plotly_chart(fig_severity, use_container_width=True)
                else:
                    st.info("No attacks to categorize by severity.")

            # --- Source/Destination IP Analysis ---
            with st.expander("📍 Source & Destination IP Analysis (for Attacks)", expanded=False):
                st.markdown("#### Top Source and Destination IPs in Attack Traffic")

                ip_cols_src = ['Src IP', 'src_ip', 'source_ip']
                found_src_ip_col = next((col for col in ip_cols_src if col in attacks_df.columns), None)

                ip_cols_dst = ['Dst IP', 'dst_ip', 'destination_ip']
                found_dst_ip_col = next((col for col in ip_cols_dst if col in attacks_df.columns), None)

                top_n_ips = st.slider("Show Top N IPs:", min_value=5, max_value=50, value=10, step=5, key="top_n_ips_slider")

                col_ip_1, col_ip_2 = st.columns(2)

                with col_ip_1:
                    if found_src_ip_col:
                        st.markdown(f"##### Top {top_n_ips} Source IPs in Attacks")
                        src_ip_counts = attacks_df[found_src_ip_col].value_counts().head(top_n_ips).reset_index()
                        src_ip_counts.columns = ['Source IP', 'Count']
                        fig_src_ip = px.bar(src_ip_counts, x='Source IP', y='Count',
                                            title=f"Top {top_n_ips} Source IPs (Attack Traffic)", template="plotly_dark",
                                            color='Count', color_continuous_scale=px.colors.sequential.Plotly3)
                        st.plotly_chart(fig_src_ip, use_container_width=True)
                    else:
                        st.info(f"Source IP column not found for analysis (looked for {', '.join(ip_cols_src)}).")

                with col_ip_2:
                    if found_dst_ip_col:
                        st.markdown(f"##### Top {top_n_ips} Destination IPs in Attacks")
                        dst_ip_counts = attacks_df[found_dst_ip_col].value_counts().head(top_n_ips).reset_index()
                        dst_ip_counts.columns = ['Destination IP', 'Count']
                        fig_dst_ip = px.bar(dst_ip_counts, x='Destination IP', y='Count',
                                            title=f"Top {top_n_ips} Destination IPs (Attack Traffic)", template="plotly_dark",
                                            color='Count', color_continuous_scale=px.colors.sequential.Dense)
                        st.plotly_chart(fig_dst_ip, use_container_width=True)
                    else:
                        st.info(f"Destination IP column not found for analysis (looked for {', '.join(ip_cols_dst)}).")

            # --- Protocol & Service Analysis Expander ---
            with st.expander("🔗 Protocol & Service Analysis (for Attacks)", expanded=False):
                st.markdown("#### Network Characteristics of Detected Attacks")

                col_proto_serv_1, col_proto_serv_2 = st.columns(2)

                protocol_cols = ['Protocol', 'proto', 'protocol']
                found_protocol_col = next((col for col in protocol_cols if col in attacks_df.columns), None)

                with col_proto_serv_1:
                    if found_protocol_col:
                        st.markdown("##### Top Protocols Involved in Attacks")
                        attack_protocol_counts = attacks_df[found_protocol_col].value_counts().head(10).reset_index()
                        attack_protocol_counts.columns = ['Protocol', 'Count']
                        fig_attack_proto = px.bar(attack_protocol_counts, x='Protocol', y='Count',
                                                  title="Top Protocols (Attack Traffic)", template="plotly_dark",
                                                  color='Count', color_continuous_scale=px.colors.sequential.Plasma)
                        st.plotly_chart(fig_attack_proto, use_container_width=True)
                    else:
                        st.info(
                            f"Protocol column not found for attack analysis (looked for {', '.join(protocol_cols)}).")

                service_cols = ['Service', 'service']
                found_service_col = next((col for col in service_cols if col in attacks_df.columns), None)

                with col_proto_serv_2:
                    if found_service_col:
                        st.markdown("##### Top Services Targeted by Attacks")
                        attack_service_counts = attacks_df[found_service_col].value_counts().head(10).reset_index()
                        attack_service_counts.columns = ['Service', 'Count']
                        fig_attack_service = px.bar(attack_service_counts, x='Service', y='Count',
                                                    title="Top Services (Attack Traffic)", template="plotly_dark",
                                                    color='Count', color_continuous_scale=px.colors.sequential.Viridis)
                        st.plotly_chart(fig_attack_service, use_container_width=True)
                    else:
                        st.info(f"Service column not found for attack analysis (looked for {', '.join(service_cols)}).")

                dst_port_cols = ['Dst Port', 'dst_port', 'destination_port']
                found_dst_port_col = next((col for col in dst_port_cols if col in attacks_df.columns), None)

                if found_dst_port_col:
                    st.markdown("##### Top 10 Destination Ports in Attacks")
                    dst_port_counts = attacks_df[found_dst_port_col].value_counts().head(10).reset_index()
                    dst_port_counts.columns = ['Destination Port', 'Count']
                    fig_dst_port = px.bar(dst_port_counts, x='Destination Port', y='Count',
                                          title="Top Destination Ports (Attack Traffic)", template="plotly_dark",
                                          color='Count', color_continuous_scale=px.colors.sequential.Magma)
                    st.plotly_chart(fig_dst_port, use_container_width=True)
                else:
                    st.info(f"Destination Port column not found for analysis (looked for {', '.join(dst_port_cols)}).")

            # --- Temporal Analysis Expander ---
            timestamp_cols = ['Timestamp', 'timestamp', 'Time', 'time']
            found_timestamp_col = next((col for col in timestamp_cols if col in filtered_results_df.columns), None)

            if found_timestamp_col:
                with st.expander("⏰ Temporal Analysis of Attacks", expanded=False):
                    st.markdown("#### Attack Trends Over Time")
                    try:
                        # Ensure the timestamp column is in a datetime format
                        filtered_results_df[found_timestamp_col] = pd.to_datetime(filtered_results_df[found_timestamp_col],
                                                                          errors='coerce')
                        # Filter for attacks and drop rows where timestamp conversion failed
                        attacks_df_time = filtered_results_df[filtered_results_df['Predicted_Attack_Type'] != 'BENIGN'].dropna(
                            subset=[found_timestamp_col]).copy()

                        if not attacks_df_time.empty:
                            # Resample to hourly/daily for plotting based on data density
                            # Decide on aggregation level based on the span of the data
                            time_span = attacks_df_time[found_timestamp_col].max() - attacks_df_time[
                                found_timestamp_col].min()

                            if time_span.days > 30:  # If data spans more than a month, aggregate daily
                                attacks_df_time['Date_Period'] = attacks_df_time[found_timestamp_col].dt.to_period('D')
                                time_unit = 'Daily'
                            elif time_span.days > 1:  # If data spans more than a day, aggregate hourly
                                attacks_df_time['Date_Period'] = attacks_df_time[found_timestamp_col].dt.to_period('H')
                                time_unit = 'Hourly'
                            else:  # Otherwise, aggregate by minute for very short spans
                                attacks_df_time['Date_Period'] = attacks_df_time[found_timestamp_col].dt.to_period('T')
                                time_unit = 'Minutely'

                            attack_timeline = attacks_df_time.groupby('Date_Period').size().reset_index(
                                name='Attack Count')
                            attack_timeline['Date_Period'] = attack_timeline[
                                'Date_Period'].dt.to_timestamp()  # Convert back to datetime for Plotly

                            fig_time_series = px.line(attack_timeline, x='Date_Period', y='Attack Count',
                                                      title=f"Attack Count Over Time ({time_unit})",
                                                      labels={'Date_Period': 'Time',
                                                              'Attack Count': 'Number of Attacks'},
                                                      template="plotly_dark")
                            st.plotly_chart(fig_time_series, use_container_width=True)
                        else:
                            st.info("No attacks with valid timestamps for temporal analysis (after confidence filtering).")
                    except Exception as e:
                        st.error(f"Error during temporal analysis: {e}")
                        st.info(
                            f"Ensure the timestamp column ({found_timestamp_col}) is in a recognized datetime format.")
            else:
                st.info(
                    f"Timestamp column not found in data for temporal analysis (looked for {', '.join(timestamp_cols)}).")

            # --- Detailed Results Table Expander ---
            with st.expander("📋 Detailed Results Table", expanded=True):
                st.markdown("#### Full Prediction Results (Filtered by Confidence)")

                # Legend for the table
                st.markdown('<div class="legend-container">', unsafe_allow_html=True)
                st.markdown("<strong>Legend:</strong>", unsafe_allow_html=True)
                for attack_type, color_code in color_map.items():
                    # Function to determine if a color is light or dark (simplified heuristic)
                    def is_light_color(hex_color):
                        hex_color = hex_color.lstrip('#')
                        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
                        # Perceived brightness (Luminance) formula
                        luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
                        return luminance > 0.5  # Threshold for light color


                    text_color = 'black' if is_light_color(color_code) else 'white'

                    st.markdown(f"""
                    <div class="legend-item">
                        <div class="legend-color-box" style="background-color: {color_code};"></div>
                        <span style="color: {text_color};">{attack_type}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


                def highlight_and_style(df_page):
                    style_df = pd.DataFrame('', index=df_page.index, columns=df_page.columns)
                    for i, row in df_page.iterrows():
                        attack_type = row.get('Predicted_Attack_Type', 'BENIGN')
                        color = color_map.get(attack_type, '#6c757d')  # Default grey if type not in map

                        # Determine text color dynamically for contrast
                        def is_light_color_internal(hex_color):
                            hex_color = hex_color.lstrip('#')
                            rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
                            luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
                            return luminance > 0.5

                        text_color = 'black' if is_light_color_internal(color) else 'white'

                        style_df.loc[i] = f'background-color: {color}; color: {text_color};'

                    if 'Confidence' in df_page.columns and not df_page.empty:
                        if pd.api.types.is_numeric_dtype(df_page['Confidence']) and not df_page['Confidence'].empty:
                            # Use iloc to get the index that corresponds to max confidence in the current slice
                            max_conf_idx_in_slice = df_page['Confidence'].idxmax()
                            current_style = style_df.loc[max_conf_idx_in_slice].iloc[0]  # Get existing style string
                            style_df.loc[max_conf_idx_in_slice] = f'{current_style} border: 3px solid #0d6efd;'  # Add border
                    return style_df


                rows_per_page = st.selectbox("Rows per page:", [10, 25, 50, 100], index=1, key="rows_per_page_pred")
                total_pages = (len(filtered_results_df) - 1) // rows_per_page + 1 if len(filtered_results_df) > 0 else 1
                page_number = st.number_input("Page:", min_value=1, max_value=total_pages, value=1,
                                               key="page_number_pred")
                df_page = filtered_results_df.iloc[(page_number - 1) * rows_per_page:page_number * rows_per_page]
                st.dataframe(df_page.style.apply(highlight_and_style, axis=None), use_container_width=True)

            # --- Top N Attack Instances Expander ---
            with st.expander("🏆 Top 10 Most Confident Attack Instances", expanded=False):
                st.markdown("#### High-Confidence Detections")
                if not attacks_df.empty:
                    top_n_attacks = attacks_df.sort_values(by='Confidence', ascending=False).head(10)
                    st.dataframe(top_n_attacks.style.format({'Confidence': '{:.4f}'}), use_container_width=True)
                else:
                    st.info("No attacks detected with sufficient confidence to show top instances.")

with tab_ranking:
    st.header("Model Ranking & Evaluation Deep Dive")
    if 'metrics_df' not in st.session_state or st.session_state.metrics_df.empty:
        st.info("No evaluation data available. Please run the full pipeline to generate or reload existing artifacts.")
    else:
        metrics_df = st.session_state.metrics_df

        if 'Model' not in metrics_df.columns:
            st.error(
                "The 'evaluation_metrics.csv' file is malformed and missing the 'Model' column. Please clear all cached results and re-run the pipeline.")
        else:
            # Metrics used for overall scoring
            metrics_to_score = ['Accuracy', 'F1-Score (macro)', 'Precision (macro)', 'Recall (macro)']
            valid_metrics_to_score = [m for m in metrics_to_score if m in metrics_df.columns]

            sorted_metrics_df = metrics_df.copy()

            # Calculate and add 'Overall Score' if valid metrics exist
            if valid_metrics_to_score:
                scaler = MinMaxScaler()
                # Ensure values are float before scaling
                scores_for_scaling = sorted_metrics_df[valid_metrics_to_score].astype(float)
                sorted_metrics_df['Overall Score'] = np.mean(
                    scaler.fit_transform(scores_for_scaling), axis=1)
                sorted_metrics_df = sorted_metrics_df.sort_values(by="Overall Score", ascending=False).reset_index(
                    drop=True)
            else:
                st.warning("Could not calculate 'Overall Score' as required metrics are missing.")
                # If no valid metrics to score, ensure 'Overall Score' column doesn't cause errors
                if 'Overall Score' not in sorted_metrics_df.columns:
                    sorted_metrics_df['Overall Score'] = np.nan # Add as NaN

            # Determine and display the best model
            if not sorted_metrics_df.empty:
                st.session_state.best_model_name = sorted_metrics_df.iloc[0]['Model']
            else:
                st.session_state.best_model_name = None # No models to rank

            st.success(
                f"🏆 **Recommended Model:** Based on overall performance, the best model is **{st.session_state.get('best_model_name', 'N/A')}**.")

            # Sub-tabs for detailed evaluation
            eval_tabs = st.tabs(["📈 Performance Ranking", "🎭 Confusion Matrix", "✨ Feature Importance", "📊 ROC Curve",
                                 "🚫 Error Analysis"])

            with eval_tabs[0]: # Performance Ranking Tab
                st.markdown("#### Model Performance Metrics")

                all_models = metrics_df['Model'].tolist()
                # Default selection: best model if available, otherwise all models
                default_selected_models = ([st.session_state.get('best_model_name')]
                                           if st.session_state.get('best_model_name') else all_models)
                selected_models_for_comparison = st.multiselect(
                    "Select models to compare:",
                    options=all_models,
                    default=default_selected_models,
                    key="model_comparison_multiselect"
                )

                if selected_models_for_comparison:
                    comparison_df = metrics_df[metrics_df['Model'].isin(selected_models_for_comparison)].copy()

                    # Columns to highlight (numerical metrics)
                    columns_to_highlight = valid_metrics_to_score.copy()
                    if 'Overall Score' in comparison_df.columns:
                        columns_to_highlight.append('Overall Score')

                    # Create a dictionary of format strings for only the numerical columns
                    format_dict = {col: '{:.4f}' for col in columns_to_highlight}

                    # Display the comparison table with highlighting and specific formatting
                    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=columns_to_highlight,
                                                                   color='#3E4C59').format(format_dict),
                                 use_container_width=True)

                    # Interactive bar chart for selected metrics comparison
                    st.markdown("##### Visual Comparison of Selected Metrics")
                    if not comparison_df.empty and valid_metrics_to_score:
                        metrics_melted_df = comparison_df.melt(id_vars=['Model'], value_vars=valid_metrics_to_score,
                                                               var_name='Metric', value_name='Score')
                        fig_metrics_compare = px.bar(metrics_melted_df, x='Metric', y='Score', color='Model',
                                                     barmode='group',
                                                     title="Comparative Model Performance",
                                                     template="plotly_dark",
                                                     height=500)
                        fig_metrics_compare.update_yaxes(range=[0, 1]) # Metrics typically between 0 and 1
                        st.plotly_chart(fig_metrics_compare, use_container_width=True)
                    else:
                        st.info("Select models and ensure metrics are available for comparison chart.")
                else:
                    st.info("Please select models to view their performance metrics.")

                # Display Hyperparameters of the Recommended Model
                st.markdown("---")
                if st.session_state.get('best_model_name'):
                    with st.expander(f"✨ Hyperparameters of Recommended Model: {st.session_state.best_model_name}",
                                     expanded=False):
                        artifacts = st.session_state.get('artifacts')
                        if artifacts and 'available_models' in artifacts:
                            recommended_model_pipeline = artifacts['available_models'].get(
                                st.session_state.best_model_name)
                            if recommended_model_pipeline:
                                st.markdown("##### Pipeline Steps and Parameters:")
                                if isinstance(recommended_model_pipeline, Pipeline):
                                    for step_name, step_estimator in recommended_model_pipeline.named_steps.items():
                                        st.markdown(f"**Step: {step_name}**")
                                        step_params = step_estimator.get_params()
                                        # Filter out internal/private params and non-serializable types
                                        filtered_params = {k: v for k, v in step_params.items() if
                                                           '__' not in k and not isinstance(v, (type, types.FunctionType))}
                                        if filtered_params:
                                            st.json(filtered_params)
                                        else:
                                            st.write("No specific parameters for this step.")
                                else: # If it's a single estimator
                                    st.markdown(f"**Model: {st.session_state.best_model_name}**")
                                    model_params = recommended_model_pipeline.get_params()
                                    filtered_params = {k: v for k, v in model_params.items() if
                                                       '__' not in k and not isinstance(v, (type, types.FunctionType))}
                                    if filtered_params:
                                        st.json(filtered_params)
                                    else:
                                        st.write("No specific parameters for this model.")
                            else:
                                st.warning("Recommended model pipeline not found in loaded artifacts. It might not have been trained or loaded correctly.")
                        else:
                            st.info("Artifacts not loaded. Run the pipeline to see model hyperparameters.")
                else:
                    st.info(
                        "Run modeling and evaluation phases to determine the best model and view its hyperparameters.")

            # --- Detailed Evaluation Visualizations (Confusion Matrix, Feature Importance, ROC Curve, Error Analysis) ---
            artifacts = st.session_state.get('artifacts') # Get artifacts again for clarity within tabs

            # Proceed only if best model and test data are available
            if (artifacts and 'available_models' in artifacts and st.session_state.get('best_model_name') and
                'X_test' in artifacts and 'y_test' in artifacts and 'label_encoder' in artifacts):

                best_model_name_to_use = st.session_state.best_model_name
                best_model = artifacts['available_models'].get(best_model_name_to_use)

                if best_model:
                    y_true_encoded = artifacts['y_test'] # Keep original encoded for internal model use
                    X_test_eval = artifacts['X_test']
                    label_encoder = artifacts['label_encoder'] # Get the label encoder
                    labels_str = label_encoder.classes_ # These are the string labels for display and cm

                    # Predict on the test set
                    try:
                        y_pred_encoded = best_model.predict(X_test_eval)
                        y_prob = best_model.predict_proba(X_test_eval)

                        # Convert true and predicted labels to strings for confusion matrix and error analysis
                        y_true_str = label_encoder.inverse_transform(y_true_encoded)
                        y_pred_str = label_encoder.inverse_transform(y_pred_encoded)

                    except Exception as pred_eval_err:
                        st.error(f"Error predicting on test data for {best_model_name_to_use}: {pred_eval_err}")
                        best_model = None # Disable further visualizations for this model if prediction fails


                if best_model: # Re-check if model prediction was successful
                    with eval_tabs[1]: # Confusion Matrix Tab
                        st.markdown(f"#### Confusion Matrix for {best_model_name_to_use}")
                        # Use string labels for confusion matrix directly
                        cm = confusion_matrix(y_true_str, y_pred_str, labels=labels_str)
                        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), x=labels_str,
                                           y=labels_str, title=f"Confusion Matrix for {best_model_name_to_use}",
                                           color_continuous_scale=px.colors.sequential.Viridis, template="plotly_dark")
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with eval_tabs[2]: # Feature Importance Tab
                        st.markdown(f"#### Feature Importance for {best_model_name_to_use}")
                        final_estimator = best_model.steps[-1][1] if isinstance(best_model, Pipeline) else best_model
                        if hasattr(final_estimator, 'feature_importances_'):
                            importances = pd.DataFrame({'feature': X_test_eval.columns,
                                                        'importance': final_estimator.feature_importances_}).sort_values(
                                'importance', ascending=False).head(20)
                            fig_imp = px.bar(importances, x='importance', y='feature', orientation='h',
                                             title=f"Top 20 Feature Importances for {best_model_name_to_use}",
                                             color='importance', color_continuous_scale=px.colors.sequential.Cividis_r,
                                             template="plotly_dark")
                            st.plotly_chart(fig_imp, use_container_width=True)
                        else:
                            st.info(f"Feature importances are not available for {type(final_estimator).__name__} model.")

                    with eval_tabs[3]: # ROC Curve Tab
                        st.markdown(f"#### ROC Curves for {best_model_name_to_use}")
                        fig_roc = go.Figure()
                        fig_roc.update_layout(template="plotly_dark", xaxis_title="False Positive Rate",
                                              yaxis_title="True Positive Rate",
                                              title_text=f"ROC Curves for {best_model_name_to_use}")
                        for i, label_str_value in enumerate(labels_str): # Iterate through string label names
                            # Binarize true labels for one-vs-rest ROC, comparing encoded true labels with integer index
                            y_true_binary = (y_true_encoded == i).astype(int)
                            # Ensure the class exists in y_true before calculating ROC
                            if np.sum(y_true_binary) > 0 and y_prob.shape[1] > i:
                                fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
                                fig_roc.add_trace(
                                    go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{label_str_value} (AUC = {auc(fpr, tpr):.2f})'))
                            else:
                                st.warning(f"Skipping ROC for class '{label_str_value}' due to no positive samples or mismatched prediction probabilities.")
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'),
                                                     name='Random (AUC = 0.50)'))
                        st.plotly_chart(fig_roc, use_container_width=True)

                    with eval_tabs[4]: # Error Analysis Tab
                        st.markdown("#### Error Analysis: Common Misclassifications")
                        # Use encoded values for comparison to identify errors
                        errors = y_pred_encoded != y_true_encoded
                        if errors.any(): # Check if there are any misclassifications
                            # Create error_df with string labels for better readability in treemap
                            error_df = pd.DataFrame({
                                'Actual': label_encoder.inverse_transform(y_true_encoded[errors]),
                                'Predicted': label_encoder.inverse_transform(y_pred_encoded[errors])
                            })
                            # Group by actual and predicted to find common misclassification pairs
                            error_counts = error_df.groupby(['Actual', 'Predicted']).size().reset_index(
                                name='Count').sort_values('Count', ascending=False)
                            if not error_counts.empty:
                                fig_err = px.treemap(error_counts, path=['Actual', 'Predicted'], values='Count',
                                                     title="Treemap of Model Errors", color_continuous_scale='Reds',
                                                     template="plotly_dark")
                                st.plotly_chart(fig_err, use_container_width=True)
                            else:
                                st.success("The model made no misclassifications on the test set!")
                        else:
                            st.success("The model made no misclassifications on the test set! Excellent performance!")

                else:
                    st.error(
                        f"Could not load or use the recommended model: {best_model_name_to_use}. It might not have been trained successfully or there was an issue during prediction.")
            else:
                st.info(
                    "Model evaluation details (Confusion Matrix, Feature Importance, ROC Curve, Error Analysis) will be available once the full ML pipeline has been successfully run and a best model is determined.")
