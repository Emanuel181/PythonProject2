# src/data_understanding.py

import os # Imports the 'os' module for interacting with the operating system, such as managing file paths and directories.
from typing import Optional # Imports `Optional` from `typing` to indicate that a function might return a value or `None`.

import matplotlib.pyplot as plt # Imports `matplotlib.pyplot` for creating static, interactive, and animated visualizations.
import numpy as np # Imports `numpy` for numerical operations, especially with arrays.
import pandas as pd # Imports `pandas` for data manipulation and analysis, primarily using DataFrames.
import seaborn as sns # Imports `seaborn` for creating informative statistical graphics based on matplotlib.

from src.config import ( # Imports specific configurations from the project's `config.py` file.
    BENIGN_LABEL, # The label for benign (non-attack) traffic.
    EXPECTED_COLUMNS_AND_DTYPES, # A dictionary defining expected column names and their data types for validation.
    HIGH_CARDINALITY_THRESHOLD, # A threshold used to identify high-cardinality features.
    PROCESSED_DATA_DIR, # The directory path where processed data will be saved.
    REPORTS_DIR, # The directory path where reports (including plots) will be saved.
    TARGET_COLUMN, # The name of the target (label) column in the dataset.
)
from src.utils import ( # Imports utility functions from the project's `utils.py` file.
    display_df_info, # Function to display comprehensive information about a DataFrame.
    load_raw_data, # Function to load raw data from CSV files.
    optimize_dataframe_dtypes, # Function to optimize DataFrame column data types to reduce memory usage.
    plot_and_save, # Function to standardize plotting and saving figures.
    save_dataframe, # Function to save a DataFrame to a CSV file.
    setup_logging, # Function to set up the logging configuration.
    validate_dataframe_schema, # Function to validate the schema of a DataFrame against expected types.
)

logger = setup_logging()  # Get the configured logger # Initializes a logger instance by calling `setup_logging()`, which configures how messages are logged (e.g., to console, file).


def data_understanding_phase() -> Optional[pd.DataFrame]: # Defines the main function for the Data Understanding phase.
    """
    Loads the raw dataset, performs initial exploration, and generates descriptive statistics.
    Identifies data quality issues (missing values, infinities, duplicates, constant columns,
    and highly correlated features). Visualizes key distributions.
    This phase provides foundational insights into the 'Volume' and 'Variety' aspects of the data.
    It also includes initial automated data validation against a predefined schema.


    Returns:
        Optional[pd.DataFrame]: The DataFrame after initial data understanding and basic column cleaning,
                                or None if data loading or critical validation failed.
                                This DataFrame is then passed to the data preparation phase.
    """
    logger.info("--- CRISP-DM Phase 2: Data Understanding ---") # Logs an informational message indicating the start of the Data Understanding phase.

    # Load raw data from multiple CSV files. This step addresses data 'Volume'.
    df_raw = load_raw_data() # Calls the utility function to load raw data into a DataFrame.

    if df_raw.empty: # Checks if the loaded DataFrame is empty.
        logger.error("No data loaded. Exiting Data Understanding phase.") # Logs an error if no data was loaded.
        return None # Returns None, indicating failure to load data.

    # --- Automated Data Validation (NEW) ---
    logger.info("\nPerforming Automated Data Validation...") # Logs the start of automated data validation.
    # Validate the raw data against the expected schema defined in config.py.
    # This ensures consistency of input data and robustness of the pipeline.
    # If validation fails critically, the function returns None, halting the pipeline.
    if not validate_dataframe_schema( # Calls the utility function to validate the DataFrame's schema.
        df_raw, EXPECTED_COLUMNS_AND_DTYPES, df_name="Raw Data" # Passes the raw DataFrame, expected schema, and a name for logging.
    ):
        logger.error( # Logs a critical error if schema validation fails.
            "Raw data schema validation failed. Review data source or expected schema. Aborting project."
        )
        return None # Returns None, halting the project due to critical validation failure.

        # --- Initial Data Overview ---
    logger.info("\nInitial Data Overview (before any cleaning):") # Logs a header for the initial data overview.
    display_df_info(df_raw, "Raw Data Overview") # Calls the utility function to display comprehensive info about the raw DataFrame.

    # Optimize dtypes early to reduce memory footprint for large datasets.
    # This is a crucial step for managing the 'Volume' aspect efficiently.
    df_raw = optimize_dataframe_dtypes(df_raw) # Calls the utility function to optimize DataFrame column data types for memory efficiency.

    # --- Data Quality Checks & Initial Cleaning ---
    logger.info( # Logs the start of data quality checks and initial cleaning.
        "\nPerforming comprehensive Data Quality Checks and Initial Cleaning..."
    )

    # 1. Clean column names (strip whitespace and handle potential duplicates that pandas creates).
    # Pandas sometimes adds '.1' to duplicate column names; this logic standardizes them.
    initial_columns_count = len(df_raw.columns) # Stores the initial count of columns.
    df_raw.columns = ( # Cleans column names by stripping leading/trailing whitespace.
        df_raw.columns.str.strip()
    )  # Remove leading/trailing whitespace from column names

    # Handle pandas-generated duplicated column names (e.g., 'Fwd Header Length' and 'Fwd Header Length.1').
    # A simple strategy: rename by appending unique suffix if an exact duplicate exists after stripping.
    cols = pd.Series( # Converts column names to a Pandas Series for easier manipulation.
        df_raw.columns
    )
    seen = {}  # Dictionary to track seen column names # Initializes a dictionary to keep track of column names encountered.
    new_cols = []  # List to build the new, unique column names # Initializes a list to store the new, standardized column names.
    for col in df_raw.columns: # Iterates through each column name in the DataFrame.
        if col in seen: # Checks if the current column name has been seen before.
            seen[col] += 1 # Increments the count for the duplicate column name.
            new_cols.append( # Appends a new name with a suffix (e.g., '_1', '_2') for duplicates.
                f"{col}_{seen[col]}"
            )  # Append a counter if name is a duplicate
        else:
            seen[col] = 0 # Initializes the count for a new (unique) column name.
            new_cols.append(col)  # Add original name if unique # Appends the original (unique) column name.
    df_raw.columns = ( # Assigns the new list of unique column names back to the DataFrame.
        new_cols
    )
    # Log if any columns were renamed due to duplicates for transparency
    if len(df_raw.columns) < initial_columns_count or any( # Checks if the number of columns changed or if any columns were suffixed.
        col.endswith("_1") or col.endswith("_2") for col in df_raw.columns
    ):
        logger.info( # Logs that column names were cleaned and made unique.
            f"Cleaned and ensured unique column names. Total columns: {len(df_raw.columns)}"
        )
    else:
        logger.info("No duplicate column names found after stripping whitespace.") # Logs if no duplicate column names were found.

    # 2. Identify and handle infinite values (known issue in CIC-IDS2017 data, typically in 'Flow Bytes/s', 'Flow Packets/s').
    problematic_cols = ["Flow Bytes/s", "Flow Packets/s"] # Defines a list of columns known to potentially contain infinite values.
    for col in problematic_cols: # Iterates through each problematic column.
        if col in df_raw.columns: # Checks if the column exists in the DataFrame.
            # First, convert any non-numeric entries (e.g., strings like 'Infinity') to NaN.
            # `errors='coerce'` will replace invalid parsing with NaN.
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce") # Converts the column to numeric, coercing unparseable values to NaN.

            # Now, replace actual infinite numerical values (np.inf, -np.inf) with NaN.
            if (df_raw[col] == np.inf).any() or (df_raw[col] == -np.inf).any(): # Checks if there are any actual infinite numerical values.
                logger.warning( # Logs a warning if infinite values were found and converted.
                    f"Column '{col}' contained infinite values, which were converted to NaN."
                )
            else:
                logger.info( # Logs if no infinite values were found or they were already handled.
                    f"Column '{col}' does not contain infinite values (or they were already handled)."
                )
        else:
            logger.warning(f"Problematic column '{col}' not found in the dataset.") # Logs a warning if a problematic column is not found.

    # 3. Check for missing values after initial handling of infinities/non-numeric conversions.
    # These newly identified NaNs will be handled in the subsequent Data Preparation phase via imputation.
    missing_values = df_raw.isnull().sum() # Calculates the sum of null values for each column.
    missing_values = missing_values[missing_values > 0].sort_values( # Filters to show only columns with missing values and sorts them.
        ascending=False
    )  # Filter to show only columns with NaNs
    if not missing_values.empty: # Checks if any columns have missing values.
        logger.info( # Logs a header for identified missing values.
            "\nMissing values identified (before imputation in Data Preparation):"
        )
        logger.info(missing_values.to_string()) # Logs the missing values count per column.

        # Plot missing values percentage to visually assess data completeness.
        missing_percent = (missing_values / len(df_raw)) * 100 # Calculates the percentage of missing values.
        plot_and_save( # Calls the utility function to plot and save the missing values percentage.
            plot_func=sns.barplot, # Uses seaborn's barplot function.
            filename="missing_values_percentage.png", # Specifies the filename for the saved plot.
            title="Percentage of Missing Values Per Column", # Sets the title of the plot.
            x=missing_percent.index,  # Column names for x-axis # Sets column names as x-axis labels.
            y=missing_percent.values,  # Percentage values for y-axis # Sets percentage values as y-axis labels.
            palette="viridis",  # Color palette for the bars # Sets the color palette for the bars.
            figsize=(12, 6),  # Figure size # Sets the size of the figure.
        )
        plt.xticks( # Adjusts x-axis tick labels.
            rotation=90
        )  # Rotate x-axis labels for better readability if many columns # Rotates labels for readability.

    else:
        logger.info("No missing values detected.") # Logs if no missing values were found.

    # 4. Check for constant columns (features with only one unique value).
    # These columns provide no information for machine learning models and are candidates for removal.
    # `dropna=False` ensures NaN counts as a unique value if present.
    constant_cols = [ # Identifies columns that have only one unique value, excluding the target column.
        col
        for col in df_raw.columns
        if df_raw[col].nunique(dropna=False) == 1 and col != TARGET_COLUMN
    ]
    if constant_cols: # Checks if any constant columns were found.
        logger.info( # Logs the names of the constant columns.
            f"\nColumns with a single unique value (constant columns): {constant_cols}"
        )
        # Note: These columns will be explicitly dropped in the Data Preparation phase.
    else:
        logger.info("No constant columns found.") # Logs if no constant columns were found.

    # 5. Check for highly correlated features among numerical columns.
    # High correlation can indicate redundancy, leading to multicollinearity issues in some models
    # and potentially increasing training time. This identifies candidates for dimensionality reduction.
    numeric_df = df_raw.select_dtypes( # Selects only numerical columns from the raw DataFrame.
        include=np.number
    )  # Select only numerical columns for correlation calculation
    # Ensure there are enough numerical columns to compute correlations
    if not numeric_df.empty and len(numeric_df.columns) > 1: # Checks if there are numerical columns and more than one for correlation.
        corr_matrix = ( # Calculates the absolute correlation matrix.
            numeric_df.corr().abs()
        )  # Calculate absolute correlation matrix to find strong linear relationships
        # Select the upper triangle of the correlation matrix to avoid duplicate pairs and self-correlations.
        upper_tri = corr_matrix.where( # Creates a mask for the upper triangle of the correlation matrix.
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        # Find features with absolute correlation greater than a high threshold (e.g., 0.95).
        to_drop_highly_correlated = [ # Identifies columns that are highly correlated with other columns.
            column for column in upper_tri.columns if any(upper_tri[column] > 0.95)
        ]
        if to_drop_highly_correlated: # Checks if any highly correlated features were found.
            logger.info( # Logs the names of the highly correlated features.
                f"\nHighly correlated features (potential candidates for removal): {to_drop_highly_correlated}"
            )
            # Plot a heatmap of a subset of features, including the highly correlated ones, for visualization.
            # This helps to visually confirm and understand the relationships.
            features_for_heatmap = list( # Creates a list of features to include in the heatmap (highly correlated ones and some others for context).
                set(to_drop_highly_correlated + list(upper_tri.columns)[:20])
            )  # Include up to 20 for context
            plot_and_save( # Calls the utility function to plot and save the heatmap.
                plot_func=sns.heatmap, # Uses seaborn's heatmap function.
                filename="highly_correlated_features_heatmap.png", # Specifies the filename.
                title="Heatmap of Highly Correlated Numerical Features", # Sets the title.
                data=corr_matrix.loc[features_for_heatmap, features_for_heatmap], # Selects a subset of the correlation matrix for plotting.
                annot=True, # Annotates the heatmap with correlation values.
                fmt=".2f", # Formats annotation values to two decimal places.
                cmap="coolwarm", # Sets the colormap.
                linewidths=0.5, # Sets the linewidths between cells.
                figsize=(15, 12), # Sets the figure size.
            )
        else:
            logger.info("No highly correlated features (above 0.95) detected.") # Logs if no highly correlated features were found.
    else:
        logger.warning( # Logs a warning if there are not enough numerical columns for correlation analysis.
            "Not enough numerical columns for correlation analysis or numerical DataFrame is empty."
        )

    # --- Target Variable Distribution ---
    # Analyze the distribution of the target 'Label' column to understand class imbalance,
    # which is a significant aspect of 'Variety' in this dataset and impacts model training strategies.
    if TARGET_COLUMN in df_raw.columns: # Checks if the target column exists in the DataFrame.
        logger.info(f"\nDistribution of '{TARGET_COLUMN}' column:") # Logs a header for the target column distribution.
        label_counts = df_raw[ # Calculates the value counts for each unique label in the target column.
            TARGET_COLUMN
        ].value_counts()  # Get counts for each unique label
        logger.info(label_counts.to_string()) # Logs the label counts.

        # Plot the overall target distribution.
        plot_and_save( # Calls the utility function to plot and save the overall target distribution.
            plot_func=sns.countplot, # Uses seaborn's countplot function.
            filename="label_distribution_raw.png", # Specifies the filename.
            title=f"Distribution of {TARGET_COLUMN} in Raw Data", # Sets the title.
            y=TARGET_COLUMN,  # Use the column name for the y-axis # Sets the target column for the y-axis.
            data=df_raw,  # The full DataFrame # Passes the full DataFrame.
            order=label_counts.index,  # Order bars by frequency (most common first) # Orders bars by frequency.
            palette="viridis", # Sets the color palette.
            figsize=(10, 7), # Sets the figure size.
        )

        # Plot distribution of attack types only (excluding the majority 'BENIGN' class).
        # This provides a clearer view of the attack class proportions.
        attack_labels = label_counts[ # Filters out the benign label to get only attack labels.
            label_counts.index != BENIGN_LABEL
        ]  # Filter out the benign label
        if not attack_labels.empty: # Checks if there are any attack labels after filtering.
            logger.info("\nAttack Type Distribution (excluding BENIGN):") # Logs a header for attack type distribution.
            logger.info(attack_labels.to_string()) # Logs the attack label counts.
            plot_and_save( # Calls the utility function to plot and save the attack type distribution.
                plot_func=sns.countplot, # Uses seaborn's countplot.
                filename="attack_type_distribution.png", # Specifies the filename.
                title="Distribution of Attack Types (Excluding BENIGN)", # Sets the title.
                y=TARGET_COLUMN,  # Use the column name for the y-axis on the filtered data # Sets the target column for the y-axis.
                data=df_raw[ # Passes the DataFrame filtered to exclude benign labels.
                    df_raw[TARGET_COLUMN] != BENIGN_LABEL
                ],  # Filtered DataFrame
                order=attack_labels.index,  # Order bars by frequency # Orders bars by frequency.
                palette="magma",  # Different color palette for distinction # Sets a different color palette.
                figsize=( # Dynamically adjusts figure height for readability.
                    10,
                    len(attack_labels) * 0.5 + 2,
                ),
            )
        else:
            logger.info("No specific attack labels found other than BENIGN.") # Logs if no specific attack labels were found.
    else:
        logger.error( # Logs an error if the target column is not found.
            f"Target column '{TARGET_COLUMN}' not found in DataFrame. Cannot analyze label distribution."
        )

    logger.info("--- Data Understanding Phase Complete ---\n") # Logs the completion of the Data Understanding phase.

    # Save the DataFrame after initial cleaning (column names, dtype optimization) for the next phase.
    save_dataframe(df_raw, PROCESSED_DATA_DIR, "data_understood.csv") # Saves the processed DataFrame to a CSV file.
    return df_raw # Returns the processed DataFrame.


if __name__ == "__main__": # Checks if the script is being run directly (not imported as a module).
    # This block allows running the phase script independently for testing/development.
    df_understood = data_understanding_phase() # Calls the data_understanding_phase function.
    if df_understood is not None: # Checks if the phase completed successfully and returned a DataFrame.
        logger.info( # Logs successful execution for standalone test.
            "Data understanding phase executed successfully for standalone test."
        )