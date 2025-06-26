# src/utils.py

import logging # Imports the standard Python logging library for logging messages.
import os # Imports the 'os' module, which provides functions for interacting with the operating system, such as path manipulation and directory creation.
from typing import Any, Callable, Dict, List, Optional, Tuple # Imports specific type hints for better code readability and maintainability.

import joblib # Imports joblib for efficient serialization and deserialization of Python objects (e.g., trained models, transformers).
import matplotlib.pyplot as plt # Imports `matplotlib.pyplot` for creating plots.
import numpy as np # Imports NumPy for numerical operations, especially array manipulations.
import pandas as pd # Imports Pandas for data manipulation and analysis using DataFrames.
import seaborn as sns # Imports Seaborn for creating informative statistical graphics based on matplotlib.

import src.config as config  # Import config to access dynamic settings like LOG_FILE, REPORTS_DIR etc. # Imports the project's configuration file to access various settings and parameters, such as file paths for logs and reports.



# --- Logging Setup ---
# Revised setup_logging function for robust global configuration
def setup_logging(log_level: Optional[str] = None) -> logging.Logger: # Defines a function to set up the logging configuration.
    """
    Sets up the logging configuration for the project's root logger.
    Logs will be written to a file and streamed to the console.
    This function should be called once, ideally early in main.py, to configure
    the logging system for all modules.

    Args:
        log_level (str, optional): The desired logging level (e.g., 'INFO', 'DEBUG', 'WARNING').
                                   If None, it defaults to config.LOG_LEVEL.
    Returns:
        logging.Logger: The configured root logger instance.
    """
    # Ensure logs directory exists
    log_dir = os.path.dirname(config.LOG_FILE) # Extracts the directory path from the configured log file path.
    os.makedirs(log_dir, exist_ok=True) # Creates the log directory if it doesn't already exist.

    # Convert string log level to logging module's constant
    # Use config.LOG_LEVEL as fallback if not explicitly provided
    effective_log_level = log_level if log_level else config.LOG_LEVEL # Determines the effective log level, prioritizing the provided argument over the config.
    numeric_log_level = getattr(logging, effective_log_level.upper(), logging.INFO) # Converts the string log level (e.g., "INFO") to its corresponding numeric constant (e.g., logging.INFO).

    # Get the root logger
    root_logger = logging.getLogger() # Gets the root logger instance.
    # Set root logger's level to NOTSET to ensure messages from all levels are passed to handlers
    root_logger.setLevel(logging.NOTSET)  # Process all messages initially # Sets the root logger's level to NOTSET, ensuring all messages are passed to its handlers, which then filter by their own levels.

    # Clear existing handlers to prevent duplicate output if called multiple times (e.g. during testing)
    # This is important if main.py is re-executed or if this function is mistakenly called multiple times.
    for handler in list(root_logger.handlers): # Iterates through a copy of the root logger's existing handlers.
        root_logger.removeHandler(handler) # Removes each existing handler to prevent duplicate log output.

    # Create formatter
    formatter = logging.Formatter( # Creates a log formatter to define the format of log messages.
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s" # Specifies the format string including timestamp, logger name, level, and message.
    )

    # File handler
    file_handler = logging.FileHandler( # Creates a file handler to write logs to a file.
        config.LOG_FILE, mode="w" # Specifies the log file path from config and sets mode to 'w' to overwrite the file on each run.
    )
    file_handler.setFormatter(formatter) # Sets the formatter for the file handler.
    file_handler.setLevel(numeric_log_level)  # Set level for file handler # Sets the logging level for messages written to the file.
    root_logger.addHandler(file_handler) # Adds the file handler to the root logger.

    # Stream handler (console)
    stream_handler = logging.StreamHandler() # Creates a stream handler to send logs to the console (stdout/stderr).
    stream_handler.setFormatter(formatter) # Sets the formatter for the stream handler.
    stream_handler.setLevel(numeric_log_level)  # Set level for console handler # Sets the logging level for messages streamed to the console.
    root_logger.addHandler(stream_handler) # Adds the stream handler to the root logger.

    # Set the level for the 'src' hierarchy to match the root.
    logging.getLogger("src").setLevel(numeric_log_level) # Ensures loggers within the 'src' module hierarchy inherit the configured level.

    return root_logger  # Return the configured root logger # Returns the configured root logger instance.


# IMPORTANT: Remove the module-level logger initialization here!
# logger = setup_logging() # This line should be REMOVED or commented out # This line is commented out as it was identified as a potential source of logging issues if not handled carefully.


# --- Data Loading and Saving ---
# All functions below will now get their logger instance when called,
# which will correctly inherit from the root logger configured in main.py.
# So, for example, `logger = logging.getLogger(__name__)` should be added inside each function or class that needs logging.

# For simplicity, and because these functions are called from main, or other functions that will have a logger,
# we can instantiate a logger for utils here, *after* the setup_logging function definition.
# This logger will pick up the root configuration when it's made.
logger = logging.getLogger(__name__) # Instantiates a logger specifically for the 'utils' module. This logger will automatically inherit the configuration set up by `setup_logging`.


def load_raw_data(chunksize: Optional[int] = None) -> pd.DataFrame: # Defines a function to load raw data from CSV files.
    """
    Loads all daily CSV files from the CIC-IDS2017 dataset.
    Handles common issues like inconsistent column names during concatenation.
    This function specifically addresses the 'Volume' and 'Variety' aspects of Big Data
    by efficiently loading potentially large and diverse datasets.

    Args:
        chunksize (int, optional): Number of rows to read at a time for large files.
                                   If None, loads the entire file. Defaults to None.
                                   Using `chunksize` (with `low_memory=True` implicitly)
                                   can help manage memory for very large files.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all loaded files. Returns an empty DataFrame
                      if no files could be loaded or an error occurred during loading.
    """
    all_dfs = [] # Initializes an empty list to store individual DataFrames loaded from CSV files.
    logger.info(f"Loading data from {config.RAW_DATA_DIR}") # Logs the directory from which raw data will be loaded.

    reference_columns = None # Initializes `reference_columns` to None, which will store the columns of the first loaded DataFrame to ensure consistency.

    for file in config.CSV_FILES: # Iterates through each CSV file specified in the project configuration.
        file_path = os.path.join(config.RAW_DATA_DIR, file) # Constructs the full path to the current CSV file.
        if not os.path.exists(file_path): # Checks if the current file exists.
            logger.warning(f"File not found: {file_path}. Skipping.") # Logs a warning if the file is not found.
            continue # Skips to the next file in the loop.
        try: # Begins a try block to handle potential errors during file loading.
            if chunksize: # Checks if a chunksize is specified for reading large files incrementally.
                reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=True) # Creates an iterator for reading the CSV file in chunks, with low_memory mode enabled.
                df_chunks = [] # Initializes a list to store DataFrame chunks.
                for i, chunk in enumerate(reader): # Iterates through each chunk from the CSV file.
                    chunk.columns = chunk.columns.str.strip() # Strips whitespace from column names in the current chunk.
                    if reference_columns is None: # For the very first chunk, sets its columns as the reference.
                        reference_columns = chunk.columns
                    else: # For subsequent chunks, reindexes them to match the reference columns, filling missing with 0.
                        chunk = chunk.reindex(columns=reference_columns, fill_value=0)
                    df_chunks.append(chunk) # Adds the processed chunk to the list.
                    logger.debug( # Logs debug information about the loaded chunk.
                        f"Loaded chunk {i + 1} from {file} with {len(chunk)} rows."
                    )
                df = pd.concat(df_chunks, ignore_index=True) # Concatenates all chunks into a single DataFrame.
            else: # If no chunksize is specified, loads the entire file at once.
                df = pd.read_csv(file_path, low_memory=False) # Reads the entire CSV file into a DataFrame, with low_memory mode disabled for better dtype inference.
                df.columns = df.columns.str.strip() # Strips whitespace from column names.
                if reference_columns is None: # For the first file, sets its columns as the reference.
                    reference_columns = df.columns
                else: # For subsequent files, reindexes them to match the reference columns.
                    df = df.reindex(columns=reference_columns, fill_value=0)

            all_dfs.append(df) # Adds the loaded and processed DataFrame to the list of all DataFrames.
            logger.info(f"Loaded {file} with {len(df)} rows.") # Logs that the file was successfully loaded.
        except pd.errors.EmptyDataError: # Catches the error if a CSV file is empty.
            logger.warning(f"File {file} is empty. Skipping.") # Logs a warning and skips the empty file.
            continue # Continues to the next file.
        except Exception as e: # Catches any other general exception during file loading.
            logger.error(f"Error loading {file}: {e}", exc_info=True) # Logs the error with traceback information.
            continue # Continues to the next file.

    if not all_dfs: # After trying to load all files, checks if the `all_dfs` list is empty (meaning no files were loaded).
        logger.error( # Logs a critical error if no CSV files could be loaded.
            "No CSV files were loaded. Please check data_dir and CSV_FILES in config.py."
        )
        return pd.DataFrame() # Returns an empty DataFrame.

    df_raw = pd.concat(all_dfs, ignore_index=True) # Concatenates all successfully loaded DataFrames into a single raw DataFrame.
    logger.info(f"Total rows after concatenation: {len(df_raw)}") # Logs the total number of rows in the concatenated DataFrame.
    logger.info(f"Total columns: {len(df_raw.columns)}") # Logs the total number of columns in the concatenated DataFrame.
    return df_raw # Returns the concatenated raw DataFrame.


def save_dataframe(df: pd.DataFrame, path: str, filename: str) -> None: # Defines a function to save a Pandas DataFrame to a CSV file.
    """
    Saves a pandas DataFrame to a specified directory path in CSV format.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        path (str): The directory path where the file will be saved.
        filename (str): The name of the CSV file (e.g., 'processed_data.csv').
    """
    os.makedirs(path, exist_ok=True) # Ensures the target directory exists, creating it if necessary.
    full_path = os.path.join(path, filename) # Constructs the full file path for saving.
    try: # Begins a try block to handle potential errors during saving.
        df.to_csv(full_path, index=False) # Saves the DataFrame to a CSV file, without writing the DataFrame index.
        logger.info(f"DataFrame saved successfully to {full_path}") # Logs a success message with the save path.
    except Exception as e: # Catches any exception during saving.
        logger.error(f"Error saving DataFrame to {full_path}: {e}", exc_info=True) # Logs an error message with traceback.


def load_dataframe(path: str, filename: str) -> Optional[pd.DataFrame]: # Defines a function to load a Pandas DataFrame from a CSV file.
    """
    Loads a pandas DataFrame from a specified directory path in CSV format.

    Args:
        path (str): The directory path from which to load the file.
        filename (str): The name of the CSV file.

    Returns:
        Optional[pd.DataFrame]: The loaded DataFrame, or None if the file is not found
                                or an error occurs during loading.
    """
    full_path = os.path.join(path, filename) # Constructs the full file path for loading.
    if not os.path.exists(full_path): # Checks if the file exists at the specified path.
        logger.error(f"Object file not found: {full_path}") # Logs an error if the file is not found.
        return None # Returns None.
    try: # Begins a try block to handle potential errors during loading.
        df = pd.read_csv(full_path) # Reads the CSV file into a DataFrame.
        logger.info(f"DataFrame loaded successfully from {full_path}") # Logs a success message.
        return df # Returns the loaded DataFrame.
    except Exception as e: # Catches any exception during loading.
        logger.error(f"Error loading DataFrame from {full_path}: {e}", exc_info=True) # Logs an error message with traceback.
        return None # Returns None.


def save_object(obj: Any, path: str, filename: str) -> None: # Defines a function to save a Python object using joblib.
    """
    Saves a Python object (e.g., trained model, scaler, label encoder, feature selector)
    to a specified directory path using `joblib` for efficient serialization.

    Args:
        obj (Any): The object to be saved.
        path (str): The directory path where the object file will be saved.
        filename (str): The name of the file (e.g., 'scaler.pkl').
    """
    os.makedirs(path, exist_ok=True) # Ensures the target directory exists.
    full_path = os.path.join(path, filename) # Constructs the full file path for saving the object.
    try: # Begins a try block for saving the object.
        joblib.dump(obj, full_path) # Uses joblib to dump (serialize) the object to the specified path.
        logger.info(f"Object saved successfully to {full_path}") # Logs a success message.
    except Exception as e: # Catches any exception during saving.
        logger.error(f"Error saving object to {full_path}: {e}", exc_info=True) # Logs an error message.


def load_object(path: str, filename: str) -> Optional[Any]: # Defines a function to load a Python object using joblib.
    """
    Loads a Python object (e.g., trained model, scaler, label encoder, feature selector)
    from a specified directory path using `joblib`.

    Args:
        path (str): The directory path from which to load the object.
        filename (str): The name of the file (e.g., 'scaler.pkl').

    Returns:
        Optional[Any]: The loaded object, or None if the file is not found
                               or an error occurs during loading.
    """
    full_path = os.path.join(path, filename) # Constructs the full file path for loading.
    if not os.path.exists(full_path): # Checks if the file exists.
        logger.error(f"Object file not found: {full_path}") # Logs an error if the file is not found.
        return None # Returns None.
    try: # Begins a try block for loading the object.
        obj = joblib.load(full_path) # Uses joblib to load (deserialize) the object from the specified path.
        logger.info(f"Object loaded successfully from {full_path}") # Logs a success message.
        return obj # Returns the loaded object.
    except Exception as e: # Catches any exception during loading.
        logger.error(f"Error loading object from {full_path}: {e}", exc_info=True) # Logs an error message.
        return None # Returns None.


# --- Data Information and Visualization Helpers ---
def display_df_info(df: pd.DataFrame, name: str = "DataFrame") -> None: # Defines a function to display basic information about a DataFrame.
    """
    Displays comprehensive basic information and data quality insights about a DataFrame.
    Includes shape, columns, memory usage, head, data types, and missing values summary.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        name (str): A descriptive name for the DataFrame (for logging purposes).
    """
    logger.info(f"\n--- {name} Info ---") # Logs a header with the DataFrame's name.
    logger.info(f"Shape: {df.shape}") # Logs the shape (number of rows, number of columns) of the DataFrame.
    logger.info(f"Columns: {df.columns.tolist()}") # Logs the list of column names.
    logger.info( # Logs the memory usage of the DataFrame in MB.
        f"Memory usage:\n{df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB"
    )
    logger.info(f"First 5 rows:\n{df.head().to_string()}") # Logs the first 5 rows of the DataFrame.
    logger.info(f"Data types:\n{df.dtypes.to_string()}") # Logs the data types of each column.

    missing_vals = df.isnull().sum() # Calculates the count of null values for each column.
    missing_vals = missing_vals[missing_vals > 0] # Filters to show only columns with at least one missing value.
    if not missing_vals.empty: # Checks if there are any missing values.
        logger.info(f"Missing values:\n{missing_vals.to_string()}") # Logs the count of missing values per column.
    else:
        logger.info("No missing values detected.") # Logs if no missing values are found.

    if config.TARGET_COLUMN in df.columns: # Checks if the target column exists in the DataFrame.
        logger.info( # Logs the unique values and their counts in the target column.
            f"Unique values in '{config.TARGET_COLUMN}':\n{df[config.TARGET_COLUMN].value_counts().to_string()}"
        )
    logger.info(f"--------------------") # Logs a separator.


def plot_and_save( # Defines a function to execute a plotting function and save the resulting plot.
    plot_func: Callable, filename: str, title: str, **kwargs: Any # Accepts the plotting function, filename, title, and arbitrary keyword arguments.
) -> None:
    """
    Executes a plotting function, saves the generated plot to the REPORTS_DIR,
    and closes the plot to free memory. This function standardizes plot saving.

    Args:
        plot_func (Callable): The plotting function (e.g., sns.histplot, plt.bar, sns.countplot, or a Series.plot method).
        filename (str): The desired filename for the saved plot (e.g., 'feature_distribution.png').
        title (str): The title for the plot.
        **kwargs: Arbitrary keyword arguments to pass directly to the plotting function.
                  Includes 'figsize' if a custom figure size is needed.
    """
    plt.figure(figsize=kwargs.pop("figsize", (10, 7))) # Creates a new matplotlib figure, setting its size, and removes 'figsize' from kwargs.

    # --- FIX for FutureWarning: Passing `palette` without assigning `hue` ---
    # This handles both sns.countplot and sns.barplot as they are commonly used with palette
    # If x or y is provided and hue is not, use x/y as hue to avoid the warning.
    # Also ensure legend is False if hue is automatically assigned this way.
    if plot_func in [sns.countplot, sns.barplot]: # Checks if the plotting function is `sns.countplot` or `sns.barplot`.
        if "hue" not in kwargs: # Checks if the 'hue' argument is not already provided in kwargs.
            if "x" in kwargs and kwargs["x"] is not None: # If 'x' is provided and not None.
                kwargs["hue"] = kwargs["x"] # Sets 'hue' to the same as 'x'.
                kwargs["legend"] = False # Disables the legend if 'hue' is automatically assigned this way.
            elif "y" in kwargs and kwargs["y"] is not None: # Else if 'y' is provided and not None.
                kwargs["hue"] = kwargs["y"] # Sets 'hue' to the same as 'y'.
                kwargs["legend"] = False # Disables the legend.
        if "palette" in kwargs and "hue" not in kwargs: # If 'palette' is provided but 'hue' is not (and not set above).
            # If palette is explicitly passed without hue, and x/y aren't suitable for hue,
            # this is where the FutureWarning usually originates.
            # In such cases, we might choose to warn or simply let seaborn handle it if it's not critical.
            # For this context, the above logic should cover common cases.
            pass # No specific action needed here if the above logic handles common warnings.

    try: # Begins a try block to handle potential errors during plotting and saving.
        plot_func(**kwargs) # Executes the provided plotting function with its arguments.
        plt.title(title) # Sets the title of the plot.
        plt.tight_layout() # Adjusts plot parameters for a tight layout.
        plot_path = os.path.join(config.REPORTS_DIR, filename) # Constructs the full file path for saving the plot in the reports directory.
        plt.savefig(plot_path) # Saves the generated plot to the specified path.
        logger.info(f"Saved plot '{title}' to {plot_path}") # Logs a success message with the plot title and path.
    except Exception as e: # Catches any exception during plotting or saving.
        logger.error(f"Error saving plot '{title}': {e}", exc_info=True) # Logs an error message with traceback.
    finally: # Ensures cleanup even if an error occurs.
        plt.close() # Closes the matplotlib figure to free up memory.


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame: # Defines a function to optimize DataFrame column data types.
    """
    Optimizes DataFrame column dtypes to reduce memory usage.
    Converts numerical columns to smaller integer/float types if possible
    and object columns to 'category' if they have low cardinality.
    This is a key step to manage the 'Volume' aspect of Big Data by reducing memory footprint.

    Args:
        df (pd.DataFrame): The DataFrame to optimize.

    Returns:
        pd.DataFrame: The DataFrame with optimized dtypes.
    """
    logger.info("Optimizing DataFrame dtypes...") # Logs the start of dtype optimization.
    initial_memory = df.memory_usage(deep=True).sum() / (1024**2) # Calculates the initial memory usage of the DataFrame in MB.

    for col in df.columns: # Iterates through each column in the DataFrame.
        if col == config.TARGET_COLUMN: # Skips the target column as its dtype might be specifically handled later.
            continue

        col_type = df[col].dtype # Gets the data type of the current column.

        if "object" in str(col_type): # Checks if the column's data type contains 'object' (e.g., strings).
            if ( # Checks if all values in the column are instances of int, float, str, or NaN.
                df[col]
                .apply(lambda x: isinstance(x, (int, float, str)) or pd.isna(x))
                .all()
            ):
                try: # Begins a try block for converting object columns.
                    temp_numeric = pd.to_numeric(df[col], errors="coerce") # Attempts to convert the column to numeric, coercing errors to NaN.
                    if ( # Checks if the column can be downcasted to an integer type.
                        not temp_numeric.isnull().all() # Ensures not all values are NaN.
                        and (temp_numeric.dropna() % 1 == 0).all() # Ensures all non-NaN values are integers.
                    ):
                        df[col] = pd.to_numeric( # Downcasts to the smallest possible integer type.
                            temp_numeric, downcast="integer", errors="ignore"
                        )
                        logger.debug( # Logs the conversion to integer type.
                            f"Converted object column '{col}' to integer type."
                        )
                    elif not temp_numeric.isnull().all(): # Checks if the column can be downcasted to a float type.
                        df[col] = pd.to_numeric( # Downcasts to the smallest possible float type (float32).
                            temp_numeric, downcast="float", errors="ignore"
                        )
                        logger.debug(f"Converted object column '{col}' to float type.") # Logs the conversion to float type.
                    else: # If not convertible to numeric, checks for low cardinality to convert to 'category'.
                        num_unique_values = df[col].nunique(dropna=False) # Counts unique values, including NaN.
                        num_total_values = len(df[col]) # Gets the total number of values in the column.
                        if num_unique_values / num_total_values < 0.5: # If the ratio of unique values is less than 0.5 (low cardinality).
                            df[col] = df[col].astype("category") # Converts the column to 'category' dtype.
                            logger.debug( # Logs the conversion to category type.
                                f"Converted object column '{col}' to category type."
                            )
                        else:
                            logger.debug( # Logs if an object column is kept as object due to high cardinality.
                                f"Column '{col}' is object type with high cardinality, kept as object."
                            )
                except Exception as e: # Catches any exception during object column optimization.
                    logger.warning( # Logs a warning if optimization fails due to unexpected values.
                        f"Failed to optimize object column '{col}' due to unexpected values: {e}"
                    )
            else:
                logger.warning( # Logs a warning if an object column contains complex non-numeric objects.
                    f"Column '{col}' contains complex objects, skipping optimization."
                )

        elif "int" in str(col_type): # Checks if the column's data type contains 'int'.
            c_min = df[col].min() # Gets the minimum value in the column.
            c_max = df[col].max() # Gets the maximum value in the column.
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: # Checks if values fit into int8.
                df[col] = df[col].astype(np.int8) # Converts to int8.
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: # Checks if values fit into int16.
                df[col] = df[col].astype(np.int16) # Converts to int16.
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: # Checks if values fit into int32.
                df[col] = df[col].astype(np.int32) # Converts to int32.
        elif "float" in str(col_type): # Checks if the column's data type contains 'float'.
            df[col] = df[col].astype(np.float32) # Converts float columns to float32 to save memory.

    final_memory = df.memory_usage(deep=True).sum() / (1024**2) # Calculates the final memory usage in MB.
    logger.info( # Logs the memory reduction achieved.
        f"Memory reduced from {initial_memory:.2f} MB to {final_memory:.2f} MB ({100 * (initial_memory - final_memory) / initial_memory:.2f}% reduction)."
    )
    return df # Returns the DataFrame with optimized data types.


def validate_dataframe_schema( # Defines a function to validate the schema of a DataFrame.
    df: pd.DataFrame, expected_schema: Dict[str, str], df_name: str = "DataFrame" # Accepts the DataFrame, expected schema, and a name for logging.
) -> bool:
    """
    Validates the schema of a DataFrame against an expected schema (column names and primary dtype).
    Logs warnings for missing columns, unexpected columns, or dtype mismatches.
    This function is part of 'Automated Data Validation' in the Data Understanding phase.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        expected_schema (Dict[str, str]): A dictionary where keys are expected column names
                                          and values are expected dtype prefixes (e.g., 'int', 'float', 'object').
        df_name (str): A descriptive name for the DataFrame being validated (for logging).

    Returns:
        bool: True if validation passes without critical errors, False otherwise.
    """
    logger.info(f"Performing schema validation for {df_name}...") # Logs the start of schema validation.
    is_valid = True # Initializes a flag to track validation success.
    current_columns = set(df.columns) # Gets the set of current column names in the DataFrame.
    expected_columns = set(expected_schema.keys()) # Gets the set of expected column names from the schema.

    # Check for missing expected columns
    missing_columns = expected_columns - current_columns # Finds columns present in expected schema but missing in the DataFrame.
    if missing_columns: # Checks if any expected columns are missing.
        logger.warning( # Logs a warning about missing expected columns.
            f"Validation Warning: Missing expected columns in {df_name}: {missing_columns}"
        )
        is_valid = False # Sets the validation flag to False.

    # Check for dtype mismatches against expected prefixes
    for col, expected_dtype_prefix in expected_schema.items(): # Iterates through each column and its expected dtype in the schema.
        if col in df.columns: # Checks if the column exists in the DataFrame.
            actual_dtype = str(df[col].dtype) # Gets the actual data type of the column as a string.
            if not actual_dtype.startswith(expected_dtype_prefix): # Checks if the actual dtype string does not start with the expected prefix.
                is_numeric_expected = expected_dtype_prefix in ["int", "float"] # Checks if the expected dtype is numeric.
                is_numeric_actual = actual_dtype.startswith( # Checks if the actual dtype is numeric.
                    "int"
                ) or actual_dtype.startswith("float")

                if is_numeric_expected and is_numeric_actual: # If both expected and actual are numeric.
                    if expected_dtype_prefix == "int" and actual_dtype.startswith( # If expected is int but actual is float.
                        "float"
                    ):
                        if not df[col].dropna().apply(lambda x: x.is_integer()).all(): # Checks if all non-null float values are actually integers.
                            logger.warning( # Logs a warning if float values are not integer-like.
                                f"Validation Warning: Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype_prefix}' in {df_name}. Non-integer float values present."
                            )
                            is_valid = False # Sets validation flag to False.
                        else:
                            logger.debug( # Logs debug info if float values are integer-like and convertible.
                                f"Validation Info: Column '{col}' dtype mismatch ({actual_dtype} vs {expected_dtype_prefix}), but appears convertible to integer. Proceeding."
                            )
                    elif expected_dtype_prefix == "float" and actual_dtype.startswith( # If expected is float and actual is int.
                        "int"
                    ):
                        logger.debug( # Logs debug info as int can be safely cast to float.
                            f"Validation Info: Column '{col}' dtype mismatch ({actual_dtype} vs {expected_dtype_prefix}), but integer can be safely cast to float. Proceeding."
                        )
                    else:
                        logger.warning( # Logs a warning for significant numerical type mismatch.
                            f"Validation Warning: Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype_prefix}' in {df_name}. Significant numerical type mismatch."
                        )
                        is_valid = False # Sets validation flag to False.
                elif not ( # Special handling for object to category conversion.
                    expected_dtype_prefix == "object" and actual_dtype == "category"
                ):
                    if actual_dtype != expected_dtype_prefix and not ( # If actual dtype doesn't match expected and isn't a float-to-int scenario.
                        expected_dtype_prefix.startswith("float") # Check if expected is float.
                        and actual_dtype.startswith("int") # Check if actual is int.
                    ):
                        logger.warning( # Logs a warning for a general type mismatch.
                            f"Validation Warning: Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype_prefix}' in {df_name}. Type mismatch."
                        )
                        is_valid = False # Sets validation flag to False.

    if is_valid: # Checks the final validation flag.
        logger.info(f"Schema validation for {df_name} passed.") # Logs success message if validation passed.
    else:
        logger.error( # Logs an error message if validation failed.
            f"Schema validation for {df_name} failed. Check warnings/errors above."
        )

    return is_valid # Returns the validation result.